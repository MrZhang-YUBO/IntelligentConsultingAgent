# -*- coding: utf-8 -*-
"""
Exp3 - SFT LoRA 推断脚本（LoRA + 基座模型）

功能：
1) 读取 experiments/exp3/data/qa_eval_500.json 的 "query" 做推断
2) 加载基座模型 + LoRA adapter（不 merge）
3) 保存到 experiments/exp3/results/
   字段：query / answer / LLM_answer
4) 路径不写死：统一使用 src.common.paths（由 LLM_EXPERIMENT 控制）
5) 关键配置前置，可用环境变量覆盖

运行方式（项目根目录）：
  export LLM_EXPERIMENT=exp3
  python -m src.exp3.sft.infer_lora
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】先设置实验选择（paths.py import 时会读环境变量）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp3")

# 现在再 import paths（非常关键）
from src.common.paths import data_path, prompt_path, results_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 配置前置区：数据 / prompt / 模型 / 生成参数 / 日志
# =========================================================
import argparse
import json
import time
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------
# 数据集：和 qa_testset_500 同目录，只是命名不同
# -------------------------
DATA_FILE = os.getenv("EVAL_FILE", "qa_eval_500.json")  # -> experiments/exp3/data/qa_eval_500.json

# -------------------------
# Prompt（可选）：不存在就用默认
# -------------------------
PROMPT_FILE = os.getenv("PROMPT_FILE", "baseline_qa.txt")  # -> experiments/exp3/prompts/baseline_qa.txt

DEFAULT_PROMPT = """你是一名法律咨询助手。

你的任务是：
针对用户提出的法律相关问题，提供法律原则说明和建议。

请严格遵守以下规则：

1. 简明解释法律原则、制度，然后给出常见处理思路，不要编造具体事实。
2. 回答避免冗长，不要超过150词。

例子：
输入："我在餐厅吃饭，滑倒摔断了腿，餐厅地板确实很滑且没放提示牌，我可以索赔吗？",
输出："根据《民法典》，宾馆、商场、餐馆等经营场所的经营者负有安全保障义务。未尽到义务导致他人损害的，应当承担侵权责任。您可以主张医药费、护理费、误工费等赔偿。"

用户问题：
{query}

请按照以上要求，给出你的回答：
"""

# -------------------------
# 模型配置（环境变量可覆盖）
# -------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("LORA_PATH", "")  # 必填：例如 saves/Qwen2.5-7B-Instruct/lora/train_xxx
MODEL_TAG = os.getenv("MODEL_TAG", "qwen2.5_7b_lora")  # 输出文件 tag

DTYPE = os.getenv("DTYPE", "float16")  # float16 / bfloat16 / float32
USE_CHAT_TEMPLATE = os.getenv("USE_CHAT_TEMPLATE", "1") == "1"

# -------------------------
# 生成参数（环境变量可覆盖）
# -------------------------
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.8"))
DO_SAMPLE = os.getenv("DO_SAMPLE", "1") == "1"

# -------------------------
# 日志/进度
# -------------------------
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
PREVIEW_EVERY = int(os.getenv("PREVIEW_EVERY", "50"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "220"))


# =========================================================
# 2) 工具函数：读取 prompt / 读取数据 / 构造输入 / 推断
# =========================================================
def load_prompt_text() -> str:
    """
    优先从 experiments/exp3/prompts/<PROMPT_FILE> 读取；
    不存在则用 DEFAULT_PROMPT。
    """
    p = prompt_path(PROMPT_FILE)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if "{query}" not in txt:
            raise ValueError(f"Prompt 文件必须包含 '{{query}}' 占位符：{p}")
        return txt
    return DEFAULT_PROMPT


def load_evalset() -> List[Dict]:
    """
    读取 qa_eval_500.json（JSON 数组）
    - 每条至少包含 query
    - answer 作为参考答案保留，但不参与模型输入
    """
    p = data_path(DATA_FILE)
    if not p.exists():
        raise FileNotFoundError(f"未找到评估集文件：{p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{DATA_FILE} 必须是 JSON 数组（list）。")

    items = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        q = obj.get("query", "")
        if isinstance(q, str) and q.strip():
            items.append({
                "query": q.strip(),
                "answer": obj.get("answer", "")
            })

    if not items:
        raise ValueError("评估集中没有找到有效的 'query' 字段。")
    return items


def build_model_input(tokenizer, prompt_tpl: str, query: str, use_chat_template: bool) -> str:
    """
    构造模型输入：
    - 优先走 chat_template（Qwen instruct 更稳）
    - 不可用则回退为纯文本 prompt
    """
    user_text = prompt_tpl.format(query=query)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass

    return user_text


@torch.inference_mode()
def generate_one(model, tokenizer, input_text: str) -> str:
    """
    单条推断：只解码“新生成”的 token，避免把 prompt 一起解码进回答。
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_len:]

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def _torch_dtype(dtype: str):
    if dtype.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype.lower() in ("fp16", "float16"):
        return torch.float16
    return torch.float32


# =========================================================
# 3) 主流程：加载 -> 推断 -> 保存
# =========================================================
def run_lora_infer(base_model: str, lora_path: str, model_tag: str) -> str:
    if not lora_path:
        raise ValueError("必须通过环境变量或参数指定 LORA_PATH（LoRA adapter 目录）。")

    items = load_evalset()
    prompt_tpl = load_prompt_text()

    out_dir = ensure_dir(results_path())  # -> experiments/exp3/results
    out_file = out_dir / f"answer_{model_tag}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype(DTYPE)

    print("=" * 80)
    print("[INFO] Exp3 SFT LoRA Inference（LoRA + 基座）")
    print(f"[INFO] LLM_EXPERIMENT = {os.environ.get('LLM_EXPERIMENT')}")
    print(f"[INFO] 评估集：{data_path(DATA_FILE)} | 样本数：{len(items)}")
    pf = prompt_path(PROMPT_FILE)
    print(f"[INFO] Prompt：{pf if pf.exists() else 'DEFAULT_PROMPT'}")
    print(f"[INFO] Base model：{base_model}")
    print(f"[INFO] LoRA path：{lora_path}")
    print(f"[INFO] device：{device} | dtype：{DTYPE} | chat_template：{USE_CHAT_TEMPLATE}")
    print(f"[INFO] gen：max_new_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, top_p={TOP_P}, do_sample={DO_SAMPLE}")
    print(f"[INFO] 输出：{out_file}")
    print("=" * 80)

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # 2) base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    # 3) attach LoRA
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()

    results: List[Dict] = []
    start = time.time()

    for idx, obj in enumerate(items, start=1):
        query = obj["query"]
        ref_answer = obj.get("answer", "")

        input_text = build_model_input(tokenizer, prompt_tpl, query, use_chat_template=USE_CHAT_TEMPLATE)
        llm_answer = generate_one(model, tokenizer, input_text)

        results.append({"query": query, "answer": ref_answer, "LLM_answer": llm_answer})

        if idx % LOG_EVERY == 0 or idx == len(items):
            elapsed = time.time() - start
            speed = idx / elapsed if elapsed > 0 else 0.0
            print(f"[PROGRESS] {idx}/{len(items)} | {speed:.2f} 条/秒 | elapsed {elapsed:.1f}s")

        if idx == 1 or idx % PREVIEW_EVERY == 0:
            pq = query[:PREVIEW_CHARS] + ("…" if len(query) > PREVIEW_CHARS else "")
            pa = llm_answer[:PREVIEW_CHARS] + ("…" if len(llm_answer) > PREVIEW_CHARS else "")
            print("\n[示例输出]")
            print("Q:", pq)
            print("A:", pa)
            print()

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 已保存：{out_file}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Exp3 SFT LoRA Inference")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="基座模型（HF repo 或本地路径）")
    parser.add_argument("--lora_path", type=str, default=LORA_PATH, help="LoRA adapter 目录（必填）")
    parser.add_argument("--model_tag", type=str, default=MODEL_TAG, help="输出文件 tag")
    args = parser.parse_args()

    run_lora_infer(args.base_model, args.lora_path, args.model_tag)


if __name__ == "__main__":
    main()
