# -*- coding: utf-8 -*-
"""
Exp3 - DPO LoRA 推断脚本（云友好）

功能：
1) 读取 experiments/exp3/data/qa_eval_500.json 的 query 做推断
2) 加载 base model + LoRA adapter（不 merge）
3) 支持两种模式（用环境变量切换）：
   - MODE=dpo:         base + DPO LoRA
   - MODE=sft_dpo:     base + SFT LoRA + DPO LoRA（两层叠加）
4) 保存到 experiments/exp3/results/
   字段：query / answer / LLM_answer
5) 路径不写死：统一使用 src.common.paths（由 LLM_EXPERIMENT 控制）
6) 关键配置前置，可用环境变量覆盖

运行方式（项目根目录）：
  export LLM_EXPERIMENT=exp3
  # 模式A：base + dpo lora
  export MODE=dpo
  export BASE_MODEL=/mnt/systemDisk/model_cache/Qwen/Qwen2.5-7B-Instruct
  export DPO_LORA_PATH=/mnt/workspace/LLM--/DPO_LoRA_exp1   # 你的 webui 输出目录
  python -m src.exp3.dpo.infer_dpo_lora

  # 模式B：base + sft lora + dpo lora
  export MODE=sft_dpo
  export SFT_LORA_PATH=/path/to/sft_lora
  export DPO_LORA_PATH=/path/to/dpo_lora
  python -m src.exp3.dpo.infer_dpo_lora
"""

from __future__ import annotations

# =========================================================
# 0) 【必须】先锁定实验目录（paths.py import 时会读环境变量）
# =========================================================
import os
os.environ.setdefault("LLM_EXPERIMENT", "exp3")

from src.common.paths import data_path, prompt_path, results_path, ensure_dir  # noqa: E402

# =========================================================
# 1) 配置前置区：数据 / prompt / 模型 / 生成参数 / 日志
# =========================================================
import argparse
import json
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -------------------------
# 运行模式：dpo / sft_dpo
# -------------------------
MODE = os.getenv("MODE", "dpo").strip().lower()  # dpo | sft_dpo

# -------------------------
# 数据集
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
1) 简明解释法律原则、制度，然后给出常见处理思路，不要编造具体事实。
2) 回答避免冗长，不要超过150词。

用户问题：
{query}

请按照以上要求，给出你的回答：
"""

# -------------------------
# 模型路径（环境变量可覆盖）
# -------------------------
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# DPO LoRA（必填：你截图里的输出目录）
DPO_LORA_PATH = os.getenv("DPO_LORA_PATH", "DPO_LoRA_exp1")

# SFT LoRA（仅 MODE=sft_dpo 需要）
SFT_LORA_PATH = os.getenv("SFT_LORA_PATH", "")

# 输出 tag（可覆盖）
MODEL_TAG = os.getenv("MODEL_TAG", "").strip()

# dtype / chat template
DTYPE = os.getenv("DTYPE", "float16")  # float16 / bfloat16 / float32
USE_CHAT_TEMPLATE = os.getenv("USE_CHAT_TEMPLATE", "1") == "1"

# -------------------------
# 生成参数
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
# 2) 工具函数：读取 prompt / 数据 / 构造输入 / 推断
# =========================================================
def load_prompt_text() -> str:
    p = prompt_path(PROMPT_FILE)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if "{query}" not in txt:
            raise ValueError(f"Prompt 文件必须包含 '{{query}}' 占位符：{p}")
        return txt
    return DEFAULT_PROMPT


def load_evalset() -> List[Dict]:
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
            items.append({"query": q.strip(), "answer": obj.get("answer", "")})
    if not items:
        raise ValueError("评估集中没有找到有效的 'query' 字段。")
    return items


def build_model_input(tokenizer, prompt_tpl: str, query: str, use_chat_template: bool) -> str:
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
    gen_ids = outputs[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _torch_dtype(dtype: str):
    d = (dtype or "").lower()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _infer_tag(mode: str, base_model: str, sft_lora: str, dpo_lora: str) -> str:
    """
    输出文件 tag 的默认规则（可用 MODEL_TAG 覆盖）：
    - dpo:      qwen2.5_7b__dpo_lora_<dpo_dirname>
    - sft_dpo:  qwen2.5_7b__sft_lora_<sft_dirname>__dpo_lora_<dpo_dirname>
    """
    if MODEL_TAG:
        return MODEL_TAG

    base_name = os.path.basename(base_model.rstrip("/")) or "base"
    base_name = base_name.replace("Qwen2.5-7B-Instruct", "qwen2.5_7b").replace("Qwen2.5-3B-Instruct", "qwen2.5_3b")
    base_name = base_name.replace("/", "_")

    dpo_name = os.path.basename(dpo_lora.rstrip("/")) or "dpo"
    if mode == "sft_dpo":
        sft_name = os.path.basename(sft_lora.rstrip("/")) or "sft"
        return f"{base_name}__sft_lora_{sft_name}__dpo_lora_{dpo_name}"
    return f"{base_name}__dpo_lora_{dpo_name}"


def _resolve_path(p: str) -> str:
    """
    允许传相对路径（相对项目根目录运行时的 cwd），也允许绝对路径。
    不存在就直接报错，避免默默用错目录。
    """
    pp = os.path.expanduser(p)
    # 如果是相对路径，保持相对，但做 exists 检查
    if os.path.isabs(pp):
        if not os.path.exists(pp):
            raise FileNotFoundError(f"LoRA 路径不存在：{pp}")
        return pp
    # 相对路径：直接用当前工作目录拼出来（你在项目根目录运行时最合理）
    abs_p = os.path.abspath(pp)
    if not os.path.exists(abs_p):
        raise FileNotFoundError(f"LoRA 路径不存在：{abs_p}（由 {pp} 解析）")
    return abs_p


# =========================================================
# 3) 主流程：加载 -> 推断 -> 保存
# =========================================================
def run_infer(base_model: str, mode: str, sft_lora_path: str, dpo_lora_path: str) -> str:
    items = load_evalset()
    prompt_tpl = load_prompt_text()

    # 解析 LoRA 路径
    dpo_lora_abs = _resolve_path(dpo_lora_path)
    sft_lora_abs: Optional[str] = None
    if mode == "sft_dpo":
        if not sft_lora_path:
            raise ValueError("MODE=sft_dpo 时必须提供 SFT_LORA_PATH。")
        sft_lora_abs = _resolve_path(sft_lora_path)

    # 输出命名
    tag = _infer_tag(mode, base_model, sft_lora_abs or "", dpo_lora_abs)

    out_dir = ensure_dir(results_path())  # -> experiments/exp3/results
    out_file = out_dir / f"answer_{tag}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _torch_dtype(DTYPE)

    print("=" * 80)
    print("[INFO] Exp3 DPO LoRA Inference")
    print(f"[INFO] LLM_EXPERIMENT = {os.environ.get('LLM_EXPERIMENT')}")
    print(f"[INFO] MODE = {mode}")
    print(f"[INFO] Eval set: {data_path(DATA_FILE)} | n={len(items)}")
    pf = prompt_path(PROMPT_FILE)
    print(f"[INFO] Prompt: {pf if pf.exists() else 'DEFAULT_PROMPT'}")
    print(f"[INFO] Base model: {base_model}")
    if sft_lora_abs:
        print(f"[INFO] SFT LoRA:  {sft_lora_abs}")
    print(f"[INFO] DPO LoRA:  {dpo_lora_abs}")
    print(f"[INFO] device={device} | dtype={DTYPE} | chat_template={USE_CHAT_TEMPLATE}")
    print(f"[INFO] gen: max_new_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE}, top_p={TOP_P}, do_sample={DO_SAMPLE}")
    print(f"[INFO] Output: {out_file}")
    print("=" * 80)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    # attach LoRA(s)
    model = base
    if sft_lora_abs:
        model = PeftModel.from_pretrained(model, sft_lora_abs)
    model = PeftModel.from_pretrained(model, dpo_lora_abs)
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
            print("\n[PREVIEW]")
            print("Q:", pq)
            print("A:", pa)
            print()

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved: {out_file}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Exp3 DPO LoRA Inference")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--mode", type=str, default=MODE, choices=["dpo", "sft_dpo"])
    parser.add_argument("--sft_lora_path", type=str, default=SFT_LORA_PATH)
    parser.add_argument("--dpo_lora_path", type=str, default=DPO_LORA_PATH)
    args = parser.parse_args()

    run_infer(args.base_model, args.mode, args.sft_lora_path, args.dpo_lora_path)


if __name__ == "__main__":
    main()
