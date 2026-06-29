# src/exp3/datasets/generate_self_instruct_qwen_api.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.common.paths import data_path, prompt_path, ensure_dir


# -----------------------
# Config (env-friendly)
# -----------------------
SEED_FILE = os.getenv("SELF_INSTRUCT_SEED_FILE", "seed_sft_200.json")
PROMPT_FILE = os.getenv("SELF_INSTRUCT_PROMPT_FILE", "self_instruct_sft.txt")

OUT_JSONL = os.getenv("SELF_INSTRUCT_OUT_JSONL", "self_instruct_gen_qwen32b.jsonl")

N_SAMPLES = int(os.getenv("SELF_INSTRUCT_N_SAMPLES", "500"))  # 先500，稳定后再加
FEWSHOT_K = int(os.getenv("SELF_INSTRUCT_FEWSHOT_K", "4"))    # 每次抽4条seed做示例
SEED_RANDOM = int(os.getenv("SELF_INSTRUCT_SEED", "42"))

TEMPERATURE = float(os.getenv("SELF_INSTRUCT_TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("SELF_INSTRUCT_TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("SELF_INSTRUCT_MAX_TOKENS", "512"))

# DashScope OpenAI-compatible
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL = os.getenv("QWEN_TEACHER_MODEL", "qwen2.5-32b-instruct")

# Retry / throttling
MAX_RETRIES = int(os.getenv("SELF_INSTRUCT_MAX_RETRIES", "5"))
SLEEP_BASE = float(os.getenv("SELF_INSTRUCT_SLEEP_BASE", "0.8"))


# -----------------------
# Helpers
# -----------------------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def dump_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def safe_strip_invisible(s: str) -> str:
    return s.replace("\u200b", "").replace("\ufeff", "").strip()

def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从模型输出中提取第一个 JSON 对象（兜底：模型偶尔会多说一句）
    """
    if not text:
        return None
    t = safe_strip_invisible(text)

    # 1) 如果本身就是JSON对象
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 正则提取第一个 {...}
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)

    # 3) 再解析
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def validate_sft_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    强制输出符合你的SFT schema:
    - instruction: str
    - input: "" (固定空串)
    - output: str
    """
    if not isinstance(item, dict):
        return None
    ins = safe_strip_invisible(str(item.get("instruction", "")))
    out = safe_strip_invisible(str(item.get("output", "")))
    if not ins or not out:
        return None

    # 过滤模板化废话（可按需加更多规则）
    bad_prefix = ["作为一个AI", "作为AI", "作为一名AI", "我无法提供法律建议"]
    if any(ins.startswith(x) for x in bad_prefix) or any(out.startswith(x) for x in bad_prefix):
        return None

    # 长度简单阈值（太短基本没训练价值）
    if len(out) < 20:
        return None

    return {"instruction": ins, "input": "", "output": out}

def build_user_prompt(prompt_template: str, seed_examples: List[Dict[str, Any]]) -> str:
    seed_json = json.dumps(seed_examples, ensure_ascii=False, indent=2)
    return prompt_template.replace("<<<SEED_EXAMPLES_JSON>>>", seed_json)


def main() -> None:
    if not API_KEY:
        raise ValueError("Missing DASHSCOPE_API_KEY env var.")

    random.seed(SEED_RANDOM)

    # paths
    seed_path = data_path("sft", SEED_FILE)
    tmpl_path = prompt_path(PROMPT_FILE)
    out_path = data_path("sft", OUT_JSONL)
    ensure_dir(out_path.parent)

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}")
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {tmpl_path}")

    seed_data = load_json(seed_path)
    if not isinstance(seed_data, list) or len(seed_data) == 0:
        raise ValueError("Seed JSON must be a non-empty list of SFT items.")

    prompt_template = tmpl_path.read_text(encoding="utf-8")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 断点续跑：统计已生成多少条（按行数）
    done = 0
    if out_path.exists():
        done = sum(1 for _ in out_path.open("r", encoding="utf-8"))
    target_total = N_SAMPLES
    if done >= target_total:
        print(f"[INFO] Already have {done} lines in {out_path}, target={target_total}. Nothing to do.")
        return

    print(f"[INFO] Seed size = {len(seed_data)}")
    print(f"[INFO] Output jsonl = {out_path}")
    print(f"[INFO] Base URL = {BASE_URL}")
    print(f"[INFO] Model = {MODEL}")
    print(f"[INFO] Resume from done={done}, generating {target_total - done} more...")

    i = done
    while i < target_total:
        fewshot = random.sample(seed_data, k=min(FEWSHOT_K, len(seed_data)))
        user_prompt = build_user_prompt(prompt_template, fewshot)

        # 调用 + 重试
        ok_item = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                )
                text = resp.choices[0].message.content or ""
                obj = extract_first_json_object(text)
                ok_item = validate_sft_item(obj or {})
                if ok_item is None:
                    raise ValueError("Invalid/empty item from model output.")
                break
            except Exception as e:
                wait = SLEEP_BASE * (2 ** (attempt - 1))
                print(f"[WARN] gen failed (i={i}, attempt={attempt}/{MAX_RETRIES}): {e} -> sleep {wait:.1f}s")
                time.sleep(wait)

        if ok_item is None:
            print(f"[ERROR] Skip i={i}: failed after retries.")
            i += 1
            continue

        dump_jsonl_line(out_path, ok_item)
        i += 1
        if i % 20 == 0:
            print(f"[OK] progress: {i}/{target_total}")

    print(f"[DONE] Generated {target_total} items -> {out_path}")


if __name__ == "__main__":
    main()
