# -*- coding: utf-8 -*-
"""
DPO Step-1: Generate candidate answers A/B using Qwen2.5-32B (API),
with prompt fully externalized.

Input:
- experiments/exp3/data/dpo/intent_2k.json
  [{"问题": "...", "类型": "法律类"}, ...]

Prompt:
- experiments/exp3/prompts/dpo_candidate_prompt.txt

Output:
- experiments/exp3/data/dpo/raw/dpo_candidates_ab_qwen32b.jsonl
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List

from openai import OpenAI

from src.common.paths import data_path, prompt_path, ensure_dir


# -------------------------
# Config (env)
# -------------------------
QUESTION_FILE = os.getenv("DPO_QUESTION_FILE", "intent_2k.json")
QUESTION_FIELD = os.getenv("DPO_QUESTION_FIELD", "问题")
TYPE_FIELD = os.getenv("DPO_TYPE_FIELD", "类型")
TYPE_VALUE = os.getenv("DPO_TYPE_VALUE", "法律类")

PROMPT_FILE = os.getenv("DPO_CAND_PROMPT_FILE", "dpo_candidate_prompt.txt")

N = int(os.getenv("DPO_N", "20"))
SEED = int(os.getenv("DPO_SEED", "42"))

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_TEACHER_MODEL", "qwen2.5-32b-instruct")

MAX_TOKENS = int(os.getenv("DPO_MAX_TOKENS", "400"))
TOP_P = float(os.getenv("DPO_TOP_P", "0.9"))

TEMP_A = float(os.getenv("DPO_TEMP_A", "0.2"))
TEMP_B = float(os.getenv("DPO_TEMP_B", "0.9"))

SLEEP = float(os.getenv("DPO_SLEEP", "0.3"))
MAX_RETRIES = int(os.getenv("DPO_MAX_RETRIES", "4"))

OUT_JSONL = os.getenv("DPO_CAND_OUT", "dpo_candidates_ab_qwen32b.jsonl")


# -------------------------
# Utils
# -------------------------
def _clean(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\u200b", "").replace("\ufeff", "").strip()


def load_questions() -> List[str]:
    path = data_path("dpo", QUESTION_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Missing question file: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Question file must be a JSON list")

    qs = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get(TYPE_FIELD) != TYPE_VALUE:
            continue
        q = _clean(item.get(QUESTION_FIELD))
        if q:
            qs.append(q)

    # 去重但保序
    seen = set()
    uniq = []
    for q in qs:
        if q not in seen:
            uniq.append(q)
            seen.add(q)

    return uniq


def load_prompt_template() -> str:
    p = prompt_path(PROMPT_FILE)
    if not p.exists():
        raise FileNotFoundError(f"Missing prompt file: {p}")
    return p.read_text(encoding="utf-8")


def call_llm(
    client: OpenAI,
    prompt_tmpl: str,
    question: str,
    temperature: float,
) -> str:
    content = prompt_tmpl.format(question=question)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            wait = 0.8 * (2 ** (attempt - 1))
            print(f"[WARN] retry {attempt}/{MAX_RETRIES}: {e} -> sleep {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"API failed after retries: {last_err}")


# -------------------------
# Main
# -------------------------
def main():
    if not API_KEY:
        raise ValueError("Missing DASHSCOPE_API_KEY")

    random.seed(SEED)

    questions = load_questions()
    if not questions:
        raise ValueError("No valid legal questions found")

    take_n = min(N, len(questions))
    sampled = random.sample(questions, k=take_n)

    prompt_tmpl = load_prompt_template()

    out_path = data_path("dpo", "raw", OUT_JSONL)
    ensure_dir(out_path.parent)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"[INFO] Questions total={len(questions)}, sample={take_n}")
    print(f"[INFO] Model={MODEL}")
    print(f"[INFO] Prompt={PROMPT_FILE}")
    print(f"[INFO] Temps A={TEMP_A}, B={TEMP_B}")
    print(f"[INFO] Output={out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(sampled, 1):
            a = call_llm(client, prompt_tmpl, q, TEMP_A)
            time.sleep(SLEEP)
            b = call_llm(client, prompt_tmpl, q, TEMP_B)
            time.sleep(SLEEP)

            row = {
                "id": f"q_{i:06d}",
                "instruction": q,
                "input": "",
                "cand_a": a,
                "cand_b": b,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == take_n:
                print(f"[OK] progress {i}/{take_n}")

    print("[DONE] DPO candidates generated.")


if __name__ == "__main__":
    main()
