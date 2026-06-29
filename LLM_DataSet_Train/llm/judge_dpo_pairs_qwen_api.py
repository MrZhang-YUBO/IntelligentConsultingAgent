# -*- coding: utf-8 -*-
"""
DPO Step-2: Judge A/B candidates -> chosen/rejected (NO reason)
Using Qwen API (DashScope OpenAI-compatible)
Prompt is externalized.

Input:
- experiments/exp3/data/dpo/raw/dpo_candidates_ab_qwen32b.jsonl

Output:
- experiments/exp3/data/dpo/processed/dpo_pairs_qwen32b.jsonl
  Each line:
  {"instruction": "...", "input": "", "chosen": "...", "rejected": "..."}
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI

from src.common.paths import data_path, prompt_path, ensure_dir


# -------------------------
# Config
# -------------------------
IN_JSONL = os.getenv("DPO_CAND_IN", "dpo_candidates_ab_qwen32b.jsonl")
OUT_JSONL = os.getenv("DPO_PAIR_OUT", "dpo_pairs_qwen32b.jsonl")

PROMPT_FILE = os.getenv("DPO_JUDGE_PROMPT_FILE", "dpo_judge_prompt.txt")

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_TEACHER_MODEL", "qwen2.5-32b-instruct")

# judge params
TEMPERATURE = float(os.getenv("DPO_JUDGE_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("DPO_JUDGE_TOP_P", "1.0"))
MAX_TOKENS = int(os.getenv("DPO_JUDGE_MAX_TOKENS", "128"))

SLEEP = float(os.getenv("DPO_JUDGE_SLEEP", "0.2"))
MAX_RETRIES = int(os.getenv("DPO_JUDGE_MAX_RETRIES", "4"))


# -------------------------
# Utils
# -------------------------
def _clean(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\u200b", "").replace("\ufeff", "").strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from model output.
    Accepts:
    - pure JSON
    - JSON wrapped by extra text
    """
    if not text:
        return None

    t = _clean(text)

    # direct parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # regex fallback
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _judge_once(
    client: OpenAI,
    prompt_tmpl: str,
    question: str,
    a: str,
    b: str,
) -> str:
    """
    Return 'A' or 'B'
    """
    prompt = prompt_tmpl.format(
        question=question,
        answer_a=a,
        answer_b=b,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            text = resp.choices[0].message.content or ""
            obj = _extract_json(text)
            if not obj:
                raise ValueError("Judge output is not valid JSON")

            better = _clean(obj.get("better", "")).upper()
            if better not in ("A", "B"):
                raise ValueError(f"Invalid better value: {better}")

            return better
        except Exception as e:
            last_err = e
            wait = 0.8 * (2 ** (attempt - 1))
            print(f"[WARN] retry {attempt}/{MAX_RETRIES}: {e} -> sleep {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Judge failed after retries: {last_err}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    if not API_KEY:
        raise ValueError("Missing DASHSCOPE_API_KEY")

    in_path = data_path("dpo", "raw", IN_JSONL)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {in_path}")

    prompt_p = prompt_path(PROMPT_FILE)
    if not prompt_p.exists():
        raise FileNotFoundError(f"Missing judge prompt: {prompt_p}")
    prompt_tmpl = prompt_p.read_text(encoding="utf-8")

    out_path = data_path("dpo", OUT_JSONL)
    ensure_dir(out_path.parent)

    # resume support
    done = 0
    if out_path.exists():
        done = sum(1 for _ in out_path.open("r", encoding="utf-8"))

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"[INFO] In:  {in_path}")
    print(f"[INFO] Out: {out_path}")
    print(f"[INFO] Model: {MODEL}")
    print(f"[INFO] Resume from line: {done}")

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("a", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx < done:
                continue

            row = json.loads(line)
            q = _clean(row.get("instruction", ""))
            a = _clean(row.get("cand_a", ""))
            b = _clean(row.get("cand_b", ""))

            # basic sanity check
            if not q or not a or not b or a == b:
                continue

            better = _judge_once(client, prompt_tmpl, q, a, b)

            chosen = a if better == "A" else b
            rejected = b if better == "A" else a

            out = {
                "instruction": q,
                "input": "",
                "chosen": chosen,
                "rejected": rejected,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"[OK] judged {idx + 1} lines")

            time.sleep(SLEEP)

    print("[DONE] DPO judge finished.")


if __name__ == "__main__":
    main()
