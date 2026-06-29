# -*- coding: utf-8 -*-
"""
Build SFT dataset from:
- alpaca_gpt4_data_zh.json
- lawzhidao_best_sft_all.json
- self_instruct_gen_qwen32b.jsonl

Output:
- experiments/exp3/data/sft/train.jsonl
- experiments/exp3/data/sft/val.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from src.common.paths import data_path, ensure_dir


SEED = 42
VAL_RATIO = 0.1  # 10% validation


def load_json(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def basic_filter(item: Dict[str, Any]) -> bool:
    """
    Minimal sanity check for SFT samples
    """
    ins = item.get("instruction", "").strip()
    out = item.get("output", "").strip()

    if not ins or not out:
        return False
    if len(out) < 15:
        return False
    return True


def main():
    random.seed(SEED)

    # -------- paths --------
    alpaca_path = data_path("sft", "alpaca_gpt4_data_zh.json")
    law_path = data_path("sft", "lawzhidao_best_sft_all.json")
    self_inst_path = data_path("sft", "self_instruct_gen_qwen32b.jsonl")

    out_dir = data_path("sft")
    ensure_dir(out_dir)

    # -------- load --------
    alpaca = load_json(alpaca_path)
    law = load_json(law_path)
    self_inst = load_jsonl(self_inst_path)

    print(f"[INFO] alpaca: {len(alpaca)}")
    print(f"[INFO] law: {len(law)}")
    print(f"[INFO] self-instruct: {len(self_inst)}")

    # -------- merge --------
    all_data = alpaca + law + self_inst

    # -------- filter --------
    all_data = [x for x in all_data if basic_filter(x)]
    print(f"[INFO] after filter: {len(all_data)}")

    # -------- shuffle --------
    random.shuffle(all_data)

    # -------- split --------
    n_total = len(all_data)
    n_val = int(n_total * VAL_RATIO)

    val_data = all_data[:n_val]
    train_data = all_data[n_val:]

    # -------- write --------
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for x in train_data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for x in val_data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"[DONE] train: {len(train_data)} -> {train_path}")
    print(f"[DONE] val:   {len(val_data)} -> {val_path}")


if __name__ == "__main__":
    main()
