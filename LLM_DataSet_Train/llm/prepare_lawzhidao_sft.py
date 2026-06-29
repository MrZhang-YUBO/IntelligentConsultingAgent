# src/exp3/datasets/prepare_lawzhidao_sft.py
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

import pandas as pd

from src.common.paths import data_path, ensure_dir

CSV_REL_PATH = ("sft", "lawzhidao_filter.csv")

OUT_SAMPLE200 = ("sft", "lawzhidao_best_sample200.json")
OUT_SFT_ALL = ("sft", "lawzhidao_best_sft_all.json")


def _clean_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # 简单去掉常见的空白/不可见字符
    s = s.replace("\u200b", "").replace("\ufeff", "").strip()
    return s


def main(seed: int = 42, sample_n: int = 200) -> None:
    csv_path: Path = data_path(*CSV_REL_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Please place it at: experiments/<LLM_EXPERIMENT>/data/sft/lawzhidao_filter.csv"
        )

    # 读取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")

    required_cols = {"title", "question", "reply", "is_best"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Got columns: {list(df.columns)}")

    # 过滤 is_best == 1（兼容字符串/数字）
    df_best = df[df["is_best"].astype(str).str.strip().isin(["1", "True", "true"])].copy()

    # 清洗关键字段
    df_best["title"] = df_best["title"].apply(_clean_text)
    df_best["reply"] = df_best["reply"].apply(_clean_text)

    # 丢弃空 title 或空 reply
    before = len(df_best)
    df_best = df_best[(df_best["title"] != "") & (df_best["reply"] != "")]
    after = len(df_best)

    if after == 0:
        raise ValueError("No valid rows after filtering is_best==1 and removing empty title/reply.")

    print(f"[INFO] Loaded: {len(df)} rows")
    print(f"[INFO] is_best==1: {before} rows")
    print(f"[INFO] valid (non-empty title/reply): {after} rows")

    # 输出 1：采样 200 条（若不足则全量）
    random.seed(seed)
    indices = list(range(after))
    random.shuffle(indices)
    take_n = min(sample_n, after)
    sample_df = df_best.iloc[indices[:take_n]]

    sample_200 = [
        {
            "query": row["title"],
            "answer": row["reply"],
        }
        for _, row in sample_df.iterrows()
    ]

    # 输出 2：全部 best，用于 SFT
    sft_all = [
        {
            "instruction": row["title"],
            "input": "",
            "output": row["reply"],
        }
        for _, row in df_best.iterrows()
    ]

    # 写文件（确保目录存在）
    out_dir = ensure_dir(data_path("sft"))
    out_sample_path = out_dir / OUT_SAMPLE200[-1]
    out_sft_all_path = out_dir / OUT_SFT_ALL[-1]

    out_sample_path.write_text(json.dumps(sample_200, ensure_ascii=False, indent=2), encoding="utf-8")
    out_sft_all_path.write_text(json.dumps(sft_all, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote sample200 JSON: {out_sample_path}  (n={len(sample_200)})")
    print(f"[OK] Wrote SFT all JSON: {out_sft_all_path}  (n={len(sft_all)})")


if __name__ == "__main__":
    main()
