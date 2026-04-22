"""Rebuild clean train/val/test splits from data/clauses.csv.

Original splits leaked: the same clause text appears in multiple splits
because ToSDR clauses are boilerplate shared across ~400 services.
This script:
  1. deduplicates by `title` (each unique clause kept once),
  2. stratified-splits 70/15/15 by `label` with a fixed seed,
  3. writes data/preprocessed/{train,val,test}.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "clauses.csv"
OUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
SEED = 42


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    before = len(df)
    # Each title maps to a single label (verified), so drop_duplicates on title
    # gives one row per clause and preserves label integrity.
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    print(f"rows after dedupe: {len(df)}  (was {before})")

    # 70 / 15 / 15 stratified by label.
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part)}  labels: {part.label.value_counts().to_dict()}")

    # Leakage sanity check.
    t = set(train.title); v = set(val.title); te = set(test.title)
    print(f"\noverlap  train∩val={len(t & v)}  train∩test={len(t & te)}  val∩test={len(v & te)}")


if __name__ == "__main__":
    main()
