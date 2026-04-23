"""Stratified 5-fold CV (macro-F1) for classical baselines.

TF-IDF is fit inside each fold via a Pipeline so vocabulary from the
held-out fold never leaks into training. Writes per-fold scores plus
mean/std to `results/cv_macro_f1.csv`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.features import build_tfidf_vectorizer
from src.models import all_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"


def cross_validate_macro_f1(
    texts,
    labels,
    models=None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    models = models or all_models()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = {}
    for name, est in models.items():
        pipe = Pipeline(
            [("tfidf", build_tfidf_vectorizer()), ("clf", clone(est))]
        )
        rows[name] = cross_val_score(
            pipe, texts, labels, cv=cv, scoring="f1_macro", n_jobs=-1
        )

    out = pd.DataFrame(rows).T
    out.columns = [f"fold_{i + 1}" for i in range(n_splits)]
    fold_cols = list(out.columns)
    out["mean"] = out[fold_cols].mean(axis=1)
    out["std"] = out[fold_cols].std(axis=1)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["title", "label"])

    cv_scores = cross_validate_macro_f1(
        train_df["title"].astype(str), train_df["label"]
    )
    print("── Classical Baselines: 5-fold CV (macro-F1) ─────")
    print(cv_scores.round(4).to_string())

    out_path = RESULTS_DIR / "cv_macro_f1.csv"
    cv_scores.to_csv(out_path)
    print(f"\nsaved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
