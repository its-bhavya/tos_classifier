"""Held-out test-set evaluation for all classical baselines.

Trains each model in `src.models.all_models()` on `train.csv`, evaluates on
`test.csv`, and writes per-model artifacts plus a combined summary into
`results/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.features import build_tfidf_vectorizer
from src.models import all_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"


def _evaluate_one(name, est, X_tr, y_tr, X_te, y_te, class_labels):
    pipe = Pipeline([("tfidf", build_tfidf_vectorizer()), ("clf", clone(est))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    prec_macro = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_te, y_pred)

    per_class_f1 = f1_score(
        y_te, y_pred, labels=class_labels, average=None, zero_division=0
    )
    per_class_f1_dict = {c: float(s) for c, s in zip(class_labels, per_class_f1)}

    full_report = classification_report(y_te, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=class_labels)

    return {
        "name": name,
        "acc": acc,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "f1_macro": f1_macro,
        "mcc": mcc,
        "per_class_f1": per_class_f1_dict,
        "full_report": full_report,
        "cm": cm,
    }


def _save_artifacts(r, class_labels, train_size, test_size, y_te_counts):
    name = r["name"]
    cm_list = r["cm"].tolist()

    metrics = {
        "model": name,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "class_labels": class_labels,
        "test_class_counts": {str(k): int(v) for k, v in y_te_counts.items()},
        "accuracy": float(r["acc"]),
        "precision_macro": float(r["prec_macro"]),
        "recall_macro": float(r["rec_macro"]),
        "f1_macro": float(r["f1_macro"]),
        "mcc": float(r["mcc"]),
        "f1_per_class": r["per_class_f1"],
        "confusion_matrix": cm_list,
    }
    (RESULTS_DIR / f"test_metrics_{name}.json").write_text(
        json.dumps(metrics, indent=2)
    )

    txt_path = RESULTS_DIR / f"test_report_{name}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Held-out test-set evaluation — {name}\n")
        f.write(f"train size: {train_size}   test size: {test_size}\n\n")
        f.write("test class counts:\n")
        f.write(y_te_counts.to_string() + "\n\n")
        f.write(f"accuracy         : {r['acc']:.4f}\n")
        f.write(f"precision (macro): {r['prec_macro']:.4f}\n")
        f.write(f"recall    (macro): {r['rec_macro']:.4f}\n")
        f.write(f"macro F1         : {r['f1_macro']:.4f}\n")
        f.write(f"MCC              : {r['mcc']:.4f}\n\n")
        f.write("per-class F1:\n")
        for cls, s in r["per_class_f1"].items():
            f.write(f"  {cls:8s}: {s:.4f}\n")
        f.write("\n" + r["full_report"])
        f.write(f"\nconfusion matrix (rows=true, cols=pred, order={class_labels}):\n")
        f.write(str(r["cm"]) + "\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        r["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_title(f"Test-set Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"test_confmat_{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv").dropna(subset=["title", "label"])
    test = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["title", "label"])

    X_tr, y_tr = train["title"].astype(str), train["label"]
    X_te, y_te = test["title"].astype(str), test["label"]
    class_labels = sorted(y_tr.unique())
    y_te_counts = y_te.value_counts()

    print(f"train size: {len(train)}   test size: {len(test)}")
    print("test class counts:")
    print(y_te_counts.to_string())
    print()

    summary_rows = []
    for name, est in all_models().items():
        r = _evaluate_one(name, est, X_tr, y_tr, X_te, y_te, class_labels)
        print(f"=== {name} ===")
        print(f"  accuracy         : {r['acc']:.4f}")
        print(f"  precision (macro): {r['prec_macro']:.4f}")
        print(f"  recall    (macro): {r['rec_macro']:.4f}")
        print(f"  macro F1         : {r['f1_macro']:.4f}")
        print(f"  MCC              : {r['mcc']:.4f}")
        print("  per-class F1:", {c: round(s, 4) for c, s in r["per_class_f1"].items()})
        print()

        _save_artifacts(r, class_labels, len(train), len(test), y_te_counts)

        row = {
            "model": name,
            "accuracy": r["acc"],
            "precision_macro": r["prec_macro"],
            "recall_macro": r["rec_macro"],
            "f1_macro": r["f1_macro"],
            "mcc": r["mcc"],
        }
        for cls, s in r["per_class_f1"].items():
            row[f"f1_{cls}"] = s
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("model").round(4)
    print("── Summary ──")
    print(summary.to_string())

    summary_csv = RESULTS_DIR / "test_summary.csv"
    summary.to_csv(summary_csv)
    print(f"\nsaved {summary_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
