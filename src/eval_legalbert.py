"""Held-out test-set evaluation for the Legal-BERT checkpoint.

Loads models/legal_bert_checkpoint/, runs inference on data/split/test.csv
(overridable via TEST_CSV env var), and writes metrics + artifacts into
results/legalbert_test/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import ClauseDataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get(
    "MODEL_DIR", PROJECT_ROOT / "models" / "legal_bert_checkpoint"
))
TEST_CSV = Path(os.environ.get(
    "TEST_CSV", PROJECT_ROOT / "data" / "split" / "test.csv"
))
RESULTS_DIR = PROJECT_ROOT / "results" / "legalbert_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 256
BATCH = 32
LABEL_NAMES = ["good", "neutral", "bad"]  # order must match id2label


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device    : {device}")
    print(f"model dir : {MODEL_DIR}")
    print(f"test csv  : {TEST_CSV}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device).eval()

    ds = ClauseDataset(str(TEST_CSV), tokenizer, MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)

    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds.extend(out.logits.argmax(dim=1).cpu().tolist())
            gold.extend(batch["labels"].tolist())

    labels_idx = [0, 1, 2]
    acc = accuracy_score(gold, preds)
    prec_macro = precision_score(gold, preds, average="macro", zero_division=0)
    rec_macro = recall_score(gold, preds, average="macro", zero_division=0)
    f1_macro = f1_score(gold, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(gold, preds)
    per_class_f1 = f1_score(gold, preds, labels=labels_idx, average=None, zero_division=0)
    report = classification_report(
        gold, preds, labels=labels_idx, target_names=LABEL_NAMES,
        digits=4, zero_division=0,
    )
    cm = confusion_matrix(gold, preds, labels=labels_idx)

    print("\n== Legal-BERT test-set results ==")
    print(f"n                : {len(gold)}")
    print(f"accuracy         : {acc:.4f}")
    print(f"precision (macro): {prec_macro:.4f}")
    print(f"recall    (macro): {rec_macro:.4f}")
    print(f"macro F1         : {f1_macro:.4f}")
    print(f"MCC              : {mcc:.4f}")
    print("per-class F1     :",
          {n: round(float(s), 4) for n, s in zip(LABEL_NAMES, per_class_f1)})
    print("\n" + report)
    print(f"confusion matrix (rows=true, cols=pred, order={LABEL_NAMES}):")
    print(cm)

    metrics = {
        "model": "legal_bert",
        "model_dir": str(MODEL_DIR),
        "test_csv": str(TEST_CSV),
        "n": int(len(gold)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "mcc": float(mcc),
        "f1_per_class": {
            n: float(s) for n, s in zip(LABEL_NAMES, per_class_f1)
        },
        "class_labels": LABEL_NAMES,
        "confusion_matrix": cm.tolist(),
    }
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    with (RESULTS_DIR / "test_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Legal-BERT test-set evaluation\n")
        f.write(f"model_dir : {MODEL_DIR}\n")
        f.write(f"test_csv  : {TEST_CSV}\n")
        f.write(f"n         : {len(gold)}\n\n")
        f.write(f"accuracy         : {acc:.4f}\n")
        f.write(f"precision (macro): {prec_macro:.4f}\n")
        f.write(f"recall    (macro): {rec_macro:.4f}\n")
        f.write(f"macro F1         : {f1_macro:.4f}\n")
        f.write(f"MCC              : {mcc:.4f}\n\n")
        f.write("per-class F1:\n")
        for n, s in zip(LABEL_NAMES, per_class_f1):
            f.write(f"  {n:8s}: {float(s):.4f}\n")
        f.write("\n" + report)
        f.write(f"\nconfusion matrix (rows=true, cols=pred, order={LABEL_NAMES}):\n")
        f.write(str(cm) + "\n")

    pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        RESULTS_DIR / "test_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_title("Legal-BERT — Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_confmat.png", dpi=150)
    plt.close(fig)

    df = pd.read_csv(TEST_CSV).dropna(subset=["title", "label"])
    df = df[df["label"].isin(LABEL_NAMES)].reset_index(drop=True)
    df["true_label"] = [LABEL_NAMES[i] for i in gold]
    df["predicted_label"] = [LABEL_NAMES[i] for i in preds]
    df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    print(f"\nsaved artifacts -> {RESULTS_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
