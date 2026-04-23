"""Generate confusion-matrix PNGs for the classical baselines.

Trains each model from `src.models.all_models()` on the train split, predicts
on the validation split, and saves `results/val/confmat_<name>.png` per model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.features import build_tfidf_vectorizer
from src.models import all_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results" / "val"


def _plot_confusion(cm, labels, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrices(
    train_csv: Path = DATA_DIR / "train.csv",
    val_csv: Path = DATA_DIR / "val.csv",
    out_dir: Path = RESULTS_DIR,
    text_col: str = "title",
    label_col: str = "label",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv).dropna(subset=[text_col, label_col])
    val_df = pd.read_csv(val_csv).dropna(subset=[text_col, label_col])

    X_train = train_df[text_col].astype(str)
    y_train = train_df[label_col]
    X_val = val_df[text_col].astype(str)
    y_val = val_df[label_col]

    class_labels = sorted(y_train.unique())

    for name, est in all_models().items():
        pipe = Pipeline(
            [("tfidf", build_tfidf_vectorizer()), ("clf", clone(est))]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        cm = confusion_matrix(y_val, y_pred, labels=class_labels)

        out_path = out_dir / f"confmat_{name}.png"
        _plot_confusion(cm, class_labels, f"Confusion Matrix — {name}", out_path)
        print(f"saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    save_confusion_matrices()
