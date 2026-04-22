# src/explain.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "models/legal_bert_checkpoint"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EXAMPLE_CLAUSES = [
    "You can delete your account and all associated data at any time.",
    "This agreement is governed by the laws of the State of California.",
    "We may share your personal data with third parties at our sole discretion.",
    "You waive your right to participate in class action lawsuits.",
    "By continuing to use this service, you accept any changes to these terms.",
]

LABEL_NAMES = ["GOOD", "NEUTRAL", "BAD"]


def plot_shap_tokens(tokens, shap_vals, clause_text, clause_idx, save_dir):
    """
    Manually draw token-level SHAP attributions as a colored bar chart.
    Red = pushes toward BAD, Blue = pushes away from BAD.
    """
    # Clean up tokens (remove special tokens like [CLS], [SEP])
    clean_tokens = []
    clean_vals = []
    for tok, val in zip(tokens, shap_vals):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        # Remove ## wordpiece prefix for display
        display_tok = tok.replace("##", "")
        clean_tokens.append(display_tok)
        clean_vals.append(val)

    if not clean_tokens:
        print(f"  No tokens to plot for clause {clause_idx+1}")
        return

    vals = np.array(clean_vals)
    colors = ["#d32f2f" if v > 0 else "#1565c0" for v in vals]  # red=BAD, blue=GOOD

    fig, ax = plt.subplots(figsize=(max(10, len(clean_tokens) * 0.55), 3.5))
    bars = ax.bar(range(len(clean_tokens)), vals, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(clean_tokens)))
    ax.set_xticklabels(clean_tokens, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("SHAP Value (impact on BAD class)", fontsize=9)
    ax.set_title(
        f"Clause {clause_idx+1} — SHAP Token Attribution for BAD class\n\"{clause_text[:80]}...\"",
        fontsize=10, pad=10
    )

    # Legend
    red_patch  = mpatches.Patch(color="#d32f2f", label="Pushes toward BAD")
    blue_patch = mpatches.Patch(color="#1565c0", label="Pushes toward GOOD/NEUTRAL")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="upper right")

    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"shap_clause_{clause_idx+1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_shap_analysis():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    def predict_proba(texts):
        inputs = tokenizer(
            list(texts), return_tensors="pt",
            truncation=True, max_length=256, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return F.softmax(logits, dim=-1).detach().numpy()

    # Use a masker so SHAP knows how to handle text
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker)

    print("Running SHAP (this may take a few minutes)...")
    shap_values = explainer(EXAMPLE_CLAUSES)

    for i, clause in enumerate(EXAMPLE_CLAUSES):
        print(f"Generating plot for clause {i+1}...")

        # shap_values[i] has shape (n_tokens, n_classes)
        # We want class index 2 = BAD
        token_shap = shap_values[i].values[:, 2]   # SHAP vals for BAD class
        token_names = shap_values[i].data           # the actual token strings

        plot_shap_tokens(
            tokens=token_names,
            shap_vals=token_shap,
            clause_text=clause,
            clause_idx=i,
            save_dir=RESULTS_DIR
        )

    print("\nDone! Check the results/ folder.")


if __name__ == "__main__":
    run_shap_analysis()