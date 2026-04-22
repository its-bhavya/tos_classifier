# src/explain.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
import torch
import matplotlib.pyplot as plt
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


def run_shap_analysis():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    def predict_proba(texts):
        inputs = tokenizer(list(texts), return_tensors="pt",
                           truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        return F.softmax(logits, dim=-1).numpy()

    explainer = shap.Explainer(predict_proba, tokenizer)
    shap_values = explainer(EXAMPLE_CLAUSES)

    for i, clause in enumerate(EXAMPLE_CLAUSES):
        print(f"Generating SHAP plot for clause {i+1}...")
        plt.figure(figsize=(12, 3))
        shap.plots.text(shap_values[i, :, 2], display=False)
        plt.title(f"Clause {i+1} — SHAP for BAD class\n'{clause[:70]}...'", fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"shap_clause_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    print("SHAP analysis complete. Check results/ folder.")


if __name__ == "__main__":
    run_shap_analysis()