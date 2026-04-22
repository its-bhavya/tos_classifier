# src/inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ID2LABEL = {0: "GOOD", 1: "NEUTRAL", 2: "BAD"}
MODEL_PATH = "models/legal_bert_checkpoint"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        print("Model loaded.")


def predict_clause(clause_text: str) -> dict:
    _load_model()
    inputs = _tokenizer(
        clause_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

    predicted_id = probs.argmax().item()
    label = ID2LABEL[predicted_id]
    confidence = probs[predicted_id].item()

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {
            "GOOD": round(probs[0].item(), 4),
            "NEUTRAL": round(probs[1].item(), 4),
            "BAD": round(probs[2].item(), 4),
        }
    }


def predict_batch(clauses: list) -> list:
    return [predict_clause(c) for c in clauses]


if __name__ == "__main__":
    test_clauses = [
        "You can delete your account at any time.",
        "We may share your data with third parties at our sole discretion.",
        "You waive your right to participate in class action lawsuits.",
    ]
    for clause in test_clauses:
        result = predict_clause(clause)
        print(f"\n{clause[:60]}")
        print(f"  → {result['label']} ({result['confidence']:.1%})")