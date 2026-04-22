import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.insert(0, "/content/tos_classifier/src")
from dataset import ClauseDataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (f1_score, accuracy_score,
                             matthews_corrcoef, confusion_matrix,
                             classification_report)
import umap

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = "/content/tos_classifier"
TEST_PATH   = f"{BASE_DIR}/data/preprocessed/test.csv"
MODEL_DIR   = f"{BASE_DIR}/models/legal_bert_checkpoint"
RESULTS_DIR = f"{BASE_DIR}/results"
MAX_LEN     = 256
BATCH_SIZE  = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load Model ─────────────────────────────────────────
print("Loading best checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model     = model.to(device)
model.eval()

# ── Test DataLoader ────────────────────────────────────
test_dataset = ClauseDataset(TEST_PATH, tokenizer, MAX_LEN)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Inference ──────────────────────────────────────────
all_preds, all_labels, all_embeddings = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        cls_emb = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)
label_names    = ["good", "neutral", "bad"]

# ── Metrics ────────────────────────────────────────────
print("\n── Test Set Results ──────────────────────")
print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
print(f"Macro F1 : {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"MCC      : {matthews_corrcoef(all_labels, all_preds):.4f}")
print("\nPer-class Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# ── Confusion Matrix ───────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix — Legal-BERT")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
print("Confusion matrix saved ✓")

# ── UMAP ───────────────────────────────────────────────
print("\nRunning UMAP...")
reducer       = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(all_embeddings)

colors = {0: "green", 1: "gold", 2: "red"}
plt.figure(figsize=(9, 6))
for label_id, label_name in enumerate(label_names):
    idx = np.array(all_labels) == label_id
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                c=colors[label_id], label=label_name, alpha=0.5, s=10)
plt.title("UMAP of Legal-BERT CLS Embeddings")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/umap_embeddings.png")
print("UMAP saved ✓")

# ── Export Predictions for Member B ───────────────────
test_df = pd.read_csv(TEST_PATH)
test_df["predicted_label"] = [label_names[p] for p in all_preds]
test_df["true_label"]      = [label_names[l] for l in all_labels]
test_df.to_csv(f"{RESULTS_DIR}/test_predictions.csv", index=False)
print("test_predictions.csv exported for Member B ✓")