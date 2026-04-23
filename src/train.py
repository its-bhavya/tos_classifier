import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, "/content/tos_classifier/src")
from dataset import ClauseDataset

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import f1_score

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = "/content/tos_classifier"
TRAIN_PATH  = f"{BASE_DIR}/data/preprocessed/train.csv"
VAL_PATH    = f"{BASE_DIR}/data/preprocessed/val.csv"
MODEL_NAME  = "nlpaueb/legal-bert-base-uncased"
SAVE_DIR    = f"{BASE_DIR}/models/legal_bert_checkpoint"
RESULTS_DIR = f"{BASE_DIR}/results"

# ── Hyperparameters ────────────────────────────────────
EPOCHS     = 15
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 256
WARMUP_PCT = 0.1

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Tokenizer ──────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Datasets ───────────────────────────────────────────
train_dataset = ClauseDataset(TRAIN_PATH, tokenizer, MAX_LEN)
val_dataset   = ClauseDataset(VAL_PATH,   tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── Class Weights ──────────────────────────────────────
label_counts = pd.read_csv(TRAIN_PATH)["label"].map(
    {"good": 0, "neutral": 1, "bad": 2}
).value_counts().sort_index()

class_weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float)
class_weights = class_weights / class_weights.sum()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class_weights = class_weights.to(device)

# ── Model ──────────────────────────────────────────────
print("Loading Legal-BERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)

# Freeze bottom 6 transformer layers
for i, layer in enumerate(model.bert.encoder.layer):
    if i < 6:
        for param in layer.parameters():
            param.requires_grad = False

model = model.to(device)

# ── Optimizer & Scheduler ──────────────────────────────
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_PCT * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ── Training Loop ──────────────────────────────────────
train_losses, val_f1s = [], []
best_val_f1 = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # Train
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(f"  Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validate
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average="macro")
    val_f1s.append(val_f1)

    print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}\n")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ Best model saved (F1={best_val_f1:.4f})")

# ── Plot Training Curves ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, marker='o', color='blue')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2.plot(range(1, EPOCHS+1), val_f1s, marker='o', color='green')
ax2.set_title("Validation Macro F1")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1 Score")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png")
print(f"\nTraining curves saved ✓")
print(f"Best Val F1: {best_val_f1:.4f}")