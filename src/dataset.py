import torch
from torch.utils.data import Dataset
import pandas as pd

LABEL_MAP = {"good": 0, "neutral": 1, "bad": 2}

class ClauseDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["title", "label"])
        self.df = self.df[self.df["label"].isin(LABEL_MAP.keys())]
        self.df = self.df.reset_index(drop=True)

        self.texts  = self.df["title"].tolist()
        self.labels = self.df["label"].map(LABEL_MAP).tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(self.df["label"].value_counts())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }