# ToS Classifier

Classify Terms-of-Service clauses as **good**, **neutral**, or **bad** with a fine-tuned Legal-BERT model, compared against three classical ML baselines (Logistic Regression, SVM-RBF, Random Forest). Ships with a Gradio demo that segments a pasted ToS / URL, labels each clause, and scores the overall document risk.

---

## Dataset

**Source.** [ToSDR](https://tosdr.org/) API. Fetched asynchronously by `src/fetch_clauses.py` and saved to `data/clauses.csv`.

**Raw size.** 11,729 clause rows across 394 services, but only **950 unique clause strings** — ToSDR clauses are boilerplate reused across many services.

**Columns.** `title` (clause text, model input) · `label` (`good` / `neutral` / `bad`) · `service_id` · `service_name` · `category`. ToSDR's `blocker` class is remapped to `bad` in the fetcher.

**Augmented data.** `data/augmented_datasets/` holds CUAD (commercial contracts) and OPP-115 (privacy policies) mapped to the same three-label schema for cross-domain experiments.

---

## Splits

Two splits live under `data/`:

| Split dir | Dedup? | Rows (train / val / test) | Use case |
|---|---|---|---|
| `data/preprocessed/` | **yes** (by `title`) | 665 / 142 / 143 | Clean, leakage-free evaluation. Canonical. |
| `data/split/` | no | 8,210 / 1,759 / 1,760 | Row-level split for Legal-BERT fine-tuning experiments. Expect leakage. |

Both are stratified 70/15/15 on `label` with seed 42. `src/split_data.py` rebuilds the deduped `data/preprocessed/` split.

### The leakage incident

Initial splits gave macro-F1 ≈ 0.99 on val. Audit found:
- Train had only 765 unique `title`s across 8,210 rows (~10× duplication).
- 76% of val clauses and 79% of test clauses appeared verbatim in train.
- 100% of val/test `service_id`s also appeared in train.

**Root cause.** ToSDR clauses are boilerplate shared across ~400 services; a random row-level split without deduplication leaks identical strings across splits, so the classifier memorizes strings instead of generalizing. Fix: `src/split_data.py` dedupes by `title` before the stratified 70/15/15.

---

## Models

### Classical baselines (`src/models.py`)
All use `class_weight='balanced'` on top of TF-IDF (unigrams + bigrams, 10k max features, `sublinear_tf=True`, English stop-words removed — `src/features.py`).

- **Logistic Regression** — `solver='lbfgs'`, `max_iter=1000`
- **SVM (RBF)** — RBF kernel
- **Random Forest** — 200 trees, `n_jobs=-1`

### Legal-BERT (`src/train.py`)
Fine-tuned from `nlpaueb/legal-bert-base-uncased` with `AutoModelForSequenceClassification` (3 labels, `id2label={0: good, 1: neutral, 2: bad}`). Full fine-tune, class-weighted cross-entropy, linear warmup schedule, AdamW. Checkpoint at `models/legal_bert_checkpoint/` (weights `model.safetensors` are git-ignored).

Train from scratch:
```bash
MODEL_NAME=nlpaueb/legal-bert-base-uncased \
EPOCHS=8 BATCH_SIZE=16 LR=1e-5 \
.venv/Scripts/python.exe -m src.train
```

### Hand-crafted features (`src/preprocess.py`)
Four per-clause scalar features, available for feature-augmentation experiments: `clause_length`, `passive_voice_ratio` (via spaCy `en_core_web_sm`), `legal_keyword_density` (32-term keyword list), `flesch_kincaid_grade` (via `textstat`).

---

## Results

All numbers on the **clean** (deduped) held-out test split, `data/preprocessed/test.csv` (n=143; neutral 57 / bad 49 / good 37).

| Model | Accuracy | Macro F1 | MCC | F1 good | F1 neutral | F1 bad |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8112 | 0.7963 | 0.7216 | 0.7097 | 0.8545 | 0.8246 |
| SVM (RBF) | 0.8392 | 0.8238 | 0.7651 | 0.7302 | 0.8991 | 0.8421 |
| Random Forest | 0.8182 | 0.7995 | 0.7250 | 0.6875 | 0.8621 | 0.8491 |
| **Legal-BERT** | **0.8601** | **0.8570** | **0.7908** | **0.8358** | 0.8649 | **0.8704** |

Legal-BERT wins on accuracy, macro-F1, MCC, and every per-class F1 except `neutral` (where SVM-RBF's 0.8991 edges it out). The biggest gap is on `good` (+0.11 F1 over SVM-RBF) — negation and conditional language benefit the most from contextual embeddings. Confusion matrices under `results/`.

### Validation metrics (for reference)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.8662 | 0.8631 |
| SVM (RBF) | 0.8732 | 0.8704 |
| Random Forest | 0.8310 | 0.8229 |

---

## Repo layout

```
data/
  clauses.csv                        # raw ToSDR dump
  preprocessed/{train,val,test}.csv  # deduped 70/15/15 (canonical)
  split/{train,val,test}.csv         # raw 70/15/15 (no dedup; leaky)
  augmented_datasets/                # CUAD + OPP-115
src/
  fetch_clauses.py    # ToSDR API fetcher
  split_data.py       # dedupe + stratified split
  features.py         # TF-IDF vectorizer
  preprocess.py       # hand-crafted text features
  models.py           # classical baselines
  evaluate.py         # 5-fold CV macro-F1 (baselines)
  confusion.py        # val-set confusion matrix PNGs (baselines)
  test_eval.py        # held-out test evaluation (baselines)
  dataset.py          # HF-style ClauseDataset for Legal-BERT
  train.py            # Legal-BERT fine-tune loop
  eval_legalbert.py   # held-out test evaluation (Legal-BERT)
  inference.py        # single-clause / batch predict via Legal-BERT
  explain.py          # SHAP token attributions for Legal-BERT
  segment.py          # clause segmentation (regex + spaCy)
  fetch_data.py       # URL fetch + segment for the demo
demo/
  app.py              # Gradio demo
models/
  legal_bert_checkpoint/   # fine-tuned weights (safetensors git-ignored)
results/
  legalbert_test/          # Legal-BERT test metrics / report / confmat
  test_*.{json,txt,png,csv}  # baseline per-model test artifacts
  test_summary.csv           # baseline combined summary
  confmat_*.png              # val-set confusion matrices (baselines)
  confusion_matrix.png, umap_embeddings.png, training_curves.png  # Legal-BERT
  shap_clause_*.png          # SHAP token attributions
```

---

## How to run

All scripts use `from src...` imports, so invoke from the project root with `-m`:

```bash
# 1. Rebuild clean splits (optional — already committed)
.venv/Scripts/python.exe -m src.split_data

# 2. Classical baselines
.venv/Scripts/python.exe -m src.evaluate     # 5-fold CV -> results/cv_macro_f1.csv
.venv/Scripts/python.exe -m src.confusion    # val confmats -> results/
.venv/Scripts/python.exe -m src.test_eval    # test-set artifacts -> results/

# 3. Legal-BERT
.venv/Scripts/python.exe -m src.train           # fine-tune -> models/legal_bert_checkpoint/
.venv/Scripts/python.exe -m src.eval_legalbert  # test metrics -> results/legalbert_test/
.venv/Scripts/python.exe -m src.explain         # SHAP plots -> results/

# 4. Gradio demo (requires models/legal_bert_checkpoint/model.safetensors)
.venv/Scripts/python.exe demo/app.py
```

The demo accepts either a pasted ToS text block or a URL, segments it into clauses, predicts `good / neutral / bad` with confidence scores, and renders an overall document risk score. URL fetching uses browser-like headers but sites behind strong bot-protection (Facebook, LinkedIn, Cloudflare challenge pages) will still block — paste the text directly in that case.

---

## Caveats

- **Dataset scale.** 950 unique clauses is small; metrics have non-trivial variance across seeds. Treat single-run numbers as indicative, not definitive.
- **Distribution shift.** ToSDR clauses are editor-surfaced snippets, not natural-distribution ToS prose. Demo predictions on long, real-world ToS paragraphs are less calibrated than test-set numbers suggest.
- **Weights not in git.** `models/legal_bert_checkpoint/model.safetensors` is `.gitignore`d due to size (~440 MB). Re-train via `src.train` or restore from an external copy before running the demo.
