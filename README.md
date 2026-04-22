# ToS Classifier

Classify Terms-of-Service clauses as **good**, **neutral**, or **bad** using classical ML baselines (Logistic Regression, SVM-RBF, Random Forest) and, on a separate track, Legal-BERT.

This README documents the classical-baseline pipeline end to end — dataset, steps, and results — including a leakage incident that was found and fixed along the way.

---

## Dataset

**Source.** [ToSDR](https://tosdr.org/) API. Fetched asynchronously by `src/fetch_clauses.py` and saved to `data/clauses.csv`.

**Raw size.** 11,729 clause rows across 394 services, but only **950 unique clause strings** — ToSDR clauses are boilerplate reused across many services.

**Columns.**
| Column | Description |
|---|---|
| `title` | The clause text (model input) |
| `label` | `good` / `neutral` / `bad` |
| `service_id` | Stable numeric id of the service |
| `service_name` | Human-readable service name |
| `category` | Free-text category from ToSDR (often identical to `title`) |

Label mapping note: ToSDR's `blocker` class is remapped to `bad` inside the fetcher.

**Augmented data.** `data/augmented_datasets/` additionally contains CUAD (commercial contracts) and OPP-115 (privacy policies) mapped to the same three-label schema. These are used on the Legal-BERT track for cross-domain experiments and are **not** part of the classical-baseline numbers below.

---

## Pipeline

### 1. Fetch raw clauses — `src/fetch_clauses.py`
Async ToSDR pull (8-way concurrency, exponential backoff). Filters for comprehensively reviewed services and writes `data/clauses.csv`.

### 2. Clean train/val/test split — `src/split_data.py`
Deduplicates by `title` (11,729 → 950 unique clauses), then stratified 70/15/15 on `label` with seed 42. Writes `data/preprocessed/{train,val,test}.csv`. Supersedes the earlier `data/preprocessing/preprocessing.ipynb` notebook.

Resulting split sizes: **665 / 142 / 143**. Zero overlap across splits.

### 3. Hand-crafted text features — `src/preprocess.py`
Four per-clause scalar features, intended to complement the TF-IDF representation:

| Feature | What it measures |
|---|---|
| `clause_length(text)` | Word count via regex. |
| `passive_voice_ratio(text)` | Fraction of sentences with a passive dependency label (`nsubjpass` / `auxpass` / `csubjpass`), via spaCy `en_core_web_sm`. Model is cached with `lru_cache`. |
| `legal_keyword_density(text)` | Hits of a 32-term legal-keyword list (`shall`, `waive`, `arbitration`, `indemnify`, …) divided by total word count. |
| `flesch_kincaid_grade(text)` | Readability grade via `textstat`. Higher = harder to read. |

### 4. TF-IDF vectorizer — `src/features.py`
`build_tfidf_vectorizer()` returns a sklearn `TfidfVectorizer` configured with:

- unigrams + bigrams (`ngram_range=(1, 2)`)
- `max_features=10000`
- English stop-words removed
- `sublinear_tf=True`

### 5. Three baseline classifiers — `src/models.py`
All use `class_weight='balanced'` (labels are skewed toward `neutral`).

- `build_logistic_regression()` — `solver='lbfgs'`, `max_iter=1000`
- `build_svm_rbf()` — RBF kernel
- `build_random_forest()` — 200 trees, `n_jobs=-1`

`all_models()` returns a `{name: estimator}` dict for batch use.

### 6. 5-fold stratified cross-validation — `src/evaluate.py`
`cross_validate_macro_f1()` runs `StratifiedKFold(5)` on the train split. TF-IDF is fit **inside each fold** via a `Pipeline` so vocabulary never leaks from val fold into train fold. Reports macro-F1 and writes `results/cv_macro_f1.csv`.

### 7. Confusion matrices — `src/confusion.py`
Trains each baseline on train, predicts on val, and writes per-model PNGs to `results/confmat_{logreg,svm_rbf,random_forest}.png`.

---

## Results

### The leakage incident (first pass)

The initial splits gave macro-F1 ≈ **0.99** for all three models on val — too good to be true. A leakage audit uncovered:

- Train had only **765 unique `title`s** out of 8,210 rows (~10× duplication).
- **76% of val clauses appeared verbatim in train**; 79% of test did too.
- Every single val/test `service_id` also appeared in train (100% service overlap).
- `title == category` for 59% of rows.

**Root cause.** ToSDR clauses are shared boilerplate across ~400 services; a random row-level split without deduplication leaks identical clause strings into val/test, so the classifier mostly memorizes strings rather than generalizes.

**Fix.** `src/split_data.py` deduplicates by `title` first, then stratified-splits the 950 unique clauses 70/15/15 by `label`.

### Clean-split validation results

Trained on `train.csv` (665 clauses), evaluated on `val.csv` (142 clauses; class counts: neutral 56 / bad 49 / good 37):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.8662 | 0.8631 |
| **SVM (RBF)** | **0.8732** | **0.8704** |
| Random Forest | 0.8310 | 0.8229 |

Per-class F1 on the clean val split ranges **0.76–0.89**. The `good` class is the hardest (smallest support: 37 clauses). These are the honest baseline numbers to compare Legal-BERT against.

---

## Repo layout

```
data/
  clauses.csv                        # raw ToSDR dump
  preprocessed/{train,val,test}.csv  # clean, deduped, stratified splits
  augmented_datasets/                # CUAD + OPP-115 (Legal-BERT track)
  preprocessing/                     # early split notebook (superseded)
src/
  fetch_clauses.py    # ToSDR API fetcher
  split_data.py       # dedupe + stratified split
  preprocess.py       # hand-crafted text features
  features.py         # TF-IDF vectorizer
  models.py           # three baseline classifiers
  evaluate.py         # 5-fold CV (+ Colab-shaped Legal-BERT eval)
  confusion.py        # val-set confusion matrix PNGs
  dataset.py, train.py, inference.py, explain.py, segment.py   # Legal-BERT track
results/
  confmat_*.png
  cv_macro_f1.csv
```

---

## How to run

All scripts import siblings via `from src...`, so always invoke from the project root with the `-m` flag (direct `python src/foo.py` will fail with `ModuleNotFoundError: No module named 'src'`):

```bash
# rebuild clean splits from data/clauses.csv
.venv/Scripts/python.exe -m src.split_data

# 5-fold CV macro-F1 for all three baselines -> results/cv_macro_f1.csv
.venv/Scripts/python.exe -m src.evaluate

# val-set confusion matrices -> results/confmat_*.png
.venv/Scripts/python.exe -m src.confusion
```

---

## Known caveats / next steps

- `src/evaluate.py` still carries Colab-hardcoded paths (`/content/tos_classifier/...`) in its Legal-BERT section. The classical `cross_validate_macro_f1` block at the bottom is portable.
- `torch`, `transformers`, `umap` are used by the Legal-BERT code but aren't declared in `pyproject.toml`.
- 950 unique clauses is a small dataset — the classical baselines are already in the 0.83–0.87 macro-F1 range, leaving a narrow ceiling. Legal-BERT's incremental value should be measured on these **same clean splits** before any comparison is made.
