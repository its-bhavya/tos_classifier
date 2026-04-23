# Baseline Model Evaluation Results

Clean 70/15/15 split (dedup by `title`, stratified by `label`, seed 42).
Sizes: **train 665 / val 142 / test 143**.

All models use TF-IDF (unigrams+bigrams, max 10k features, `sublinear_tf=True`)
with `class_weight='balanced'`. Reproduce via `python -m src.test_eval`.

---

## Held-out test results (`test.csv`, n=143)

Source: [`test/test_summary.csv`](test/test_summary.csv)

| Model | Accuracy | Macro F1 | MCC | F1 good | F1 neutral | F1 bad |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8112 | 0.7963 | 0.7216 | 0.7097 | 0.8545 | 0.8246 |
| **SVM (RBF)** | **0.8392** | **0.8238** | **0.7651** | 0.7302 | 0.8991 | 0.8421 |
| Random Forest | 0.8182 | 0.7995 | 0.7250 | 0.6875 | 0.8621 | 0.8491 |

Winner: **SVM (RBF)** on accuracy, macro-F1, and MCC.

---

## Validation results (`val.csv`, n=142)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.8662 | 0.8631 |
| **SVM (RBF)** | **0.8732** | **0.8704** |
| Random Forest | 0.8310 | 0.8229 |

---

## Layout

```
results/
  BASELINES.md                         # this file
  val/
    confmat_{logreg,svm_rbf,random_forest}.png
  test/
    test_summary.csv                   # combined metrics
    test_metrics_{model}.json          # per-model full metrics + confusion matrix
    test_report_{model}.txt            # per-model sklearn classification_report
    test_confmat_{model}.png           # per-model confusion matrix
  eda/
    1_class_distribution_of_TOS_clauses.png
    2_clause_length_distribution_by_risk_category.png
    3_word_count.png
```

---

## Notes

- `good` is consistently the hardest class (smallest support: 37 val / ~37 test).
- Per-class F1 on test ranges **0.69–0.90**.
- An earlier leakage incident (macro-F1 ≈ 0.99 from duplicated boilerplate
  clauses across splits) was the reason for dedup-first splitting. See main
  `README.md` for the full write-up.
