"""Model factories for ToS clause classification.

Each function returns an unfitted sklearn estimator configured for this task.
Class weights are set to 'balanced' because the ToS label distribution is
skewed (neutral clauses dominate).
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Logistic Regression with balanced class weights."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )


def build_svm_rbf(random_state: int = 42) -> SVC:
    """Support Vector Classifier with RBF kernel and balanced class weights."""
    return SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=random_state,
    )
