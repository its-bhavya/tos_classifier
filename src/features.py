"""Vectorizer / feature-matrix builders for ToS clause classification."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 10_000) -> TfidfVectorizer:
    """Return an unfitted TF-IDF vectorizer for clause text.

    - unigrams + bigrams
    - lowercased
    - English stop words removed
    - capped at `max_features` most-frequent terms
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        lowercase=True,
        stop_words="english",
        sublinear_tf=True,
    )
