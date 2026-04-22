"""Feature extraction for ToS clause classification."""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
from spacy.language import Language


_WORD_RE = re.compile(r"\b\w+\b")


def clause_length(text: str) -> int:
    """Return the number of words in a clause."""
    if not isinstance(text, str):
        return 0
    return len(_WORD_RE.findall(text))


@lru_cache(maxsize=1)
def _nlp() -> Language:
    # parser is needed for dependency labels (nsubjpass, auxpass); disable NER for speed.
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


_PASSIVE_DEPS = {"nsubjpass", "auxpass", "csubjpass"}


def passive_voice_ratio(text: str) -> float:
    """Return the ratio of passive-voice sentences in a clause.

    A sentence counts as passive if any token bears a passive dependency
    label (nsubjpass, auxpass, csubjpass). Returns 0.0 for empty input.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    doc = _nlp()(text)
    sents = list(doc.sents)
    if not sents:
        return 0.0
    passive = sum(
        1 for s in sents if any(t.dep_ in _PASSIVE_DEPS for t in s)
    )
    return passive / len(sents)
