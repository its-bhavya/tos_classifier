"""Feature extraction for ToS clause classification."""

from __future__ import annotations

import re
from functools import lru_cache

import spacy
import textstat
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


# ~30 terms commonly associated with legal / ToS-style obligation, liability,
# dispute resolution, and IP language. Matched case-insensitively as whole words;
# multi-word phrases are matched as a sequence.
LEGAL_KEYWORDS: tuple[str, ...] = (
    "shall",
    "waive",
    "waiver",
    "arbitration",
    "arbitrate",
    "indemnify",
    "indemnity",
    "liability",
    "liable",
    "warranty",
    "warranties",
    "disclaim",
    "terminate",
    "termination",
    "jurisdiction",
    "governing law",
    "class action",
    "binding",
    "hereby",
    "herein",
    "consent",
    "confidential",
    "proprietary",
    "license",
    "sublicense",
    "perpetual",
    "irrevocable",
    "royalty-free",
    "third party",
    "breach",
    "dispute",
    "remedy",
    "enforceable",
)

_LEGAL_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in LEGAL_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def legal_keyword_density(text: str) -> float:
    """Return the fraction of words in `text` that are legal keywords.

    Multi-word phrases (e.g. "class action") count as one hit but are divided
    by total word count, giving a 0..1 density value.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    total_words = len(_WORD_RE.findall(text))
    if total_words == 0:
        return 0.0
    hits = len(_LEGAL_KEYWORD_RE.findall(text))
    return hits / total_words


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level for `text`.

    Higher values indicate text that is harder to read (roughly the US
    school grade needed to understand it). Returns 0.0 for empty input.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return 0.0


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
