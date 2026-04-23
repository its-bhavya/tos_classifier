# src/summarize.py
"""Group BAD-labeled clauses into themes and render an HTML digest."""

from __future__ import annotations

from html import escape

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

THEMES: dict[str, list[str]] = {
    "Data sharing & sale": [
        "share", "sell", "third part", "advertis", "marketing partner", "affiliate",
    ],
    "Tracking & cookies": [
        "cookie", "track", "fingerprint", "analytics", "pixel",
    ],
    "Forced arbitration / class-action waiver": [
        "arbitrat", "class action", "class-action", "waive", "jury trial",
    ],
    "Unilateral changes": [
        "modify", "change these terms", "at any time", "sole discretion", "without notice",
    ],
    "Account termination": [
        "terminate", "suspend", "delete your account", "without cause",
    ],
    "Content ownership / licensing": [
        "license to", "royalty-free", "perpetual", "your content", "user content",
    ],
    "Liability / warranty": [
        "as is", "as-is", "no warrant", "not liable", "limit", "indemnif",
    ],
    "Jurisdiction / governing law": [
        "governed by", "jurisdiction", "venue", "exclusive forum",
    ],
}

OTHER = "Other concerns"

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
_sum_tokenizer = None
_sum_model = None


def _get_summarizer():
    global _sum_tokenizer, _sum_model
    if _sum_tokenizer is None or _sum_model is None:
        print(f"Loading summarizer {SUMMARIZER_MODEL}...")
        _sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
        _sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
        _sum_model.eval()
        print("Summarizer loaded.")
    return _sum_tokenizer, _sum_model


def group_bad_clauses(
    clauses: list[str],
    predictions: list[dict],
) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for clause, pred in zip(clauses, predictions):
        if pred.get("label") != "BAD":
            continue
        text_lc = clause.lower()
        conf = float(pred.get("confidence", 0.0))
        matched = False
        for theme, keywords in THEMES.items():
            if any(kw in text_lc for kw in keywords):
                grouped.setdefault(theme, []).append((clause, conf))
                matched = True
        if not matched:
            grouped.setdefault(OTHER, []).append((clause, conf))
    return grouped


def summarize_theme(clauses: list[str]) -> str:
    joined = " ".join(c.strip() for c in clauses).strip()
    if not joined:
        return ""
    # Single short clause — skip the model to avoid degenerate output.
    if len(clauses) == 1 and len(joined.split()) < 25:
        return joined
    tokenizer, model = _get_summarizer()
    inputs = tokenizer(
        joined,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=60,
            min_length=15,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_theme_summaries(
    grouped: dict[str, list[tuple[str, float]]],
) -> dict[str, tuple[str, int, float]]:
    summaries: dict[str, tuple[str, int, float]] = {}
    for theme, items in grouped.items():
        clauses = [c for c, _ in items]
        confs = [conf for _, conf in items]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        summary_text = summarize_theme(clauses)
        summaries[theme] = (summary_text, len(items), avg_conf)
    return summaries


def render_bad_summary_html(
    summaries: dict[str, tuple[str, int, float]],
) -> str:
    if not summaries:
        return ""

    total = sum(count for _, count, _ in summaries.values())

    theme_blocks = ""
    for theme, (summary_text, count, avg_conf) in summaries.items():
        theme_blocks += f"""
        <div style="margin:12px 0 16px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:4px;">
                <div style="font-weight:600;color:#c0392b;font-size:15px;">
                    • {escape(theme)} <span style="color:#888;font-weight:400;">({count} clause{'s' if count != 1 else ''})</span>
                </div>
                <span style="background:#fadbd8;color:#c0392b;font-size:12px;font-weight:600;
                             padding:2px 8px;border-radius:10px;white-space:nowrap;">avg {avg_conf:.0%}</span>
            </div>
            <p style="margin:4px 0 0 16px;color:#2c3e50;font-size:14px;line-height:1.5;">
                {escape(summary_text)}
            </p>
        </div>"""

    return f"""
    <div style="font-family:Arial,sans-serif;padding:16px;border-radius:8px;
                background:#fdf3f2;border-left:6px solid #c0392b;margin-bottom:16px;">
        <h2 style="margin:0 0 8px 0;color:#c0392b;">⚠️ {total} red flag{'s' if total != 1 else ''} found</h2>
        <p style="margin:0 0 4px 0;color:#555;font-size:13px;">
            AI-generated summary of BAD-labeled clauses, grouped by theme.
        </p>
        {theme_blocks}
    </div>
    """
