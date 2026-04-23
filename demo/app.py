# demo/app.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.segment import segment_into_clauses
from src.fetch_data import fetch_and_segment
from src.inference import predict_batch
from src.summarize import group_bad_clauses, build_theme_summaries, render_bad_summary_html


def classify_tos(input_text: str, input_url: str) -> tuple:
    if input_url and input_url.strip():
        try:
            clauses, _ = fetch_and_segment(input_url.strip())
            source = f"URL: {input_url.strip()}"
        except Exception as e:
            return "", "", f"❌ Error fetching URL: {str(e)}"
    elif input_text and input_text.strip():
        clauses = segment_into_clauses(input_text.strip())
        source = "Pasted text"
    else:
        return "", "", "⚠️ Please paste text or enter a URL."

    if not clauses:
        return "", "", "⚠️ No clauses found. Try a different document."

    predictions = predict_batch(clauses)

    bad_count = sum(1 for p in predictions if p["label"] == "BAD")
    good_count = sum(1 for p in predictions if p["label"] == "GOOD")
    total = len(predictions)
    risk_score = round((bad_count / total) * 100)

    if risk_score >= 40:
        risk_level, risk_color = "🔴 HIGH RISK", "#c0392b"
    elif risk_score >= 20:
        risk_level, risk_color = "🟡 MODERATE RISK", "#e67e22"
    else:
        risk_level, risk_color = "🟢 LOW RISK", "#27ae60"

    risk_html = f"""
    <div style="font-family:Arial,sans-serif;padding:16px;border-radius:8px;
                background:#f8f9fa;border-left:6px solid {risk_color};margin-bottom:16px;">
        <h2 style="margin:0;color:{risk_color};">Document Risk Score: {risk_score}/100</h2>
        <p style="margin:4px 0 0 0;font-size:16px;">{risk_level}</p>
        <p style="margin:4px 0 0 0;color:#555;font-size:14px;">
            Source: {source} | Total: {total} clauses |
            🔴 Bad: {bad_count} | 🟢 Good: {good_count} |
            🟡 Neutral: {total - bad_count - good_count}
        </p>
    </div>
    """

    COLOR_MAP = {
        "GOOD":    ("#d5f5e3", "#1e8449"),
        "NEUTRAL": ("#fef9e7", "#9a7d0a"),
        "BAD":     ("#fadbd8", "#c0392b"),
    }
    EMOJI_MAP = {"GOOD": "🟢", "NEUTRAL": "🟡", "BAD": "🔴"}

    rows_html = ""
    for i, (clause, pred) in enumerate(zip(clauses, predictions), 1):
        label = pred["label"]
        confidence = pred["confidence"]
        bg, fg = COLOR_MAP[label]
        emoji = EMOJI_MAP[label]
        rows_html += f"""
        <tr style="background-color:{bg};">
            <td style="padding:8px;text-align:center;font-weight:bold;color:{fg};width:40px;">{i}</td>
            <td style="padding:8px;color:#2c3e50;font-size:14px;">{clause}</td>
            <td style="padding:8px;text-align:center;font-weight:bold;color:{fg};white-space:nowrap;">
                {emoji} {label}
            </td>
            <td style="padding:8px;text-align:center;color:#555;font-size:13px;white-space:nowrap;">
                {confidence:.1%}
            </td>
        </tr>"""

    table_html = f"""
    <div style="font-family:Arial,sans-serif;overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <thead>
                <tr style="background-color:#2c3e50;color:white;">
                    <th style="padding:10px;">#</th>
                    <th style="padding:10px;text-align:left;">Clause</th>
                    <th style="padding:10px;">Risk Label</th>
                    <th style="padding:10px;">Confidence</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""

    summary_html = render_bad_summary_html(build_theme_summaries(group_bad_clauses(clauses, predictions)))

    return summary_html + risk_html, table_html, f"✅ Done. {total} clauses analysed."


with gr.Blocks(title="ToS Risk Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ Terms of Service Clause Risk Classifier")
    gr.Markdown("Paste a ToS URL or raw text. Each clause will be labelled 🟢 GOOD, 🟡 NEUTRAL, or 🔴 BAD.")

    with gr.Row():
        url_input = gr.Textbox(label="Option 1 — Enter a ToS URL",
                               placeholder="https://www.example.com/terms")
        text_input = gr.Textbox(label="Option 2 — Paste raw ToS text",
                                placeholder="Paste your Terms of Service here...", lines=8)

    submit_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")
    status_output = gr.Textbox(label="Status", interactive=False)
    risk_score_output = gr.HTML(label="Document Risk Score")
    results_output = gr.HTML(label="Clause-by-Clause Analysis")

    submit_btn.click(
        fn=classify_tos,
        inputs=[text_input, url_input],
        outputs=[risk_score_output, results_output, status_output]
    )

    gr.Markdown("---\n🟢 **GOOD** — Protects the user | 🟡 **NEUTRAL** — Standard boilerplate | 🔴 **BAD** — Harms user rights")


if __name__ == "__main__":
    demo.launch(share=False)