"""
ML Observability Platform
Single HF Space with:
  - Tab 1: Sentiment Inference
  - Tab 2: Monitoring Dashboard

Runs on free CPU tier. Loads model from HF Hub.
"""

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from drift_detector import detect_drift, load_reference_stats
from prediction_store import get_predictions, log_prediction

# ── Configuration ──────────────────────────────────────────────────
MODEL_ID = os.environ.get("HF_MODEL_ID", "DynamicKarabo/sentiment-distilbert")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_DIR = Path("model")

# ── Load Model ─────────────────────────────────────────────────────
print(f"📥 Loading model from {MODEL_ID}...")
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN or None,
        ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt"],
    )

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()
print("✅ Model loaded successfully!")

LABELS = ["NEGATIVE", "POSITIVE"]


# ── Inference ──────────────────────────────────────────────────────
def predict(text: str, progress=gr.Progress()):
    """Run inference on input text."""
    if not text or not text.strip():
        return "", 0.0, "Please enter some text."

    start = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    confidence, pred_idx = torch.max(probs, dim=-1)
    label = LABELS[pred_idx.item()]
    confidence_val = confidence.item()
    latency_ms = (time.time() - start) * 1000

    # Log prediction for monitoring
    log_prediction(text, label, confidence_val, latency_ms)

    details = f"⚡ {latency_ms:.1f}ms | Neg: {probs[0,0]:.3f} | Pos: {probs[0,1]:.3f}"

    return label, round(confidence_val, 4), details


# ── Monitoring Dashboard ───────────────────────────────────────────
def build_monitor():
    """Build the monitoring dashboard with drift metrics and charts."""
    predictions = get_predictions(2000)
    ref_stats = load_reference_stats()
    drift = detect_drift(predictions, ref_stats)

    n_total = len(predictions)

    # Status colors
    status_config = {
        "normal": ("🟢", "#1a3a1a", "#2ecc71"),
        "mild": ("🟡", "#3a2a1a", "#f39c12"),
        "warning": ("🟠", "#3a2a1a", "#e67e22"),
        "critical": ("🔴", "#3a1a1a", "#e74c3c"),
    }
    icon, bg, border = status_config.get(drift["status"], ("⚪", "#1e1e1e", "#666"))

    def gauge_color(val, thresholds=(0.05, 0.1)):
        if val < thresholds[0]:
            return "#2ecc71"
        elif val < thresholds[1]:
            return "#f39c12"
        return "#e74c3c"

    # Status banner
    status_html = f"""
    <div style="padding:16px; border-radius:8px; background:{bg}; border:1px solid {border}; margin-bottom:16px;">
        <h2 style="margin:0 0 8px 0;">{icon} Drift Status: <strong>{drift['status'].upper()}</strong></h2>
        <p style="margin:4px 0;">{drift['message']}</p>
        <p style="margin:4px 0; font-size:0.85em; opacity:0.7;">Composite Score: {drift['drift_score']:.4f}</p>
    </div>
    """

    # Metrics cards
    psi_color = gauge_color(drift["psi"], (0.1, 0.25))
    conf_shift_color = gauge_color(drift["confidence_shift"], (0.05, 0.1))
    label_shift_color = gauge_color(drift["label_shift"], (0.05, 0.1))

    metrics_html = f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px;">
        <div style="padding:12px; background:#1e1e1e; border-radius:8px; text-align:center;">
            <div style="font-size:0.8em; opacity:0.7;">Total Predictions</div>
            <div style="font-size:1.8em; font-weight:bold;">{n_total}</div>
        </div>
        <div style="padding:12px; background:#1e1e1e; border-radius:8px; text-align:center;">
            <div style="font-size:0.8em; opacity:0.7;">PSI</div>
            <div style="font-size:1.8em; font-weight:bold; color:{psi_color};">{drift['psi']:.4f}</div>
        </div>
        <div style="padding:12px; background:#1e1e1e; border-radius:8px; text-align:center;">
            <div style="font-size:0.8em; opacity:0.7;">Confidence Shift</div>
            <div style="font-size:1.8em; font-weight:bold; color:{conf_shift_color};">{drift['confidence_shift']:.4f}</div>
        </div>
        <div style="padding:12px; background:#1e1e1e; border-radius:8px; text-align:center;">
            <div style="font-size:0.8em; opacity:0.7;">Label Shift</div>
            <div style="font-size:1.8em; font-weight:bold; color:{label_shift_color};">{drift['label_shift']:.4f}</div>
        </div>
    </div>
    """

    # Charts
    charts_html = "<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;'>"

    if predictions:
        timestamps = [p["timestamp"][:19] for p in predictions]
        confidences = [p["confidence"] for p in predictions]
        labels = [p["label"] for p in predictions]
        pos_count = sum(1 for l in labels if l == "POSITIVE")
        neg_count = len(labels) - pos_count

        # Confidence scatter
        fig1 = go.Figure()
        colors = ["#2ecc71" if l == "POSITIVE" else "#e74c3c" for l in labels]
        fig1.add_trace(go.Scatter(
            x=timestamps, y=confidences, mode="markers",
            marker=dict(color=colors, size=4, opacity=0.6),
        ))
        fig1.update_layout(
            title="Confidence Over Time",
            xaxis_title="Time", yaxis_title="Confidence",
            yaxis_range=[0, 1], template="plotly_dark",
            height=300, margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
        )

        # Label distribution pie
        fig2 = go.Figure(data=[go.Pie(
            labels=["POSITIVE", "NEGATIVE"],
            values=[pos_count, neg_count],
            marker=dict(colors=["#2ecc71", "#e74c3c"]),
            hole=0.4,
        )])
        fig2.update_layout(
            title=f"Label Distribution ({pos_count}P / {neg_count}N)",
            template="plotly_dark", height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        charts_html += f"<div>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
        charts_html += f"<div>{fig2.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    else:
        charts_html += '<div style="padding:40px; text-align:center; opacity:0.5;">No predictions yet. Use the Inference tab to get started.</div><div></div>'

    charts_html += "</div>"

    # Drift gauge
    fig3 = go.Figure(data=[go.Indicator(
        mode="gauge+number",
        value=drift["drift_score"] * 100,
        title={"text": "Drift Score (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2ecc71" if drift["drift_score"] < 0.4 else "#f39c12" if drift["drift_score"] < 0.7 else "#e74c3c"},
            "steps": [
                {"range": [0, 15], "color": "#1a3a1a"},
                {"range": [15, 40], "color": "#3a2a1a"},
                {"range": [40, 70], "color": "#3a1a1a"},
                {"range": [70, 100], "color": "#4a0a0a"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 40},
        },
    )])
    fig3.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))

    gauge_html = f"<div>{fig3.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

    # Recent predictions table
    table_html = ""
    if predictions:
        recent = predictions[-20:][::-1]
        rows = "".join(
            f"""<tr style="border-bottom:1px solid #333;">
                <td style="padding:4px 8px; font-size:0.85em;">{p['timestamp'][:19]}</td>
                <td style="padding:4px 8px; color:{'#2ecc71' if p['label']=='POSITIVE' else '#e74c3c'}; font-weight:bold;">{p['label']}</td>
                <td style="padding:4px 8px;">{p['confidence']:.3f}</td>
                <td style="padding:4px 8px;">{p['latency_ms']:.0f}ms</td>
            </tr>"""
            for p in recent
        )
        table_html = f"""
        <div style="margin-top:16px;">
            <h3>📋 Recent Predictions (last 20)</h3>
            <div style="max-height:400px; overflow-y:auto; border:1px solid #333; border-radius:8px;">
                <table style="width:100%; border-collapse:collapse;">
                    <thead><tr style="background:#1e1e1e; position:sticky; top:0;">
                        <th style="padding:8px; text-align:left;">Timestamp</th>
                        <th style="padding:8px; text-align:left;">Label</th>
                        <th style="padding:8px; text-align:left;">Confidence</th>
                        <th style="padding:8px; text-align:left;">Latency</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>"""

    return status_html + metrics_html + charts_html + gauge_html + table_html


# ── Build Gradio Interface ─────────────────────────────────────────

with gr.Blocks(
    title="ML Observability Platform",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
    css="footer {display:none !important;} .gradio-container {max-width: 1200px !important;}",
) as demo:

    gr.Markdown(
        "# 🧠 ML Observability Platform\n**Sentiment Analysis + Real-Time Drift Monitoring**"
    )

    # ── Tab 1: Inference ──
    with gr.Tab("🔮 Inference"):
        gr.Markdown("### Sentiment Analysis")
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Enter text",
                    placeholder="This movie was absolutely incredible! The acting was top-notch.",
                    lines=4,
                )
                with gr.Row():
                    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
                    clear_btn = gr.ClearButton([text_input], size="lg")

            with gr.Column(scale=2):
                label_output = gr.Label(label="Sentiment", num_top_classes=2)
                confidence_output = gr.Number(label="Confidence")
                details_output = gr.Textbox(label="Details", interactive=False)

        with gr.Row():
            gr.Examples(
                examples=[
                    ["This movie was absolutely incredible! The acting was top-notch."],
                    ["Worst film I've ever seen. Complete waste of time and money."],
                    ["It was okay, nothing special but not terrible either."],
                ],
                inputs=text_input,
                outputs=[label_output, confidence_output, details_output],
                fn=predict,
                cache_examples=False,
            )

    # ── Tab 2: Monitoring ──
    with gr.Tab("📊 Monitor"):
        gr.Markdown("### Model Monitoring Dashboard")
        refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary", size="lg")
        monitor_output = gr.HTML(label="Dashboard")

    # Wire events
    predict_btn.click(
        fn=predict, inputs=text_input,
        outputs=[label_output, confidence_output, details_output],
    )
    refresh_btn.click(fn=build_monitor, inputs=[], outputs=monitor_output)

    # Load dashboard on page load
    demo.load(fn=build_monitor, inputs=[], outputs=monitor_output)

print("🚀 ML Observability Platform ready!")
if __name__ == "__main__":
    demo.launch()
