"""
Drift detection — compares production predictions to reference distribution.
Uses Population Stability Index (PSI) and confidence distribution shifts.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import ks_2samp

REFERENCE_STATS_FILE = Path("model/reference_stats.json")


def load_reference_stats() -> Optional[dict]:
    """Load reference statistics saved during training."""
    if not REFERENCE_STATS_FILE.exists():
        return None
    with open(REFERENCE_STATS_FILE) as f:
        stats = json.load(f)
    # Handle placeholder stats from CI (no training yet)
    if stats.get("status") in ("pending_first_train", "no_reference"):
        return None
    return stats


def compute_psi(expected_counts: list, actual_counts: list, epsilon: float = 1e-6) -> float:
    """
    Population Stability Index.
    PSI = Σ (actual - expected) * ln(actual / expected)
    """
    psi = 0.0
    for exp, act in zip(expected_counts, actual_counts):
        exp = max(exp, epsilon)
        act = max(act, epsilon)
        psi += (act - exp) * math.log(act / exp)
    return psi


def detect_drift(
    predictions: list[dict],
    reference_stats: Optional[dict] = None,
) -> dict:
    """
    Compute drift metrics from recent predictions vs reference distribution.
    Returns a dict with drift scores and severity flags.
    """
    if reference_stats is None:
        reference_stats = load_reference_stats()

    if reference_stats is None:
        return {
            "status": "no_reference",
            "message": "No reference distribution available. Run training first.",
            "drift_score": 0.0,
            "psi": 0.0,
            "confidence_shift": 0.0,
            "label_shift": 0.0,
            "n_production": len(predictions),
        }

    if len(predictions) < 50:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(predictions)} predictions logged. Need at least 50.",
            "drift_score": 0.0,
            "psi": 0.0,
            "confidence_shift": 0.0,
            "label_shift": 0.0,
            "n_production": len(predictions),
        }

    # Production statistics
    prod_confidences = [p["confidence"] for p in predictions]
    prod_labels = [p["label"] for p in predictions]
    prod_pos_ratio = sum(1 for l in prod_labels if l == "POSITIVE") / len(prod_labels)

    # Reference statistics
    ref_conf_hist = reference_stats.get("prediction_histogram", {})
    ref_class_props = reference_stats.get("class_proportions", {})
    ref_conf_mean = reference_stats.get("confidence_mean", 0.5)

    # 1. PSI on confidence distribution
    ref_pos_counts = ref_conf_hist.get("pos_counts", [0] * 10)
    prod_pos_hist, _ = np.histogram(prod_confidences, bins=10, range=(0, 1))
    prod_pos_counts = prod_pos_hist.tolist()

    # Normalize to percentages
    ref_pos_pct = np.array(ref_pos_counts, dtype=float)
    prod_pos_pct = np.array(prod_pos_counts, dtype=float)
    ref_pos_pct = ref_pos_pct / ref_pos_pct.sum() if ref_pos_pct.sum() > 0 else ref_pos_pct
    prod_pos_pct = prod_pos_pct / prod_pos_pct.sum() if prod_pos_pct.sum() > 0 else prod_pos_pct

    psi = compute_psi(ref_pos_pct.tolist(), prod_pos_pct.tolist())

    # 2. Confidence distribution shift (mean difference)
    confidence_shift = abs(np.mean(prod_confidences) - ref_conf_mean)

    # 3. Label distribution shift (class proportion change)
    ref_pos_ratio = ref_class_props.get("positive", 0.5)
    label_shift = abs(prod_pos_ratio - ref_pos_ratio)

    # 4. KS test on confidence distributions
    # Simulate reference confidences from histogram (approximate)
    ref_sample = []
    for i, count in enumerate(ref_pos_counts):
        bin_center = (i + 0.5) / 10
        ref_sample.extend([bin_center] * count)
    if len(ref_sample) > 10:
        ks_stat, ks_pval = ks_2samp(ref_sample, prod_confidences)
    else:
        ks_stat, ks_pval = 0.0, 1.0

    # Composite drift score (weighted)
    drift_score = 0.4 * min(psi / 0.2, 1.0) + 0.3 * min(confidence_shift / 0.1, 1.0) + 0.3 * min(label_shift / 0.15, 1.0)

    # Severity thresholds
    if drift_score > 0.7:
        severity = "critical"
        message = "Significant drift detected — consider retraining."
    elif drift_score > 0.4:
        severity = "warning"
        message = "Moderate drift detected — monitor closely."
    elif drift_score > 0.15:
        severity = "mild"
        message = "Minor drift detected — no action needed."
    else:
        severity = "normal"
        message = "No significant drift. Model is performing as expected."

    return {
        "status": severity,
        "message": message,
        "drift_score": round(drift_score, 4),
        "psi": round(psi, 4),
        "confidence_shift": round(confidence_shift, 4),
        "label_shift": round(label_shift, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_p_value": round(ks_pval, 4),
        "n_production": len(predictions),
        "prod_pos_ratio": round(prod_pos_ratio, 4),
        "ref_pos_ratio": round(ref_pos_ratio, 4),
    }
