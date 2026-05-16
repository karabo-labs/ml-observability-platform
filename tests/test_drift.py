"""Tests for drift detection logic."""

import json
import sys
import tempfile
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from drift_detector import compute_psi, detect_drift


def test_compute_psi_identical():
    """PSI should be ~0 for identical distributions."""
    a = [0.2, 0.3, 0.3, 0.2]
    b = [0.2, 0.3, 0.3, 0.2]
    psi = compute_psi(a, b)
    assert abs(psi) < 0.001, f"Expected ~0, got {psi}"


def test_compute_psi_different():
    """PSI should be >0 for different distributions."""
    a = [0.5, 0.3, 0.1, 0.1]
    b = [0.1, 0.1, 0.3, 0.5]
    psi = compute_psi(a, b)
    assert psi > 0.1, f"Expected >0.1, got {psi}"


def test_detect_drift_no_reference():
    """Should return 'no_reference' status when no ref stats."""
    result = detect_drift([], reference_stats=None)
    assert result["status"] == "no_reference"


def test_detect_drift_insufficient_data():
    """Should return 'insufficient_data' when < 50 predictions."""
    ref = {
        "class_proportions": {"positive": 0.5, "negative": 0.5},
        "confidence_mean": 0.8,
        "confidence_std": 0.1,
        "prediction_histogram": {
            "pos_counts": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "neg_counts": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "pos_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "neg_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "n_samples": 100,
    }
    predictions = [{"confidence": 0.9, "label": "POSITIVE"} for _ in range(5)]
    result = detect_drift(predictions, reference_stats=ref)
    assert result["status"] == "insufficient_data"


def test_detect_drift_normal():
    """Should detect normal (low drift) when distributions match."""
    ref = {
        "class_proportions": {"positive": 0.5, "negative": 0.5},
        "confidence_mean": 0.85,
        "confidence_std": 0.1,
        "prediction_histogram": {
            "pos_counts": [0, 0, 0, 0, 0, 0, 0, 5, 10, 35],
            "neg_counts": [0, 0, 0, 0, 0, 0, 0, 5, 10, 35],
            "pos_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "neg_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "n_samples": 100,
    }
    # High confidence predictions matching reference
    predictions = [
        {"confidence": 0.85 + (i % 5) * 0.02, "label": "POSITIVE" if i % 2 == 0 else "NEGATIVE"}
        for i in range(100)
    ]
    result = detect_drift(predictions, reference_stats=ref)
    assert result["status"] in ("normal", "mild"), f"Expected normal/mild, got {result['status']}"
    assert result["drift_score"] < 0.5


def test_detect_drift_critical():
    """Should detect critical drift when distributions diverge."""
    ref = {
        "class_proportions": {"positive": 0.5, "negative": 0.5},
        "confidence_mean": 0.85,
        "confidence_std": 0.1,
        "prediction_histogram": {
            "pos_counts": [0, 0, 0, 0, 0, 0, 0, 5, 10, 35],
            "neg_counts": [0, 0, 0, 0, 0, 0, 0, 5, 10, 35],
            "pos_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "neg_bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "n_samples": 100,
    }
    # Low confidence, all negative predictions — major drift
    predictions = [
        {"confidence": 0.4 + (i % 5) * 0.03, "label": "NEGATIVE"}
        for i in range(100)
    ]
    result = detect_drift(predictions, reference_stats=ref)
    assert result["status"] in ("warning", "critical"), f"Expected warning/critical, got {result['status']}"
