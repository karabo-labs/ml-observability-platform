"""
Prediction logging — stores every prediction locally for drift monitoring.
Uses JSONL format. Safe for concurrent reads/writes on HF Spaces.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

PREDICTIONS_FILE = Path("data/predictions.jsonl")
LOCK = threading.Lock()
MAX_RECORDS = 50_000  # keep last 50k predictions


def log_prediction(text: str, label: str, confidence: float, latency_ms: float):
    """Store a single prediction record."""
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_length": len(text),
        "label": label,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }

    with LOCK:
        with open(PREDICTIONS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Trim if too large
        if PREDICTIONS_FILE.stat().st_size > 10_000_000:  # 10MB cap
            _trim_records()


def _trim_records():
    """Keep only the last MAX_RECORDS entries."""
    try:
        with open(PREDICTIONS_FILE) as f:
            lines = f.readlines()
        if len(lines) > MAX_RECORDS:
            with open(PREDICTIONS_FILE, "w") as f:
                f.writelines(lines[-MAX_RECORDS:])
    except Exception:
        pass


def load_predictions() -> list[dict]:
    """Load all stored predictions."""
    if not PREDICTIONS_FILE.exists():
        return []
    with LOCK:
        with open(PREDICTIONS_FILE) as f:
            return [json.loads(line) for line in f if line.strip()]


def get_predictions(n: int = 1000) -> list[dict]:
    """Get the most recent N predictions."""
    records = load_predictions()
    return records[-n:]
