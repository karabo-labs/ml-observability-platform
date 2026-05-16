"""
Push trained model to Hugging Face Hub.
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

HUB_MODEL_ID = os.environ.get("HF_MODEL_ID", "karaboLLM/sentiment-distilbert")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = Path("model_output")


def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set. Skipping push.")
        sys.exit(1)

    if not OUTPUT_DIR.exists():
        print(f"❌ Model output directory {OUTPUT_DIR} not found. Run train.py first.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    print(f"📦 Ensuring repo exists: {HUB_MODEL_ID}")
    create_repo(
        repo_id=HUB_MODEL_ID,
        token=HF_TOKEN,
        repo_type="model",
        exist_ok=True,
    )

    # Upload model files
    print(f"📤 Uploading model to {HUB_MODEL_ID}...")
    api.upload_folder(
        repo_id=HUB_MODEL_ID,
        folder_path=str(OUTPUT_DIR),
        path_in_repo=".",
        token=HF_TOKEN,
        commit_message="Update model with reference statistics",
    )

    # Upload metrics as model card metadata
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        accuracy = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        card_content = f"""---
tags:
- distilbert
- sentiment
- ml-observability
library_name: transformers
metrics:
- accuracy: {accuracy:.4f}
- f1: {f1:.4f}
---

# Sentiment Classifier (DistilBERT)

Fine-tuned on IMDB for the ML Observability Platform.

**Accuracy:** {accuracy:.4f}  
**F1 Score:** {f1:.4f}

Part of the [ml-observability-platform](https://github.com/DynamicKarabo/ml-observability-platform) MLOps pipeline.
"""
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=HUB_MODEL_ID,
            token=HF_TOKEN,
            commit_message="Update model card with metrics",
        )

    print(f"✅ Model pushed to https://huggingface.co/{HUB_MODEL_ID}")


if __name__ == "__main__":
    main()
