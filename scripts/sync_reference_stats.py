"""
Sync reference statistics from HF Hub to app/model/ for drift detection.
Used in CI/CD when no fresh training run happened.
"""

import json
import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    os.system("pip install huggingface-hub requests")
    from huggingface_hub import hf_hub_download

MODEL_ID = os.environ.get("HF_MODEL_ID", "DynamicKarabo/sentiment-distilbert")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    os.makedirs("app/model", exist_ok=True)

    # Check if fresh stats already exist from training
    if os.path.exists("model_output/reference_stats.json"):
        import shutil
        shutil.copy("model_output/reference_stats.json", "app/model/reference_stats.json")
        print("✅ Using fresh reference stats from training")
        return

    # Otherwise download from HF Hub
    try:
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="reference_stats.json",
            token=HF_TOKEN or None,
        )
        with open(path) as f:
            stats = json.load(f)
        with open("app/model/reference_stats.json", "w") as f:
            json.dump(stats, f)
        print(f"✅ Downloaded existing reference stats from {MODEL_ID}")
    except Exception as e:
        print(f"⚠️ Could not download reference stats: {e}")
        # Create minimal default — monitor will show 'no_reference'
        with open("app/model/reference_stats.json", "w") as f:
            json.dump({}, f)
        print("⚠️ Created empty reference stats. Monitor will show 'no reference' status.")


if __name__ == "__main__":
    main()
