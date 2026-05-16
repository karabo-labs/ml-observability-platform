"""
Sync reference statistics from HF Hub to app/model/ for drift detection.
Used in CI/CD when no fresh training run happened.
"""

import json
import os
import sys

try:
    from huggingface_hub import hf_hub_download, HfApi
except ImportError:
    os.system("pip install huggingface-hub requests")
    from huggingface_hub import hf_hub_download, HfApi

MODEL_ID = os.environ.get("HF_MODEL_ID", "DynamicKarabo/sentiment-distilbert")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "app/model"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if fresh stats already exist from training (artifact download)
    if os.path.exists("model_output/reference_stats.json"):
        import shutil
        shutil.copy("model_output/reference_stats.json", os.path.join(OUTPUT_DIR, "reference_stats.json"))
        print("✅ Using fresh reference stats from training")
        return

    # Check if model repo exists on HF Hub
    if HF_TOKEN:
        try:
            api = HfApi(token=HF_TOKEN)
            api.repo_info(repo_id=MODEL_ID, repo_type="model")
            print(f"✅ Model repo {MODEL_ID} exists on Hub")
        except Exception:
            print(f"ℹ️  Model repo {MODEL_ID} not found on Hub yet. First CI run will create it.")
            print(f"   Monitor will show 'no reference' until first train completes.")
            # Write empty stats
            with open(os.path.join(OUTPUT_DIR, "reference_stats.json"), "w") as f:
                json.dump({"status": "pending_first_train"}, f)
            return

    # Download existing reference stats from HF Hub
    try:
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="reference_stats.json",
            token=HF_TOKEN or None,
        )
        with open(path) as f:
            stats = json.load(f)
        with open(os.path.join(OUTPUT_DIR, "reference_stats.json"), "w") as f:
            json.dump(stats, f)
        print(f"✅ Downloaded existing reference stats from {MODEL_ID}")
    except Exception as e:
        print(f"⚠️ Could not download reference stats: {e}")
        print(f"   Monitor will show 'no reference' — this is fine for first run.")
        # Create minimal placeholder
        with open(os.path.join(OUTPUT_DIR, "reference_stats.json"), "w") as f:
            json.dump({"status": "no_reference"}, f)


if __name__ == "__main__":
    main()
