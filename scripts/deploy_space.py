"""
Update HF Space with app/ directory contents.
Space must already exist (create manually in HF UI).
"""

import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    os.system("pip install huggingface-hub>=0.20")
    from huggingface_hub import HfApi

SPACE_ID = os.environ.get("HF_SPACE_ID", "karaboLLM/ml-observability")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set. Skipping deploy.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Verify Space exists
    print(f"🔍 Checking Space exists: {SPACE_ID}")
    try:
        info = api.repo_info(repo_id=SPACE_ID, repo_type="space")
        runtime = info.cardData if hasattr(info, 'cardData') else {}
        print(f"✅ Space found: {SPACE_ID} (SDK: {info.sdk if hasattr(info, 'sdk') else 'unknown'})")
    except Exception as e:
        print(f"\n❌ Space '{SPACE_ID}' not found on Hugging Face.")
        print(f"\n👉 You need to create it manually first:")
        print(f"   1. Go to https://huggingface.co/new-space")
        print(f"   2. Owner: DynamicKarabo")
        print(f"   3. Space Name: ml-observability")
        print(f"   4. SDK: Gradio")
        print(f"   5. Hardware: CPU basic (free)")
        print(f"   6. Accept Spaces Terms")
        print(f"   7. Click 'Create Space'")
        print(f"\n   Then re-run this workflow.")
        sys.exit(1)

    # Upload app directory
    app_dir = "app"
    if not os.path.exists(app_dir):
        print(f"❌ {app_dir}/ directory not found. Run from repo root.")
        sys.exit(1)

    print(f"📤 Uploading {app_dir}/ to Space...")
    try:
        api.upload_folder(
            repo_id=SPACE_ID,
            folder_path=app_dir,
            path_in_repo=".",
            token=HF_TOKEN,
            repo_type="space",
            commit_message="Deploy: update app from CI/CD",
            delete_patterns=["model/*.bin", "model/*.safetensors"],
        )
        print(f"\n✅ Files deployed to https://huggingface.co/spaces/{SPACE_ID}")
        print(f"   Space will auto-build in a few seconds.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
