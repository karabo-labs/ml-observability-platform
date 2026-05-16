"""
Create or update HF Space with app/ directory contents.
"""

import os
import sys

try:
    from huggingface_hub import HfApi, create_repo, SpaceHardware
except ImportError:
    os.system("pip install huggingface-hub>=0.20")
    from huggingface_hub import HfApi, create_repo, SpaceHardware

SPACE_ID = os.environ.get("HF_SPACE_ID", "DynamicKarabo/ml-observability")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set. Skipping deploy.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create Space if it doesn't exist
    print(f"📦 Ensuring space exists: {SPACE_ID}")
    try:
        create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            token=HF_TOKEN,
            exist_ok=True,
            space_sdk="gradio",
            space_hardware=SpaceHardware.CPU_BASIC,
        )
        print(f"✅ Space ready: https://huggingface.co/spaces/{SPACE_ID}")
    except Exception as e:
        print(f"⚠️ Space creation warning: {e}")

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
            commit_message="Deploy: update app from CI/CD",
            delete_patterns=["model/*.bin", "model/*.safetensors"],
        )
        print(f"✅ Deployed to https://huggingface.co/spaces/{SPACE_ID}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
