"""
Check HF Space deployment status after CI/CD deploy.
Polls the HF API until the Space reaches RUNNING state,
then confirms the Gradio app is actually serving requests.
"""

import os
import sys
import time
import requests

SPACE_ID = os.environ.get("HF_SPACE_ID", "karaboLLM/ml-observability")
MAX_RETRIES = 48          # 48 × 15s = 12 min max (free tier builds can be slow)
RETRY_DELAY = 15
GRADIO_TIMEOUT = 60       # Extra time to wait for Gradio to serve after Space is RUNNING


def check_gradio_ready(space_id: str, max_wait: int = GRADIO_TIMEOUT) -> bool:
    """Hit the Gradio app root to confirm the app is actually serving."""
    gradio_url = f"https://{space_id.replace('/', '-')}.hf.space/"
    deadline = time.time() + max_wait

    while time.time() < deadline:
        try:
            r = requests.get(gradio_url, timeout=10)
            if r.status_code == 200:
                print(f"   ✅ Gradio app responding (HTTP {r.status_code})")
                return True
            print(f"   ⏳ Gradio returned HTTP {r.status_code} — waiting...")
        except requests.RequestException as e:
            print(f"   ⏳ Gradio not ready yet: {e}")
        time.sleep(5)

    print(f"   ⚠️ Gradio did not respond within {max_wait}s")
    return False


def main():
    api_url = f"https://huggingface.co/api/spaces/{SPACE_ID}"

    print(f"🔍 Checking Space status: {SPACE_ID}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(api_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                runtime = data.get("runtime", {})
                status = runtime.get("stage", "unknown")
                print(f"⏳ Attempt {attempt}/{MAX_RETRIES}: status = {status}")

                if status == "RUNNING":
                    print(f"\n✅ Space is RUNNING on HF API!")
                    print(f"   Verifying Gradio app is serving...")
                    if check_gradio_ready(SPACE_ID):
                        print(f"\n✅ Space is LIVE!")
                        print(f"   https://huggingface.co/spaces/{SPACE_ID}")
                        return
                    else:
                        # Gradio didn't come up — don't fail here, but warn
                        print(f"\n⚠️ Space API reports RUNNING but Gradio endpoint not responding.")
                        print(f"   Check manually: https://huggingface.co/spaces/{SPACE_ID}")
                        sys.exit(1)
                elif status == "BUILD_ERROR":
                    print(f"\n❌ Space build failed. Check logs:")
                    print(f"   https://huggingface.co/spaces/{SPACE_ID}/logs")
                    sys.exit(1)
            else:
                print(f"⏳ Attempt {attempt}/{MAX_RETRIES}: API returned {resp.status_code}")
        except requests.RequestException as e:
            print(f"⏳ Attempt {attempt}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    print(f"\n⚠️ Space did not reach RUNNING within {MAX_RETRIES * RETRY_DELAY}s.")
    print(f"   Check dashboard: https://huggingface.co/spaces/{SPACE_ID}")
    sys.exit(1)


if __name__ == "__main__":
    main()
