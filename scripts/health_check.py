"""
Check HF Space deployment status after CI/CD deploy.
Polls the HF API until the Space reaches RUNNING state.
"""

import os
import sys
import time
import requests

SPACE_ID = os.environ.get("HF_SPACE_ID", "karaboLLM/ml-observability")
MAX_RETRIES = 24
RETRY_DELAY = 15


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
                    print(f"\n✅ Space is LIVE!")
                    print(f"   https://huggingface.co/spaces/{SPACE_ID}")
                    return
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
