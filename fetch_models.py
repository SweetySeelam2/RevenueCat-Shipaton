import os
import requests

# URLs from Hugging Face Hub
CANCEL_URL = os.getenv("CANCEL_MODEL_URL")
DELAY_URL = os.getenv("DELAY_MODEL_URL")

os.makedirs("model", exist_ok=True)

def download(url, out_path):
    if not url:
        print(f"⚠️ Skipping {out_path}, no URL provided")
        return
    print(f"⬇️ Downloading {url} → {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Saved {out_path}")

download(CANCEL_URL, "model/rf_cancel_model_fixed.pkl")
download(DELAY_URL, "model/rf_delay_model_fixed.pkl")
