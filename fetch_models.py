# fetch_models.py
import os, urllib.request, pathlib

CANCEL_URL = os.getenv("CANCEL_MODEL_URL", "").strip()
DELAY_URL  = os.getenv("DELAY_MODEL_URL", "").strip()

def dl(url: str, out_path: str):
    if not url:
        print(f"SKIP: no URL for {out_path}")
        return
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ downloading {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"✅ wrote {out_path} ({p.stat().st_size} bytes)")

if __name__ == "__main__":
    # Keep these in sync with main.py defaults (or set via env in your host)
    cancel_out = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
    delay_out  = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")

    dl(CANCEL_URL, cancel_out)
    dl(DELAY_URL,  delay_out)

    print("Done fetching models.")
