# fetch_models.py
import os, pathlib, urllib.request

CANCEL_URL = os.getenv("MODEL_URL")            # remote .pkl for cancellation
DELAY_URL  = os.getenv("MODEL_DELAY_URL")      # remote .pkl for delay

CANCEL_PATH = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
DELAY_PATH  = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")

def _download(url: str, out_path: str):
    if not url: 
        return
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_models] downloading {url} -> {out_path} …")
    urllib.request.urlretrieve(url, out_path)
    print(f"[fetch_models] saved {out_path} ({os.path.getsize(out_path)} bytes)")

# Only download if the file is missing
if not os.path.exists(CANCEL_PATH):
    _download(CANCEL_URL, CANCEL_PATH)
else:
    print(f"[fetch_models] found {CANCEL_PATH}, skipping")

if not os.path.exists(DELAY_PATH):
    _download(DELAY_URL, DELAY_PATH)
else:
    print(f"[fetch_models] found {DELAY_PATH}, skipping")
