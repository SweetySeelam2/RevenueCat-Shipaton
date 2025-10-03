# fetch_models.py
import os, sys, urllib.request, pathlib, time

MODEL_DIR = os.getenv("MODEL_DIR", "model")
CANCEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "rf_cancel_model_fixed.pkl"))
DELAY_PATH  = os.getenv("MODEL_DELAY_PATH", os.path.join(MODEL_DIR, "rf_delay_model_fixed.pkl"))

CANCEL_URL = os.getenv("MODEL_URL", "")
DELAY_URL  = os.getenv("MODEL_DELAY_URL", "")

def _download(url: str, out_path: str, name: str, retries=3, backoff=2):
    if not url or not url.startswith("http"):
        print(f"[fetch_models] {name}: invalid or empty URL: {url!r}", file=sys.stderr)
        return False
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, retries+1):
        try:
            print(f"[fetch_models] downloading {url} -> {out_path} …", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(out_path, "wb") as f:
                f.write(r.read())
            size = os.path.getsize(out_path)
            print(f"[fetch_models] saved {out_path} ({size} bytes)", flush=True)
            return True
        except Exception as e:
            print(f"[fetch_models] attempt {i}/{retries} failed for {name}: {e}", file=sys.stderr, flush=True)
            time.sleep(backoff * i)
    return False

def ensure_models():
    ok = True
    if not os.path.isfile(CANCEL_PATH):
        ok = _download(CANCEL_URL, CANCEL_PATH, "cancel") and ok
    if not os.path.isfile(DELAY_PATH):
        ok = _download(DELAY_URL, DELAY_PATH, "delay") and ok
    return ok

# Run at import during cold start (so Railway startup pulls the files)
if __name__ == "__main__" or True:
    try:
        ensure_models()
    except Exception as e:
        print(f"[startup] fetch_models failed (will still try lazy loads): {e}", file=sys.stderr)