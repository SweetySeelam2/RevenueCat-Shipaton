# main.py
import os, threading, concurrent.futures
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# -------------------------
# NumPy 2.x compatibility shim (old libs may reference np.bool/np.int/np.float)
# -------------------------
try:
    if not hasattr(np, "bool"):  np.bool  = bool   # type: ignore[attr-defined]
    if not hasattr(np, "int"):   np.int   = int    # type: ignore[attr-defined]
    if not hasattr(np, "float"): np.float = float  # type: ignore[attr-defined]
except Exception as _e:
    print(f"[startup] numpy shim warning: {_e}")

# --- try to ensure models are on disk (HF download) ---
try:
    from fetch_models import ensure_models
    ensure_models()
except Exception as e:
    print(f"[startup] ensure_models warning: {e}")

# also try lazy import if still missing
if not (os.path.exists("model/rf_cancel_model_fixed.pkl") and os.path.exists("model/rf_delay_model_fixed.pkl")):
    try:
        import fetch_models  # runs on import
    except Exception as e:
        print(f"[startup] fetch_models failed (will still try lazy loads): {e}")

# ============== Env / paths ==============
MODEL_PATH = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
MODEL_DELAY_PATH = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")
CANCEL_FEATURE_ORDER_CSV = os.getenv("CANCEL_FEATURE_ORDER_CSV", "model/cancel_feature_order_fixed.csv")
DELAY_FEATURE_ORDER_CSV = os.getenv("DELAY_FEATURE_ORDER_CSV", "model/delay_feature_order_fixed.csv")

# SHAP toggles / timeouts
DISABLE_SHAP = os.getenv("DISABLE_SHAP", "0") == "1"    # <-- set to 1 on Railway free
SHAP_TIMEOUT_S = int(os.getenv("SHAP_TIMEOUT_S", "12")) # hard cap

# ============== Globals / locks ==============
_cancel_model = None
_delay_model = None
_cancel_features: List[str] = []
_delay_features: List[str] = []

_explainer_cancel = None
_explainer_delay = None
_shap_cancel_error: Optional[str] = None
_shap_delay_error: Optional[str] = None

_features_lock = threading.Lock()
_model_lock = threading.Lock()
_shap_lock = threading.Lock()

# small thread pool for SHAP calls (so we can enforce a timeout)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False

def _load_feature_names_from_csv(one_col_csv_path: str) -> List[str]:
    if not _file_exists(one_col_csv_path):
        raise FileNotFoundError(f"Feature order CSV not found: {one_col_csv_path}")
    return pd.read_csv(one_col_csv_path, header=None)[0].tolist()

def get_cancel_features() -> List[str]:
    global _cancel_features
    if _cancel_features:
        return _cancel_features
    with _features_lock:
        if _cancel_features:
            return _cancel_features
        _cancel_features = _load_feature_names_from_csv(CANCEL_FEATURE_ORDER_CSV)
    return _cancel_features

def get_delay_features() -> List[str]:
    global _delay_features
    if _delay_features:
        return _delay_features
    with _features_lock:
        if _delay_features:
            return _delay_features
        _delay_features = _load_feature_names_from_csv(DELAY_FEATURE_ORDER_CSV)
    return _delay_features

def get_cancel_model():
    global _cancel_model
    if _cancel_model is not None:
        return _cancel_model
    with _model_lock:
        if _cancel_model is None:
            print("[lazy] loading cancel model …")
            _cancel_model = joblib.load(MODEL_PATH)
    return _cancel_model

def get_delay_model():
    global _delay_model
    if _delay_model is not None:
        return _delay_model
    with _model_lock:
        if _delay_model is None:
            print("[lazy] loading delay model …")
            _delay_model = joblib.load(MODEL_DELAY_PATH)
    return _delay_model

def _build_tree_explainer(model, which: str):
    global _shap_cancel_error, _shap_delay_error
    try:
        import shap  # import only after numpy shim
        expl = shap.TreeExplainer(
            model,
            feature_perturbation="interventional",
            model_output="probability",
            check_additivity=False
        )
        return expl, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if which == "cancel":
            _shap_cancel_error = err
        else:
            _shap_delay_error = err
        return None, err

def get_cancel_explainer():
    global _explainer_cancel
    if DISABLE_SHAP:
        return None
    if _explainer_cancel is not None:
        return _explainer_cancel
    with _shap_lock:
        if _explainer_cancel is None:
            print("[lazy] building SHAP explainer (cancel) …")
            expl, _ = _build_tree_explainer(get_cancel_model(), "cancel")
            _explainer_cancel = expl
    return _explainer_cancel

def get_delay_explainer():
    global _explainer_delay
    if DISABLE_SHAP:
        return None
    if _explainer_delay is not None:
        return _explainer_delay
    with _shap_lock:
        if _explainer_delay is None:
            print("[lazy] building SHAP explainer (delay) …")
            expl, _ = _build_tree_explainer(get_delay_model(), "delay")
            _explainer_delay = expl
    return _explainer_delay

def _as_row(payload: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
    row = {k: payload.get(k, 0) for k in columns}
    return pd.DataFrame([row], columns=columns)

def _safe_topk_contrib(names: List[str], values: np.ndarray, Xrow: pd.Series, topk: int):
    pairs = list(zip(names, values))
    pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:topk]
    out = []
    for name, val in pairs:
        v = Xrow[name]
        out.append({
            "feature": name,
            "shap_value": float(val),
            "value": float(v) if np.issubdtype(type(v), np.number) else v
        })
    return out

def _fallback_importances(model, feats: List[str], Xrow: pd.Series, topk: int):
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_, dtype=float)
        top = _safe_topk_contrib(feats, fi, Xrow, topk)
        return {"mode": "feature_importances", "top_contributions": top}
    return {"mode": "none", "reason": "no feature_importances_ on model"}

def _shap_values_with_timeout(explainer, X):
    # run SHAP in a worker so we can time out
    future = _executor.submit(explainer.shap_values, X)
    return future.result(timeout=SHAP_TIMEOUT_S)

# ============== API ==============
app = FastAPI(
    title="CancelSense API",
    version="1.7.0",
    description="Predict cancellation & delay risk. SHAP can be disabled and is time-limited."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "message": "CancelSense API is running. See /health."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cancel_model_path": MODEL_PATH,
        "delay_model_path": MODEL_DELAY_PATH,
        "cancel_features_count": len(_cancel_features) or None,
        "delay_features_count": len(_delay_features) or None,
        "cancel_model_loaded": _cancel_model is not None,
        "delay_model_loaded": _delay_model is not None,
        "cancel_model_file_exists": _file_exists(MODEL_PATH),
        "delay_model_file_exists": _file_exists(MODEL_DELAY_PATH),
        "shap_cancel_ready": (not DISABLE_SHAP) and (_explainer_cancel is not None),
        "shap_delay_ready":  (not DISABLE_SHAP) and (_explainer_delay is not None),
        "shap_disabled": DISABLE_SHAP,
        "shap_timeout_sec": SHAP_TIMEOUT_S
    }

@app.get("/debug/shap_status")
def shap_status():
    return {
        "shap_cancel_ready": (not DISABLE_SHAP) and (_explainer_cancel is not None),
        "shap_delay_ready":  (not DISABLE_SHAP) and (_explainer_delay is not None),
        "shap_cancel_error": _shap_cancel_error,
        "shap_delay_error": _shap_delay_error,
        "shap_disabled": DISABLE_SHAP,
        "shap_timeout_sec": SHAP_TIMEOUT_S
    }

# ----- schema -----
class FlightFeatures(BaseModel):
    DOT_CODE: int
    FL_NUMBER: int
    AIRLINE: int
    ORIGIN: int
    DEST: int
    CRS_DEP_TIME: float
    CRS_ARR_TIME: float
    DIVERTED: int
    DISTANCE: float
    YEAR: int
    MONTH: int
    DAY: int
    DAY_OF_WEEK: int
    NIGHT_FLIGHT: int
    LONG_AIR_TIME: Optional[int] = 0

# ----- predictions -----
@app.post("/predict")
def predict(f: FlightFeatures):
    model = get_cancel_model()
    feats = get_cancel_features()
    X = _as_row(f.dict(), feats)
    proba = model.predict_proba(X)[0, 1]
    return {"cancel_probability": float(proba), "predicted_label": int(proba >= 0.5)}

@app.post("/predict_delay")
def predict_delay(f: FlightFeatures):
    model = get_delay_model()
    feats = get_delay_features()
    X = _as_row(f.dict(), feats)
    proba = model.predict_proba(X)[0, 1]
    return {"delay_probability": float(proba), "predicted_label": int(proba >= 0.5)}

# ----- explanations -----
@app.post("/explain")
def explain(f: FlightFeatures, topk: int = 8):
    model = get_cancel_model()
    feats = get_cancel_features()
    X = _as_row(f.dict(), feats)

    if DISABLE_SHAP:
        return _fallback_importances(model, feats, X.iloc[0], topk)

    try:
        explainer = get_cancel_explainer()
        if explainer is None:
            raise RuntimeError(_shap_cancel_error or "SHAP explainer unavailable")
        sv = _shap_values_with_timeout(explainer, X)
        if isinstance(sv, list):
            shap_vals = sv[1][0]
            base_value = float(getattr(explainer, "expected_value", [0.0, 0.0])[1])
        else:
            shap_vals = sv[0]
            base_value = float(getattr(explainer, "expected_value", 0.0))
        top = _safe_topk_contrib(feats, shap_vals, X.iloc[0], topk)
        return {"mode": "tree_shap", "base_value": base_value,
                "cancel_probability": float(model.predict_proba(X)[0, 1]),
                "top_contributions": top}
    except Exception as e:
        return _fallback_importances(model, feats, X.iloc[0], topk) | {"fallback_reason": f"{type(e).__name__}: {e}"}

@app.post("/explain_delay")
def explain_delay(f: FlightFeatures, topk: int = 8):
    model = get_delay_model()
    feats = get_delay_features()
    X = _as_row(f.dict(), feats)

    if DISABLE_SHAP:
        return _fallback_importances(model, feats, X.iloc[0], topk)

    try:
        explainer = get_delay_explainer()
        if explainer is None:
            raise RuntimeError(_shap_delay_error or "SHAP explainer unavailable")
        sv = _shap_values_with_timeout(explainer, X)
        if isinstance(sv, list):
            shap_vals = sv[1][0]
            base_value = float(getattr(explainer, "expected_value", [0.0, 0.0])[1])
        else:
            shap_vals = sv[0]
            base_value = float(getattr(explainer, "expected_value", 0.0))
        top = _safe_topk_contrib(feats, shap_vals, X.iloc[0], topk)
        return {"mode": "tree_shap", "base_value": base_value,
                "delay_probability": float(model.predict_proba(X)[0, 1]),
                "top_contributions": top}
    except Exception as e:
        return _fallback_importances(model, feats, X.iloc[0], topk) | {"fallback_reason": f"{type(e).__name__}: {e}"}

# Background warmup (models only; SHAP builds on-demand)
def _background_warm():
    try:
        get_cancel_features()
        get_delay_features()
        get_cancel_model()
        get_delay_model()
        print("[warmup] models loaded.")
    except Exception as e:
        print(f"[warmup] skipped/failed: {e}")

@app.on_event("startup")
def on_startup():
    threading.Thread(target=_background_warm, daemon=True).start()