# main.py
import os
import threading
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# =========================
# Early: ensure models exist (startup self-heal)
# =========================
try:
    from fetch_models import ensure_models
    ensure_models()
except Exception as e:
    print(f"[startup] ensure_models warning: {e}")

# =========================
# Env / paths
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
MODEL_DELAY_PATH = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")

CANCEL_FEATURE_ORDER_CSV = os.getenv(
    "CANCEL_FEATURE_ORDER_CSV", "model/cancel_feature_order_fixed.csv"
)
DELAY_FEATURE_ORDER_CSV = os.getenv(
    "DELAY_FEATURE_ORDER_CSV", "model/delay_feature_order_fixed.csv"
)

# =========================
# Globals (lazy cache)
# =========================
_cancel_model = None
_delay_model = None
_cancel_features: List[str] = []
_delay_features: List[str] = []
_explainer_cancel = None
_explainer_delay = None

_features_lock = threading.Lock()
_model_lock = threading.Lock()
_shap_lock = threading.Lock()

# =========================
# Helpers
# =========================
def _file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False

def ensure_cancel_ready():
    """If the cancel model file is missing, try to download it now."""
    if not _file_exists(MODEL_PATH):
        try:
            from fetch_models import ensure_models
            ensure_models()
        except Exception as e:
            print(f"[lazy] ensure_cancel_ready failed: {e}")

def ensure_delay_ready():
    """If the delay model file is missing, try to download it now."""
    if not _file_exists(MODEL_DELAY_PATH):
        try:
            from fetch_models import ensure_models
            ensure_models()
        except Exception as e:
            print(f"[lazy] ensure_delay_ready failed: {e}")

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
    """Lazy-load cancel model, self-healing if file is missing."""
    global _cancel_model
    if _cancel_model is not None:
        return _cancel_model
    with _model_lock:
        if _cancel_model is None:
            ensure_cancel_ready()
            print("[lazy] loading cancel model …")
            _cancel_model = joblib.load(MODEL_PATH)
    return _cancel_model

def get_delay_model():
    """Lazy-load delay model, self-healing if file is missing."""
    global _delay_model
    if _delay_model is not None:
        return _delay_model
    with _model_lock:
        if _delay_model is None:
            ensure_delay_ready()
            print("[lazy] loading delay model …")
            _delay_model = joblib.load(MODEL_DELAY_PATH)
    return _delay_model

def get_cancel_explainer():
    global _explainer_cancel
    if _explainer_cancel is not None:
        return _explainer_cancel
    with _shap_lock:
        if _explainer_cancel is None:
            print("[lazy] building SHAP explainer (cancel) …")
            import shap  # import only when needed
            _explainer_cancel = shap.TreeExplainer(get_cancel_model())
    return _explainer_cancel

def get_delay_explainer():
    global _explainer_delay
    if _explainer_delay is not None:
        return _explainer_delay
    with _shap_lock:
        if _explainer_delay is None:
            print("[lazy] building SHAP explainer (delay) …")
            import shap  # import only when needed
            _explainer_delay = shap.TreeExplainer(get_delay_model())
    return _explainer_delay

def _as_row(payload: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
    row = {k: payload.get(k, 0) for k in columns}
    return pd.DataFrame([row], columns=columns)

# =========================
# API
# =========================
app = FastAPI(
    title="CancelSense API",
    version="1.5.0",
    description="Predict flight cancellation & delay risk with lazy-loaded, self-healing models and SHAP explanations."
)

# Simple root to confirm app is alive fast
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
        # Show both in-memory and on-disk status to diagnose issues quickly
        "cancel_model_loaded": _cancel_model is not None,
        "delay_model_loaded": _delay_model is not None,
        "cancel_model_file_exists": _file_exists(MODEL_PATH),
        "delay_model_file_exists": _file_exists(MODEL_DELAY_PATH),
        "shap_cancel_ready": _explainer_cancel is not None,
        "shap_delay_ready": _explainer_delay is not None,
    }

# =========================
# Schemas
# =========================
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

# =========================
# Endpoints
# =========================
@app.post("/predict")
def predict(f: FlightFeatures):
    model = get_cancel_model()
    feats = get_cancel_features()
    X = _as_row(f.dict(), feats)
    proba = model.predict_proba(X)[0, 1]
    return {"cancel_probability": float(proba), "predicted_label": int(proba >= 0.5)}

@app.post("/explain")
def explain(f: FlightFeatures, topk: int = 8):
    model = get_cancel_model()
    feats = get_cancel_features()
    X = _as_row(f.dict(), feats)
    explainer = get_cancel_explainer()
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer.expected_value)
    contrib = sorted(zip(feats, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:topk]
    return {
        "base_value": base_value,
        "cancel_probability": float(model.predict_proba(X)[0, 1]),
        "top_contributions": [
            {
                "feature": name,
                "shap_value": float(val),
                "value": float(X.iloc[0][name]) if np.issubdtype(type(X.iloc[0][name]), np.number) else X.iloc[0][name],
            }
            for name, val in contrib
        ],
    }

@app.post("/predict_delay")
def predict_delay(f: FlightFeatures):
    model = get_delay_model()
    feats = get_delay_features()
    X = _as_row(f.dict(), feats)
    proba = model.predict_proba(X)[0, 1]
    return {"delay_probability": float(proba), "predicted_label": int(proba >= 0.5)}

@app.post("/explain_delay")
def explain_delay(f: FlightFeatures, topk: int = 8):
    model = get_delay_model()
    feats = get_delay_features()
    X = _as_row(f.dict(), feats)
    explainer = get_delay_explainer()
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer.expected_value)
    contrib = sorted(zip(feats, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:topk]
    return {
        "base_value": base_value,
        "delay_probability": float(model.predict_proba(X)[0, 1]),
        "top_contributions": [
            {
                "feature": name,
                "shap_value": float(val),
                "value": float(X.iloc[0][name]) if np.issubdtype(type(X.iloc[0][name]), np.number) else X.iloc[0][name],
            }
            for name, val in contrib
        ],
    }

# =========================
# Background warm-up
# =========================
def _background_warm():
    try:
        get_cancel_features()
        get_delay_features()
        # Only load models here; build SHAP on demand to keep warmup fast
        get_cancel_model()
        get_delay_model()
        print("[warmup] models loaded.")
    except Exception as e:
        print(f"[warmup] skipped/failed: {e}")

@app.on_event("startup")
def on_startup():
    # Kick off background warmup after the server is listening
    threading.Thread(target=_background_warm, daemon=True).start()