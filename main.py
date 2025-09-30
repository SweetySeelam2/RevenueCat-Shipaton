from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import pandas as pd
import numpy as np
import joblib
import shap

# =========================================================
# Paths (override via env if you deploy)
# =========================================================
# Models
MODEL_PATH = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
MODEL_DELAY_PATH = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")  # or rf_delay_model.pkl

# Small CSVs that only contain the exact feature order used at training
#   - one feature name per line, no header
CANCEL_FEATURE_ORDER_CSV = os.getenv("CANCEL_FEATURE_ORDER_CSV", "model/cancel_feature_order_fixed.csv")
DELAY_FEATURE_ORDER_CSV  = os.getenv("DELAY_FEATURE_ORDER_CSV",  "model/delay_feature_order_fixed.csv")

# =========================================================
# Load models
# =========================================================
rf_cancel = joblib.load(MODEL_PATH)

try:
    rf_delay = joblib.load(MODEL_DELAY_PATH)
except Exception as e:
    rf_delay = None
    print(f"⚠️ Could not load delay model at {MODEL_DELAY_PATH}: {e}")

# =========================================================
# Load feature orders (no big datasets needed)
# =========================================================
def _read_feature_list(path: str) -> List[str]:
    names = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
    # strip accidental whitespace
    return [n.strip() for n in names]

cancel_feature_names: List[str] = _read_feature_list(CANCEL_FEATURE_ORDER_CSV)
delay_feature_names: List[str]  = _read_feature_list(DELAY_FEATURE_ORDER_CSV)

# =========================================================
# SHAP explainers
# =========================================================
explainer_cancel = shap.TreeExplainer(rf_cancel)
explainer_delay  = shap.TreeExplainer(rf_delay) if rf_delay is not None else None

# =========================================================
# Schemas (fields common to both models; OPTIONALs are filled with 0 if unused)
# Make sure these names match your encoded features; extra fields are ignored.
# =========================================================
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

    # Optional; will be ignored if not present in the model's feature list
    LONG_AIR_TIME: Optional[int] = 0

# =========================================================
# Helpers
# =========================================================
def _row_to_df(f: FlightFeatures, feature_list: List[str]) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact column order the model expects.
    Missing columns are filled with 0. Extra fields in the schema are ignored."""
    payload: Dict[str, Any] = {name: getattr(f, name, 0) for name in feature_list}
    # numbers only; if any None slips in, default to 0
    for k, v in list(payload.items()):
        if v is None:
            payload[k] = 0
    return pd.DataFrame([payload], columns=feature_list)

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="CancelSense API",
    version="1.3.0",
    description="Predict flight cancellation & delay risk with explanations (delay model is leakage-free)."
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cancel_model_path": MODEL_PATH,
        "delay_model_path": MODEL_DELAY_PATH,
        "delay_model_loaded": rf_delay is not None,
        "cancel_features": cancel_feature_names,
        "delay_features": delay_feature_names
    }

# =======================
# Cancellation Endpoints
# =======================
@app.post("/predict")
def predict_cancel(f: FlightFeatures):
    X = _row_to_df(f, cancel_feature_names)
    proba = float(rf_cancel.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {"cancel_probability": proba, "predicted_label": pred}

@app.post("/explain")
def explain_cancel(f: FlightFeatures, topk: int = 8):
    X = _row_to_df(f, cancel_feature_names)

    # SHAP for class 1
    sv = explainer_cancel.shap_values(X)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer_cancel.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer_cancel.expected_value)

    contrib = list(zip(cancel_feature_names, shap_vals))
    contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:max(1, topk)]

    return {
        "base_value": base_value,
        "cancel_probability": float(rf_cancel.predict_proba(X)[0, 1]),
        "top_contributions": [
            {
                "feature": name,
                "shap_value": float(val),
                "value": float(X.iloc[0][name]) if np.issubdtype(type(X.iloc[0][name]), np.number) else X.iloc[0][name]
            }
            for name, val in contrib_sorted
        ]
    }

# ==============
# Delay Endpoints
# ==============
@app.post("/predict_delay")
def predict_delay(f: FlightFeatures):
    if rf_delay is None:
        return {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}
    Xd = _row_to_df(f, delay_feature_names)
    proba = float(rf_delay.predict_proba(Xd)[0, 1])
    pred = int(proba >= 0.5)
    return {"delay_probability": proba, "predicted_label": pred}

@app.post("/explain_delay")
def explain_delay(f: FlightFeatures, topk: int = 8):
    if rf_delay is None or explainer_delay is None:
        return {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}
    Xd = _row_to_df(f, delay_feature_names)

    sv = explainer_delay.shap_values(Xd)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer_delay.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer_delay.expected_value)

    contrib = list(zip(delay_feature_names, shap_vals))
    contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:max(1, topk)]

    return {
        "base_value": base_value,
        "delay_probability": float(rf_delay.predict_proba(Xd)[0, 1]),
        "top_contributions": [
            {
                "feature": name,
                "shap_value": float(val),
                "value": float(Xd.iloc[0][name]) if np.issubdtype(type(Xd.iloc[0][name]), np.number) else Xd.iloc[0][name]
            }
            for name, val in contrib_sorted
        ]
    }

# ======================
# Combined (both models)
# ======================
@app.post("/predict_both")
def predict_both(f: FlightFeatures):
    # Cancellation
    Xc = _row_to_df(f, cancel_feature_names)
    cancel_proba = float(rf_cancel.predict_proba(Xc)[0, 1])
    cancel_pred  = int(cancel_proba >= 0.5)

    # Delay
    if rf_delay is None:
        delay_payload: Dict[str, Any] = {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}
    else:
        Xd = _row_to_df(f, delay_feature_names)
        delay_proba = float(rf_delay.predict_proba(Xd)[0, 1])
        delay_pred  = int(delay_proba >= 0.5)
        delay_payload = {"delay_probability": delay_proba, "predicted_label": delay_pred}

    return {
        "cancellation": {"cancel_probability": cancel_proba, "predicted_label": cancel_pred},
        "delay": delay_payload
    }

# ======================
# How to run locally
# ======================
#   pip install -r requirements.txt
#   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000