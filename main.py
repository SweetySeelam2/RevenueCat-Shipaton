from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import shap
import numpy as np
import os

# =========================
# Paths (env-overridable)
# =========================
# Cancellation
MODEL_PATH = os.getenv("MODEL_PATH", "model/rf_cancel_model_fixed.pkl")
FEATURES_SAMPLE_PATH = os.getenv("FEATURES_SAMPLE_PATH", "data/flight_cancel_features.csv")
CANCEL_FEATURE_ORDER_CSV = os.getenv("CANCEL_FEATURE_ORDER_CSV", "model/cancel_feature_order_fixed.csv")

# Delay (LEAKAGE-FREE)
MODEL_DELAY_PATH = os.getenv("MODEL_DELAY_PATH", "model/rf_delay_model_fixed.pkl")
DELAY_FEATURES_SAMPLE_PATH = os.getenv("DELAY_FEATURES_SAMPLE_PATH", "data/flight_features.csv")
DELAY_FEATURE_ORDER_CSV = os.getenv("DELAY_FEATURE_ORDER_CSV", "model/delay_feature_order_fixed.csv")

# =========================
# Helpers
# =========================
def file_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False

def load_feature_names_cancel() -> list[str]:
    """
    Prefer dedicated feature-order CSV. If not present, fall back to header of data CSV.
    """
    if file_exists(CANCEL_FEATURE_ORDER_CSV):
        names = pd.read_csv(CANCEL_FEATURE_ORDER_CSV, header=None)[0].tolist()
        return names
    # Fallback for local dev only
    if not file_exists(FEATURES_SAMPLE_PATH):
        raise FileNotFoundError(
            f"Missing both {CANCEL_FEATURE_ORDER_CSV} and {FEATURES_SAMPLE_PATH} for cancel features."
        )
    df = pd.read_csv(FEATURES_SAMPLE_PATH)
    # keep in sync with your training cleanup
    for col in ['MISSING_AIR_TIME', 'MISSING_ELAPSED_TIME', 'MISSING_CRS_ELAPSED_TIME', 'LONG_AIR_TIME']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return [c for c in df.columns if c != 'CANCELLED']

def load_feature_names_delay() -> list[str]:
    """
    Prefer dedicated feature-order CSV. If not present, fall back to data CSV header
    with leakage columns dropped (clean production-safe set).
    """
    if file_exists(DELAY_FEATURE_ORDER_CSV):
        names = pd.read_csv(DELAY_FEATURE_ORDER_CSV, header=None)[0].tolist()
        return names
    # Fallback for local dev only
    if not file_exists(DELAY_FEATURES_SAMPLE_PATH):
        raise FileNotFoundError(
            f"Missing both {DELAY_FEATURE_ORDER_CSV} and {DELAY_FEATURES_SAMPLE_PATH} for delay features."
        )
    df = pd.read_csv(DELAY_FEATURES_SAMPLE_PATH)
    delay_drop = [
        'CANCELLED', 'DEP_DELAY_15', 'ARR_DELAY_15',
        'AIRLINE_DOT', 'AIRLINE_CODE', 'ORIGIN_CITY', 'DEST_CITY',
        # leakage / post-event signals removed for production-safe model:
        'AIR_TIME', 'ELAPSED_TIME', 'CRS_ELAPSED_TIME', 'IS_DELAYED',
        'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN',
        'DEP_TIME', 'ARR_TIME',
    ]
    for c in delay_drop:
        if c in df.columns:
            df = df.drop(columns=[c])
    return [c for c in df.columns if c != 'DEP_DELAY_15']

# =========================
# Load models + features
# =========================
# Cancel model
rf_cancel = joblib.load(MODEL_PATH)
cancel_feature_names = load_feature_names_cancel()
explainer_cancel = shap.TreeExplainer(rf_cancel)

# Delay model
try:
    rf_delay = joblib.load(MODEL_DELAY_PATH)
    delay_feature_names = load_feature_names_delay()
    explainer_delay = shap.TreeExplainer(rf_delay)
except Exception as e:
    rf_delay = None
    delay_feature_names = []
    explainer_delay = None
    print(f"⚠️ Delay model unavailable: {e}")

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
    # Optional field (ignored if not in feature list)
    LONG_AIR_TIME: Optional[int] = 0

# =========================
# App
# =========================
app = FastAPI(
    title="CancelSense API",
    version="1.3.0",
    description="Predict flight cancellation & delay risk with explanations (leakage-free delay)."
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cancel_features_count": len(cancel_feature_names),
        "delay_features_count": len(delay_feature_names),
        "cancel_model_path": MODEL_PATH,
        "delay_model_path": MODEL_DELAY_PATH,
        "delay_model_loaded": rf_delay is not None
    }

# ========== Cancellation ==========
@app.post("/predict")
def predict(f: FlightFeatures):
    row = {k: getattr(f, k) for k in f.__fields__.keys() if k in cancel_feature_names}
    for col in cancel_feature_names:
        row.setdefault(col, 0)
    X = pd.DataFrame([row], columns=cancel_feature_names)

    proba = rf_cancel.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    return {"cancel_probability": float(proba), "predicted_label": pred}

@app.post("/explain")
def explain(f: FlightFeatures, topk: int = 8):
    row = {k: getattr(f, k) for k in f.__fields__.keys() if k in cancel_feature_names}
    for col in cancel_feature_names:
        row.setdefault(col, 0)
    X = pd.DataFrame([row], columns=cancel_feature_names)

    sv = explainer_cancel.shap_values(X)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer_cancel.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer_cancel.expected_value)

    contrib = list(zip(cancel_feature_names, shap_vals))
    contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:topk]

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

# ============= Delay =============
@app.post("/predict_delay")
def predict_delay(f: FlightFeatures):
    if rf_delay is None:
        return {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}

    row = {k: getattr(f, k) for k in f.__fields__.keys() if k in delay_feature_names}
    for col in delay_feature_names:
        row.setdefault(col, 0)
    Xd = pd.DataFrame([row], columns=delay_feature_names)

    proba = rf_delay.predict_proba(Xd)[0, 1]
    pred = int(proba >= 0.5)
    return {"delay_probability": float(proba), "predicted_label": pred}

@app.post("/explain_delay")
def explain_delay(f: FlightFeatures, topk: int = 8):
    if rf_delay is None or explainer_delay is None:
        return {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}

    row = {k: getattr(f, k) for k in f.__fields__.keys() if k in delay_feature_names}
    for col in delay_feature_names:
        row.setdefault(col, 0)
    Xd = pd.DataFrame([row], columns=delay_feature_names)

    sv = explainer_delay.shap_values(Xd)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
        base_value = float(explainer_delay.expected_value[1])
    else:
        shap_vals = sv[0]
        base_value = float(explainer_delay.expected_value)

    contrib = list(zip(delay_feature_names, shap_vals))
    contrib_sorted = sorted(contrib, key=lambda x: abs(x[1]), reverse=True)[:topk]

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

# ======== Combined (both) ========
@app.post("/predict_both")
def predict_both(f: FlightFeatures):
    # Cancellation
    row_c = {k: getattr(f, k) for k in f.__fields__.keys() if k in cancel_feature_names}
    for col in cancel_feature_names:
        row_c.setdefault(col, 0)
    Xc = pd.DataFrame([row_c], columns=cancel_feature_names)
    cancel_proba = float(rf_cancel.predict_proba(Xc)[0, 1])
    cancel_pred  = int(cancel_proba >= 0.5)

    # Delay
    if rf_delay is None:
        delay = {"error": f"Delay model not loaded at {MODEL_DELAY_PATH}"}
    else:
        row_d = {k: getattr(f, k) for k in f.__fields__.keys() if k in delay_feature_names}
        for col in delay_feature_names:
            row_d.setdefault(col, 0)
        Xd = pd.DataFrame([row_d], columns=delay_feature_names)
        delay_proba = float(rf_delay.predict_proba(Xd)[0, 1])
        delay_pred  = int(delay_proba >= 0.5)
        delay = {"delay_probability": delay_proba, "predicted_label": delay_pred}

    return {
        "cancellation": {"cancel_probability": cancel_proba, "predicted_label": cancel_pred},
        "delay": delay
    }

# To run locally:
#   uvicorn main:app --reload --host 0.0.0.0 --port 8000