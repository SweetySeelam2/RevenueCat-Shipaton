
CancelSense API — Quickstart
============================
Prereqs:
  pip install fastapi uvicorn shap scikit-learn pandas

Files created:
  - /mnt/data/main.py
  - /mnt/data/flight_cancel_report.html

Env vars (optional):
  - MODEL_PATH (default: model/rf_cancel_model_fixed.pkl)
  - FEATURES_SAMPLE_PATH (default: data/flight_cancel_features.csv)

Run the API:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Sample request (predict):
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
    "DOT_CODE": 19790,
    "FL_NUMBER": 2486,
    "AIRLINE": 3,
    "ORIGIN": 96,
    "DEST": 23,
    "CRS_DEP_TIME": 815,
    "CRS_ARR_TIME": 1000,
    "DIVERTED": 0,
    "DISTANCE": 547,
    "YEAR": 2021,
    "MONTH": 6,
    "DAY": 13,
    "DAY_OF_WEEK": 6,
    "NIGHT_FLIGHT": 0
  }'

Sample request (explain):
  curl -X POST "http://localhost:8000/explain?topk=8" -H "Content-Type: application/json" -d '@sample.json'

Notes:
- Ensure your model and data live at the expected paths, or set MODEL_PATH / FEATURES_SAMPLE_PATH.
- The /explain route returns the top SHAP contributors for class=1 (cancelled).
