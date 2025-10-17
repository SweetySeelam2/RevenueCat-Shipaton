# ✈️ RevenueCat-Shipaton (CancelSense API)

Predict **flight cancellation** and **delay probabilities** with explainable AI - powered by Random Forest models and served through a **FastAPI** backend on **Railway**.

---

## 🚀 Live API

**Base URL:**  
👉 [https://revenuecat-shipaton-production.up.railway.app/](https://revenuecat-shipaton-production.up.railway.app/)

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Root ping — confirms the API is online |
| `/health` | GET | System status & model metadata |
| `/predict` | POST | Predict flight **cancellation risk** |
| `/predict_delay` | POST | Predict flight **delay risk** |
| `/explain_safe` | POST | Fast feature-importance explanations (fallback mode) |
| `/explain_delay_safe` | POST | Same for delay model |
| `/debug/shap_status` | GET | Internal diagnostics for SHAP explainers |
| `/docs` | GET | Interactive Swagger UI (OpenAPI spec) |

---

## 🧠 Model Overview

Two **Random Forest Classifiers** trained on FAA / BTS-style flight records:

| Model | Target | File | Features |
|--------|---------|------|-----------|
| `rf_cancel_model_fixed.pkl` | Flight cancelled (0 / 1) | `model/rf_cancel_model_fixed.pkl` | 14 |
| `rf_delay_model_fixed.pkl` | Flight delayed (0 / 1) | `model/rf_delay_model_fixed.pkl` | 15 |

Both models load automatically from **Hugging Face Hub** using `fetch_models.py`.

---

## ⚙️ Deployment (on Railway)

### ▶️ Start Command
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 5
```

**🌍 Environment Variables**
| Variable          | Example Value                                                                                           | Description                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `MODEL_URL`       | `https://huggingface.co/spaces/sweetyseelam/RevenueCat-Shipaton/resolve/main/rf_cancel_model_fixed.pkl` | cancel model                                    |
| `MODEL_DELAY_URL` | `https://huggingface.co/spaces/sweetyseelam/RevenueCat-Shipaton/resolve/main/rf_delay_model_fixed.pkl`  | delay model                                     |
| `DISABLE_SHAP`    | `1`                                                                                                     | disable SHAP (for speed/stability on free tier) |
| `SHAP_TIMEOUT_S`  | `12`                                                                                                    | max seconds before SHAP timeout                 |
| `PORT`            | *(auto-set by Railway)*                                                                                 | internal port                                   |


**✅ Recommendation:**

Set DISABLE_SHAP=1 on the Railway free plan to avoid timeouts (use /explain_safe & /explain_delay_safe).

Re-enable SHAP later by setting DISABLE_SHAP=0.

---

## 🧾 Example Requests

**PowerShell**
```
$body = @'
{
  "DOT_CODE": 19805,
  "FL_NUMBER": 123,
  "AIRLINE": 1,
  "ORIGIN": 11298,
  "DEST": 12892,
  "CRS_DEP_TIME": 945.0,
  "CRS_ARR_TIME": 1205.0,
  "DIVERTED": 0,
  "DISTANCE": 733.0,
  "YEAR": 2024,
  "MONTH": 10,
  "DAY": 2,
  "DAY_OF_WEEK": 3,
  "NIGHT_FLIGHT": 0,
  "LONG_AIR_TIME": 0
}
'@

Invoke-WebRequest -Method POST `
  -Uri "https://revenuecat-shipaton-production.up.railway.app/predict" `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body | Select -Expand Content
```

**Python**
```
import requests, json

url = "https://revenuecat-shipaton-production.up.railway.app/predict_delay"
payload = {
    "DOT_CODE": 19805, "FL_NUMBER": 123, "AIRLINE": 1,
    "ORIGIN": 11298, "DEST": 12892,
    "CRS_DEP_TIME": 945.0, "CRS_ARR_TIME": 1205.0,
    "DIVERTED": 0, "DISTANCE": 733.0,
    "YEAR": 2024, "MONTH": 10, "DAY": 2, "DAY_OF_WEEK": 3,
    "NIGHT_FLIGHT": 0, "LONG_AIR_TIME": 0
}
res = requests.post(url, json=payload, timeout=20)
print(res.json())
```

**JavaScript (fetch)**
```
const payload = {
  DOT_CODE: 19805, FL_NUMBER: 123, AIRLINE: 1,
  ORIGIN: 11298, DEST: 12892,
  CRS_DEP_TIME: 945.0, CRS_ARR_TIME: 1205.0,
  DIVERTED: 0, DISTANCE: 733.0,
  YEAR: 2024, MONTH: 10, DAY: 2, DAY_OF_WEEK: 3,
  NIGHT_FLIGHT: 0, LONG_AIR_TIME: 0
};

fetch("https://revenuecat-shipaton-production.up.railway.app/predict", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify(payload)
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

**Response Example**

json
```
{
  "delay_probability": 0.4313381711663397,
  "predicted_label": 0
}
```

---

## 📊 Explanation Endpoints (SHAP-safe mode)
When DISABLE_SHAP=1, the API uses feature_importances_ for explanations.

**Endpoints**

bash
```
POST /explain_safe?topk=8
POST /explain_delay_safe?topk=8
```

**Sample Response**

json
```
{
  "mode": "feature_importances",
  "top_contributions": [
    {"feature": "MONTH", "shap_value": 0.26, "value": 10.0},
    {"feature": "NIGHT_FLIGHT", "shap_value": 0.19, "value": 0.0}
  ]
}
```

**If you re-enable SHAP (DISABLE_SHAP=0):**

json
```
{
  "mode": "tree_shap",
  "base_value": 0.47,
  "cancel_probability": 0.32,
  "top_contributions": [...]
}
```

---

## 🧩 OpenAPI Docs

Interactive documentation: https://revenuecat-shipaton-production.up.railway.app/docs

🕓 Keep-Alive (Pinger Workflow)                                                                            
Prevent Railway from idling by adding this workflow at .github/workflows/ping.yml:

yaml
```
name: keepalive
on:
  schedule: [{ cron: "*/10 * * * *" }]
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Hit health endpoint
        run: |
          curl -sS --max-time 20 https://revenuecat-shipaton-production.up.railway.app/health || true
```

This pings the app every 10 minutes to keep it awake.

---

## 📦 Repository Structure
css

RevenueCat-Shipaton/                                                        
├── README.md                                   
├── LICENSE                                                     
├── Flight_Cancellation_Notification.ipynb                                                          
├── .gitignore                                                       
├── main.py                  ← FastAPI app entry point                                                            
├── fetch_models.py          ← Downloads models from Hugging Face                                                     
├── model/                                                 
│   ├── rf_cancel_model_fixed.pkl          ← Downloads models from Hugging Face                                            
│   ├── rf_delay_model_fixed.pkl           ← Downloads models from Hugging Face                                                     
│   ├── cancel_feature_order_fixed.csv                                                      
│   └── delay_feature_order_fixed.csv                                                
├── requirements.txt                                                                              
└── .github/                                                                               
    └── workflows/ping.yml    ← keep-alive GitHub Action

---

## 🧭 Author

**Sweety Seelam** - Business Analyst | Aspiring Data Scientist      

📧 Email: sweetyseelam2@gmail.com

· LinkedIn: https://www.linkedin.com/in/sweetyrao670/

· Portfolio: https://sweetyseelam2.github.io/SweetySeelam.github.io/

· Github: https://github.com/SweetySeelam2

· Medium:  https://medium.com/@sweetyseelam

---

**✅ Status:** Live and stable (200 OK on /predict & /predict_delay)

**🧩 Backend:** FastAPI + Uvicorn

**☁️ Hosting:** Railway (Python 3.11)

**💾 Models:** Random Forest (Hugging Face PKL files)

**🧠 Explainability:** SHAP (automatic fallback mode)

---

## ⚖️ License

MIT License - Copyright (c) 2025 Sweety Seelam.
