import pandas as pd
import joblib
import os

# 1) CANCEL feature order from your existing sample CSV (just this once)
CANCEL_FEATURE_SAMPLE_CSV = "data/flight_cancel_features.csv"
df_cancel = pd.read_csv(CANCEL_FEATURE_SAMPLE_CSV)
# Mirror your API’s drop:
for col in ['MISSING_AIR_TIME','MISSING_ELAPSED_TIME','MISSING_CRS_ELAPSED_TIME','LONG_AIR_TIME']:
    if col in df_cancel.columns:
        df_cancel = df_cancel.drop(columns=[col])
cancel_feature_names = [c for c in df_cancel.columns if c != 'CANCELLED']
pd.Series(cancel_feature_names).to_csv("model/cancel_feature_order_fixed.csv", index=False, header=False)
print("Wrote model/cancel_feature_order_fixed.csv with", len(cancel_feature_names), "features.")

print("Done.")