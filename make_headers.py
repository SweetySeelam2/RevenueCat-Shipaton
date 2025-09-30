import pandas as pd

# Delay: read feature order and write an empty CSV with only headers
delay_cols = pd.read_csv("model/delay_feature_order_fixed.csv", header=None)[0].tolist()
pd.DataFrame(columns=delay_cols).to_csv("data/flight_features.csv", index=False)

# Cancellation: read feature order and write an empty CSV with only headers (+ target for tooling)
cancel_cols = pd.read_csv("model/cancel_feature_order_fixed.csv", header=None)[0].tolist()
pd.DataFrame(columns=cancel_cols + ["CANCELLED"]).to_csv("data/flight_cancel_features.csv", index=False)

print("Wrote header-only CSVs.")
