import pandas as pd
import time
import os

df = pd.read_csv("data/production_drifted.csv")

os.makedirs("data/stream", exist_ok=True)

for i in range(len(df)):
    df.iloc[[i]].to_csv(f"data/stream/row_{i}.csv", index=False)
    print(f"Sent row {i}")
    time.sleep(1)  # simulate real-time