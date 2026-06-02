import pandas as pd
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv(r"resources\yk_tlv_stat_mock.csv", encoding="windows-1255")

# Randomly assign each parcel a statistical area (1-200)
df["ms_ezor"] = rng.integers(1, 201, size=len(df))

# Set ms_ezor_shetach = actual sum of parcel areas per statistical area
actual_totals = df.groupby("ms_ezor")["Shape_Area"].transform("sum")
df["ms_ezor_shetach"] = actual_totals

out_path = r"resources\yk_tlv_stat_mock.csv"
df.to_csv(out_path, index=False, encoding="windows-1255")
print(f"Done. {len(df)} parcels assigned to 200 statistical areas.")
print(df[["oid_migrash", "ms_ezor", "ms_ezor_shetach"]].head(10))
