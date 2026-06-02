import pandas as pd

df = pd.read_csv(r"resources\yk_tlv_stat.csv", encoding="utf-8-sig")

# Area of each statistical area (consistent per ms_ezor)
stat_area_totals = df.groupby("ms_ezor")["ms_ezor_shetach"].first()

# Sum parcel area per (statistical area, land-use type)
grouped = df.groupby(["ms_ezor", "k_yeud_rashi"])["Shape_Area"].sum().reset_index()

# Pivot: rows = statistical areas, columns = land-use types
pivot = grouped.pivot(index="ms_ezor", columns="k_yeud_rashi", values="Shape_Area").fillna(0)

# Divide each land-use area by the statistical area's total area
feature_matrix = pivot.div(stat_area_totals, axis=0)

# Rename columns to x_<code> for clarity
feature_matrix.columns = [f"x_{col}" for col in feature_matrix.columns]
feature_matrix.index.name = "ms_ezor"

out_path = r"resources\feature_matrix.csv"
feature_matrix.to_csv(out_path, encoding="utf-8-sig")

print(f"Feature matrix: {feature_matrix.shape[0]} statistical areas x {feature_matrix.shape[1]} land-use types")
print(feature_matrix.head())
