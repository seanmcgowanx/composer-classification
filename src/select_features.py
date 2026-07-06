"""Select a model-ready feature set from data/processed/features.csv.

EDA (notebooks/eda.ipynb) found two exact linear dependencies to drop:

  - pitch_range == pitch_max - pitch_min       -> drop pitch_range
  - pc_0..pc_11 are normalized and sum to 1.0  -> drop one bin (pc_11)

`filename` is kept as a leading id column but is not a feature. Reads
data/processed/features.csv and writes data/processed/features_model.csv,
leaving the raw feature file untouched.
"""
import pandas as pd
from pathlib import Path

SRC = Path("data/processed/features.csv")
OUT = Path("data/processed/features_model.csv")

DROP = ["pitch_range", "pc_11"]

df = pd.read_csv(SRC)
selected = df.drop(columns=DROP)
OUT.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(OUT, index=False)

print(f"dropped {DROP}; wrote {OUT}: {selected.shape[0]} rows x {selected.shape[1]} cols")
