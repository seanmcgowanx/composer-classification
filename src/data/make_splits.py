"""Assign every song to a cross validation fold, stratified by composer.

Reads data/processed/rolls_manifest.csv (never the rolls directory) and writes
data/processed/splits.csv with one row per song: filename, composer, fold. Folds
are assigned by song, so a song's roll and feature vector always land on the same
side of any split. The assignment is seeded and rerunning the script reproduces
the file byte for byte.

train.py holds out one fold at a time: it trains on the other folds and uses the
held out fold for early stopping and metrics. Nothing here fits on data; scaling
and imputation belong to the modeling pipeline, per fold, train side only.
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MANIFEST = "data/processed/rolls_manifest.csv"
OUT = "data/processed/splits.csv"
N_FOLDS = 5
SEED = 42

df = pd.read_csv(MANIFEST)[["filename", "composer"]]
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["fold"] = -1
for k, (_, test_idx) in enumerate(skf.split(df, df["composer"])):
    df.loc[test_idx, "fold"] = k

assert (df["fold"] >= 0).all()
df.to_csv(OUT, index=False)

print(f"wrote {OUT}: {len(df)} rows, {N_FOLDS} folds, seed {SEED}")
