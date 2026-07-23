"""Shared constants for the modeling stage.

This module holds only what more than one module needs: file paths, the
composer label order, the feature column lists, and the three data constants
(crop length, fold count, seed). The training hyperparameters live at the top
of train.py and the model sizes at the top of model.py, next to the code that
uses them; all of them are frozen at the winners of the hyperparameter sweep.
Each training run still writes the values it used to config.json in its
experiments/ folder, so every recorded result stays traceable to its exact
settings.

The feature column lists encode settled modeling decisions from the EDA, not
knobs: DROP_COLS removes the two compositional dependencies, POWER_COLS gets
yeo-johnson for heavy right tails, and everything else gets a plain
StandardScaler. All fitting happens on the training folds only, inside
train.py.
"""
# paths, relative to the repo root (scripts run from there)
MANIFEST_CSV = "data/processed/rolls_manifest.csv"
FEATURES_CSV = "data/processed/features.csv"
SPLITS_CSV = "data/processed/splits.csv"
EXPERIMENTS_DIR = "experiments"

COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]  # fixed label order

CROP_FRAMES = 300  # 30 seconds at 10 frames per second
N_FOLDS = 5
SEED = 42

# the 39 extracted features, in features.csv column order
FEATURE_COLS = [
    # melodic intervals from the skyline melody: share of each interval size in
    # semitones (0 to 12, then 13 and larger), upward share, histogram entropy
    "mi_0", "mi_1", "mi_2", "mi_3", "mi_4", "mi_5", "mi_6", "mi_7", "mi_8",
    "mi_9", "mi_10", "mi_11", "mi_12", "mi_13plus", "mi_up_ratio", "mi_entropy",
    # vertical intervals: share of each interval class between simultaneously
    # sounding notes, plus the dissonant to consonant ratio
    "vi_0", "vi_1", "vi_2", "vi_3", "vi_4", "vi_5", "vi_6", "vi_7", "vi_8",
    "vi_9", "vi_10", "vi_11", "vi_dissonance",
    # key strength: pitch class entropy and Krumhansl key fit
    "pc_entropy", "key_fit", "key_major_leaning",
    # rhythm on the beat grid, plus tempo
    "dur_sixteenth", "dur_eighth", "dur_quarter", "dur_half", "dur_entropy",
    "offbeat_rate", "tempo",
]

# decided in the EDA (notebooks/eda.ipynb)
DROP_COLS = ["mi_11", "vi_11"]  # compositional dependencies of their histograms
POWER_COLS = ["vi_2", "vi_dissonance", "mi_12", "vi_1", "mi_13plus", "mi_8"]

MODEL_COLS = [c for c in FEATURE_COLS if c not in DROP_COLS]  # 37 model inputs
SCALE_COLS = [c for c in MODEL_COLS if c not in POWER_COLS]  # StandardScaler group


if __name__ == "__main__":
    assert len(FEATURE_COLS) == 39
    assert len(MODEL_COLS) == 37
    assert set(POWER_COLS) | set(SCALE_COLS) == set(MODEL_COLS)
    assert not set(POWER_COLS) & set(SCALE_COLS)
    assert not set(DROP_COLS) & set(MODEL_COLS)
    print(f"{len(FEATURE_COLS)} extracted, {len(DROP_COLS)} dropped, "
          f"{len(POWER_COLS)} power transformed, {len(SCALE_COLS)} scaled")
