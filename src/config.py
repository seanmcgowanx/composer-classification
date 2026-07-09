"""Shared configuration for the modeling stage.

Config holds every setting the pipeline and model use: file paths, data
constants, and the hyperparameters of the final model. The values are frozen at
the winners of the hyperparameter sweep (see the decisions log); the
experimentation phase is over, so there are no command line overrides. Changing
a hyperparameter means editing this file. Each training run still writes the
values it used to config.json in its experiments/ folder, so every recorded
result stays traceable to its exact settings.

The feature column lists are module constants rather than Config fields because
they are settled modeling decisions from the EDA, not knobs: DROP_COLS removes
the two compositional dependencies, POWER_COLS gets yeo-johnson for heavy right
tails, and everything else gets a plain StandardScaler. All fitting happens on
the training folds only, inside train.py.
"""
from dataclasses import dataclass

COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]  # fixed label order

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

# decided in the EDA (notebooks/eda.ipynb) and the decisions log
DROP_COLS = ["mi_11", "vi_11"]  # compositional dependencies of their histograms
POWER_COLS = ["vi_2", "vi_dissonance", "mi_12", "vi_1", "mi_13plus", "mi_8"]

MODEL_COLS = [c for c in FEATURE_COLS if c not in DROP_COLS]  # 37 model inputs
SCALE_COLS = [c for c in MODEL_COLS if c not in POWER_COLS]  # StandardScaler group


@dataclass
class Config:
    # paths, relative to the repo root (scripts run from there)
    manifest_csv: str = "data/processed/rolls_manifest.csv"
    features_csv: str = "data/processed/features.csv"
    splits_csv: str = "data/processed/splits.csv"
    experiments_dir: str = "experiments"

    # data
    crop_frames: int = 300  # 30 seconds at 10 frames per second
    n_folds: int = 5
    seed: int = 42

    # training
    batch_size: int = 32
    lr: float = 1e-3  # sweep winner; the first baseline used 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100  # a ceiling; early stopping decides the real length
    patience: int = 10  # stop after this many epochs without improvement
    num_workers: int = 0  # safest default on MPS

    # model (the CNN channel sizes are fixed in model.py)
    lstm_hidden: int = 128
    dropout: float = 0.3


if __name__ == "__main__":
    from dataclasses import asdict

    assert len(FEATURE_COLS) == 39
    assert len(MODEL_COLS) == 37
    assert set(POWER_COLS) | set(SCALE_COLS) == set(MODEL_COLS)
    assert not set(POWER_COLS) & set(SCALE_COLS)
    assert not set(DROP_COLS) & set(MODEL_COLS)
    print(f"{len(FEATURE_COLS)} extracted, {len(DROP_COLS)} dropped, "
          f"{len(POWER_COLS)} power transformed, {len(SCALE_COLS)} scaled")
    print(asdict(Config()))
