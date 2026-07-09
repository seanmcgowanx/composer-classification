"""Data access for training: table join, feature preprocessing, and roll crops.

load_table merges the rolls manifest, the feature CSV, and the fold assignments
into one DataFrame with one row per song: its roll path, its 39 features, its
label, and its fold. Everything downstream reads from that table.

build_preprocessor returns the (unfitted) sklearn pipeline for the 37 model
features. train.py fits it on the training folds only and saves it per fold, so
nothing about the validation songs ever leaks into the scaling.

CropDataset feeds training: each epoch it serves one random 30 second crop of
every song, so a shuffled DataLoader keeps the batch class mix equal to the song
class mix and the class weights stay valid. song_windows feeds evaluation: it
cuts a whole song into fixed windows so the model can score each window and
average the results into one song prediction.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import Dataset

from src.modeling.config import (COMPOSERS, FEATURES_CSV, MANIFEST_CSV,
                                 POWER_COLS, SCALE_COLS, SPLITS_CSV)


def load_table():
    """Join manifest, features, and splits into one row per song."""
    manifest = pd.read_csv(MANIFEST_CSV)
    features = pd.read_csv(FEATURES_CSV)
    splits = pd.read_csv(SPLITS_CSV)
    df = manifest.merge(features, on=["filename", "composer"], validate="1:1")
    df = df.merge(splits, on=["filename", "composer"], validate="1:1")
    assert len(df) == len(manifest), "manifest, features, and splits disagree"
    df["label"] = df["composer"].map(COMPOSERS.index)
    return df


def build_preprocessor():
    """The feature pipeline: fill NaNs, then scale every column.

    Takes and returns DataFrames so the transformers can name their columns.
    The two steps: fill the few missing values with the training median, then
    yeo-johnson the heavy tailed columns and standardize the rest. Fit on
    training songs only; train.py enforces that.
    """
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", ColumnTransformer([
            ("yeo-johnson", PowerTransformer(method="yeo-johnson",
                                             standardize=True), POWER_COLS),
            ("standardize", StandardScaler(), SCALE_COLS),
        ])),
    ])
    pipeline.set_output(transform="pandas")  # keep column names between steps
    return pipeline


def load_roll(path):
    """Load one roll npz as float32 (2, 88, T) with both channels in [0, 1]."""
    arr = np.load(path)["roll"].astype(np.float32)
    arr[1] /= 127.0  # channel 1 is MIDI velocity, 0 to 127
    return arr


def pad_to(arr, n_frames):
    """Right pad a (2, 88, T) roll with zeros so it is at least n_frames long."""
    t = arr.shape[2]
    if t >= n_frames:
        return arr
    out = np.zeros((arr.shape[0], arr.shape[1], n_frames), dtype=arr.dtype)
    out[:, :, :t] = arr
    return out


class CropDataset(Dataset):
    """One random fixed length crop per song per epoch, plus its feature vector.

    feats must already be transformed by the fold's fitted preprocessor, as a
    float32 array aligned row for row with df.
    """

    def __init__(self, df, feats, crop_frames, seed):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.feats = feats
        self.crop_frames = crop_frames
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        roll = pad_to(load_roll(self.paths[i]), self.crop_frames)
        # pick a random window; short songs were padded so start can only be 0
        start = self.rng.integers(0, roll.shape[2] - self.crop_frames + 1)
        crop = roll[:, :, start:start + self.crop_frames]
        return torch.tensor(crop), torch.tensor(self.feats[i]), self.labels[i]


def song_windows(roll, crop_frames):
    """Cut one roll into evaluation windows, stacked as (n, 2, 88, crop_frames).

    Windows tile the song left to right without overlap. When the length is not
    an exact multiple, one extra window covers the final crop_frames so the tail
    of the song is still scored.
    """
    roll = pad_to(roll, crop_frames)
    t = roll.shape[2]
    starts = list(range(0, t - crop_frames + 1, crop_frames))
    if starts[-1] + crop_frames < t:
        starts.append(t - crop_frames)
    windows = np.stack([roll[:, :, s:s + crop_frames] for s in starts])
    return torch.tensor(windows)
