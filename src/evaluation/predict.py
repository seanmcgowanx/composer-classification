"""Out of fold predictions from a finished cross validation run.

oof_predictions rebuilds the predictions behind a run's reported scores: for
each fold it loads that fold's fitted feature preprocessor and best checkpoint,
then scores every song in the held out fold exactly the way train.py did
(window the roll, average the window probabilities into one song prediction).

window_predictions is the same scoring but one row per window instead of one per
song, keeping each window's own probabilities rather than averaging them. It
answers a different question: how well the model classifies a single 30 second
clip (the Shazam style task) versus a whole piece. The 03_model_evaluation
notebook uses it to compare the two.

Each song is scored by the one model that never trained on it, so pooling the
five folds gives one honest prediction per song for all 1,628 songs. That
pooled table is what the 03_model_evaluation notebook analyzes.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.modeling.config import COMPOSERS, CROP_FRAMES, MODEL_COLS, N_FOLDS
from src.modeling.dataset import load_roll, load_table, song_windows
from src.modeling.model import ComposerNet


def oof_predictions(run_dir, device):
    """One row per song: fold, true composer, class probabilities, prediction."""
    run_dir = Path(run_dir)
    df = load_table()
    rows = []
    for k in range(N_FOLDS):
        fold_dir = run_dir / f"fold{k}"
        # the fold's fitted preprocessor and best weights, exactly as train.py
        # saved them at that fold's best epoch
        pre = joblib.load(fold_dir / "preprocessing.joblib")
        net = ComposerNet().to(device)
        net.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
        net.eval()

        val_df = df[df["fold"] == k].reset_index(drop=True)
        val_feats = pre.transform(val_df[MODEL_COLS]).to_numpy(np.float32)

        with torch.no_grad():
            for i, path in enumerate(val_df["path"]):
                # same scoring as train.py's evaluate: every window of the song
                # in one batch, softmax to probabilities, average the windows
                windows = song_windows(load_roll(path), CROP_FRAMES).to(device)
                feats = torch.tensor(val_feats[i], device=device)
                feats = feats.expand(len(windows), -1)
                probs = torch.softmax(net(windows, feats), dim=1)
                probs = probs.mean(dim=0).cpu().numpy()
                rows.append({
                    "filename": val_df.at[i, "filename"],
                    "composer": val_df.at[i, "composer"],
                    "fold": k,
                    "n_frames": val_df.at[i, "n_frames"],
                    "prob_bach": probs[0],
                    "prob_beethoven": probs[1],
                    "prob_chopin": probs[2],
                    "prob_mozart": probs[3],
                    "pred": COMPOSERS[int(probs.argmax())],
                })
        print(f"fold {k}: scored {len(val_df)} songs")
    return pd.DataFrame(rows)


def window_predictions(run_dir, device):
    """One row per window: fold, song, window index, class probabilities, pred.

    Same fold models and preprocessors as oof_predictions, but each 30 second
    window keeps its own prediction instead of being averaged into the song. The
    window index runs left to right; the last window of a song may overlap the
    previous one, exactly as song_windows tiles it for scoring.
    """
    run_dir = Path(run_dir)
    df = load_table()
    rows = []
    for k in range(N_FOLDS):
        fold_dir = run_dir / f"fold{k}"
        pre = joblib.load(fold_dir / "preprocessing.joblib")
        net = ComposerNet().to(device)
        net.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
        net.eval()

        val_df = df[df["fold"] == k].reset_index(drop=True)
        val_feats = pre.transform(val_df[MODEL_COLS]).to_numpy(np.float32)

        with torch.no_grad():
            for i, path in enumerate(val_df["path"]):
                # the same windows as oof scoring, but scored one at a time; the
                # feature vector is the song's, repeated for every window
                windows = song_windows(load_roll(path), CROP_FRAMES).to(device)
                feats = torch.tensor(val_feats[i], device=device)
                feats = feats.expand(len(windows), -1)
                probs = torch.softmax(net(windows, feats), dim=1).cpu().numpy()
                for w, p in enumerate(probs):
                    rows.append({
                        "filename": val_df.at[i, "filename"],
                        "composer": val_df.at[i, "composer"],
                        "fold": k,
                        "window": w,
                        "prob_bach": p[0],
                        "prob_beethoven": p[1],
                        "prob_chopin": p[2],
                        "prob_mozart": p[3],
                        "pred": COMPOSERS[int(p.argmax())],
                    })
        print(f"fold {k}: scored {len(val_df)} songs")
    return pd.DataFrame(rows)
