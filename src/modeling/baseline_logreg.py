"""Classical floor for the core comparison: multinomial logistic regression.

A linear classifier on the handcrafted features alone, the simplest model in the
paper's ladder. It answers what a plain linear model on the music theory vector
reaches before any deep roll branch is added, so the CNN, LSTM, and hybrid arms
can be read against it rather than against zero.

Everything is held identical to the neural runs so the numbers are comparable:
the same table, the same 5 folds, the same feature preprocessor refit on the
training folds only, and the same class balancing (sklearn's "balanced" uses the
same n_samples / (n_classes * count) weights train.py computes by hand). It uses
no roll, so each song is one feature vector and one prediction, with no windowing
to average. The three scores match evaluate() in train.py: log loss, macro-F1,
and balanced accuracy. Writes experiments/baseline_logreg/summary.json in the
same shape as a train.py run.

    /opt/miniconda3/envs/composer-classification/bin/python -m src.modeling.baseline_logreg
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss)

from src.modeling.config import COMPOSERS, EXPERIMENTS_DIR, MODEL_COLS, N_FOLDS, SEED
from src.modeling.dataset import build_preprocessor, load_table

RUN_NAME = "baseline_logreg"
MAX_ITER = 1000  # lbfgs headroom so the fit converges on every fold


def score_fold(df, k):
    """Fit on every fold except k, score fold k, return its three metrics."""
    train_df = df[df["fold"] != k]
    val_df = df[df["fold"] == k]

    # refit the preprocessor on the training folds only, exactly as train.py
    # does, so the held out fold's statistics never leak into the scaling
    pre = build_preprocessor()
    train_feats = pre.fit_transform(train_df[MODEL_COLS]).to_numpy(np.float32)
    val_feats = pre.transform(val_df[MODEL_COLS]).to_numpy(np.float32)

    # class_weight balanced matches the per composer weighting the neural runs
    # apply, so rare Chopin is not drowned out by Bach
    clf = LogisticRegression(class_weight="balanced", max_iter=MAX_ITER,
                             random_state=SEED)
    clf.fit(train_feats, train_df["label"].to_numpy())

    probs = clf.predict_proba(val_feats)
    labels = val_df["label"].to_numpy()
    preds = probs.argmax(axis=1)
    return {
        "log_loss": log_loss(labels, probs, labels=range(len(COMPOSERS))),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "balanced_acc": balanced_accuracy_score(labels, preds),
    }


def main():
    run_dir = Path(EXPERIMENTS_DIR) / RUN_NAME
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; delete it to rerun")
    run_dir.mkdir(parents=True)

    df = load_table()
    print(f"run {RUN_NAME}: {len(df)} songs, {len(MODEL_COLS)} features, "
          f"logistic regression on features only")

    results = {k: score_fold(df, k) for k in range(N_FOLDS)}

    f1s = [r["macro_f1"] for r in results.values()]
    bal_accs = [r["balanced_acc"] for r in results.values()]
    summary = {
        "folds": {str(k): r for k, r in results.items()},
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s)),
        "mean_balanced_acc": float(np.mean(bal_accs)),
    }
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n{RUN_NAME}: macro-F1 {summary['mean_macro_f1']:.4f} "
          f"+/- {summary['std_macro_f1']:.4f} over {N_FOLDS} folds")


if __name__ == "__main__":
    main()
