"""Train the CNN/LSTM fusion model with stratified 5 fold cross validation.

One invocation is one experiment: it trains a model per fold and aggregates the
scores. All hyperparameters are fixed in src/modeling/config.py (the sweep
winners; the experimentation phase is over). The only argument is a name for the
run:

    /opt/miniconda3/envs/composer-classification/bin/python -m src.modeling.train final

For each fold: the feature preprocessor is fit on the training folds only, the
model trains on one random crop per song per epoch with class weighted cross
entropy, and the held out fold is scored by cutting each song into windows and
averaging the window probabilities into one song prediction. The held out fold
drives both early stopping and the reported metrics, so cross validation
estimates are mildly optimistic (disclosed in the decisions log).

Artifacts land in experiments/<run_name>/: config.json (the settings used),
summary.json (per fold and average scores), and one folder per fold holding
metrics.csv (one row per epoch), best.pt (the weights at the best val macro-F1),
and preprocessing.joblib (the fitted feature pipeline).
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader

from src.modeling.config import COMPOSERS, MODEL_COLS, Config
from src.modeling.dataset import CropDataset, build_preprocessor, load_roll, load_table, song_windows
from src.modeling.model import ComposerNet


def evaluate(net, val_df, val_feats, cfg, device):
    """Score a held out fold: one prediction per song, averaged over windows."""
    # eval mode changes how dropout and batch norm behave: dropout stops
    # zeroing activations and batch norm uses its saved running averages,
    # so the same song always gets the same prediction
    net.eval()
    all_probs = []
    # no_grad tells torch not to track gradients; we are only predicting,
    # and skipping the bookkeeping makes this loop faster and lighter
    with torch.no_grad():
        for i, path in enumerate(val_df["path"]):
            # cut the whole song into fixed windows and run them through the
            # model as one batch (even the longest song, 174 windows, fits)
            windows = song_windows(load_roll(path), cfg.crop_frames).to(device)
            # the feature vector describes the whole song, so every window of
            # this song gets the same copy of it
            feats = torch.tensor(val_feats[i], device=device)
            feats = feats.expand(len(windows), -1)
            # softmax turns the model's raw scores (logits) into probabilities
            # that sum to 1; averaging those over the windows gives one
            # probability per composer for the whole song
            probs = torch.softmax(net(windows, feats), dim=1)
            all_probs.append(probs.mean(dim=0).cpu().numpy())
    all_probs = np.array(all_probs)
    labels = val_df["label"].to_numpy()
    # the predicted composer is whichever probability is highest
    preds = all_probs.argmax(axis=1)
    # three scores per epoch: log_loss judges the probabilities themselves
    # (confidently wrong costs more than unsure), macro-F1 averages F1 over the
    # 4 composers so Bach's size cannot hide bad Chopin predictions, and
    # balanced accuracy is the average of the 4 per composer recall rates
    return (log_loss(labels, all_probs, labels=range(len(COMPOSERS))),
            f1_score(labels, preds, average="macro"),
            balanced_accuracy_score(labels, preds))


def train_fold(cfg, df, k, out_dir, device):
    """Train on every fold except k, early stop on fold k, save the artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # seed torch's random numbers (weight init, batch shuffling, dropout) so
    # rerunning this fold reproduces the same result; each fold gets its own
    # seed so the five models do not start from identical weights
    torch.manual_seed(cfg.seed + k)

    # fold k is held out for scoring; the other four folds are the training set
    train_df = df[df["fold"] != k].reset_index(drop=True)
    val_df = df[df["fold"] == k].reset_index(drop=True)

    # fit the feature preprocessing (median fill, yeo-johnson, standardize) on
    # the training songs only, then apply it to both sides; fitting on all
    # songs would leak the held out fold's statistics into training. saving
    # the fitted pipeline lets later analysis apply the exact same transform.
    pre = build_preprocessor()
    train_feats = pre.fit_transform(train_df[MODEL_COLS]).to_numpy(np.float32)
    val_feats = pre.transform(val_df[MODEL_COLS]).to_numpy(np.float32)
    joblib.dump(pre, out_dir / "preprocessing.joblib")

    # the dataset serves one random 30 second crop per song; the loader
    # shuffles song order each epoch and hands the model batches of 32
    dataset = CropDataset(train_df, train_feats, cfg.crop_frames, seed=cfg.seed + k)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers)

    # class weights counter the imbalance: each composer's weight is the size
    # of an average class divided by that composer's size, so mistakes on
    # rare Chopin (about 8% of songs) cost roughly 8 times more than mistakes
    # on Bach (63%), and the loss cannot be minimized by just guessing Bach
    counts = train_df["label"].value_counts().sort_index().to_numpy()
    weights = len(train_df) / (len(COMPOSERS) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))

    net = ComposerNet(cfg).to(device)
    # AdamW is the optimizer that updates the weights after each batch;
    # weight_decay gently shrinks weights toward zero as regularization
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    best_f1, best_bal_acc, best_epoch = -1.0, -1.0, -1
    epochs_since_best = 0
    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss",
                         "val_macro_f1", "val_balanced_acc", "seconds"])
        for epoch in range(cfg.epochs):
            t0 = time.time()

            # one training pass: every song once, as one random crop
            net.train()  # the opposite of net.eval(): dropout active again
            losses = []
            for roll, feats, label in loader:
                # move the batch to the same device (GPU) as the model
                roll, feats, label = roll.to(device), feats.to(device), label.to(device)
                # the standard pytorch step: clear old gradients, run the
                # batch forward, measure the loss, run backward to compute
                # gradients, and let the optimizer nudge every weight
                optimizer.zero_grad()
                loss = criterion(net(roll, feats), label)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            # after each training pass, score the held out fold and append
            # one row to metrics.csv (flush makes it readable mid run)
            val_loss, macro_f1, bal_acc = evaluate(net, val_df, val_feats, cfg, device)
            seconds = time.time() - t0
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{macro_f1:.4f}", f"{bal_acc:.4f}", f"{seconds:.1f}"])
            fh.flush()

            # early stopping: whenever val macro-F1 sets a new best, save the
            # weights and reset the counter; after patience epochs with no
            # new best, assume the model has peaked and stop this fold.
            # best.pt therefore always holds the peak epoch's weights, not
            # the weights from whenever training happened to end.
            marker = ""
            if macro_f1 > best_f1:
                best_f1, best_bal_acc, best_epoch = macro_f1, bal_acc, epoch
                torch.save(net.state_dict(), out_dir / "best.pt")
                epochs_since_best = 0
                marker = " *"  # marks new best epochs in the console output
            else:
                epochs_since_best += 1
            print(f"fold {k} epoch {epoch}: train {train_loss:.4f} "
                  f"val {val_loss:.4f} macro-F1 {macro_f1:.4f} "
                  f"bal-acc {bal_acc:.4f} ({seconds:.0f}s){marker}")
            if epochs_since_best >= cfg.patience:
                print(f"fold {k}: early stop at epoch {epoch} "
                      f"(best epoch {best_epoch})")
                break
    return {"macro_f1": best_f1, "balanced_acc": best_bal_acc, "epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="name for the experiments/ output folder")
    run_name = parser.parse_args().run_name

    cfg = Config()
    # refuse to overwrite an existing run so results are never silently lost
    run_dir = Path(cfg.experiments_dir) / run_name
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; pick a new run name")
    run_dir.mkdir(parents=True)
    # record the exact settings this run used, for traceability
    with open(run_dir / "config.json", "w") as fh:
        json.dump({**asdict(cfg), "run_name": run_name}, fh, indent=2)

    # use the Apple GPU (MPS) when available, otherwise fall back to CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    df = load_table(cfg)
    print(f"run {run_name}: device {device}, {len(df)} songs, "
          f"{len(MODEL_COLS)} features")

    # train one model per fold; each fold's score comes from songs that model
    # never trained on, so together the five scores cover every song once
    results = {}
    for k in range(cfg.n_folds):
        results[k] = train_fold(cfg, df, k, run_dir / f"fold{k}", device)

    # the experiment's headline numbers: the average score across folds, with
    # the standard deviation showing how much the folds disagree
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
    print(f"\n{run_name}: macro-F1 {summary['mean_macro_f1']:.4f} "
          f"+/- {summary['std_macro_f1']:.4f} over {cfg.n_folds} folds")


if __name__ == "__main__":
    main()
