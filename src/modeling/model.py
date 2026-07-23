"""The hybrid classifier: CNN over the piano roll, LSTM over time, late fusion.

ComposerNet takes two inputs per example: a piano roll crop of shape
(batch, 2, 88, 300) and a preprocessed feature vector of shape (batch, 37).
The CNN reads the roll like an image and shrinks it step by step. The LSTM then
reads the CNN's output left to right as a sequence over time. Its final hidden
state, a summary of the whole crop, is concatenated with the handcrafted
feature vector (late fusion, per the decisions log) and mapped to 4 composer
logits.

The architecture is frozen at the hyperparameter sweep winner: three conv
blocks with 16, 32, and 64 channels, a one directional LSTM with 128 hidden
units, and dropout 0.3. Larger and smaller variants were tried and lost; the
sweep results live in experiments/ and the decisions log.

branch is an ablation switch, not a hyperparameter. It defaults to "both", the
frozen fusion model. "roll" drops the feature vector so the head sees only the
LSTM summary, and "features" drops the CNN and LSTM so the head sees only the
handcrafted vector. The two ablated arms measure what fusion buys over either
input alone; the decisions log records the comparison.
"""
import torch
import torch.nn as nn

from src.modeling.config import COMPOSERS, CROP_FRAMES, MODEL_COLS

LSTM_HIDDEN = 128
DROPOUT = 0.3
BRANCHES = ("both", "roll", "features")


class ComposerNet(nn.Module):
    def __init__(self, branch="both"):
        super().__init__()
        assert branch in BRANCHES, branch
        self.branch = branch
        # the roll branch (CNN then LSTM) exists in every mode except features
        # only, where there is no roll to read
        if branch != "features":
            # each block is convolution, batch norm, relu, then a pool that
            # halves both axes, so (88, 300) leaves as (11, 37) with 64 channels
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.lstm = nn.LSTM(input_size=64, hidden_size=LSTM_HIDDEN,
                                batch_first=True)
        # the head's width is whatever the active branches feed it: the LSTM
        # summary, the feature vector, or both concatenated
        head_in = {"both": LSTM_HIDDEN + len(MODEL_COLS),
                   "roll": LSTM_HIDDEN,
                   "features": len(MODEL_COLS)}[branch]
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, len(COMPOSERS)),
        )

    def roll_summary(self, roll):
        """The LSTM's final hidden state: one 128 vector summarizing the crop."""
        x = self.cnn(roll)        # (batch, 64, 11, 37)
        x = x.mean(dim=2)         # average over the pitch axis: (batch, 64, 37)
        x = x.permute(0, 2, 1)    # the LSTM wants (batch, time, channels)
        _, (h, _) = self.lstm(x)  # h[-1] is the final hidden state
        return h[-1]

    def forward(self, roll, feats):
        # feed the head only what the active branches produce; the frozen
        # "both" model concatenates the roll summary and the feature vector
        if self.branch == "features":
            return self.head(feats)
        if self.branch == "roll":
            return self.head(self.roll_summary(roll))
        return self.head(torch.cat([self.roll_summary(roll), feats], dim=1))


if __name__ == "__main__":
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for branch in BRANCHES:
            net = ComposerNet(branch).to(device)
            roll = torch.rand(8, 2, 88, CROP_FRAMES, device=device)
            feats = torch.randn(8, len(MODEL_COLS), device=device)
            logits = net(roll, feats)
            assert logits.shape == (8, len(COMPOSERS)), logits.shape
            print(f"{device} {branch}: forward OK, logits {tuple(logits.shape)}")
    n_params = sum(p.numel() for p in ComposerNet().parameters())
    print(f"parameters (both): {n_params:,}")
