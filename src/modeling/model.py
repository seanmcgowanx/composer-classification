"""The hybrid classifier: CNN over the piano roll, LSTM over time, late fusion.

ComposerNet takes two inputs per example: a piano roll crop of shape
(batch, 2, 88, 300) and a preprocessed feature vector of shape (batch, 37).
The CNN reads the roll like an image and shrinks it step by step. The LSTM then
reads the CNN's output left to right as a sequence over time. Its final hidden
state, a summary of the whole crop, is concatenated with the handcrafted
feature vector (late fusion, per the decisions log) and mapped to 4 composer
logits.

The architecture is frozen at the hyperparameter sweep winner: three conv
blocks with 16, 32, and 64 channels, and a one directional LSTM. Larger and
smaller variants were tried and lost; the sweep results live in experiments/
and the decisions log.
"""
import torch
import torch.nn as nn

from src.modeling.config import COMPOSERS, MODEL_COLS, Config


class ComposerNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # each block is convolution, batch norm, relu, then a pool that halves
        # both axes, so the (88, 300) input leaves as (11, 37) with 64 channels
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
        self.lstm = nn.LSTM(input_size=64, hidden_size=cfg.lstm_hidden,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.lstm_hidden + len(MODEL_COLS), 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, len(COMPOSERS)),
        )

    def forward(self, roll, feats):
        x = self.cnn(roll)        # (batch, 64, 11, 37)
        x = x.mean(dim=2)         # average over the pitch axis: (batch, 64, 37)
        x = x.permute(0, 2, 1)    # the LSTM wants (batch, time, channels)
        _, (h, _) = self.lstm(x)  # h[-1] is the final hidden state
        return self.head(torch.cat([h[-1], feats], dim=1))


if __name__ == "__main__":
    cfg = Config()
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        net = ComposerNet(cfg).to(device)
        roll = torch.rand(8, 2, 88, cfg.crop_frames, device=device)
        feats = torch.randn(8, len(MODEL_COLS), device=device)
        logits = net(roll, feats)
        assert logits.shape == (8, len(COMPOSERS)), logits.shape
        print(f"{device}: forward OK, logits {tuple(logits.shape)}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")
