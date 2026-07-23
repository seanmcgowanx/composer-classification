"""The hybrid classifier: CNN over the piano roll, LSTM over time, late fusion.

ComposerNet takes two inputs per example: a piano roll crop of shape
(batch, 2, 88, 300) and a preprocessed feature vector of shape (batch, 37). A
roll encoder turns the roll into a fixed 128 wide summary; that summary is
concatenated with the handcrafted feature vector (late fusion, per the decisions
log) and mapped to 4 composer logits. The frozen model uses the hybrid encoder:
the CNN reads the roll like an image and shrinks it step by step, then the LSTM
reads the CNN's output left to right as a sequence over time and returns its
final hidden state as the summary.

The architecture is frozen at the hyperparameter sweep winner: three conv
blocks with 16, 32, and 64 channels, a one directional LSTM with 128 hidden
units, and dropout 0.3. Larger and smaller variants were tried and lost; the
sweep results live in experiments/ and the decisions log.

The two single input arms (roll only and features only) were ablated to measure
what fusion buys over either input alone; fusion won on both comparisons (see
the decisions log). That input ablation is settled, so its branch switch is gone
and ComposerNet always fuses both inputs.

RollCnnEncoder and RollLstmEncoder are the plain single architecture models in
the paper's core comparison: hybrid against CNN only against LSTM only, all
fusing the same handcrafted features, asking whether the CNN and LSTM together
earn their complexity over either sequence model alone. On our corpus the three
are statistically indistinguishable (see the decisions log), so this comparison
is a first class result, not throwaway scaffolding. The CNN only and LSTM only
arms reuse the hybrid's frozen hyperparameters, which were swept for the hybrid,
so their scores are a floor for each architecture rather than its own best case;
the decisions log records this caveat, and it only strengthens the null since
both match the hybrid despite the handicap.
"""
import torch
import torch.nn as nn

from src.modeling.config import COMPOSERS, CROP_FRAMES, MODEL_COLS

LSTM_HIDDEN = 128
DROPOUT = 0.3


def _conv_stack():
    """The three frozen conv blocks, shared by the hybrid and CNN only encoders.

    Each block is convolution, batch norm, relu, then a pool that halves both
    axes, so the (88, 300) input leaves as (11, 37) with 64 channels.
    """
    return nn.Sequential(
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


class RollHybridEncoder(nn.Module):
    """The frozen roll encoder: CNN, then LSTM over time. Returns a 128 vector."""
    def __init__(self):
        super().__init__()
        self.cnn = _conv_stack()
        self.lstm = nn.LSTM(input_size=64, hidden_size=LSTM_HIDDEN,
                            batch_first=True)

    def forward(self, roll):
        x = self.cnn(roll)        # (batch, 64, 11, 37)
        x = x.mean(dim=2)         # average over the pitch axis: (batch, 64, 37)
        x = x.permute(0, 2, 1)    # the LSTM wants (batch, time, channels)
        _, (h, _) = self.lstm(x)  # h[-1] is the final hidden state
        return h[-1]


class RollCnnEncoder(nn.Module):
    """The plain CNN arm of the core comparison: conv stack, pool over time.

    Replaces the LSTM's ordered read of the frame sequence with a mean over
    time, so this arm measures what sequence modeling buys on top of the CNN. A
    linear projection matches the hybrid's 128 wide summary so the head is
    unchanged.
    """
    def __init__(self):
        super().__init__()
        self.cnn = _conv_stack()
        self.project = nn.Linear(64, LSTM_HIDDEN)

    def forward(self, roll):
        x = self.cnn(roll)        # (batch, 64, 11, 37)
        x = x.mean(dim=(2, 3))    # average over pitch and time: (batch, 64)
        return self.project(x)    # (batch, 128)


class RollLstmEncoder(nn.Module):
    """The plain LSTM arm of the core comparison: LSTM over raw roll frames.

    Each time frame's 2 channels by 88 pitches is flattened to 176 inputs and
    read left to right at full pitch resolution, so this arm measures what the
    CNN's learned features buy on top of recurrence.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2 * 88, hidden_size=LSTM_HIDDEN,
                            batch_first=True)

    def forward(self, roll):
        b, c, p, t = roll.shape      # (batch, 2, 88, 300)
        x = roll.permute(0, 3, 1, 2)  # (batch, time, 2, 88)
        x = x.reshape(b, t, c * p)   # (batch, 300, 176)
        _, (h, _) = self.lstm(x)
        return h[-1]                 # (batch, 128)


# the roll encoders train.py can select; "hybrid" is the frozen model, the other
# two are the plain single architecture arms of the core comparison
ENCODERS = {
    "hybrid": RollHybridEncoder,
    "cnn": RollCnnEncoder,
    "lstm": RollLstmEncoder,
}


class ComposerNet(nn.Module):
    """Fuse a roll summary with the feature vector and map to composer logits.

    Defaults to the frozen hybrid encoder, so ComposerNet() reproduces
    experiments/final. The roll_encoder argument selects which arm of the core
    comparison runs (hybrid, cnn, or lstm); every encoder returns a 128 wide
    summary, so the fused head is identical across them.
    """
    def __init__(self, roll_encoder="hybrid"):
        super().__init__()
        assert roll_encoder in ENCODERS, roll_encoder
        self.roll_encoder = ENCODERS[roll_encoder]()
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN + len(MODEL_COLS), 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, len(COMPOSERS)),
        )

    def forward(self, roll, feats):
        summary = self.roll_encoder(roll)
        return self.head(torch.cat([summary, feats], dim=1))


if __name__ == "__main__":
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for name in ENCODERS:
            net = ComposerNet(name).to(device)
            roll = torch.rand(8, 2, 88, CROP_FRAMES, device=device)
            feats = torch.randn(8, len(MODEL_COLS), device=device)
            logits = net(roll, feats)
            assert logits.shape == (8, len(COMPOSERS)), logits.shape
            n_params = sum(p.numel() for p in net.parameters())
            print(f"{device} {name}: forward OK, logits {tuple(logits.shape)}, "
                  f"{n_params:,} parameters")
