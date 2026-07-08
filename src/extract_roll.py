"""Extract a piano roll tensor from each MIDI file for the CNN/LSTM.

Reads MIDI files from data/interim/<composer>/ and writes one compressed npz
per song to data/processed/rolls/: a uint8 array of shape (2, 88, T) with an
onset channel (1 where a note starts) and a frame channel (MIDI velocity while
the note sounds), over the 88 piano keys at FS samples per second. One row of
metadata per song goes to data/processed/rolls_manifest.csv, which is what
loaders and splits should read (never the directory itself).
"""
import numpy as np
import pandas as pd
import pretty_midi
from pathlib import Path

SRC = Path("data/interim")
OUT_DIR = Path("data/processed/rolls")
MANIFEST = Path("data/processed/rolls_manifest.csv")
COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]
FS = 10  # samples per second
PITCH_LO = 21  # the 88 piano keys: MIDI notes 21..108
PITCH_HI = 108


def roll(pm):
    """Return a uint8 array of shape (2, 88, T) for one parsed MIDI file."""
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return None

    T = int(np.ceil(pm.get_end_time() * FS))
    if T == 0:
        return None

    arr = np.zeros((2, PITCH_HI - PITCH_LO + 1, T), dtype=np.uint8)
    for n in notes:
        p = min(max(n.pitch, PITCH_LO), PITCH_HI) - PITCH_LO
        start = min(int(n.start * FS), T - 1)
        end = max(int(n.end * FS), start + 1)
        arr[0, p, start] = 1
        arr[1, p, start:end] = np.maximum(arr[1, p, start:end], n.velocity)
    return arr


OUT_DIR.mkdir(parents=True, exist_ok=True)
rows = []
skipped = 0
for composer in COMPOSERS:
    folder = SRC / composer
    files = list(folder.glob("*.mid")) + list(folder.glob("*.MID"))
    for path in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
            arr = roll(pm)
        except Exception as e:
            arr = None
            print(f"skip (parse error): {path.name}: {e}")
        if arr is None:
            skipped += 1
            continue
        out = OUT_DIR / f"{composer}__{path.name}.npz"
        np.savez_compressed(out, roll=arr)
        rows.append({"filename": path.name, "composer": composer,
                     "path": str(out), "n_frames": arr.shape[2]})

df = pd.DataFrame(rows)
df.to_csv(MANIFEST, index=False)

print(f"\nwrote {len(df)} rolls to {OUT_DIR} (skipped {skipped})")
print(df["composer"].value_counts().to_string())
