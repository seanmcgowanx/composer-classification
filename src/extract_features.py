"""Extract numeric features from per-composer MIDI files into a single CSV.

Reads MIDI files from data/interim/<composer>/ and writes one row per file
(features + composer target) to data/processed/features.csv.
"""
import numpy as np
import pandas as pd
import pretty_midi
from pathlib import Path

SRC = Path("data/interim")
OUT = Path("data/processed/features.csv")
COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]


def features(pm):
    """Return a dict of numeric features for one parsed MIDI file."""
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return None

    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    total_duration = pm.get_end_time()

    # average simultaneous pitches over frames that have at least one note
    roll = pm.get_piano_roll(fs=10) > 0
    active_per_frame = roll.sum(axis=0)
    sounding = active_per_frame[active_per_frame > 0]
    polyphony_mean = float(sounding.mean()) if sounding.size else 0.0

    try:
        tempo = pm.estimate_tempo()
    except (ValueError, IndexError):
        tempo = np.nan

    # normalized pitch-class histogram (key/tonality signal)
    pc = np.bincount(pitches % 12, minlength=12) / len(pitches)

    f = {
        "n_notes": len(notes),
        "total_duration": total_duration,
        "note_density": len(notes) / total_duration if total_duration else 0.0,
        "n_instruments": len(pm.instruments),
        "pitch_mean": pitches.mean(),
        "pitch_std": pitches.std(),
        "pitch_min": int(pitches.min()),
        "pitch_max": int(pitches.max()),
        "pitch_range": int(pitches.max() - pitches.min()),
        "dur_mean": durations.mean(),
        "dur_std": durations.std(),
        "vel_mean": velocities.mean(),
        "vel_std": velocities.std(),
        "tempo": tempo,
        "polyphony_mean": polyphony_mean,
    }
    for i in range(12):
        f[f"pc_{i}"] = pc[i]
    return f


rows = []
skipped = 0
for composer in COMPOSERS:
    folder = SRC / composer
    files = list(folder.glob("*.mid")) + list(folder.glob("*.MID"))
    for path in files:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
            f = features(pm)
        except Exception as e:
            f = None
            print(f"skip (parse error): {path.name}: {e}")
        if f is None:
            skipped += 1
            continue
        f["filename"] = path.name
        f["composer"] = composer
        rows.append(f)

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"\nwrote {OUT}: {len(df)} rows x {df.shape[1]} cols (skipped {skipped})")
print(df["composer"].value_counts().to_string())
