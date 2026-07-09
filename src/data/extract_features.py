"""Extract a handcrafted music theory feature vector from each MIDI file into a CSV.

Reads MIDI files from data/interim/<composer>/ and writes one row per song
(39 features + composer target) to data/processed/features.csv. The vector is
the handcrafted half of the hybrid model (the piano roll from
src/data/extract_roll.py is the other half) and enters at the dense head.

Scope: only features encoding music theory the CNN cannot read off a fixed rate
roll. Melodic intervals need skyline extraction, vertical intervals need interval
class knowledge, key fit needs the Krumhansl templates, and the rhythm features
need a beat grid that a 10 frames per second roll does not expose. Descriptive
statistics visible in the roll itself (pitch range, polyphony, note density,
velocity) are deliberately not extracted.

Features that can fail on edge cases (tempo, key fit, beat grid) are written as
NaN; imputation is the modeling pipeline's job, fit on the train split only.
"""
import numpy as np
import pandas as pd
import pretty_midi
from pathlib import Path

SRC = Path("data/interim")
OUT = Path("data/processed/features.csv")
COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]
ROLL_FS = 10  # piano roll samples per second, matches FS in extract_roll.py
ONSET_TOL = 0.01  # seconds; notes starting closer than this share an onset
DISSONANT = [1, 2, 6, 10, 11]  # interval classes: seconds, sevenths, tritone
CONSONANT = [0, 3, 4, 5, 7, 8, 9]

# Krumhansl (1990) key profiles, C major / C minor
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def entropy(p):
    """Shannon entropy (base 2) of a probability vector."""
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def skyline_intervals(pm):
    """Signed melodic intervals between consecutive skyline notes, pooled over tracks.

    The skyline of a track is the highest pitch among notes starting together
    (within ONSET_TOL), a standard cheap stand in for the melody line.
    """
    intervals = []
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        notes = sorted(inst.notes, key=lambda n: (n.start, -n.pitch))
        skyline = []
        for n in notes:
            if skyline and n.start - skyline[-1][0] < ONSET_TOL:
                continue
            skyline.append((n.start, n.pitch))
        pitches = [p for _, p in skyline]
        intervals.extend(int(b) - int(a) for a, b in zip(pitches, pitches[1:]))
    return np.array(intervals)


def melodic_features(pm, f):
    """Interval histogram, direction ratio, and entropy from the skyline melody."""
    iv = skyline_intervals(pm)
    absi = np.abs(iv)
    hist = np.zeros(14)
    if absi.size:
        hist = np.bincount(np.minimum(absi, 13), minlength=14) / absi.size
    for i in range(13):
        f[f"mi_{i}"] = hist[i]
    f["mi_13plus"] = hist[13]
    ups, downs = (iv > 0).sum(), (iv < 0).sum()
    f["mi_up_ratio"] = ups / (ups + downs) if (ups + downs) else np.nan
    f["mi_entropy"] = entropy(hist)


def vertical_features(binroll, f):
    """Wrapped histogram of intervals between simultaneously sounding pitches."""
    counts = np.zeros(12)
    for d in range(1, binroll.shape[0]):
        counts[d % 12] += np.logical_and(binroll[:-d, :], binroll[d:, :]).sum()
    total = counts.sum()
    hist = counts / total if total else counts
    for i in range(12):
        f[f"vi_{i}"] = hist[i]
    dis, con = hist[DISSONANT].sum(), hist[CONSONANT].sum()
    f["vi_dissonance"] = dis / con if con else np.nan


def key_features(notes, f):
    """Pitch class entropy and Krumhansl key fit.

    Reports how strongly the piece commits to any one key, not which key it is:
    transposition is an arbitrary choice and says nothing about the composer.
    """
    # duration weighted pitch class distribution
    pc_dur = np.zeros(12)
    for n in notes:
        pc_dur[n.pitch % 12] += n.end - n.start
    pc_p = pc_dur / pc_dur.sum() if pc_dur.sum() else pc_dur
    f["pc_entropy"] = entropy(pc_p)

    # correlate against all 12 rotations of each profile, keep the best match
    if np.count_nonzero(pc_dur) >= 2:
        best_maj = max(np.corrcoef(pc_dur, np.roll(MAJOR_PROFILE, k))[0, 1]
                       for k in range(12))
        best_min = max(np.corrcoef(pc_dur, np.roll(MINOR_PROFILE, k))[0, 1]
                       for k in range(12))
        f["key_fit"] = max(best_maj, best_min)
        f["key_major_leaning"] = best_maj - best_min
    else:
        # a correlation needs at least two distinct values to be defined
        f["key_fit"] = np.nan
        f["key_major_leaning"] = np.nan


def rhythm_features(pm, notes, f):
    """Note durations on the beat grid, syncopation proxy, and tempo.

    Beat relative, not raw seconds: the same rhythm at half tempo must score the
    same. This is what the fixed rate roll cannot express.
    """
    durations = np.array([n.end - n.start for n in notes])
    beats = pm.get_beats()
    dur_bins = ["dur_sixteenth", "dur_eighth", "dur_quarter", "dur_half"]
    if len(beats) >= 2:
        beat_len = np.diff(beats)
        idx = np.clip(np.searchsorted(beats, [n.start for n in notes], side="right") - 1,
                      0, len(beat_len) - 1)
        local = beat_len[idx]
        dur_beats = durations / local
        # bins centered on sixteenth, eighth, quarter, half, whole; the whole bin
        # feeds the entropy but is not emitted, being 1 minus the other four
        hist = np.histogram(dur_beats, bins=[0, 0.375, 0.75, 1.5, 3, np.inf])[0]
        hist = hist / hist.sum() if hist.sum() else hist
        for name, v in zip(dur_bins, hist):
            f[name] = v
        f["dur_entropy"] = entropy(hist)

        # onsets further than 15% of the local beat from the beat/half beat grid
        grid = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
        onsets = np.array([n.start for n in notes])
        pos = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
        dist = np.minimum(np.abs(onsets - grid[pos - 1]), np.abs(onsets - grid[pos]))
        f["offbeat_rate"] = float((dist > 0.15 * local).mean())
    else:
        for name in dur_bins:
            f[name] = np.nan
        f["dur_entropy"] = np.nan
        f["offbeat_rate"] = np.nan

    try:
        f["tempo"] = pm.estimate_tempo()
    except (ValueError, IndexError):
        f["tempo"] = np.nan


def features(pm):
    """Return a dict of 39 numeric features for one parsed MIDI file."""
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return None

    binroll = pm.get_piano_roll(fs=ROLL_FS) > 0
    if binroll.shape[1] == 0:
        return None

    f = {}
    melodic_features(pm, f)
    vertical_features(binroll, f)
    key_features(notes, f)
    rhythm_features(pm, notes, f)
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
        # same skip rule as extract_roll.py: parse failure or zero notes
        if f is None:
            skipped += 1
            continue
        f["filename"] = path.name
        f["composer"] = composer
        rows.append(f)

df = pd.DataFrame(rows)
df = df[["filename"] + [c for c in df.columns if c not in ("filename", "composer")] + ["composer"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"\nwrote {OUT}: {len(df)} rows x {df.shape[1]} cols (skipped {skipped})")
print(df["composer"].value_counts().to_string())
print("NaNs per column (nonzero only):")
na = df.isna().sum()
print(na[na > 0].to_string() if na.any() else "none")
