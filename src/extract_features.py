"""Extract a handcrafted feature vector from each MIDI file into a single CSV.

Reads MIDI files from data/interim/<composer>/ and writes one row per song
(~66 features + composer target) to data/processed/features.csv. The vector
is the handcrafted half of the hybrid model (the piano roll from
src/extract_roll.py is the other half) and enters at the dense head. Feature
definitions follow docs/input-pipeline-design.md: melodic intervals from the
skyline of each track, vertical intervals from the roll, pitch and key fit,
texture, rhythm on the beat grid, and dynamics. Features that can fail on
edge cases (tempo, key fit, beat grid) are written as NaN; imputation is the
modeling pipeline's job, fit on the train split only.
"""
import numpy as np
import pandas as pd
import pretty_midi
from pathlib import Path

SRC = Path("data/interim")
OUT = Path("data/processed/features.csv")
COMPOSERS = ["bach", "beethoven", "chopin", "mozart"]
ROLL_FS = 10  # piano roll samples per second
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
    """Interval histogram and motion ratios from the skyline melody."""
    iv = skyline_intervals(pm)
    absi = np.abs(iv)
    hist = np.zeros(14)
    if absi.size:
        hist = np.bincount(np.minimum(absi, 13), minlength=14) / absi.size
    for i in range(13):
        f[f"mi_{i}"] = hist[i]
    f["mi_13plus"] = hist[13]
    f["mi_stepwise"] = hist[1] + hist[2]
    f["mi_leap"] = float((absi >= 7).mean()) if absi.size else 0.0
    ups, downs = (iv > 0).sum(), (iv < 0).sum()
    f["mi_up_ratio"] = ups / (ups + downs) if (ups + downs) else np.nan
    f["mi_mean"] = float(absi.mean()) if absi.size else 0.0
    f["mi_std"] = float(absi.std()) if absi.size else 0.0
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
    f["vi_tritone"] = hist[6]
    f["vi_perfect"] = hist[[0, 5, 7]].sum()


def pitch_features(notes, binroll, f):
    """Pitch statistics, pitch class entropy, and Krumhansl key fit."""
    pitches = np.array([n.pitch for n in notes])
    f["pitch_mean"] = pitches.mean()
    f["pitch_std"] = pitches.std()
    f["pitch_min"] = int(pitches.min())
    f["pitch_max"] = int(pitches.max())
    f["pitch_range"] = int(pitches.max() - pitches.min())
    f["pitch_median"] = float(np.median(pitches))

    # duration weighted pitch class distribution
    pc_dur = np.zeros(12)
    for n in notes:
        pc_dur[n.pitch % 12] += n.end - n.start
    pc_p = pc_dur / pc_dur.sum() if pc_dur.sum() else pc_dur
    f["pc_entropy"] = entropy(pc_p)

    if np.count_nonzero(pc_dur) >= 2:
        best_maj = max(np.corrcoef(pc_dur, np.roll(MAJOR_PROFILE, k))[0, 1]
                       for k in range(12))
        best_min = max(np.corrcoef(pc_dur, np.roll(MINOR_PROFILE, k))[0, 1]
                       for k in range(12))
        f["key_fit"] = max(best_maj, best_min)
        f["key_major_leaning"] = best_maj - best_min
    else:
        f["key_fit"] = np.nan
        f["key_major_leaning"] = np.nan

    # bass register: mean lowest sounding pitch over sounding frames
    sounding = binroll.any(axis=0)
    if sounding.any():
        f["bass_mean"] = float(binroll[:, sounding].argmax(axis=0).mean())
    else:
        f["bass_mean"] = np.nan


def texture_features(pm, notes, binroll, f):
    """Polyphony, voice count, and note density."""
    end = pm.get_end_time()
    active = binroll.sum(axis=0)
    sounding = active[active > 0]
    f["poly_mean"] = float(sounding.mean()) if sounding.size else 0.0
    f["poly_std"] = float(sounding.std()) if sounding.size else 0.0
    f["mono_frac"] = float((sounding == 1).mean()) if sounding.size else 0.0

    # how many instrument tracks sound at once
    T = binroll.shape[1]
    inst_active = np.zeros(T)
    for inst in pm.instruments:
        if inst.is_drum or not inst.notes:
            continue
        track = np.zeros(T, dtype=bool)
        for n in inst.notes:
            s = min(int(n.start * ROLL_FS), T - 1)
            track[s:max(int(n.end * ROLL_FS), s + 1)] = True
        inst_active += track
    f["voices_mean"] = float(inst_active[active > 0].mean()) if sounding.size else 0.0

    f["note_density"] = len(notes) / end if end else 0.0
    onsets = np.array([n.start for n in notes])
    windows = np.histogram(onsets, bins=np.arange(0, end + 10, 10))[0]
    f["density_std"] = float(windows.std()) if windows.size else 0.0


def rhythm_features(pm, notes, f):
    """Durations on the beat grid, syncopation proxy, and tempo."""
    durations = np.array([n.end - n.start for n in notes])
    f["dur_mean"] = durations.mean()
    f["dur_std"] = durations.std()

    beats = pm.get_beats()
    dur_bins = ["dur_sixteenth", "dur_eighth", "dur_quarter", "dur_half", "dur_whole"]
    if len(beats) >= 2:
        beat_len = np.diff(beats)
        idx = np.clip(np.searchsorted(beats, [n.start for n in notes], side="right") - 1,
                      0, len(beat_len) - 1)
        local = beat_len[idx]
        dur_beats = durations / local
        # bins centered on sixteenth, eighth, quarter, half, whole
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


def dynamics_features(pm, notes, f):
    """Velocity statistics and the flat velocity flag."""
    vel = np.array([n.velocity for n in notes])
    f["vel_mean"] = vel.mean()
    f["vel_std"] = vel.std()
    f["vel_range"] = int(vel.max() - vel.min())
    f["vel_flat"] = float(vel.std() == 0)
    track_stds = [np.std([n.velocity for n in inst.notes])
                  for inst in pm.instruments if not inst.is_drum and inst.notes]
    f["vel_track_std"] = float(np.mean(track_stds)) if track_stds else 0.0


def features(pm):
    """Return a dict of ~66 numeric features for one parsed MIDI file."""
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return None

    binroll = pm.get_piano_roll(fs=ROLL_FS) > 0
    if binroll.shape[1] == 0:
        return None

    f = {}
    melodic_features(pm, f)
    vertical_features(binroll, f)
    pitch_features(notes, binroll, f)
    texture_features(pm, notes, binroll, f)
    rhythm_features(pm, notes, f)
    dynamics_features(pm, notes, f)
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
df = df[["filename"] + [c for c in df.columns if c not in ("filename", "composer")] + ["composer"]]
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"\nwrote {OUT}: {len(df)} rows x {df.shape[1]} cols (skipped {skipped})")
print(df["composer"].value_counts().to_string())
print("NaNs per column (nonzero only):")
na = df.isna().sum()
print(na[na > 0].to_string() if na.any() else "none")
