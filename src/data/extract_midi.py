"""Copy Bach, Beethoven, Chopin, and Mozart MIDI files into data/interim/<composer>/."""
import shutil
from pathlib import Path

SRC = Path("data/raw/archive/midiclassics")
DST = Path("data/interim")
COMPOSERS = ["Bach", "Beethoven", "Chopin", "Mozart"]

for composer in COMPOSERS:
    out = DST / composer.lower()
    out.mkdir(parents=True, exist_ok=True)

    files = list((SRC / composer).rglob("*.mid")) + list((SRC / composer).rglob("*.MID"))
    for f in files:
        shutil.copy2(f, out / f.name)

    print(f"{composer}: {len(files)} files")
