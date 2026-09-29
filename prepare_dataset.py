"""
Reorganize an unzipped low-light dataset into the layout datasets.py expects:

    data/
        train/low/*.png   train/high/*.png
        test/low/*.png    test/high/*.png

Works by scanning for folder pairs whose names look like low/high
(Low, low, input, Normal, normal, high, target, gt ...) and matching files by
their numeric id, so it copes with LOL-v2's `low00001.png` vs `normal00001.png`
naming. Train vs test is decided by whether "test" appears in the folder path.

Usage (run once per dataset, with a unique --prefix so filenames never clash):
    python prepare_dataset.py --src raw/LOL-v2 --dst data --prefix lolv2
    python prepare_dataset.py --src raw/LOL     --dst data --prefix lol
    python prepare_dataset.py --src raw/VE-LOL-L --dst data --prefix velol
"""
import argparse
import re
import shutil
from pathlib import Path

LOW_NAMES = {"low", "input", "lq", "dark"}
HIGH_NAMES = {"normal", "high", "target", "gt", "hq", "bright"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def file_key(path: Path) -> str:
    """Pairing key: filename stem minus a leading low/normal/high/... tag.

    'low00012' and 'normal00012' -> '00012'  (LOL, LOL-v2 real)
    'r1e69b067t' in both folders  -> 'r1e69b067t' (LOL-v2 synthetic, same names)
    """
    return re.sub(r"^(low|normal|high|input|target|gt)[_-]?", "", path.stem, flags=re.I).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Unzipped dataset folder")
    ap.add_argument("--dst", default="data", help="Output root (default: data)")
    ap.add_argument("--prefix", required=True, help="Short dataset tag, e.g. lolv2")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")

    def is_low(d: Path) -> bool:
        n = d.name.lower()
        return n in LOW_NAMES or ("low" in n and "normal" not in n and "high" not in n)

    def find_high(low_dir: Path):
        sibs = [s for s in low_dir.parent.iterdir() if s.is_dir() and s != low_dir]
        for s in sibs:  # LOL style: Low/ next to Normal/
            if s.name.lower() in HIGH_NAMES:
                return s
        # VE-LOL style: ...-Low_test next to ...-Normal_test
        want = re.sub("low", "normal", low_dir.name, flags=re.I).lower()
        return next((s for s in sibs if s.name.lower() == want), None)

    low_dirs = [d for d in src.rglob("*") if d.is_dir() and is_low(d)]
    total = 0
    for low_dir in low_dirs:
        high_dir = find_high(low_dir)
        if high_dir is None:
            continue

        rel = str(low_dir.relative_to(src)).lower()
        split = "test" if "test" in rel else "train"
        sub = "synth" if "syn" in rel else "real"
        out_low = dst / split / "low"
        out_high = dst / split / "high"
        out_low.mkdir(parents=True, exist_ok=True)
        out_high.mkdir(parents=True, exist_ok=True)

        highs = {file_key(p): p for p in high_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        count = 0
        for lp in sorted(low_dir.iterdir()):
            if lp.suffix.lower() not in IMG_EXT:
                continue
            hp = highs.get(file_key(lp))
            if hp is None:
                continue
            name = f"{args.prefix}_{sub}_{file_key(lp)}.png"
            shutil.copy2(lp, out_low / name)
            shutil.copy2(hp, out_high / name)
            count += 1
        print(f"[{split}/{sub}] {low_dir.relative_to(src)} -> {count} pairs")
        total += count

    if total == 0:
        raise SystemExit(
            "No low/high pairs found. Check that you unzipped first and that the "
            "folders are named like Low/Normal (or input/target)."
        )
    print(f"\nDone: {total} pairs copied into {dst}/")


if __name__ == "__main__":
    main()