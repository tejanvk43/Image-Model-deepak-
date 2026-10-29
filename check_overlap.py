"""
Check whether any TEST image also appears (byte for byte) among the TRAINING images.

    python check_overlap.py --test raw/LOL-v1 --train data/train

--test and --train take one or more folders; they are searched recursively for images.
The script stops with an error if a folder is missing or holds no images, so a wrong path can
never look like "no overlap".
"""
import argparse
import hashlib
import sys
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".bmp"}


def images(folders):
    found = []
    for f in folders:
        p = Path(f).expanduser()
        if not p.exists():
            sys.exit(f"ERROR: folder not found: {p}\nCheck the path (run `ls` to see where you are).")
        files = [x for x in p.rglob("*") if x.suffix.lower() in IMG]
        if not files:
            sys.exit(f"ERROR: no images found under {p}")
        found += files
    return found


def digest(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="+", required=True, help="test image folder(s), e.g. the LOL-v1 eval15 folder")
    ap.add_argument("--train", nargs="+", required=True, help="the folder(s) you trained on")
    a = ap.parse_args()

    test, train = images(a.test), images(a.train)
    train_hash = {}
    for p in train:
        train_hash.setdefault(digest(p), p)
    hits = [(p, train_hash[h]) for p in test if (h := digest(p)) in train_hash]

    print(f"test images checked : {len(test)}")
    print(f"training images     : {len(train)}")
    print(f"test images that also appear in the training folders: {len(hits)}")
    for t, tr in hits[:20]:
        print(f"  {t}  ==  {tr}")
    if hits:
        print("\nThese images were seen in training, so scores on them are inflated. Exclude them or say so in the report.")
    else:
        print("\nNo identical files found. (Byte-identical check only: a re-saved or resized copy would not be detected.)")


if __name__ == "__main__":
    main()