#!/usr/bin/env python3
"""Check that all model assets are present and match the SHA-256 in ASSETS.json.

    python check_weights.py                 # checks weights/ (or $HANDPOSE_WEIGHTS)
    python check_weights.py --weights DIR
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=None)
    args = p.parse_args()
    weights = args.weights or Path(os.environ.get("HANDPOSE_WEIGHTS", ROOT / "weights"))
    manifest = json.loads((ROOT / "ASSETS.json").read_text())
    ok = True
    for group in ("bundled", "user_provided"):
        for name, expected in manifest[group].items():
            path = weights / Path(name).relative_to("weights")
            if not path.is_file():
                ok = False
                hint = "  -> see README.md, MANO setup" if group == "user_provided" else ""
                print(f"MISSING   {path}{hint}")
                continue
            matches = sha256(path) == expected
            # A bundled file must match; a user conversion of MANO may differ
            # with another MANO release and is reported only.
            if not matches and group == "bundled":
                ok = False
            print(f"{'OK' if matches else 'MISMATCH':9s} {path}")
    if not ok:
        sys.exit(1)
    print("all required assets present")


if __name__ == "__main__":
    main()
