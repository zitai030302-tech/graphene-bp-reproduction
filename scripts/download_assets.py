#!/usr/bin/env python3
"""Download the minimal upstream assets needed for the reproduction."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/TAMU-ESP/Graphene_BP/main"

ASSETS = {
    "BP_BioZ_v0.py": "upstream/BP_BioZ_v0.py",
    "UtilityFuncs.py": "upstream/UtilityFuncs.py",
    "config.hjson": "upstream/config.hjson",
    "requirements.txt": "upstream/requirements.txt",
    "Data/features/2020-11-05/f2_ma20_mn1_mean_all.csv": (
        "upstream/Data/features/2020-11-05/f2_ma20_mn1_mean_all.csv"
    ),
}


def download(url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url} -> {destination}")
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for upstream_path, local_path in ASSETS.items():
        download(f"{BASE_URL}/{upstream_path}", args.root / local_path, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

