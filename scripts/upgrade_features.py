"""
upgrade_features.py

Adds homoglyph_count and bigram_log_prob columns to an existing features CSV,
and recomputes min_lev_distance against BRAND_TARGETS (instead of the old
Tranco-derived top-100 SLD list).

Usage:
    python scripts/upgrade_features.py
    python scripts/upgrade_features.py --input data/features_clean.csv --output data/features_clean_v2.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain_checker import (
    BRAND_TARGETS, get_sld, get_hostname,
    _homoglyph_count, _bigram_log_prob,
)
from Levenshtein import distance as lev_distance

INPUT_CSV  = ROOT / "data" / "features_clean.csv"
OUTPUT_CSV = ROOT / "data" / "features_clean_v2.csv"


def _min_lev(sld: str) -> int:
    return min(lev_distance(sld, ref) for ref in BRAND_TARGETS) if BRAND_TARGETS else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(INPUT_CSV))
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Brand reference set: {len(BRAND_TARGETS)} targets")

    min_levs, homoglyphs, bigrams = [], [], []

    print(f"\nComputing new features ...")
    for url in tqdm(df["url"], unit="url", smoothing=0.05):
        host = get_hostname(str(url))
        sld  = get_sld(host)
        min_levs.append(_min_lev(sld))
        homoglyphs.append(_homoglyph_count(sld))
        bigrams.append(_bigram_log_prob(sld))

    df["min_lev_distance"] = min_levs   # overwrite with brand-target values
    df["homoglyph_count"]  = homoglyphs
    df["bigram_log_prob"]  = bigrams

    df.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}  ({len(df):,} rows, {len(df.columns)} columns)")

    print("\nNew feature stats:")
    print(df[["min_lev_distance", "homoglyph_count", "bigram_log_prob"]].describe().T.to_string())


if __name__ == "__main__":
    main()
