"""
extract_features.py

Reads data/urls.csv and extracts features for each URL.
Output: data/features.csv

Feature groups:
  Domain-level (13): domain_length, subdomain_count, digit_count, hyphen_count,
                     non_alnum_count, vowel_consonant_ratio, entropy,
                     char_continuity_rate, digit_ratio, tld_risk_score,
                     min_lev_distance, homoglyph_count, bigram_log_prob
  URL-level    ( 8): url_length, path_length, path_depth, has_query,
                     query_length, path_entropy, at_in_url, double_slash_in_path
"""

import argparse
import math
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import shared constants and helpers from domain_checker to stay in sync.
from domain_checker import (
    FEATURE_COLS, BRAND_TARGETS, HOMOGLYPH_CHARS,
    TLD_RISK, VOWELS, CONSONANTS,
    _shannon_entropy, _char_continuity_rate, _vowel_consonant_ratio,
    _homoglyph_count, _bigram_log_prob,
    get_sld, get_tld,
)
from Levenshtein import distance as lev_distance

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data"
INPUT_CSV  = DATA_DIR / "urls.csv"
OUTPUT_CSV = DATA_DIR / "features.csv"


# ═══════════════════════════════════════════════════════════════════════════
# feature extraction (row-level)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    domain   = hostname[4:] if hostname.startswith("www.") else hostname

    length    = len(domain)
    digits    = sum(1 for c in domain if c.isdigit())
    hyphens   = domain.count("-")
    non_alnum = sum(1 for c in domain if not c.isalnum() and c != ".")
    tld       = get_tld(domain)
    sld       = get_sld(domain)
    min_lev   = min(lev_distance(sld, ref) for ref in BRAND_TARGETS) if BRAND_TARGETS else 0

    path  = parsed.path if parsed.path else "/"
    query = parsed.query or ""
    path_q_str = path + ("?" + query if query else "")

    return {
        "domain_length":         length,
        "subdomain_count":       max(domain.count(".") - 1, 0),
        "digit_count":           digits,
        "hyphen_count":          hyphens,
        "non_alnum_count":       non_alnum,
        "vowel_consonant_ratio": round(_vowel_consonant_ratio(domain), 6),
        "entropy":               round(_shannon_entropy(domain), 6),
        "char_continuity_rate":  round(_char_continuity_rate(domain), 6),
        "digit_ratio":           round(digits / length, 6) if length else 0.0,
        "tld_risk_score":        TLD_RISK.get(tld, 2),
        "min_lev_distance":      min_lev,
        "homoglyph_count":       _homoglyph_count(sld),
        "bigram_log_prob":       _bigram_log_prob(sld),
        "url_length":            len(url),
        "has_https":             int(parsed.scheme == "https"),
        "path_length":           len(path),
        "path_depth":            max(path.count("/") - 1, 0),
        "has_query":             int(bool(query)),
        "query_length":          len(query),
        "path_entropy":          round(_shannon_entropy(path_q_str), 6),
        "at_in_url":             int("@" in url),
        "double_slash_in_path":  int("//" in path),
    }


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(INPUT_CSV),  help="Input CSV (url, label, source columns)")
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output features CSV")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print("Loading dataset ...")
    df = pd.read_csv(input_path, dtype={"url": str, "label": int, "source": str})
    print(f"  {len(df):,} rows loaded from {input_path.name}")
    print(f"  Levenshtein reference: {len(BRAND_TARGETS)} brand targets")

    print(f"\nExtracting features for {len(df):,} URLs ...")
    feature_rows: list[dict] = []
    for url in tqdm(df["url"], unit="url", smoothing=0.05):
        feature_rows.append(extract_features(str(url)))

    feat_df = pd.DataFrame(feature_rows)

    result = pd.concat(
        [df[["url", "label", "source"]].reset_index(drop=True), feat_df],
        axis=1,
    )

    print("\nFeature stats:")
    stats = result[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())

    result.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}  ({len(result):,} rows, {len(result.columns)} columns)")


if __name__ == "__main__":
    main()
