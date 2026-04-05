"""
extract_features.py

Reads data/domains.csv and extracts features for each domain.
Output: data/features.csv
"""

import math
import re
from pathlib import Path

import pandas as pd
from Levenshtein import distance as lev_distance
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_CSV = DATA_DIR / "domains.csv"
OUTPUT_CSV = DATA_DIR / "features.csv"

# ── TLD risk map ─────────────────────────────────────────────────────────────
TLD_RISK: dict[str, int] = {
    ".com": 0, ".org": 0, ".net": 0, ".edu": 0, ".gov": 0,
    ".io":  1, ".co":  1, ".info": 1,
}
# everything else -> 2

# ── character sets ────────────────────────────────────────────────────────────
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_sld(domain: str) -> str:
    """
    Extract the second-level domain label.
    'google.com'      -> 'google'
    'mail.google.com' -> 'google'
    'foo.co.uk'       -> 'foo'   (naive: always second-to-last label)
    """
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return domain


def get_tld(domain: str) -> str:
    """Return the TLD including the leading dot, e.g. '.com'."""
    idx = domain.rfind(".")
    return domain[idx:] if idx != -1 else ""


def shannon_entropy(s: str) -> float:
    """Shannon entropy of a string in bits."""
    if not s:
        return 0.0
    n = len(s)
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def char_continuity_rate(s: str) -> float:
    """
    Fraction of adjacent pairs (c_i, c_{i+1}) where both characters
    belong to the same class (both letters or both digits).
    Dots, hyphens, etc. are treated as 'other' and never match.
    """
    if len(s) < 2:
        return 0.0
    def cls(c: str) -> str:
        if c.isalpha():
            return "a"
        if c.isdigit():
            return "d"
        return "o"
    pairs = len(s) - 1
    same = sum(1 for i in range(pairs) if cls(s[i]) == cls(s[i + 1]) != "o")
    return same / pairs


def vowel_consonant_ratio(s: str) -> float:
    """Vowels / consonants; returns 0.0 when there are no consonants."""
    letters = s.lower()
    v = sum(1 for c in letters if c in VOWELS)
    c = sum(1 for c in letters if c in CONSONANTS)
    return v / c if c else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# feature extraction (row-level)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(domain: str, top100_slds: list[str]) -> dict:
    length = len(domain)
    digits = sum(1 for c in domain if c.isdigit())
    hyphens = domain.count("-")
    non_alnum = sum(1 for c in domain if not c.isalnum() and c != ".")
    tld = get_tld(domain)
    sld = get_sld(domain)

    min_lev = min(lev_distance(sld, ref) for ref in top100_slds)

    return {
        "domain_length":        length,
        "subdomain_count":      max(domain.count(".") - 1, 0),
        "digit_count":          digits,
        "hyphen_count":         hyphens,
        "non_alnum_count":      non_alnum,
        "vowel_consonant_ratio": vowel_consonant_ratio(domain),
        "entropy":              round(shannon_entropy(domain), 6),
        "char_continuity_rate": round(char_continuity_rate(domain), 6),
        "digit_ratio":          round(digits / length, 6) if length else 0.0,
        "tld_risk_score":       TLD_RISK.get(tld, 2),
        "min_lev_distance":     min_lev,
    }


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading dataset ...")
    df = pd.read_csv(INPUT_CSV, dtype={"domain": str, "label": int, "source": str})
    print(f"  {len(df):,} rows loaded from {INPUT_CSV.name}")

    # ── precompute top-100 Tranco SLDs ────────────────────────────────────
    tranco_rows = df[df["source"] == "tranco_top10k"]["domain"]
    top100_slds: list[str] = [get_sld(d) for d in tranco_rows.head(100)]
    print(f"  Reference SLDs for Levenshtein: {top100_slds[:5]} ...")

    # ── extract features ──────────────────────────────────────────────────
    print(f"\nExtracting features for {len(df):,} domains ...")
    feature_rows: list[dict] = []
    for domain in tqdm(df["domain"], unit="domain", smoothing=0.05):
        feature_rows.append(extract_features(str(domain), top100_slds))

    feat_df = pd.DataFrame(feature_rows)

    # ── combine with original columns ────────────────────────────────────
    result = pd.concat(
        [df[["domain", "label", "source"]].reset_index(drop=True), feat_df],
        axis=1,
    )

    # ── summary ───────────────────────────────────────────────────────────
    print("\nFeature stats:")
    feature_cols = list(feat_df.columns)
    stats = result[feature_cols].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())

    # ── save ──────────────────────────────────────────────────────────────
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}  ({len(result):,} rows, {len(result.columns)} columns)")


if __name__ == "__main__":
    main()
