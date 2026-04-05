"""
predict.py

Predict whether a domain is benign or malicious using both trained models.

Usage:
    python scripts/predict.py gooogle.com
    python scripts/predict.py google.com
"""

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from Levenshtein import distance as lev_distance

# ── paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "data"   / "features.csv"
RF_MODEL     = ROOT / "models" / "random_forest.joblib"
XGB_MODEL    = ROOT / "models" / "xgboost.joblib"

# ── constants (must match extract_features.py exactly) ───────────────────────
TLD_RISK: dict[str, int] = {
    ".com": 0, ".org": 0, ".net": 0, ".edu": 0, ".gov": 0,
    ".io":  1, ".co":  1, ".info": 1,
}
VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

FEATURE_COLS = [
    "domain_length",
    "subdomain_count",
    "digit_count",
    "hyphen_count",
    "non_alnum_count",
    "vowel_consonant_ratio",
    "entropy",
    "char_continuity_rate",
    "digit_ratio",
    "tld_risk_score",
    "min_lev_distance",
]

# ── risk tiers ────────────────────────────────────────────────────────────────
def risk_tier(confidence: float) -> str:
    if confidence < 0.50:
        return "ALLOW"
    if confidence < 0.80:
        return "WARNING"
    return "BLOCK"

TIER_STYLE = {
    "ALLOW":   "[  ALLOW  ]",
    "WARNING": "[WARNING  ]",
    "BLOCK":   "[ BLOCK!! ]",
}


# ═══════════════════════════════════════════════════════════════════════════
# feature helpers (identical logic to extract_features.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_sld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-2] if len(parts) >= 2 else domain


def get_tld(domain: str) -> str:
    idx = domain.rfind(".")
    return domain[idx:] if idx != -1 else ""


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def char_continuity_rate(s: str) -> float:
    if len(s) < 2:
        return 0.0
    def cls(c: str) -> str:
        if c.isalpha():  return "a"
        if c.isdigit():  return "d"
        return "o"
    pairs = len(s) - 1
    same  = sum(1 for i in range(pairs) if cls(s[i]) == cls(s[i + 1]) != "o")
    return same / pairs


def vowel_consonant_ratio(s: str) -> float:
    v = sum(1 for c in s.lower() if c in VOWELS)
    c = sum(1 for c in s.lower() if c in CONSONANTS)
    return v / c if c else 0.0


def extract_features(domain: str, top100_slds: list[str]) -> dict[str, float]:
    length  = len(domain)
    digits  = sum(1 for c in domain if c.isdigit())
    hyphens = domain.count("-")
    non_alnum = sum(1 for c in domain if not c.isalnum() and c != ".")
    tld = get_tld(domain)
    sld = get_sld(domain)
    min_lev = min(lev_distance(sld, ref) for ref in top100_slds)

    return {
        "domain_length":         length,
        "subdomain_count":       max(domain.count(".") - 1, 0),
        "digit_count":           digits,
        "hyphen_count":          hyphens,
        "non_alnum_count":       non_alnum,
        "vowel_consonant_ratio": vowel_consonant_ratio(domain),
        "entropy":               round(shannon_entropy(domain), 6),
        "char_continuity_rate":  round(char_continuity_rate(domain), 6),
        "digit_ratio":           round(digits / length, 6) if length else 0.0,
        "tld_risk_score":        TLD_RISK.get(tld, 2),
        "min_lev_distance":      min_lev,
    }


# ═══════════════════════════════════════════════════════════════════════════
# feature importance explanation
# ═══════════════════════════════════════════════════════════════════════════

def top_contributing_features(
    feature_values: dict[str, float],
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[tuple[str, float, float]]:
    """
    Returns top_n (feature_name, feature_value, importance) tuples,
    ranked by model importance.
    """
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(name, feature_values[name], imp) for name, imp in ranked[:top_n]]


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <domain>")
        print("Example: python scripts/predict.py gooogle.com")
        sys.exit(1)

    domain = sys.argv[1].strip().lower()

    # ── load reference SLDs ───────────────────────────────────────────────
    df_feat     = pd.read_csv(FEATURES_CSV, dtype=str)
    tranco_rows = df_feat[df_feat["source"] == "tranco_top10k"]["domain"]
    top100_slds = [get_sld(d) for d in tranco_rows.head(100)]

    # ── extract features ──────────────────────────────────────────────────
    feat_dict = extract_features(domain, top100_slds)
    X = np.array([[feat_dict[col] for col in FEATURE_COLS]])

    # ── load models ───────────────────────────────────────────────────────
    rf  = joblib.load(RF_MODEL)
    xgb = joblib.load(XGB_MODEL)

    # ── predict ───────────────────────────────────────────────────────────
    rf_proba  = rf.predict_proba(X)[0]   # [p_benign, p_malicious]
    xgb_proba = xgb.predict_proba(X)[0]

    rf_conf  = rf_proba[1]    # probability of malicious
    xgb_conf = xgb_proba[1]

    rf_label  = "malicious" if rf_conf  >= 0.5 else "benign"
    xgb_label = "malicious" if xgb_conf >= 0.5 else "benign"

    # final tier: use the higher malicious confidence of the two
    max_conf = max(rf_conf, xgb_conf)
    tier     = risk_tier(max_conf)
    tier_str = TIER_STYLE[tier]

    # ── print results ─────────────────────────────────────────────────────
    W = 60
    print()
    print("=" * W)
    print(f"  Domain: {domain}")
    print("=" * W)
    print(f"  {'Model':<20}  {'Prediction':<12}  {'Confidence':>10}")
    print("  " + "-" * (W - 2))
    print(f"  {'Random Forest':<20}  {rf_label:<12}  {rf_conf:>9.1%}")
    print(f"  {'XGBoost':<20}  {xgb_label:<12}  {xgb_conf:>9.1%}")
    print("  " + "-" * (W - 2))
    print(f"  Final risk tier (max confidence {max_conf:.1%}):  {tier_str}")
    print("=" * W)

    # ── feature breakdown ─────────────────────────────────────────────────
    print()
    print("  Feature breakdown:")
    print(f"  {'Feature':<25}  {'Value':>10}  {'RF imp':>8}  {'XGB imp':>8}")
    print("  " + "-" * (W - 2))

    rf_imps  = dict(zip(FEATURE_COLS, rf.feature_importances_))
    xgb_imps = dict(zip(FEATURE_COLS, xgb.feature_importances_))

    # rank by average importance across both models
    avg_imps = {f: (rf_imps[f] + xgb_imps[f]) / 2 for f in FEATURE_COLS}
    ranked_feats = sorted(FEATURE_COLS, key=lambda f: avg_imps[f], reverse=True)

    for feat in ranked_feats:
        val  = feat_dict[feat]
        ri   = rf_imps[feat]
        xi   = xgb_imps[feat]
        # format integers cleanly, floats to 4dp
        val_str = f"{int(val)}" if val == int(val) else f"{val:.4f}"
        print(f"  {feat:<25}  {val_str:>10}  {ri:>8.4f}  {xi:>8.4f}")

    print("=" * W)
    print()


if __name__ == "__main__":
    main()
