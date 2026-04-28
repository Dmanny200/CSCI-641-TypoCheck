"""
predict.py  —  CLI domain checker

Usage:
    python scripts/predict.py google.com
    python scripts/predict.py gooogle.com --whois
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from domain_checker import (
    FEATURE_COLS, WHOIS_BAND_LO,
    load_model, extract_features, risk_tier,
    whois_features, adjust_tier, get_sld,
)

RF_MODEL     = ROOT / "models" / "random_forest.joblib"
XGB_MODEL    = ROOT / "models" / "xgboost.joblib"
FEATURES_CSV = ROOT / "data"   / "features.csv"

TIER_FMT = {"ALLOW": "[  ALLOW  ]", "WARNING": "[WARNING  ]", "BLOCK": "[ BLOCK!! ]"}
W = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    parser.add_argument(
        "--whois", action="store_true",
        help="Run a WHOIS lookup when the domain is flagged malicious or confidence is uncertain"
    )
    args = parser.parse_args()

    domain = args.domain.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]

    df = pd.read_csv(FEATURES_CSV, dtype=str)
    top100_slds = [get_sld(d) for d in df[df["source"] == "tranco_top10k"]["domain"].head(100)]

    rf_clf,  rf_thr  = load_model(RF_MODEL)
    xgb_clf, xgb_thr = load_model(XGB_MODEL)

    feat = extract_features(domain, top100_slds)
    X    = np.array([[feat[col] for col in FEATURE_COLS]])

    rf_conf  = float(rf_clf.predict_proba(X)[0][1])
    xgb_conf = float(xgb_clf.predict_proba(X)[0][1])
    max_conf = max(rf_conf, xgb_conf)
    ml_tier  = risk_tier(max_conf)

    print(f"\n{'=' * W}")
    print(f"  Domain: {domain}")
    print(f"{'=' * W}")
    print(f"  {'Model':<20}  {'Label':<12}  {'Conf':>6}  {'Threshold':>9}")
    print(f"  {'-' * (W - 2)}")
    print(f"  {'Random Forest':<20}  {'malicious' if rf_conf >= rf_thr else 'benign':<12}  {rf_conf:>5.1%}  (thr {rf_thr:.2f})")
    print(f"  {'XGBoost':<20}  {'malicious' if xgb_conf >= xgb_thr else 'benign':<12}  {xgb_conf:>5.1%}  (thr {xgb_thr:.2f})")
    print(f"  {'-' * (W - 2)}")
    print(f"  ML verdict (max {max_conf:.1%}): {TIER_FMT[ml_tier]}")
    print(f"{'=' * W}")

    rf_imps  = dict(zip(FEATURE_COLS, rf_clf.feature_importances_))
    xgb_imps = dict(zip(FEATURE_COLS, xgb_clf.feature_importances_))
    avg_imps = {f: (rf_imps[f] + xgb_imps[f]) / 2 for f in FEATURE_COLS}
    print(f"\n  {'Feature':<25}  {'Value':>10}  {'RF imp':>8}  {'XGB imp':>8}")
    print(f"  {'-' * (W - 2)}")
    for f in sorted(FEATURE_COLS, key=lambda f: avg_imps[f], reverse=True):
        v = feat[f]
        print(f"  {f:<25}  {int(v) if v == int(v) else v:.4f:>10}  {rf_imps[f]:>8.4f}  {xgb_imps[f]:>8.4f}")
    print(f"{'=' * W}")

    # WHOIS is opt-in; eligible when flagged malicious (WARNING/BLOCK) or confidence is uncertain
    whois_eligible = ml_tier in ("WARNING", "BLOCK") or max_conf >= WHOIS_BAND_LO
    final_tier = ml_tier

    if args.whois and whois_eligible:
        print(f"\n  Running WHOIS lookup for {domain} ...")
        wf = whois_features(domain)
        final_tier, reasons = adjust_tier(ml_tier, wf)

        print(f"\n  {'=' * (W - 2)}")
        print(f"  WHOIS")
        print(f"  {'-' * (W - 2)}")
        if not wf.get("available"):
            print("  Lookup timed out or record unavailable")
        else:
            age     = wf.get("domain_age_days")
            privacy = wf.get("has_privacy_protection")
            known   = wf.get("registrar_known")
            expires = wf.get("expires_soon")
            print(f"  {'Domain age':<25}  {str(age) + ' days' if age is not None else 'unknown':>10}")
            print(f"  {'Privacy protection':<25}  {'Yes' if privacy else 'No':>10}")
            print(f"  {'Known registrar':<25}  {'Yes' if known else 'No':>10}")
            print(f"  {'Expires within 30d':<25}  {'Yes' if expires else 'No':>10}")
        print(f"  {'-' * (W - 2)}")
        for r in reasons:
            print(f"    - {r}")
        if final_tier != ml_tier:
            print(f"    Tier adjusted: {TIER_FMT[ml_tier]} -> {TIER_FMT[final_tier]}")
        print(f"  {'=' * (W - 2)}")

    elif whois_eligible and not args.whois:
        print(f"\n  Tip: re-run with --whois to look up registration data for this domain")

    print(f"\n{'=' * W}")
    print(f"  FINAL: {TIER_FMT[final_tier]}")
    if final_tier != ml_tier:
        print(f"  (ML said {TIER_FMT[ml_tier]}, adjusted by WHOIS)")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
