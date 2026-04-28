"""
domain_checker.py

Shared feature extraction, WHOIS lookup, and risk-tier logic.
Imported by server.py and scripts/predict.py.
"""

import concurrent.futures
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from Levenshtein import distance as lev_distance

# ── constants ─────────────────────────────────────────────────────────────────

TLD_RISK: dict[str, int] = {
    ".com": 0, ".org": 0, ".net": 0, ".edu": 0, ".gov": 0,
    ".io":  1, ".co":  1, ".info": 1,
}
VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

FEATURE_COLS = [
    "domain_length", "subdomain_count", "digit_count", "hyphen_count",
    "non_alnum_count", "vowel_consonant_ratio", "entropy",
    "char_continuity_rate", "digit_ratio", "tld_risk_score", "min_lev_distance",
]

WHOIS_BAND_LO = 0.40
WHOIS_BAND_HI = 0.70

KNOWN_REGISTRARS = {
    "godaddy", "namecheap", "google", "cloudflare", "markmonitor",
    "network solutions", "amazon registrar",
}
PRIVACY_KEYWORDS = {"privacy", "whoisguard", "redacted", "protected", "withheld"}

_TIERS = ["ALLOW", "WARNING", "BLOCK"]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(path: Path):
    """Return (classifier, threshold). Handles plain clf or {"model", "threshold"} dict."""
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["model"], obj.get("threshold", 0.5)
    return obj, 0.5


# ── feature extraction ────────────────────────────────────────────────────────

def get_sld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-2] if len(parts) >= 2 else domain


def get_tld(domain: str) -> str:
    idx = domain.rfind(".")
    return domain[idx:] if idx != -1 else ""


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _char_continuity_rate(s: str) -> float:
    if len(s) < 2:
        return 0.0
    def cls(c):
        if c.isalpha(): return "a"
        if c.isdigit(): return "d"
        return "o"
    pairs = len(s) - 1
    same  = sum(1 for i in range(pairs) if cls(s[i]) == cls(s[i + 1]) != "o")
    return same / pairs


def _vowel_consonant_ratio(s: str) -> float:
    v = sum(1 for c in s.lower() if c in VOWELS)
    c = sum(1 for c in s.lower() if c in CONSONANTS)
    return v / c if c else 0.0


def extract_features(domain: str, top100_slds: list[str]) -> dict[str, float]:
    length    = len(domain)
    digits    = sum(1 for c in domain if c.isdigit())
    hyphens   = domain.count("-")
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
        "vowel_consonant_ratio": round(_vowel_consonant_ratio(domain), 6),
        "entropy":               round(_shannon_entropy(domain), 6),
        "char_continuity_rate":  round(_char_continuity_rate(domain), 6),
        "digit_ratio":           round(digits / length, 6) if length else 0.0,
        "tld_risk_score":        TLD_RISK.get(tld, 2),
        "min_lev_distance":      min_lev,
    }


# ── risk tier ─────────────────────────────────────────────────────────────────

def risk_tier(confidence: float) -> str:
    if confidence < 0.50: return "ALLOW"
    if confidence < 0.80: return "WARNING"
    return "BLOCK"


# ── WHOIS ─────────────────────────────────────────────────────────────────────

_WHOIS_NULL = {
    "available": False,
    "domain_age_days": None,
    "has_privacy_protection": None,
    "registrar_known": None,
    "expires_soon": None,
}


def _to_naive_utc(dt) -> datetime | None:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def whois_features(domain: str, timeout: int = 5) -> dict:
    import whois
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(whois.whois, domain)
            try:
                w = future.result(timeout=timeout)
            except Exception:
                return dict(_WHOIS_NULL)
    except Exception:
        return dict(_WHOIS_NULL)

    if w is None:
        return dict(_WHOIS_NULL)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    result: dict = {"available": True}

    creation = w.creation_date
    if isinstance(creation, list):
        creation = creation[0] if creation else None
    creation = _to_naive_utc(creation)
    result["domain_age_days"] = max((now - creation).days, 0) if creation else None

    candidates = []
    for attr in ("registrant_name", "registrant_org", "org", "name", "emails"):
        val = getattr(w, attr, None)
        if val:
            candidates.append(str(val).lower())
    combined = " ".join(candidates)
    result["has_privacy_protection"] = any(kw in combined for kw in PRIVACY_KEYWORDS)

    registrar = str(getattr(w, "registrar", "") or "").lower()
    result["registrar_known"] = any(r in registrar for r in KNOWN_REGISTRARS)

    expiry = w.expiration_date
    if isinstance(expiry, list):
        expiry = expiry[0] if expiry else None
    expiry = _to_naive_utc(expiry)
    result["expires_soon"] = (expiry - now).days <= 30 if expiry else None

    return result


def adjust_tier(tier: str, wf: dict) -> tuple[str, list[str]]:
    if not wf.get("available"):
        return tier, ["WHOIS unavailable -- tier unchanged"]

    idx = _TIERS.index(tier)
    reasons: list[str] = []
    age    = wf.get("domain_age_days")
    privacy = wf.get("has_privacy_protection")
    known   = wf.get("registrar_known")

    if age is not None and age < 30:
        idx = min(idx + 1, len(_TIERS) - 1)
        reasons.append(f"domain age {age}d < 30d -> bump up")

    if privacy and age is not None and age < 90:
        idx = min(idx + 1, len(_TIERS) - 1)
        reasons.append(f"privacy protection + age {age}d < 90d -> bump up")

    if known and age is not None and age > 365:
        idx = max(idx - 1, 0)
        reasons.append(f"known registrar + age {age}d > 365d -> bump down")

    if not reasons:
        reasons.append("no WHOIS rules triggered")

    return _TIERS[idx], reasons


# ── inference helper ──────────────────────────────────────────────────────────

def run_inference(domain: str, top100_slds: list[str], rf_clf, rf_thr, xgb_clf, xgb_thr):
    """Run feature extraction + both models. Returns (feat, rf_conf, xgb_conf)."""
    feat = extract_features(domain, top100_slds)
    X = np.array([[feat[col] for col in FEATURE_COLS]])
    rf_conf  = float(rf_clf.predict_proba(X)[0][1])
    xgb_conf = float(xgb_clf.predict_proba(X)[0][1])
    return feat, rf_conf, xgb_conf
