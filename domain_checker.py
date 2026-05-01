"""
domain_checker.py

Shared feature extraction, WHOIS lookup, and risk-tier logic.
Imported by server.py and scripts/predict.py.

All public functions now accept a full URL (e.g. "https://example.com/path?q=1").
Bare hostnames / domains are also accepted and normalized to http://host/.
"""

import concurrent.futures
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

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
    # ── domain-level ────────────────────────────────────────────────────
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
    "homoglyph_count",
    "bigram_log_prob",
    # ── URL-level ────────────────────────────────────────────────────────
    "url_length",
    "path_length",
    "path_depth",
    "has_query",
    "query_length",
    "path_entropy",
    "at_in_url",
    "double_slash_in_path",
]

# ── brand reference set for typosquat detection ───────────────────────────────
# Used by min_lev_distance: these are the SLDs most commonly impersonated by
# typosquatters. Replacing the old Tranco-derived top-100 list ensures that
# amaz0n, paypa1, micros0ft etc. all compute a small edit distance.
BRAND_TARGETS: list[str] = [
    # Search / portals
    "google", "bing", "yahoo", "duckduckgo", "baidu", "yandex",
    # Social media
    "facebook", "instagram", "twitter", "tiktok", "snapchat",
    "linkedin", "reddit", "pinterest", "discord", "telegram",
    # Video / streaming
    "youtube", "twitch", "netflix", "spotify", "hulu",
    # E-commerce / retail
    "amazon", "ebay", "walmart", "etsy", "target", "shopify",
    "aliexpress", "bestbuy",
    # Finance / banking / payment
    "paypal", "venmo", "stripe", "coinbase", "binance", "robinhood",
    "chase", "wellsfargo", "bankofamerica", "citibank", "capitalone",
    # Big tech
    "microsoft", "apple", "meta", "nvidia",
    # Dev / cloud
    "github", "gitlab", "stackoverflow", "docker",
    "cloudflare", "digitalocean",
    # Productivity / collaboration
    "dropbox", "notion", "figma", "zoom", "slack",
    "outlook", "office", "icloud", "onedrive",
    # News / reference
    "cnn", "bbc", "nytimes", "wikipedia",
    # Gaming / logistics
    "steam", "roblox", "fedex", "ups",
]

# Digits frequently substituted for visually similar letters in typosquats:
# 0→o, 1→l/i, 3→e, 4→a, 5→s, 6→g/b, 7→t, 8→b, 9→g/q
HOMOGLYPH_CHARS: frozenset[str] = frozenset("013456789")

# English letter bigram frequencies (percentage of all letter pairs in English text).
# Source: standard corpus analysis. Used to compute bigram_log_prob.
_EN_BIGRAMS: dict[str, float] = {
    'th': 3.56, 'he': 3.07, 'in': 2.43, 'er': 2.05, 'an': 1.99,
    're': 1.85, 'nd': 1.81, 'at': 1.49, 'on': 1.45, 'nt': 1.42,
    'ha': 1.40, 'es': 1.38, 'st': 1.34, 'en': 1.34, 'ed': 1.30,
    'to': 1.29, 'it': 1.27, 'ou': 1.27, 'ea': 1.27, 'hi': 1.22,
    'is': 1.22, 'or': 1.21, 'ti': 1.20, 'as': 1.19, 'te': 1.18,
    'et': 1.15, 'ng': 1.13, 'of': 1.13, 'al': 1.12, 'de': 1.11,
    'se': 1.10, 'le': 1.05, 'sa': 1.02, 'si': 1.01, 'ar': 0.99,
    've': 0.94, 'ra': 0.91, 'ld': 0.89, 'ur': 0.86, 'we': 0.86,
    'ne': 0.85, 'ss': 0.85, 'el': 0.83, 'ro': 0.83, 'li': 0.82,
    'ri': 0.82, 'io': 0.81, 'co': 0.79, 'il': 0.79, 'me': 0.78,
    'ic': 0.77, 'la': 0.75, 'ma': 0.74, 'om': 0.74, 'no': 0.73,
    'ca': 0.72, 'wi': 0.72, 'fo': 0.71, 'ge': 0.70, 'ta': 0.69,
    'tr': 0.68, 'be': 0.66, 'lo': 0.65, 'di': 0.63, 'ly': 0.63,
    'pa': 0.62, 'do': 0.60, 'rs': 0.60, 'ce': 0.59, 'so': 0.57,
    'wa': 0.56, 'pr': 0.55, 'po': 0.55, 'na': 0.54, 'mi': 0.54,
    'pe': 0.54, 'mo': 0.53, 'ac': 0.53, 'pl': 0.52, 'ts': 0.51,
    'rt': 0.50, 'cu': 0.49, 'cl': 0.49, 'wh': 0.48, 'ec': 0.46,
    'ab': 0.45, 'ho': 0.45, 'ch': 0.44, 'sh': 0.43, 'ow': 0.42,
    'oo': 0.41, 'ad': 0.40, 'ai': 0.40, 'go': 0.40, 'fi': 0.39,
    'pu': 0.38, 'sp': 0.38, 'bu': 0.37, 'ut': 0.37, 'ie': 0.37,
    'ue': 0.36, 'ry': 0.35, 'wn': 0.35, 'ee': 0.35, 'ap': 0.34,
    'ay': 0.34, 'ga': 0.33, 'ag': 0.32, 'ui': 0.32, 'em': 0.32,
    'ci': 0.31, 'ck': 0.31, 'am': 0.30, 'tu': 0.30, 'id': 0.29,
    'if': 0.28, 'im': 0.28, 'ep': 0.27, 'op': 0.27, 'ob': 0.26,
    'eg': 0.25, 'ok': 0.25, 'ip': 0.24, 'sc': 0.24, 'ev': 0.24,
    'ew': 0.23, 'ex': 0.23, 'ug': 0.22, 'um': 0.22, 'lu': 0.21,
    'un': 0.20, 'ye': 0.20, 'mp': 0.19, 'ig': 0.18, 'ib': 0.17,
    'ot': 0.16, 'gi': 0.16, 'nc': 0.15, 'nk': 0.15, 'aw': 0.14,
    'av': 0.13, 'ov': 0.13, 'ol': 0.12, 'gl': 0.12, 'fl': 0.12,
    'bl': 0.11, 'br': 0.11, 'cr': 0.10, 'dr': 0.10, 'fr': 0.10,
    'gr': 0.10, 'sk': 0.09, 'sl': 0.09, 'sm': 0.09, 'sn': 0.09,
    'sw': 0.09, 'tw': 0.08, 'wr': 0.08, 'ph': 0.08,
}
# English letter unigram frequencies (%) — fallback for bigrams not in table.
_EN_UNIGRAMS: dict[str, float] = {
    'a': 8.17, 'b': 1.49, 'c': 2.78, 'd': 4.25, 'e': 12.70,
    'f': 2.23, 'g': 2.02, 'h': 6.09, 'i': 6.97, 'j': 0.15,
    'k': 0.77, 'l': 4.03, 'm': 2.41, 'n': 6.75, 'o': 7.51,
    'p': 1.93, 'q': 0.10, 'r': 5.99, 's': 6.33, 't': 9.06,
    'u': 2.76, 'v': 0.98, 'w': 2.36, 'x': 0.15, 'y': 1.97,
    'z': 0.07,
}
_BIGRAM_FLOOR = 0.01  # floor for bigrams absent from both tables

WHOIS_BAND_LO = 0.40
WHOIS_BAND_HI = 0.70

# TLDs that are strictly regulated and cannot be registered by threat actors.
# .edu  → accredited US institutions only (EDUCAUSE)
# .gov  → US government entities only (GSA)
# .mil  → US military only
SAFE_TLDS: frozenset[str] = frozenset({".edu", ".gov", ".mil"})

# Registered domains (sld.tld) that are always safe regardless of ML score.
# These are major, well-established sites whose traffic patterns cause
# false positives in the ML models.
KNOWN_SAFE_DOMAINS: frozenset[str] = frozenset({
    # Search / portals
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com", "baidu.com",
    # Social
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "linkedin.com", "reddit.com", "pinterest.com", "snapchat.com",
    "discord.com", "tumblr.com",
    # Video / streaming
    "youtube.com", "twitch.tv", "vimeo.com", "netflix.com",
    "spotify.com", "hulu.com", "disneyplus.com",
    # Tech / dev
    "github.com", "gitlab.com", "stackoverflow.com", "npmjs.com",
    "pypi.org", "rust-lang.org",
    # Microsoft
    "microsoft.com", "office.com", "outlook.com", "live.com",
    "azure.com", "visualstudio.com",
    # Apple
    "apple.com", "icloud.com",
    # Amazon / AWS
    "amazon.com", "amazonaws.com", "aws.amazon.com",
    # Commerce
    "ebay.com", "walmart.com", "etsy.com", "shopify.com",
    # Finance
    "paypal.com", "chase.com", "bankofamerica.com", "wellsfargo.com",
    "coinbase.com", "stripe.com",
    # News
    "cnn.com", "bbc.com", "nytimes.com", "theguardian.com",
    "reuters.com", "apnews.com",
    # Cloud / productivity
    "cloudflare.com", "dropbox.com", "notion.so", "figma.com",
    "zoom.us", "slack.com", "atlassian.com", "salesforce.com",
    # Infra / CDN
    "wikipedia.org", "wikimedia.org",
})


def is_known_safe(url: str) -> bool:
    """Return True if the URL belongs to a regulated TLD or known-safe domain."""
    url = _normalize_url(url)
    try:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        if len(parts) >= 2:
            tld = f".{parts[-1]}"
            if tld in SAFE_TLDS:
                return True
            reg = f"{parts[-2]}.{parts[-1]}"
            return reg in KNOWN_SAFE_DOMAINS
    except Exception:
        pass
    return False


KNOWN_REGISTRARS = {
    "godaddy", "namecheap", "google", "cloudflare", "markmonitor",
    "network solutions", "amazon registrar",
}
PRIVACY_KEYWORDS = {"privacy", "whoisguard", "redacted", "protected", "withheld"}

_TIERS = ["ALLOW", "WARNING", "BLOCK"]


# ── URL helpers ───────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Ensure the URL has a scheme so urlparse works correctly."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def get_hostname(url: str) -> str:
    """Extract bare hostname (no www.) from a URL or domain string."""
    url = _normalize_url(url)
    try:
        h = (urlparse(url).hostname or "").lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def url_to_sld(url: str) -> str:
    """SLD from a URL or bare hostname — used to build Levenshtein reference set."""
    return get_sld(get_hostname(url))


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(path: Path):
    """Return (classifier, threshold). Handles plain clf or {"model","threshold"} dict."""
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["model"], obj.get("threshold", 0.5)
    return obj, 0.5


# ── feature helpers ───────────────────────────────────────────────────────────

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


def _homoglyph_count(s: str) -> int:
    """Count digits that are common letter substitutions (0,1,3-9) in s."""
    return sum(1 for c in s if c in HOMOGLYPH_CHARS)


def _bigram_log_prob(s: str) -> float:
    """Average log-probability of consecutive alphabetic pairs.

    Uses a lookup table for common English bigrams; falls back to the product
    of English unigram frequencies for pairs not in the table. DGA domains
    score much lower than natural-language domain names.
    """
    letters = [c for c in s.lower() if c.isalpha()]
    if len(letters) < 2:
        return 0.0
    log_sum = 0.0
    for i in range(len(letters) - 1):
        bg = letters[i] + letters[i + 1]
        if bg in _EN_BIGRAMS:
            p = _EN_BIGRAMS[bg]
        else:
            # Independence model: P(xy) ≈ P(x)%×P(y)%÷100 keeps units consistent
            # with the bigram table (both in "% of all letter pairs").
            p = max(
                _EN_UNIGRAMS.get(letters[i], 0.01) * _EN_UNIGRAMS.get(letters[i + 1], 0.01) / 100,
                _BIGRAM_FLOOR,
            )
        log_sum += math.log(p)
    return round(log_sum / (len(letters) - 1), 6)


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(url: str) -> dict[str, float]:
    """Extract all features from a full URL (or bare hostname).

    Domain-level features are computed on the URL's hostname.
    URL-level features capture path/query characteristics.
    min_lev_distance uses BRAND_TARGETS (curated typosquat-target brand list).
    """
    url = _normalize_url(url)
    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()
    domain   = hostname[4:] if hostname.startswith("www.") else hostname

    # domain-level
    length    = len(domain)
    digits    = sum(1 for c in domain if c.isdigit())
    hyphens   = domain.count("-")
    non_alnum = sum(1 for c in domain if not c.isalnum() and c != ".")
    tld = get_tld(domain)
    sld = get_sld(domain)
    min_lev = min(lev_distance(sld, ref) for ref in BRAND_TARGETS) if BRAND_TARGETS else 0

    # URL-level
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


# ── risk tier ─────────────────────────────────────────────────────────────────

def risk_tier(confidence: float, tld: str = "") -> str:
    # .org/.net are lower-risk TLDs; require higher confidence before blocking.
    block_thr = 0.93 if tld in {".org", ".net"} else 0.80
    if confidence < 0.50:       return "ALLOW"
    if confidence < block_thr:  return "WARNING"
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


def whois_features(url: str, timeout: int = 5) -> dict:
    """Run a WHOIS lookup for the hostname in the given URL."""
    import whois
    domain = get_hostname(url)
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
    age     = wf.get("domain_age_days")
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


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(url: str, rf_clf, rf_thr, xgb_clf, xgb_thr):
    """Feature extraction + both models on a full URL. Returns (feat, rf_conf, xgb_conf)."""
    feat = extract_features(url)
    X = np.array([[feat[col] for col in FEATURE_COLS]])
    rf_conf  = float(rf_clf.predict_proba(X)[0][1])
    xgb_conf = float(xgb_clf.predict_proba(X)[0][1])
    return feat, rf_conf, xgb_conf
