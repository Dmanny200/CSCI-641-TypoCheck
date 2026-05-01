"""
server.py

FastAPI backend for the domain threat detector browser extension.
Run with:  uvicorn server:app --host 127.0.0.1 --port 8000
"""

import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from domain_checker import (
    FEATURE_COLS, WHOIS_BAND_LO,
    load_model, run_inference, risk_tier, whois_features, adjust_tier,
    get_hostname, get_tld, is_known_safe,
)

ROOT      = Path(__file__).resolve().parent
RF_MODEL  = ROOT / "models" / "random_forest_clean_v2.joblib"
XGB_MODEL = ROOT / "models" / "xgboost_clean_v2.joblib"

# ── cache ─────────────────────────────────────────────────────────────────────

PREDICT_TTL = timedelta(hours=1)
WHOIS_TTL   = timedelta(hours=6)

@dataclass
class _Entry:
    data:    dict
    expires: datetime

_predict_cache: dict[str, _Entry] = {}
_whois_cache:   dict[str, _Entry] = {}
_cache_lock = threading.Lock()


def _cache_get(store: dict[str, _Entry], key: str) -> dict | None:
    with _cache_lock:
        entry = store.get(key)
        if entry and datetime.now(timezone.utc) < entry.expires:
            return entry.data
        store.pop(key, None)
        return None


def _cache_set(store: dict[str, _Entry], key: str, data: dict, ttl: timedelta) -> None:
    with _cache_lock:
        store[key] = _Entry(data, datetime.now(timezone.utc) + ttl)


# Domains pre-warmed at startup so first visits are instant.
PREWARM_URLS = [
    # ── major benign sites ────────────────────────────────────────────────
    "https://google.com/", "https://youtube.com/", "https://facebook.com/",
    "https://instagram.com/", "https://twitter.com/", "https://x.com/",
    "https://reddit.com/", "https://wikipedia.org/",
    "https://amazon.com/", "https://ebay.com/", "https://etsy.com/",
    "https://walmart.com/", "https://netflix.com/", "https://spotify.com/",
    "https://twitch.tv/", "https://tiktok.com/",
    "https://apple.com/", "https://microsoft.com/", "https://github.com/",
    "https://stackoverflow.com/", "https://linkedin.com/",
    "https://discord.com/", "https://slack.com/", "https://zoom.us/",
    "https://paypal.com/", "https://chase.com/", "https://bankofamerica.com/",
    "https://wellsfargo.com/", "https://cnn.com/", "https://bbc.com/",
    "https://nytimes.com/", "https://theguardian.com/",
    "https://cloudflare.com/", "https://dropbox.com/",
    "https://notion.so/", "https://figma.com/",
    # ── common phishing / typosquat targets ──────────────────────────────
    "http://g00gle.com/", "http://gooogle.com/", "http://googie.com/",
    "http://gogle.com/", "http://arnazon.com/", "http://amaz0n.com/",
    "http://paypa1.com/", "http://micros0ft.com/", "http://faceb00k.com/",
    "http://netfl1x.com/", "http://appleid-verify.com/",
    "http://secure-paypal-login.com/", "http://amazon-security-alert.com/",
    "http://login-microsoft-secure.com/", "http://bankofamerica-secure.com/",
]


def _safe_payload(url: str) -> dict:
    """Return a pre-built ALLOW payload for allowlisted domains."""
    from domain_checker import get_hostname
    return {
        "url":            url,
        "domain":         get_hostname(url),
        "rf_conf":        0.0,
        "rf_label":       "benign",
        "rf_threshold":   0.5,
        "xgb_conf":       0.0,
        "xgb_label":      "benign",
        "xgb_threshold":  0.5,
        "max_conf":       0.0,
        "ml_tier":        "ALLOW",
        "final_tier":     "ALLOW",
        "features":       [],
        "whois_eligible": False,
        "allowlisted":    True,
        "cached":         True,
    }


def _prewarm(urls: list[str]) -> None:
    """Run inference on each URL and populate the predict cache."""
    hits = 0
    for url in urls:
        try:
            if is_known_safe(url):
                _cache_set(_predict_cache, url, _safe_payload(url), PREDICT_TTL)
                hits += 1
                continue

            feat, rf_conf, xgb_conf = run_inference(
                url,
                state.rf_clf, state.rf_thr,
                state.xgb_clf, state.xgb_thr,
            )
            rf_label  = "malicious" if rf_conf  >= state.rf_thr  else "benign"
            xgb_label = "malicious" if xgb_conf >= state.xgb_thr else "benign"
            max_conf  = max(rf_conf, xgb_conf)
            ml_tier   = risk_tier(max_conf, get_tld(get_hostname(url)))
            avg_imp   = {f: float(state.rf_imps[f] + state.xgb_imps[f]) / 2 for f in FEATURE_COLS}
            features  = [
                {
                    "name":    f,
                    "value":   float(feat[f]),
                    "rf_imp":  round(float(state.rf_imps[f]), 4),
                    "xgb_imp": round(float(state.xgb_imps[f]), 4),
                    "avg_imp": round(avg_imp[f], 4),
                }
                for f in sorted(FEATURE_COLS, key=lambda f: avg_imp[f], reverse=True)
            ]
            payload = {
                "url":            url,
                "rf_conf":        round(rf_conf, 4),
                "rf_label":       rf_label,
                "rf_threshold":   state.rf_thr,
                "xgb_conf":       round(xgb_conf, 4),
                "xgb_label":      xgb_label,
                "xgb_threshold":  state.xgb_thr,
                "max_conf":       round(max_conf, 4),
                "ml_tier":        ml_tier,
                "final_tier":     ml_tier,
                "features":       features,
                "whois_eligible": max_conf >= WHOIS_BAND_LO,
                "cached":         True,
            }
            _cache_set(_predict_cache, url, payload, PREDICT_TTL)
            hits += 1
        except Exception:
            pass
    print(f"Cache pre-warm complete: {hits}/{len(urls)} URLs cached")


# ── app state ─────────────────────────────────────────────────────────────────

class State:
    rf_clf = None;  rf_thr:  float = 0.5
    xgb_clf = None; xgb_thr: float = 0.5
    rf_imps:  dict = {}
    xgb_imps: dict = {}

state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.rf_clf,  state.rf_thr  = load_model(RF_MODEL)
    state.xgb_clf, state.xgb_thr = load_model(XGB_MODEL)
    state.rf_imps  = dict(zip(FEATURE_COLS, state.rf_clf.feature_importances_))
    state.xgb_imps = dict(zip(FEATURE_COLS, state.xgb_clf.feature_importances_))
    print("Models loaded. Listening on http://127.0.0.1:8000")

    # Pre-warm cache in background — doesn't block the server from accepting requests
    threading.Thread(target=_prewarm, args=(PREWARM_URLS,), daemon=True).start()
    print(f"Pre-warming cache for {len(PREWARM_URLS)} URLs in background...")

    yield


app = FastAPI(title="Domain Threat Detector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    with _cache_lock:
        return {
            "status":         "ok",
            "predict_cached": len(_predict_cache),
            "whois_cached":   len(_whois_cache),
        }


@app.get("/predict")
async def predict(url: str = Query(...), domain: str = Query(None)):
    # Accept either ?url= (full URL) or legacy ?domain= (bare hostname)
    raw = url if url else (f"http://{domain}/" if domain else "")
    raw = raw.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Provide url or domain parameter")
    if "://" not in raw:
        raw = "http://" + raw
    host = get_hostname(raw)
    if not host or "." not in host or len(host) < 4:
        raise HTTPException(status_code=400, detail="Invalid URL or domain")

    cached = _cache_get(_predict_cache, raw)
    if cached:
        return cached

    if is_known_safe(raw):
        result = _safe_payload(raw)
        result["cached"] = False
        _cache_set(_predict_cache, raw, result, PREDICT_TTL)
        return result

    feat, rf_conf, xgb_conf = await run_in_threadpool(
        run_inference, raw,
        state.rf_clf, state.rf_thr, state.xgb_clf, state.xgb_thr,
    )

    rf_label  = "malicious" if rf_conf  >= state.rf_thr  else "benign"
    xgb_label = "malicious" if xgb_conf >= state.xgb_thr else "benign"
    max_conf  = max(rf_conf, xgb_conf)
    ml_tier   = risk_tier(max_conf, get_tld(host))

    avg_imp = {f: float(state.rf_imps[f] + state.xgb_imps[f]) / 2 for f in FEATURE_COLS}
    features = [
        {
            "name":    f,
            "value":   float(feat[f]),
            "rf_imp":  round(float(state.rf_imps[f]), 4),
            "xgb_imp": round(float(state.xgb_imps[f]), 4),
            "avg_imp": round(avg_imp[f], 4),
        }
        for f in sorted(FEATURE_COLS, key=lambda f: avg_imp[f], reverse=True)
    ]

    whois_eligible = max_conf >= WHOIS_BAND_LO

    result = {
        "url":             raw,
        "domain":          host,
        "rf_conf":         round(rf_conf, 4),
        "rf_label":        rf_label,
        "rf_threshold":    state.rf_thr,
        "xgb_conf":        round(xgb_conf, 4),
        "xgb_label":       xgb_label,
        "xgb_threshold":   state.xgb_thr,
        "max_conf":        round(max_conf, 4),
        "ml_tier":         ml_tier,
        "final_tier":      ml_tier,
        "features":        features,
        "whois_eligible":  whois_eligible,
        "cached":          False,
    }
    _cache_set(_predict_cache, raw, result, PREDICT_TTL)
    return result


@app.get("/whois")
async def whois_lookup(url: str = Query(...), domain: str = Query(None)):
    raw = url if url else (f"http://{domain}/" if domain else "")
    raw = raw.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Provide url or domain parameter")
    if "://" not in raw:
        raw = "http://" + raw
    host = get_hostname(raw)
    if not host or "." not in host or len(host) < 4:
        raise HTTPException(status_code=400, detail="Invalid URL or domain")

    cached = _cache_get(_whois_cache, raw)
    if cached:
        return cached

    feat, rf_conf, xgb_conf = await run_in_threadpool(
        run_inference, raw,
        state.rf_clf, state.rf_thr, state.xgb_clf, state.xgb_thr,
    )
    max_conf = max(rf_conf, xgb_conf)
    ml_tier  = risk_tier(max_conf, get_tld(host))

    wf = await run_in_threadpool(whois_features, raw)
    final_tier, reasons = adjust_tier(ml_tier, wf)

    result = {
        "url":                    raw,
        "domain":                 host,
        "available":              wf.get("available", False),
        "domain_age_days":        wf.get("domain_age_days"),
        "has_privacy_protection": wf.get("has_privacy_protection"),
        "registrar_known":        wf.get("registrar_known"),
        "expires_soon":           wf.get("expires_soon"),
        "original_tier":          ml_tier,
        "final_tier":             final_tier,
        "reasons":                reasons,
    }
    _cache_set(_whois_cache, raw, result, WHOIS_TTL)
    return result


@app.get("/cache")
def cache_status():
    """Lists all currently cached domains and their expiry times."""
    now = datetime.now(timezone.utc)
    with _cache_lock:
        predict = [
            {
                "domain":     k,
                "tier":       v.data.get("final_tier"),
                "max_conf":   v.data.get("max_conf"),
                "expires_in": str(v.expires - now).split(".")[0],
            }
            for k, v in _predict_cache.items()
        ]
        whois = [
            {
                "domain":     k,
                "expires_in": str(v.expires - now).split(".")[0],
            }
            for k, v in _whois_cache.items()
        ]
    return {
        "predict": sorted(predict, key=lambda x: x["domain"]),
        "whois":   sorted(whois,   key=lambda x: x["domain"]),
    }
