"""
merge_datasets.py

Merges data/urls.csv (full URLs) and data/domains.csv (bare domains) into
a single data/combined.csv that extract_features.py can consume.

domains.csv rows are converted to http://domain/ format so the combined
file has a uniform `url` column.  Duplicate URLs are dropped, keeping the
first occurrence (urls.csv wins on collision so its labels are preserved).
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
URLS_CSV   = DATA_DIR / "urls.csv"
DOMAINS_CSV = DATA_DIR / "domains.csv"
OUTPUT_CSV = DATA_DIR / "combined.csv"


def normalize_url(raw: str) -> str | None:
    raw = raw.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if not host or "." not in host or " " in host:
            return None
        scheme = parsed.scheme.lower()
        path   = parsed.path if parsed.path else "/"
        port   = f":{parsed.port}" if parsed.port else ""
        query  = f"?{parsed.query}" if parsed.query else ""
        frag   = f"#{parsed.fragment}" if parsed.fragment else ""
        return f"{scheme}://{host}{port}{path}{query}{frag}"
    except Exception:
        return None


def main() -> None:
    print("Loading urls.csv ...")
    urls_df = pd.read_csv(URLS_CSV, dtype=str)
    urls_df["label"] = urls_df["label"].astype(int)
    print(f"  {len(urls_df):,} rows")

    print("Loading domains.csv ...")
    dom_df = pd.read_csv(DOMAINS_CSV, dtype=str)
    dom_df["label"] = dom_df["label"].astype(int)
    print(f"  {len(dom_df):,} rows")

    # Convert domain column → url column
    dom_df["url"] = dom_df["domain"].apply(
        lambda d: normalize_url(str(d)) or f"http://{d.strip()}/"
    )
    dom_df = dom_df[["url", "label", "source"]]

    # urls.csv first so it wins on dedup
    combined = pd.concat([urls_df[["url", "label", "source"]], dom_df], ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    print(f"\nDropped {before - len(combined):,} duplicate URLs")

    n_ben = int((combined["label"] == 0).sum())
    n_mal = int((combined["label"] == 1).sum())
    print(f"Combined: {n_ben:,} benign  |  {n_mal:,} malicious  |  {len(combined):,} total")

    combined.to_csv(OUTPUT_CSV, index=False, escapechar="\\")
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
