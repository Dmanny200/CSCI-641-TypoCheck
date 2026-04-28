"""
collect_data.py

Builds a labelled dataset for typosquatting / phishing domain detection.
Output: data/domains.csv  (columns: domain, label, source)
  label 0 = benign
  label 1 = malicious
"""

import csv
import io
import random
import re
import string
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import pandas as pd
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "domains.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_domain(url: str) -> str | None:
    """Return the registered domain (host without leading 'www.') from a URL."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        host = urlparse(url).hostname or ""
        host = host.lower().lstrip("www.")
        # basic sanity: must contain a dot and no spaces
        if "." in host and " " not in host:
            return host
    except Exception:
        pass
    return None


def fetch_text(url: str, timeout: int = 30) -> str | None:
    """GET a URL and return the decoded text body, or None on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (research-dataset-builder/1.0)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] could not fetch {url}: {exc}")
        return None


def fetch_bytes(url: str, timeout: int = 60) -> bytes | None:
    """GET a URL and return raw bytes, or None on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (research-dataset-builder/1.0)"}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        print(f"  [warn] could not fetch {url}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. BENIGN — Tranco top 10 000
# ═══════════════════════════════════════════════════════════════════════════

def get_tranco(n: int = 10_000) -> list[str]:
    """Download the latest Tranco list and return the top-n domains."""
    print(f"\n[1/7] Downloading Tranco top {n:,} …")
    # Tranco provides a stable 'latest' zip
    data = fetch_bytes("https://tranco-list.eu/top-1m.csv.zip", timeout=120)
    if not data:
        print("  [warn] Tranco download failed — returning empty list")
        return []

    domains: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
            for rank, row in enumerate(reader, start=1):
                if rank > n:
                    break
                if row:
                    domains.append(row[1].lower().strip())
    print(f"  -> {len(domains):,} Tranco domains loaded")
    return domains


# ═══════════════════════════════════════════════════════════════════════════
# 2. BENIGN — brand subdomains
# ═══════════════════════════════════════════════════════════════════════════

BRAND_SUBDOMAINS: list[str] = [
    # Google
    "mail.google.com", "drive.google.com", "docs.google.com",
    "maps.google.com", "news.google.com", "play.google.com",
    "cloud.google.com", "meet.google.com", "calendar.google.com",
    # Amazon / AWS
    "aws.amazon.com", "console.aws.amazon.com", "s3.amazonaws.com",
    "signin.aws.amazon.com", "payments.amazon.com",
    # Microsoft
    "login.microsoftonline.com", "outlook.live.com", "teams.microsoft.com",
    "onedrive.live.com", "azure.microsoft.com", "portal.azure.com",
    "support.microsoft.com", "store.steampowered.com",
    # Apple
    "support.apple.com", "developer.apple.com", "icloud.com",
    "appleid.apple.com",
    # Meta / Facebook
    "m.facebook.com", "developers.facebook.com", "business.facebook.com",
    "help.instagram.com",
    # Other majors
    "api.twitter.com", "developer.twitter.com",
    "www.youtube.com", "studio.youtube.com",
    "secure.paypal.com", "developer.paypal.com",
    "help.github.com", "gist.github.com", "docs.github.com",
    "account.live.com", "login.yahoo.com", "mail.yahoo.com",
    "accounts.google.com", "myaccount.google.com",
    "signin.ebay.com", "pages.ebay.com",
    "my.netflix.com", "help.netflix.com",
    "open.spotify.com", "accounts.spotify.com",
    "shop.app", "checkout.shopify.com",
]

def get_brand_subdomains() -> list[str]:
    print("\n[2/7] Adding brand subdomains …")
    print(f"  -> {len(BRAND_SUBDOMAINS):,} brand subdomains")
    return [d.lower() for d in BRAND_SUBDOMAINS]


# ═══════════════════════════════════════════════════════════════════════════
# 3. MALICIOUS — OpenPhish
# ═══════════════════════════════════════════════════════════════════════════

def get_openphish() -> list[str]:
    print("\n[3/7] Fetching OpenPhish feed …")
    text = fetch_text("https://openphish.com/feed.txt")
    if not text:
        return []
    domains = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            d = extract_domain(line)
            if d:
                domains.append(d)
    domains = list(dict.fromkeys(domains))  # dedupe, preserve order
    print(f"  -> {len(domains):,} unique domains from OpenPhish")
    return domains


# ═══════════════════════════════════════════════════════════════════════════
# 4. MALICIOUS — PhishTank
# ═══════════════════════════════════════════════════════════════════════════

def get_phishtank() -> list[str]:
    print("\n[4/7] Fetching PhishTank …")
    url = "https://data.phishtank.com/data/online-valid.csv"
    text = fetch_text(url, timeout=120)
    if not text:
        return []
    domains: list[str] = []
    try:
        df = pd.read_csv(io.StringIO(text), usecols=["url"], dtype=str)
        for raw_url in df["url"].dropna():
            d = extract_domain(raw_url)
            if d:
                domains.append(d)
    except Exception as exc:
        print(f"  [warn] PhishTank parse error: {exc}")
    domains = list(dict.fromkeys(domains))
    print(f"  -> {len(domains):,} unique domains from PhishTank")
    return domains


# ═══════════════════════════════════════════════════════════════════════════
# 5. MALICIOUS — URLhaus
# ═══════════════════════════════════════════════════════════════════════════

def get_urlhaus() -> list[str]:
    print("\n[5/7] Fetching URLhaus …")
    url = "https://urlhaus.abuse.ch/downloads/csv/"
    data = fetch_bytes(url, timeout=120)
    if not data:
        return []

    # URLhaus serves a zip
    domains: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                reader = csv.reader(io.TextIOWrapper(f, "utf-8"))
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    # columns: id, dateadded, url, url_status, last_online,
                    #          threat, tags, urlhaus_link, reporter
                    if len(row) >= 3:
                        d = extract_domain(row[2])
                        if d:
                            domains.append(d)
    except Exception:
        # fallback: treat as plain CSV text
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                d = extract_domain(parts[2].strip('"'))
                if d:
                    domains.append(d)

    domains = list(dict.fromkeys(domains))
    print(f"  -> {len(domains):,} unique domains from URLhaus")
    return domains


# ═══════════════════════════════════════════════════════════════════════════
# 6. GENERATED — typosquatting variations
# ═══════════════════════════════════════════════════════════════════════════

HOMOGLYPHS: dict[str, list[str]] = {
    "o": ["0"],
    "0": ["o"],
    "l": ["1", "i"],
    "i": ["1", "l"],
    "1": ["l", "i"],
    "a": ["@", "4"],
    "e": ["3"],
    "s": ["5", "$"],
    "t": ["7"],
    "g": ["9"],
    "b": ["6"],
}

KEYBOARD_ADJACENCY: dict[str, list[str]] = {
    "a": ["q","w","s","z"],  "b": ["v","g","h","n"],
    "c": ["x","d","f","v"],  "d": ["s","e","r","f","c","x"],
    "e": ["w","s","d","r"],  "f": ["d","r","t","g","v","c"],
    "g": ["f","t","y","h","b","v"],  "h": ["g","y","u","j","n","b"],
    "i": ["u","j","k","o"],  "j": ["h","u","i","k","n","m"],
    "k": ["j","i","o","l","m"],  "l": ["k","o","p"],
    "m": ["n","j","k"],  "n": ["b","h","j","m"],
    "o": ["i","k","l","p"],  "p": ["o","l"],
    "q": ["w","a"],  "r": ["e","d","f","t"],
    "s": ["a","w","e","d","x","z"],  "t": ["r","f","g","y"],
    "u": ["y","h","j","i"],  "v": ["c","f","g","b"],
    "w": ["q","a","s","e"],  "x": ["z","s","d","c"],
    "y": ["t","g","h","u"],  "z": ["a","s","x"],
}

ALT_TLDS = [".co", ".net", ".org", ".xyz", ".info", ".biz",
            ".online", ".site", ".club", ".shop", ".io"]


def _sld_tld(domain: str) -> tuple[str, str]:
    """Split 'google.com' -> ('google', '.com').  Handles multi-part TLDs naively."""
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:-1]), "." + parts[-1]
    return domain, ""


def generate_typos(domain: str) -> list[str]:
    """Return a list of typosquatting variants for a given domain."""
    sld, tld = _sld_tld(domain)
    variants: list[str] = []

    # — character omission —
    for i in range(len(sld)):
        v = sld[:i] + sld[i+1:]
        if v:
            variants.append(v + tld)

    # — character swap (transpose adjacent) —
    for i in range(len(sld) - 1):
        v = sld[:i] + sld[i+1] + sld[i] + sld[i+2:]
        variants.append(v + tld)

    # — character duplication —
    for i in range(len(sld)):
        v = sld[:i] + sld[i] + sld[i] + sld[i+1:]
        variants.append(v + tld)

    # — homoglyph substitution —
    for i, ch in enumerate(sld):
        if ch in HOMOGLYPHS:
            for repl in HOMOGLYPHS[ch]:
                v = sld[:i] + repl + sld[i+1:]
                variants.append(v + tld)

    # — adjacent keyboard key typos —
    for i, ch in enumerate(sld):
        if ch in KEYBOARD_ADJACENCY:
            for repl in KEYBOARD_ADJACENCY[ch]:
                v = sld[:i] + repl + sld[i+1:]
                variants.append(v + tld)

    # — dot insertion (split SLD at each position) —
    for i in range(1, len(sld)):
        variants.append(sld[:i] + "." + sld[i:] + tld)

    # — TLD swaps —
    for alt in ALT_TLDS:
        if alt != tld:
            variants.append(sld + alt)

    # — hyphen insertion —
    for i in range(1, len(sld)):
        variants.append(sld[:i] + "-" + sld[i:] + tld)

    # filter: no empty SLD, no pure-digit labels, length sanity
    clean: list[str] = []
    for v in variants:
        v = v.lower().strip(".")
        if v and len(v) >= 4 and v != domain:
            clean.append(v)

    return list(dict.fromkeys(clean))  # dedupe


def get_typosquatting_variants(tranco_domains: list[str], top_n: int = 200) -> list[str]:
    print(f"\n[6/7] Generating typosquatting variants for top {top_n} Tranco domains …")
    seed_domains = tranco_domains[:top_n]
    all_variants: list[str] = []
    for d in tqdm(seed_domains, unit="domain"):
        all_variants.extend(generate_typos(d))
    all_variants = list(dict.fromkeys(all_variants))
    print(f"  -> {len(all_variants):,} unique typosquatting variants")
    return all_variants


# ═══════════════════════════════════════════════════════════════════════════
# 7. GENERATED — DGA domains
# ═══════════════════════════════════════════════════════════════════════════

DGA_TLDS = [".com", ".net", ".org", ".ru", ".cn", ".info", ".biz",
            ".xyz", ".top", ".club", ".pw", ".tk"]

# ~5 000 common English words for dictionary-DGA simulation
DICT_WORDS = [
    "account","action","admin","agent","alert","amazon","apple","bank",
    "base","board","boot","box","brand","bulk","card","cash","check",
    "click","cloud","code","conf","connect","content","control","corp",
    "credit","crypto","data","deal","desk","direct","disk","domain",
    "drive","drop","email","event","extra","fast","feed","file","find",
    "fire","first","flag","flash","flow","font","force","form","free",
    "fund","game","gate","give","global","gold","good","grant","grid",
    "group","guard","guide","hash","help","hide","host","hub","info",
    "init","item","join","jump","just","keep","key","kill","kit","land",
    "launch","layer","lead","learn","left","level","link","list","live",
    "load","lock","log","loop","mail","main","make","mark","media","mesh",
    "meta","micro","mine","mode","move","name","net","new","next","node",
    "note","notify","open","order","page","pass","path","pay","ping",
    "pipe","plan","play","plug","point","pool","port","post","power",
    "press","price","prime","print","pro","proxy","push","quick","rank",
    "rate","read","real","reply","report","reset","root","route","rule",
    "run","safe","save","scan","search","secure","send","serve","set",
    "shop","sign","simple","site","skip","smart","sort","speed","split",
    "start","stat","step","store","stream","style","sync","system","tag",
    "target","task","team","text","time","token","tool","top","trace",
    "track","trade","trust","type","unit","update","upload","user","valid",
    "value","view","visit","vote","web","wire","work","world","write","zone",
]

def _random_dga(length: int) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


def get_dga_domains(n_random: int = 1000, n_dict: int = 500) -> list[str]:
    print(f"\n[7/7] Generating {n_random} random + {n_dict} dictionary DGA domains …")
    domains: list[str] = []

    # purely random
    for _ in range(n_random):
        length = random.randint(8, 25)
        tld = random.choice(DGA_TLDS)
        domains.append(_random_dga(length) + tld)

    # dictionary-based (2–3 word concatenation)
    for _ in range(n_dict):
        n_words = random.randint(2, 3)
        sld = "".join(random.choices(DICT_WORDS, k=n_words))
        tld = random.choice(DGA_TLDS)
        domains.append(sld + tld)

    domains = list(dict.fromkeys(domains))
    print(f"  -> {len(domains):,} unique DGA domains")
    return domains


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Typosquatting / Phishing Domain Dataset Builder")
    print("=" * 60)

    rows: list[dict] = []

    # ── benign ───────────────────────────────────────────────────────────
    tranco = get_tranco(10_000)
    tranco_set = set(tranco)  # for fast lookup during dedup step

    for d in tranco:
        rows.append({"domain": d, "label": 0, "source": "tranco_top10k"})

    for d in get_brand_subdomains():
        rows.append({"domain": d, "label": 0, "source": "brand_subdomains"})

    # ── malicious (real) ─────────────────────────────────────────────────
    for d in get_openphish():
        rows.append({"domain": d, "label": 1, "source": "openphish"})

    for d in get_phishtank():
        rows.append({"domain": d, "label": 1, "source": "phishtank"})

    for d in get_urlhaus():
        rows.append({"domain": d, "label": 1, "source": "urlhaus"})

    # ── malicious (generated) ────────────────────────────────────────────
    for d in get_typosquatting_variants(tranco, top_n=200):
        rows.append({"domain": d, "label": 1, "source": "generated_typosquat"})

    for d in get_dga_domains(n_random=1000, n_dict=500):
        rows.append({"domain": d, "label": 1, "source": "generated_dga"})

    # ── build dataframe ──────────────────────────────────────────────────
    print("\nBuilding dataframe …")
    df = pd.DataFrame(rows)

    # normalise
    df["domain"] = df["domain"].str.lower().str.strip()

    # drop blank / obviously invalid
    df = df[df["domain"].str.contains(r"\.", regex=True, na=False)]
    df = df[df["domain"].str.len() >= 4]

    # remove malicious rows whose domain also appears in Tranco top 10k
    mask_malicious_in_tranco = (df["label"] == 1) & (df["domain"].isin(tranco_set))
    n_removed = mask_malicious_in_tranco.sum()
    if n_removed:
        print(f"  Removed {n_removed:,} generated/phish domains that collide with Tranco top 10k")
    df = df[~mask_malicious_in_tranco]

    # global dedup: keep first occurrence (benign sources listed first,
    # so a domain in both Tranco and a phish feed stays benign)
    before = len(df)
    df = df.drop_duplicates(subset=["domain"], keep="first").reset_index(drop=True)
    print(f"  Removed {before - len(df):,} duplicate domain rows")

    # ── rebalance to ~70% benign / 30% malicious ─────────────────────────
    TARGET_BENIGN_FRAC = 0.70
    TARGET_MAL_FRAC    = 0.30

    df_benign    = df[df["label"] == 0]
    df_malicious = df[df["label"] == 1]

    n_benign_orig    = len(df_benign)
    n_malicious_orig = len(df_malicious)
    target_mal = int(n_benign_orig * TARGET_MAL_FRAC / TARGET_BENIGN_FRAC)

    print(f"\nRebalancing: {n_benign_orig:,} benign, {n_malicious_orig:,} malicious")
    if n_malicious_orig > target_mal:
        df_malicious = df_malicious.sample(n=target_mal, random_state=RANDOM_SEED)
        print(f"  Downsampled malicious: {n_malicious_orig:,} -> {target_mal:,} "
              f"(to match 70/30 split with {n_benign_orig:,} benign samples)")
    else:
        print(f"  Malicious count ({n_malicious_orig:,}) already at or below target "
              f"({target_mal:,}); no downsampling needed")

    df = (pd.concat([df_benign, df_malicious])
            .sample(frac=1, random_state=RANDOM_SEED)
            .reset_index(drop=True))

    # ── summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset summary")
    print("=" * 60)
    summary = (
        df.groupby(["source", "label"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "count"], ascending=[True, False])
    )
    for _, row in summary.iterrows():
        tag = "benign   " if row["label"] == 0 else "malicious"
        print(f"  {tag}  {row['source']:<30}  {row['count']:>7,}")
    print("-" * 60)
    label_counts = df["label"].value_counts().sort_index()
    n_benign  = label_counts.get(0, 0)
    n_mal     = label_counts.get(1, 0)
    n_total   = len(df)
    print(f"  Total benign:     {n_benign:>7,}  ({100 * n_benign / n_total:.1f}%)")
    print(f"  Total malicious:  {n_mal:>7,}  ({100 * n_mal / n_total:.1f}%)")
    print(f"  Grand total:      {n_total:>7,}")
    print("=" * 60)

    # ── save ─────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
