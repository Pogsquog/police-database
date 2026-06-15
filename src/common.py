"""Shared utilities: paths, env/key loading, HTTP caching, logging to findings.md."""
from __future__ import annotations

import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"
FINDINGS = ROOT / "findings.md"

for _d in (RAW, PROCESSED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Study window
YEARS = list(range(2015, 2025))  # 2015..2024 inclusive

# Census state FIPS -> USPS abbreviation (50 states + DC). Excludes territories.
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
ABBR_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}

# Census region (used for North-vs-South and regional controls).
# South = Census "South" region (incl. DC, TX, OK, the old Confederacy + border).
CENSUS_REGION = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South",
    "SC": "South", "VA": "South", "DC": "South", "WV": "South", "AL": "South",
    "KY": "South", "MS": "South", "TN": "South", "AR": "South", "LA": "South",
    "OK": "South", "TX": "South",
    # West
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "UT": "West", "WY": "West", "AK": "West", "CA": "West",
    "HI": "West", "OR": "West", "WA": "West",
}


# ---------------------------------------------------------------------------
# Environment / API keys (loaded from .env without extra deps)
# ---------------------------------------------------------------------------
def load_dotenv(path: Path | None = None) -> None:
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


load_dotenv()


def census_key() -> str | None:
    return os.environ.get("CENSUS_API_KEY") or None


def data_gov_key() -> str:
    return os.environ.get("DATA_GOV_API_KEY") or "DEMO_KEY"


# ---------------------------------------------------------------------------
# Cached HTTP GET — pulls land in data/raw and are reused on re-runs (offline)
# ---------------------------------------------------------------------------
def cached_get(url: str, cache_name: str, *, params: dict | None = None,
               force: bool = False, timeout: int = 90,
               binary: bool = False) -> bytes | str | None:
    """GET `url`, caching the body to data/raw/<cache_name>.

    Returns the body (str or bytes) or None on failure. Never raises on network
    errors so the pipeline degrades gracefully and logs the gap.
    """
    dest = RAW / cache_name
    if dest.exists() and not force:
        return dest.read_bytes() if binary else dest.read_text()

    for attempt in range(1, 4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                if binary:
                    dest.write_bytes(r.content)
                    return r.content
                dest.write_text(r.text)
                return r.text
            log(f"  ! HTTP {r.status_code} for {cache_name} ({url[:80]}...)")
            if r.status_code in (403, 404):
                return None
        except requests.RequestException as e:
            log(f"  ! request error ({attempt}/3) for {cache_name}: {e}")
        time.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------------------
# findings.md activity log
# ---------------------------------------------------------------------------
def log(msg: str, *, console: bool = True) -> None:
    if console:
        print(msg, file=sys.stderr)


def section(title: str, body: str = "", *, reset: bool = False) -> None:
    """Append a section to findings.md (activity log + results)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"\n\n## {title}\n_logged {ts}_\n\n"
    mode = "w" if reset else "a"
    with open(FINDINGS, mode) as f:
        if reset:
            f.write("# Findings — Modeling Factors Influencing US Police Shootings\n")
            f.write("\nA reproducible analysis of the Washington Post Fatal Police "
                    "Shootings dataset (2015–2024). This file is the running activity "
                    "log and results record, generated by the scripts in `src/`.\n")
        f.write(header)
        f.write(body.rstrip() + "\n")
