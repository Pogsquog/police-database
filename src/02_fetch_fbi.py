"""02 — Fetch FBI Crime Data Explorer violent-crime rates into a state-year table.

Uses the CDE "summarized" endpoint (one call per state covers 2015-2024). Annual
violent-crime rate = sum of monthly offense counts / that year's population * 100k.

Output: data/raw/fbi_state_year.csv  (state, year, violent_crime_rate, vc_count)
"""
from __future__ import annotations

import json

import pandas as pd

from common import (RAW, YEARS, STATE_FIPS, data_gov_key, cached_get, section, log)

BASE = "https://api.usa.gov/crime/fbi/cde/summarized/state/{abbr}/violent-crime"


def fetch_state(abbr: str, key: str) -> pd.DataFrame | None:
    url = BASE.format(abbr=abbr)
    params = {"from": "01-2015", "to": "12-2024", "API_KEY": key}
    body = cached_get(url, f"fbi_vc_{abbr}.json", params=params)
    if not body:
        return None
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        (RAW / f"fbi_vc_{abbr}.json").unlink(missing_ok=True)
        return None

    name = state_name(abbr)
    off_key = f"{name} Offenses"
    actuals = d.get("offenses", {}).get("actuals", {}).get(off_key, {})
    pop_block = d.get("populations", {}).get("population", {}).get(name, {})
    # participated_population reflects agencies that actually reported — the correct
    # denominator given the 2021 NIBRS-transition coverage gap.
    part_block = d.get("populations", {}).get("participated_population", {}).get(name, {})
    if not actuals:
        return None

    rows = []
    for y in YEARS:
        months = [f"{m:02d}-{y}" for m in range(1, 13)]
        counts = [actuals.get(mm) for mm in months if actuals.get(mm) is not None]
        if not counts:
            continue
        total = sum(counts)
        full_pop = next((pop_block.get(mm) for mm in months if pop_block.get(mm)), None)
        # average participated population across reporting months
        parts = [part_block.get(mm) for mm in months if part_block.get(mm)]
        part_pop = sum(parts) / len(parts) if parts else None
        denom = part_pop or full_pop
        rate = (total / denom * 100_000) if denom else None
        coverage = (part_pop / full_pop) if (part_pop and full_pop) else None
        rows.append({"state": abbr, "year": y, "vc_count": total,
                     "violent_crime_rate": rate,
                     "crime_coverage": coverage})
    return pd.DataFrame(rows)


# CDE labels offenses by full state name
_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def state_name(abbr: str) -> str:
    return _STATE_NAMES[abbr]


def main():
    key = data_gov_key()
    frames, missing = [], []
    for abbr in STATE_FIPS.values():
        df = fetch_state(abbr, key)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            missing.append(abbr)

    if not frames:
        section("3. FBI violent-crime fetch",
                "**FAILED** — no FBI CDE data retrieved.")
        return

    result = pd.concat(frames, ignore_index=True).sort_values(["state", "year"])
    result.to_csv(RAW / "fbi_state_year.csv", index=False)

    note = [f"FBI CDE violent-crime rates fetched for {result['state'].nunique()} "
            f"states, {result['year'].nunique()} years ({len(result)} rows)."]
    if missing:
        note.append(f"States with no/partial data: {missing}.")
    note.append(f"National mean violent-crime rate across panel: "
                f"{result['violent_crime_rate'].mean():.0f} per 100k.")
    section("3. FBI violent-crime fetch", "\n".join(note))
    log(f"Wrote fbi_state_year.csv ({len(result)} rows). Missing: {missing}")


if __name__ == "__main__":
    main()
