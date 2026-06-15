"""01 — Fetch Census demographics into a tidy state-year table.

Primary source: ACS 5-year API (needs CENSUS_API_KEY) — population by race/Hispanic
origin, poverty, median household income, unemployment, for each year 2015-2024.

Fallback (key-free): Population Estimates Program "nst-est" totals give total
population per state-year so the overall rate model can run even before the Census
key activates. Race-specific populations and socio-economic confounders require ACS.

Output: data/raw/census_state_year.csv  (tidy: one row per state-year)
"""
from __future__ import annotations

import io
import json

import pandas as pd
import requests

from common import (RAW, YEARS, STATE_FIPS, census_key, cached_get, section, log)

# ACS5 detailed-table variables (B03002 = Hispanic origin by race)
ACS_VARS = {
    "B03002_001E": "pop_total",
    "B03002_003E": "pop_white_nh",
    "B03002_004E": "pop_black_nh",
    "B03002_006E": "pop_asian_nh",
    "B03002_005E": "pop_native_nh",   # American Indian / Alaska Native alone, non-Hisp
    "B03002_012E": "pop_hispanic",
    "B17001_001E": "pov_denom",
    "B17001_002E": "pov_below",
    "B19013_001E": "median_income",
    "B23025_003E": "labor_force",
    "B23025_005E": "unemployed",
}


def fetch_acs_year(year: int, key: str) -> pd.DataFrame | None:
    var_list = ",".join(ACS_VARS)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": f"NAME,{var_list}", "for": "state:*", "key": key}
    # Cache the raw JSON keyed by year (key not included in cache name)
    body = cached_get(url, f"census_acs5_{year}.json", params=params)
    if not body:
        return None
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        log(f"  ! ACS {year}: non-JSON response (likely invalid key)")
        # Don't poison the cache with an error page
        (RAW / f"census_acs5_{year}.json").unlink(missing_ok=True)
        return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={**ACS_VARS, "state": "fips"})
    df["state"] = df["fips"].map(STATE_FIPS)
    df = df[df["state"].notna()].copy()
    num_cols = list(ACS_VARS.values())
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    # Census uses large negative sentinels for missing income
    df.loc[df["median_income"] < 0, "median_income"] = pd.NA
    df["poverty_rate"] = df["pov_below"] / df["pov_denom"]
    df["unemployment_rate"] = df["unemployed"] / df["labor_force"]
    df["year"] = year
    df["source"] = "ACS5"
    keep = ["state", "year", "pop_total", "pop_white_nh", "pop_black_nh",
            "pop_hispanic", "pop_asian_nh", "pop_native_nh", "poverty_rate",
            "median_income", "unemployment_rate", "source"]
    return df[keep]


# ---- key-free fallback: total population only ------------------------------
PEP_TOTALS = {
    "2010-2019": "https://www2.census.gov/programs-surveys/popest/datasets/"
                 "2010-2019/national/totals/nst-est2019-alldata.csv",
    "2020-2024": "https://www2.census.gov/programs-surveys/popest/datasets/"
                 "2020-2024/state/totals/NST-EST2024-ALLDATA.csv",
}


def fetch_pep_totals() -> pd.DataFrame | None:
    frames = []
    for tag, url in PEP_TOTALS.items():
        body = cached_get(url, f"pep_nst_{tag}.csv")
        if not body:
            continue
        d = pd.read_csv(io.StringIO(body))
        d = d[d["STATE"] != 0]  # drop national/region rollups (STATE fips 0)
        # USPS abbr via state name
        for y in YEARS:
            col = f"POPESTIMATE{y}"
            if col not in d.columns:
                continue
            sub = d[["NAME", col]].copy()
            sub["year"] = y
            sub = sub.rename(columns={col: "pop_total"})
            frames.append(sub)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    name_to_abbr = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
        "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
        "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
        "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
        "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    }
    out["state"] = out["NAME"].map(name_to_abbr)
    out = out[out["state"].notna()]
    for c in ["pop_white_nh", "pop_black_nh", "pop_hispanic", "pop_asian_nh",
              "pop_native_nh", "poverty_rate", "median_income", "unemployment_rate"]:
        out[c] = pd.NA
    out["source"] = "PEP_total_only"
    return out[["state", "year", "pop_total", "pop_white_nh", "pop_black_nh",
                "pop_hispanic", "pop_asian_nh", "pop_native_nh", "poverty_rate",
                "median_income", "unemployment_rate", "source"]]


def main():
    key = census_key()
    acs_frames, missing_years = [], []
    if key:
        for y in YEARS:
            df = fetch_acs_year(y, key)
            if df is not None:
                acs_frames.append(df)
                log(f"  ACS5 {y}: {len(df)} states")
            else:
                missing_years.append(y)
    else:
        log("  No CENSUS_API_KEY found; skipping ACS.")
        missing_years = list(YEARS)

    note_lines = []
    if acs_frames:
        result = pd.concat(acs_frames, ignore_index=True)
        note_lines.append(f"ACS5 demographics fetched for years: "
                          f"{sorted(set(result.year))}.")
        if missing_years:
            note_lines.append(f"ACS years unavailable (no endpoint yet / error): "
                              f"{missing_years} — back-filled with PEP totals.")
            pep = fetch_pep_totals()
            if pep is not None:
                pep = pep[pep["year"].isin(missing_years)]
                result = pd.concat([result, pep], ignore_index=True)
    else:
        note_lines.append("**ACS unavailable (Census key not active).** Falling back "
                          "to PEP total population only — the overall per-capita rate "
                          "model can run, but race-specific and socio-economic "
                          "confounders are pending the Census key.")
        result = fetch_pep_totals()

    if result is None or result.empty:
        section("2. Census demographics fetch",
                "**FAILED** — no Census data could be retrieved (no key and PEP "
                "fallback unreachable). Rate model is blocked.")
        return

    result = result.sort_values(["state", "year"]).reset_index(drop=True)
    result.to_csv(RAW / "census_state_year.csv", index=False)

    cov = result.groupby("source").size().to_dict()
    note_lines.append(f"\nRows by source: {cov}. "
                      f"States×years = {result['state'].nunique()}×"
                      f"{result['year'].nunique()}.")
    section("2. Census demographics fetch", "\n".join(note_lines))
    log(f"Wrote census_state_year.csv ({len(result)} rows).")


if __name__ == "__main__":
    main()
