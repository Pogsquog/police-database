"""02b — FBI arrests by race, per state-year (encounter/exposure proxy).

WaPo is numerator-only and the per-resident rate cannot separate "more shootings"
from "more police contact". Total arrests by race are the standard *exposure
benchmark*: shootings per arrest, rather than per resident, partials out
differences in how often each group is arrested/encountered.

Source: FBI Crime Data Explorer arrest endpoint, `type=totals`, which returns an
`Arrestee Race` breakdown. One call per state per year (cached). FBI race
categories carry no Hispanic ethnicity, so this benchmark is **Black vs White
only**. Caveat (documented downstream): arrests are themselves a product of
policing, so this benchmark can absorb upstream disparities — it answers a
narrower question than the per-resident rate, not a "truer" one.
"""
from __future__ import annotations

import json

import pandas as pd

from common import STATE_FIPS, YEARS, RAW, data_gov_key, cached_get, section, log

BASE = "https://api.usa.gov/crime/fbi/cde/arrest/state/{abbr}/all"
# FBI Arrestee Race label -> our group
RACE_MAP = {"White": "White", "Black or African American": "Black"}


def fetch_state_year(abbr: str, year: int, key: str) -> dict | None:
    url = BASE.format(abbr=abbr)
    params = {"from": f"01-{year}", "to": f"12-{year}", "type": "totals",
              "API_KEY": key}
    body = cached_get(url, f"fbi_arrests_{abbr}_{year}.json", params=params)
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def main():
    key = data_gov_key()
    rows = []
    for abbr in STATE_FIPS.values():
        for year in YEARS:
            d = fetch_state_year(abbr, year, key)
            race = (d or {}).get("Arrestee Race", {})
            if not race:
                log(f"  ! no arrest race data for {abbr} {year}")
                continue
            for fbi_lbl, grp in RACE_MAP.items():
                n = race.get(fbi_lbl)
                if n is None:
                    continue
                rows.append({"state": abbr, "year": year, "race": grp,
                             "arrests": float(n)})
        log(f"  arrests fetched: {abbr}")

    df = pd.DataFrame(rows)
    out = RAW / "fbi_arrests_state_year.csv"
    df.to_csv(out, index=False)

    wide = df.pivot_table(index=["state", "year"], columns="race",
                          values="arrests").reset_index()
    nat = df.groupby("race")["arrests"].sum()
    body = (
        f"FBI arrest totals by race fetched for {df['state'].nunique()} states × "
        f"{df['year'].nunique()} years ({len(wide)} state-years; "
        f"{len(df)} state-year-race rows).\n\n"
        f"National totals 2015–2024 — White: {nat.get('White',0):,.0f}, "
        f"Black: {nat.get('Black',0):,.0f} "
        f"(Black = {nat.get('Black',0)/(nat.get('Black',0)+nat.get('White',1)):.1%} "
        f"of Black+White arrests).\n\n"
        f"Used as the **arrest-exposure denominator** in the disparity benchmark "
        f"(script 05). FBI arrest race has no Hispanic ethnicity ⇒ Black-vs-White "
        f"only. Same NIBRS-coverage caveats as violent crime apply."
    )
    section("3b. FBI arrests-by-race fetch (encounter proxy)", body)
    log(f"Wrote {out} ({len(df)} rows).")


if __name__ == "__main__":
    main()
