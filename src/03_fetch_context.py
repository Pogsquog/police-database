"""03 — Fetch contextual confounders into tidy state(-year) tables.

  * Gun availability  : NICS firearm background checks per state-month (BuzzFeedNews
                        mirror of FBI NICS) -> annual checks per state-year.
  * Alcohol use       : CDC Chronic Disease Indicators — "Binge drinking prevalence
                        among adults" (BRFSS), state-year.
  * Mental health     : CDC CDI — "Frequent mental distress among adults", state-year.
  * Urbanisation      : population density = pop / land area; land area (sq mi) summed
                        from the Census Gazetteer county file.
  * Region / South    : static Census region map (see common.py).

Outputs:
  data/raw/nics_state_year.csv, data/raw/cdi_state_year.csv,
  data/raw/state_land_area.csv
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from common import RAW, STATE_FIPS, cached_get, section, log

ABBRS = set(STATE_FIPS.values())

# ---------------------------------------------------------------------------
NICS_URL = ("https://raw.githubusercontent.com/BuzzFeedNews/"
            "nics-firearm-background-checks/master/data/"
            "nics-firearm-background-checks.csv")

_NAME_TO_ABBR = None


def name_to_abbr():
    global _NAME_TO_ABBR
    if _NAME_TO_ABBR is None:
        import json
        # reuse the mapping embedded in 01 via a tiny inline table
        names = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
            "California": "CA", "Colorado": "CO", "Connecticut": "CT",
            "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
            "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
            "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
            "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
            "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
            "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
            "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
            "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
            "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
            "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
            "Wisconsin": "WI", "Wyoming": "WY",
        }
        _NAME_TO_ABBR = names
    return _NAME_TO_ABBR


def fetch_nics() -> pd.DataFrame | None:
    body = cached_get(NICS_URL, "nics_raw.csv")
    if not body:
        return None
    d = pd.read_csv(io.StringIO(body))
    d["year"] = d["month"].str.slice(0, 4).astype(int)
    d["state"] = d["state"].map(name_to_abbr())
    d = d[d["state"].isin(ABBRS) & d["year"].between(2015, 2024)]
    out = (d.groupby(["state", "year"])["totals"].sum()
             .reset_index().rename(columns={"totals": "nics_checks"}))
    out.to_csv(RAW / "nics_state_year.csv", index=False)
    return out


# ---------------------------------------------------------------------------
CDI_URL = "https://data.cdc.gov/resource/hksd-2xuw.json"
CDI_QUESTIONS = {
    "Binge drinking prevalence among adults": "alcohol_binge_pct",
    "Frequent mental distress among adults": "mental_distress_pct",
}


def fetch_cdi() -> pd.DataFrame | None:
    frames = []
    for question, colname in CDI_QUESTIONS.items():
        params = {
            "question": question,
            "stratificationcategory1": "Overall",
            "datavaluetypeid": "CRDPREV",
            "$limit": "50000",
            "$select": "yearstart,locationabbr,datavalue",
        }
        body = cached_get(CDI_URL, f"cdi_{colname}.json", params=params)
        if not body:
            continue
        d = pd.read_json(io.StringIO(body))
        if d.empty:
            continue
        d = d.rename(columns={"yearstart": "year", "locationabbr": "state",
                              "datavalue": colname})
        d = d[d["state"].isin(ABBRS)][["state", "year", colname]]
        d[colname] = pd.to_numeric(d[colname], errors="coerce")
        d = d.dropna().groupby(["state", "year"], as_index=False)[colname].mean()
        frames.append(d.set_index(["state", "year"]))
    if not frames:
        return None
    out = pd.concat(frames, axis=1).reset_index()
    out.to_csv(RAW / "cdi_state_year.csv", index=False)
    return out


# ---------------------------------------------------------------------------
GAZ_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_Gaz_counties_national.zip")


def fetch_land_area() -> pd.DataFrame | None:
    raw = cached_get(GAZ_URL, "gaz_counties.zip", binary=True)
    if not raw:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fn = [n for n in z.namelist() if n.endswith(".txt")][0]
        txt = z.read(fn).decode("latin-1")
    d = pd.read_csv(io.StringIO(txt), sep="\t")
    d.columns = [c.strip() for c in d.columns]
    out = (d.groupby("USPS")["ALAND_SQMI"].sum()
             .reset_index().rename(columns={"USPS": "state",
                                            "ALAND_SQMI": "land_area_sqmi"}))
    out = out[out["state"].isin(ABBRS)]
    out.to_csv(RAW / "state_land_area.csv", index=False)
    return out


def main():
    notes = []

    nics = fetch_nics()
    if nics is not None:
        notes.append(f"NICS background checks: {len(nics)} state-years "
                     f"({nics['year'].min()}-{nics['year'].max()}).")
    else:
        notes.append("NICS: **unavailable**.")

    cdi = fetch_cdi()
    if cdi is not None:
        cols = [c for c in cdi.columns if c not in ("state", "year")]
        yr = f"{cdi['year'].min()}-{cdi['year'].max()}"
        notes.append(f"CDC CDI: {cols} across {yr} "
                     f"({cdi['state'].nunique()} states; BRFSS, not every year per "
                     f"state — gaps filled with state means at panel build).")
    else:
        notes.append("CDC CDI alcohol/mental-health: **unavailable**.")

    land = fetch_land_area()
    if land is not None:
        notes.append(f"Land area (Gazetteer): {len(land)} states — basis for "
                     f"population density (rural/urban proxy).")
    else:
        notes.append("Land area: **unavailable**.")

    section("4. Contextual confounders fetch (guns, alcohol, mental health, density)",
            "\n".join(f"- {n}" for n in notes))
    log("Context fetch complete.")


if __name__ == "__main__":
    main()
