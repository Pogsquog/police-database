"""02d — State accident-mortality proxies for the "distance to medical care"
hypothesis (rural shootings are likelier to be *fatal* because trauma care is far).

The rural-paradox section (05b) flagged a fatal-only artifact: WaPo records only
fatal shootings, and a gunshot is likelier to prove fatal where a Level-I trauma
centre is an hour away. We can't measure shooting case-fatality directly (no
non-fatal denominator), but we can test the *mechanism* with other injuries whose
lethality is governed by the same EMS/trauma-access factor.

The test is a discriminant (placebo) design:
  * Motor-vehicle and other unintentional-injury deaths (falls, etc.) are
    **trauma-access-sensitive** — fast EMS and a nearby trauma centre change who
    survives. If the medical-access channel is real, these should track the police
    shooting rate across states.
  * Drug-overdose deaths are a **negative control**: their lethality is about
    addiction and response time to *naloxone*, not trauma-centre distance, and they
    are concentrated in (often denser) Appalachian/urban areas. They should NOT track
    the shooting rate if the channel is specifically trauma access rather than a
    generic "deadly/poor places" story.

Sources (all CDC, state level, free Socrata APIs — WONDER's API is national-only):
  * Unintentional-injury (all accidents) deaths & age-adjusted rate, 2015-2017:
    NCHS Leading Causes of Death, `bi63-dtpu`.
  * Motor-vehicle occupant death rate (per 100k), 2012 & 2014 averaged: `rqg5-mkef`
    (a stable cross-sectional state proxy; MV state rankings move little year to year).
  * Drug-poisoning (overdose) death rate, 2015-2017: NCHS Drug Poisoning Mortality by
    State, `44rk-q6r2`.
Caveat carried downstream: per-capita accident deaths blend *more accidents*
(exposure) with *accidents being more lethal* (case-fatality) — so this is a
consistent-with test, not a clean isolation of medical access.
"""
from __future__ import annotations

import pandas as pd

from common import RAW, section, log

SOCRATA = "https://data.cdc.gov/resource"
YEARS = ("2015", "2016", "2017")

NAME_TO_ABBR = {
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


def _grab(resource: str, params: dict) -> pd.DataFrame:
    import requests
    r = requests.get(f"{SOCRATA}/{resource}.json",
                     params={**params, "$limit": 5000}, timeout=120)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def _fetch() -> pd.DataFrame:
    yrs = "','".join(YEARS)
    # all accidents (unintentional injuries): deaths + age-adjusted rate
    ui = _grab("bi63-dtpu", {"cause_name": "Unintentional injuries",
                             "$where": f"year in('{yrs}')"})
    ui["deaths"] = ui["deaths"].astype(float); ui["aadr"] = ui["aadr"].astype(float)
    ui = ui.groupby("state").agg(accident_deaths=("deaths", "sum"),
                                 accident_aadr=("aadr", "mean")).reset_index()

    # motor-vehicle occupant death rate (2012 & 2014 averaged)
    mv = _grab("rqg5-mkef", {})
    mv["state"] = mv["state_not_geocoded"]
    mv["mv_rate"] = (mv["all_ages"].astype(float)
                     + mv["all_ages_2014"].astype(float)) / 2
    mv = mv[["state", "mv_rate"]]

    # drug-overdose death rate (negative control)
    od = _grab("44rk-q6r2", {"sex": "Both Sexes", "age": "All Ages",
                             "race": "All Races-All Origins",
                             "$where": f"year in('{yrs}')"})
    od["rate"] = od["rate"].astype(float)
    od = od.groupby("state")["rate"].mean().reset_index().rename(
        columns={"rate": "overdose_rate"})

    d = ui.merge(mv, on="state", how="outer").merge(od, on="state", how="outer")
    d["state_abbr"] = d["state"].map(NAME_TO_ABBR)
    d = d[d["state_abbr"].notna()].copy()
    return d[["state_abbr", "accident_deaths", "accident_aadr",
              "mv_rate", "overdose_rate"]].rename(columns={"state_abbr": "state"})


def main():
    out = RAW / "cdc_injury_state.csv"
    if out.exists():
        d = pd.read_csv(out)
    else:
        try:
            d = _fetch()
        except Exception as e:  # degrade gracefully like the other fetch steps
            log(f"  ! CDC injury fetch failed ({e}); medical-access test skipped")
            section("3d. CDC accident-mortality fetch (medical-access proxy)",
                    "CDC Socrata request failed; the medical-access proxy is "
                    "unavailable this run.")
            return
        d.to_csv(out, index=False)

    body = (
        "State accident-mortality proxies for the 'distance to medical care' test "
        "(see 05b §7f). Per-capita accident deaths whose lethality depends on fast "
        "trauma care (motor vehicle, falls) vs a negative control whose lethality does "
        "not (drug overdose).\n\n"
        f"| Source | Measure | States |\n|---|---|---|\n"
        f"| NCHS Leading Causes `bi63-dtpu` | Unintentional-injury deaths & age-adj "
        f"rate, 2015–2017 | {d['accident_aadr'].notna().sum()} |\n"
        f"| NCHS `rqg5-mkef` | Motor-vehicle occupant death rate /100k (2012 & 2014) | "
        f"{d['mv_rate'].notna().sum()} |\n"
        f"| NCHS Drug Poisoning by State `44rk-q6r2` | Overdose death rate /100k, "
        f"2015–2017 | {d['overdose_rate'].notna().sum()} |\n\n"
        "WONDER's API serves national data only, so these state-level series come from "
        "CDC's Socrata endpoints. Analysed in script 05b."
    )
    section("3d. CDC accident-mortality fetch (medical-access proxy)", body)
    log(f"Wrote {out}: {len(d)} states "
        f"(MV mean {d['mv_rate'].mean():.1f}, overdose mean {d['overdose_rate'].mean():.1f}).")


if __name__ == "__main__":
    main()
