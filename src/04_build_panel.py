"""04 — Assemble the analysis datasets.

Spine = 51 states (50 + DC) x 10 years (2015-2024) = 510 rows, so state-years with
zero shootings are represented. Joins denominators (ACS), violent crime (FBI),
gun/alcohol/mental-health/density confounders, and region.

Outputs:
  data/processed/state_year_panel.parquet   (overall rate model)
  data/processed/race_panel.parquet         (long by race group — disparity model)
  data/processed/state_cross_section.parquet (10-yr aggregate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (RAW, PROCESSED, YEARS, STATE_FIPS, CENSUS_REGION, section, log)

RACE_GROUPS = {  # race_label -> (count alias, population column)
    "Black": "pop_black_nh",
    "White": "pop_white_nh",
    "Hispanic": "pop_hispanic",
}


def _spine() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([sorted(STATE_FIPS.values()), YEARS],
                                     names=["state", "year"])
    return pd.DataFrame(index=idx).reset_index()


def _gap_fill(df, cols, by="state"):
    """Fill missing covariate cells: within-state mean, then national mean. Flag."""
    flag = df[cols].isna().any(axis=1)
    for c in cols:
        df[c] = df.groupby(by)[c].transform(lambda s: s.fillna(s.mean()))
        df[c] = df[c].fillna(df[c].mean())
    return df, flag


def main():
    inc = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    inc = inc[inc["in_panel_state"] & inc["year"].notna()].copy()
    inc["year"] = inc["year"].astype(int)

    panel = _spine()

    # --- numerator: total shootings + by-race counts ----------------------
    tot = inc.groupby(["state", "year"]).size().rename("shootings").reset_index()
    panel = panel.merge(tot, on=["state", "year"], how="left")
    panel["shootings"] = panel["shootings"].fillna(0).astype(int)

    for grp in RACE_GROUPS:
        c = inc[inc["race_label"] == grp].groupby(["state", "year"]).size()
        panel = panel.merge(c.rename(f"shootings_{grp.lower()}").reset_index(),
                            on=["state", "year"], how="left")
        panel[f"shootings_{grp.lower()}"] = (
            panel[f"shootings_{grp.lower()}"].fillna(0).astype(int))

    # --- denominators + socio-economic (ACS) ------------------------------
    census = pd.read_csv(RAW / "census_state_year.csv")
    panel = panel.merge(census, on=["state", "year"], how="left")

    # --- violent crime (FBI) ----------------------------------------------
    fbi = pd.read_csv(RAW / "fbi_state_year.csv")
    panel = panel.merge(fbi, on=["state", "year"], how="left")
    # null out violent-crime rate where agency coverage is too low to trust
    panel["crime_reliable"] = panel["crime_coverage"] >= 0.80
    panel.loc[~panel["crime_reliable"], "violent_crime_rate"] = np.nan

    # --- contextual confounders -------------------------------------------
    nics = pd.read_csv(RAW / "nics_state_year.csv")
    panel = panel.merge(nics, on=["state", "year"], how="left")
    # NICS mirror lacks 2024: carry forward last available year within state
    panel = panel.sort_values(["state", "year"])
    panel["nics_checks"] = panel.groupby("state")["nics_checks"].ffill()

    cdi = pd.read_csv(RAW / "cdi_state_year.csv")
    panel = panel.merge(cdi, on=["state", "year"], how="left")

    land = pd.read_csv(RAW / "state_land_area.csv")
    panel = panel.merge(land, on="state", how="left")

    # --- derived features --------------------------------------------------
    panel["region"] = panel["state"].map(CENSUS_REGION)
    panel["is_south"] = (panel["region"] == "South").astype(int)
    panel["shootings_per_100k"] = panel["shootings"] / panel["pop_total"] * 1e5
    panel["pct_black"] = panel["pop_black_nh"] / panel["pop_total"]
    panel["pct_hispanic"] = panel["pop_hispanic"] / panel["pop_total"]
    panel["pct_white_nh"] = panel["pop_white_nh"] / panel["pop_total"]
    panel["pop_density"] = panel["pop_total"] / panel["land_area_sqmi"]
    panel["log_density"] = np.log(panel["pop_density"])
    panel["nics_per_1k"] = panel["nics_checks"] / panel["pop_total"] * 1e3
    panel["median_income_k"] = panel["median_income"] / 1000.0

    # gap-fill slow-moving confounders (CrimeRate handled via reliability flag)
    panel, alc_flag = _gap_fill(panel, ["alcohol_binge_pct", "mental_distress_pct"])
    panel["context_imputed"] = alc_flag.values

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)
    panel.to_parquet(PROCESSED / "state_year_panel.parquet")

    # --- long race panel for the disparity model --------------------------
    rows = []
    for grp, popcol in RACE_GROUPS.items():
        sub = panel[["state", "year", "is_south", "violent_crime_rate",
                     "poverty_rate", "median_income_k", "log_density",
                     "nics_per_1k", "alcohol_binge_pct", "mental_distress_pct"]].copy()
        sub["race"] = grp
        sub["count"] = panel[f"shootings_{grp.lower()}"].values
        sub["group_pop"] = panel[popcol].values
        rows.append(sub)
    race_panel = pd.concat(rows, ignore_index=True)
    race_panel = race_panel[race_panel["group_pop"] > 0]

    # arrests (encounter/exposure proxy) — Black/White only; left-join, may be null
    arr_path = RAW / "fbi_arrests_state_year.csv"
    if arr_path.exists():
        arr = pd.read_csv(arr_path)
        race_panel = race_panel.merge(arr, on=["state", "year", "race"], how="left")
    else:
        race_panel["arrests"] = float("nan")
    race_panel.to_parquet(PROCESSED / "race_panel.parquet")

    # --- 10-year cross-section --------------------------------------------
    cs = panel.groupby("state").agg(
        shootings=("shootings", "sum"),
        pop_total=("pop_total", "mean"),
        violent_crime_rate=("violent_crime_rate", "mean"),
        poverty_rate=("poverty_rate", "mean"),
        median_income_k=("median_income_k", "mean"),
        pct_black=("pct_black", "mean"),
        pct_hispanic=("pct_hispanic", "mean"),
        log_density=("log_density", "mean"),
        nics_per_1k=("nics_per_1k", "mean"),
        alcohol_binge_pct=("alcohol_binge_pct", "mean"),
        mental_distress_pct=("mental_distress_pct", "mean"),
        is_south=("is_south", "first"),
    ).reset_index()
    cs["shootings_per_100k_yr"] = cs["shootings"] / cs["pop_total"] * 1e5 / len(YEARS)
    cs.to_parquet(PROCESSED / "state_cross_section.parquet")

    # --- log -------------------------------------------------------------
    miss = panel["shootings"].sum()
    top = cs.nlargest(5, "shootings")[["state", "shootings"]].to_dict("records")
    toprate = cs.nlargest(5, "shootings_per_100k_yr")[
        ["state", "shootings_per_100k_yr"]].round(2).to_dict("records")
    body = [
        f"Panel assembled: **{len(panel)} state-years** "
        f"({panel['state'].nunique()} states × {panel['year'].nunique()} years).",
        f"Total shootings in panel: **{miss:,}** (territories excluded).",
        f"Crime-rate cells nulled for low FBI coverage (<80%): "
        f"{int((~panel['crime_reliable']).sum())}.",
        f"Context-imputed state-years (alcohol/mental health): "
        f"{int(panel['context_imputed'].sum())}.",
        "",
        f"**Most shootings (count):** {top}",
        f"**Highest per-capita (annual per 100k):** {toprate}",
        "",
        "Covariate coverage (non-null %):",
    ]
    for c in ["pop_total", "violent_crime_rate", "poverty_rate", "median_income_k",
              "pct_black", "nics_per_1k", "alcohol_binge_pct", "mental_distress_pct",
              "log_density"]:
        body.append(f"- {c}: {panel[c].notna().mean()*100:.0f}%")
    section("5. State-year panel assembly", "\n".join(body))
    log(f"Wrote panel ({len(panel)} rows), race_panel ({len(race_panel)}), "
        f"cross-section ({len(cs)}).")


if __name__ == "__main__":
    main()
