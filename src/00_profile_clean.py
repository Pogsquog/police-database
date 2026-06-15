"""00 — Profile and clean the WaPo shootings + agencies files.

Outputs:
  data/processed/incidents_clean.parquet
  data/processed/agencies_clean.parquet
and a data-profile section appended to findings.md.
"""
from __future__ import annotations

import pandas as pd

from common import ROOT, PROCESSED, CENSUS_REGION, STATE_FIPS, section, log

RACE_LABELS = {
    "W": "White", "B": "Black", "H": "Hispanic", "A": "Asian",
    "N": "Native American", "O": "Other",
}

SHOOTINGS_CSV = ROOT / "05_WaPo_Police_Shootings.csv"
AGENCIES_CSV = ROOT / "05b_WaPo_Police_Agencies.csv"


def clean_incidents() -> pd.DataFrame:
    df = pd.read_csv(SHOOTINGS_CSV, dtype=str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["was_mental_illness_related"] = df["was_mental_illness_related"] == "True"
    df["body_camera"] = df["body_camera"] == "True"

    # Race: collapse multi-race (e.g. "W;B") into a flag; single race -> label
    df["race_multi"] = df["race"].fillna("").str.contains(";")
    single = df["race"].where(~df["race_multi"])
    df["race_label"] = single.map(RACE_LABELS)
    df["race_label"] = df["race_label"].where(~df["race_multi"], "Multiple")

    # Keep only the 50 states + DC (drop territories like PR/GU) for panel joins
    df["in_panel_state"] = df["state"].isin(STATE_FIPS.values())
    df["region"] = df["state"].map(CENSUS_REGION)
    df["is_south"] = (df["region"] == "South").astype("Int64")

    # Derived binary outcomes for the within-incident models
    df["unarmed"] = (df["armed_with"] == "unarmed").astype("Int64")
    df["unarmed"] = df["unarmed"].where(df["armed_with"].notna())
    df["fleeing"] = df["flee_status"].isin(["car", "foot", "other"]).astype("Int64")
    df["fleeing"] = df["fleeing"].where(df["flee_status"].notna())
    return df


def clean_agencies() -> pd.DataFrame:
    a = pd.read_csv(AGENCIES_CSV, dtype=str)
    a["total_shootings"] = pd.to_numeric(a["total_shootings"], errors="coerce")
    return a


def profile(df: pd.DataFrame, agencies: pd.DataFrame) -> str:
    n = len(df)
    lines = []
    lines.append(f"**Incidents:** {n:,} fatal shootings, "
                 f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}.")
    lines.append(f"**Agencies:** {len(agencies):,} agencies "
                 f"({agencies['type'].nunique()} types).\n")

    lines.append("**Records per year:**\n")
    yr = df.groupby("year").size()
    lines.append("| Year | Shootings |\n|---|---|")
    for y, c in yr.items():
        lines.append(f"| {int(y)} | {c:,} |")
    lines.append("")

    def vc_table(col, title, label_map=None):
        out = [f"**{title}** (missing = {df[col].isna().sum():,}):\n",
               "| Value | Count | % |\n|---|---|---|"]
        vc = df[col].value_counts(dropna=False)
        for k, v in vc.items():
            disp = "(missing)" if pd.isna(k) else (label_map.get(k, k) if label_map else k)
            out.append(f"| {disp} | {v:,} | {100*v/n:.1f}% |")
        out.append("")
        return "\n".join(out)

    lines.append(vc_table("race_label", "Race / ethnicity"))
    lines.append(vc_table("gender", "Gender"))
    lines.append(vc_table("armed_with", "Armed with"))
    lines.append(vc_table("threat_type", "Threat type"))
    lines.append(vc_table("flee_status", "Flee status"))
    lines.append(f"**Mental-illness-related:** {df['was_mental_illness_related'].mean():.1%} "
                 f"of incidents.\n")
    lines.append(f"**Body camera present:** {df['body_camera'].mean():.1%} of incidents.\n")
    lines.append(f"**Age:** mean {df['age'].mean():.1f}, median {df['age'].median():.0f}, "
                 f"missing {df['age'].isna().sum():,}.\n")

    terr = (~df["in_panel_state"]).sum()
    lines.append(f"**Geography:** {terr:,} incidents are outside the 50 states + DC "
                 f"(territories); these are dropped from the state-year panel but kept "
                 f"for incident-level models where relevant.\n")
    lines.append("**Key caveat:** this file is *numerator-only* — it records shootings, "
                 "not the population at risk. Rates and confounder adjustment require the "
                 "external state-year denominators assembled in later steps.")
    return "\n".join(lines)


def main():
    incidents = clean_incidents()
    agencies = clean_agencies()

    incidents.to_parquet(PROCESSED / "incidents_clean.parquet")
    agencies.to_parquet(PROCESSED / "agencies_clean.parquet")
    log(f"Wrote {len(incidents):,} incidents and {len(agencies):,} agencies.")

    # reset=True initialises findings.md with the report header for this run
    section("1. Data profile & cleaning (WaPo files)",
            profile(incidents, agencies), reset=True)


if __name__ == "__main__":
    main()
