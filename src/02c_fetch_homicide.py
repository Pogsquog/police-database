"""02c — CDC homicide victimization by race (police-independent offending proxy).

The arrest benchmark (02b) can't separate "Black people offend more" from "Black
people are arrested/policed more", because arrests are a policing output. Homicide
is the least-discretionary crime: it is almost always recorded regardless of
policing, and is ~80-90% intra-racial, so the race distribution of homicide
*victims* closely tracks the race distribution of homicide *offenders*. The
Black/White homicide-victimization rate ratio is therefore a benchmark for
involvement in lethal violence that does not pass through police discretion.

Source: CDC WONDER, Underlying Cause of Death 1999-2020 (database D76), filtered to
homicide ICD-10 codes (X85-Y09, Y87.1) and the WaPo-overlap window 2015-2020.
WONDER's API only serves **national** data (state/county grouping is blocked
server-side), so this benchmark is national; it is compared against the
state-aggregated arrest and shooting disparities. Race here does not net out
Hispanic ethnicity (matching the FBI arrest race coding), so it is Black vs White.
"""
from __future__ import annotations

import re

import pandas as pd

from common import RAW, section, log

WONDER_URL = "https://wonder.cdc.gov/controller/datarequest/D76"
YEARS = ["2015", "2016", "2017", "2018", "2019", "2020"]
HOMICIDE_ICD = ([f"X{n}" for n in range(85, 100)]
                + [f"Y0{n}" for n in range(0, 10)] + ["Y87.1"])


def _plist(d: dict) -> str:
    out = ""
    for k, v in d.items():
        out += f"<parameter><name>{k}</name>"
        vals = v if isinstance(v, list) else [v]
        out += "".join(f"<value>{x}</value>" for x in vals)
        out += "</parameter>"
    return out


def _request_xml() -> str:
    b = {"B_1": "D76.V8", "B_2": "*None*", "B_3": "*None*",
         "B_4": "*None*", "B_5": "*None*"}                 # group by race
    m = {"M_1": "D76.M1", "M_2": "D76.M2", "M_3": "D76.M3"}  # deaths, pop, rate
    f = {"F_D76.V1": YEARS, "F_D76.V10": ["*All*"], "F_D76.V2": HOMICIDE_ICD,
         "F_D76.V27": ["*All*"], "F_D76.V9": ["*All*"]}
    i = {"I_D76.V1": " ".join(YEARS), "I_D76.V10": "*All* (The United States)",
         "I_D76.V2": "Homicide", "I_D76.V27": "*All* (The United States)",
         "I_D76.V9": "*All* (The United States)"}
    o = {"O_V10_fmode": "freg", "O_V1_fmode": "freg", "O_V27_fmode": "freg",
         "O_V2_fmode": "freg", "O_V9_fmode": "freg", "O_aar": "aar_none",
         "O_aar_pop": "0000", "O_age": "D76.V5", "O_javascript": "on",
         "O_location": "D76.V9", "O_precision": "1", "O_rate_per": "100000",
         "O_show_totals": "false", "O_timeout": "300",
         "O_title": "Homicide by race", "O_ucd": "D76.V2", "O_urban": "D76.V19"}
    vm = {"VM_D76.M6_D76.V10": "", "VM_D76.M6_D76.V17": "*All*",
          "VM_D76.M6_D76.V1_S": "*All*", "VM_D76.M6_D76.V7": "*All*",
          "VM_D76.M6_D76.V8": "*All*"}
    v = {"V_D76.V1": "", "V_D76.V10": "", "V_D76.V11": "*All*", "V_D76.V12": "*All*",
         "V_D76.V17": "*All*", "V_D76.V19": "*All*", "V_D76.V2": "", "V_D76.V20": "*All*",
         "V_D76.V21": "*All*", "V_D76.V22": "*All*", "V_D76.V23": "*All*",
         "V_D76.V24": "*All*", "V_D76.V25": "*All*", "V_D76.V27": "", "V_D76.V4": "*All*",
         "V_D76.V5": "*All*", "V_D76.V51": "*All*", "V_D76.V52": "*All*",
         "V_D76.V6": "*All*", "V_D76.V7": "*All*", "V_D76.V8": "*All*", "V_D76.V9": ""}
    misc = {"action-Send": "Send", "finder-stage-D76.V1": "codeset",
            "finder-stage-D76.V2": "codeset", "finder-stage-D76.V27": "codeset",
            "finder-stage-D76.V9": "codeset", "stage": "request"}
    return ("<request-parameters>" + _plist(b) + _plist(m) + _plist(f) + _plist(i)
            + _plist(o) + _plist(vm) + _plist(v) + _plist(misc)
            + "</request-parameters>")


def _parse(xml: str) -> pd.DataFrame:
    rows = []
    for row in re.findall(r"<r>(.*?)</r>", xml, re.S):
        cells = re.findall(r'<c (?:l="([^"]*)"|v="([^"]*)")', row)
        vals = [a or b for a, b in cells]
        if len(vals) >= 4 and vals[0]:
            rows.append(vals[:4])
    df = pd.DataFrame(rows, columns=["race", "deaths", "population", "rate"])
    for c in ["deaths", "population", "rate"]:
        df[c] = df[c].str.replace(",", "", regex=False).astype(float)
    return df


def main():
    dest = RAW / "cdc_homicide_race.xml"
    if dest.exists():
        xml = dest.read_text()
    else:
        import requests
        r = requests.post(WONDER_URL,
                          data={"request_xml": _request_xml(),
                                "accept_datause_restrictions": "true"}, timeout=180)
        if r.status_code != 200:
            log(f"  ! CDC WONDER returned {r.status_code}; homicide benchmark skipped")
            section("3c. CDC homicide-by-race fetch (offending proxy)",
                    "CDC WONDER request failed; offending benchmark unavailable this run.")
            return
        xml = r.text
        dest.write_text(xml)

    df = _parse(xml)
    out = RAW / "cdc_homicide_race.csv"
    df.to_csv(out, index=False)

    bw = df[df["race"].isin(["Black or African American", "White"])].set_index("race")
    rr = (bw.loc["Black or African American", "rate"] / bw.loc["White", "rate"]
          if len(bw) == 2 else float("nan"))
    body = (
        f"CDC WONDER homicide victimization by race, US 2015–2020 (ICD-10 X85–Y09, "
        f"Y87.1). National only — WONDER's API blocks state grouping.\n\n"
        "| Race | Homicide deaths | Crude rate /100k |\n|---|---|---|\n"
        + "".join(f"| {row['race']} | {row['deaths']:,.0f} | {row['rate']:.1f} |\n"
                  for _, row in df.iterrows())
        + f"\n**Black/White homicide-victimization ratio ≈ {rr:.1f}×** — a "
        f"police-independent proxy for involvement in lethal violence (homicide is "
        f"~80–90% intra-racial). Compared against the arrest and shooting disparities "
        f"in the offending-benchmark ladder (script 05)."
    )
    section("3c. CDC homicide-by-race fetch (offending proxy)", body)
    log(f"Wrote {out}; Black/White homicide victimization RR ≈ {rr:.2f}.")


if __name__ == "__main__":
    main()
