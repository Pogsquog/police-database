"""06 — Within-incident logistic models (cases only).

Among people fatally shot by police, what is associated with the incident having a
given characteristic? These are *conditional on being shot* — they describe the
character of shootings, not the risk of being shot, and cannot support causal or
rate claims. Reported as odds ratios (OR) vs a reference category.

Outcomes: unarmed, body_camera present, mental-illness-related, fleeing.
Predictors: race (ref White), age (per +10 yrs), gender, region (South), year FE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import PROCESSED, section, log

RACES = ["White", "Black", "Hispanic", "Asian", "Native American"]
OUTCOMES = {
    "unarmed": "Victim unarmed",
    "body_camera": "Body camera present",
    "was_mental_illness_related": "Mental-illness-related",
    "fleeing": "Was fleeing",
}


def prep(inc):
    d = inc[inc["race_label"].isin(RACES)].copy()
    d = d[d["gender"].isin(["male", "female"])]
    d = d[d["age"].notna() & d["year"].notna()]
    d["race_label"] = pd.Categorical(d["race_label"], RACES)
    d["age10"] = d["age"] / 10.0
    d["female"] = (d["gender"] == "female").astype(int)
    d["body_camera"] = d["body_camera"].astype(int)
    d["was_mental_illness_related"] = d["was_mental_illness_related"].astype(int)
    return d


def fit_logit(d, outcome):
    sub = d[d[outcome].notna()].copy()
    sub[outcome] = sub[outcome].astype(int)
    f = (f"{outcome} ~ C(race_label, Treatment('White')) + age10 + female + "
         f"is_south + C(year)")
    res = smf.logit(f, data=sub).fit(disp=0)
    return res, len(sub), sub[outcome].mean()


def or_row(res, term, label):
    if term not in res.params:
        return None
    orr = np.exp(res.params[term])
    lo, hi = np.exp(res.conf_int().loc[term])
    p = res.pvalues[term]
    star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    return f"| {label} | {orr:.2f} | {lo:.2f}–{hi:.2f} | {p:.3f}{star} |"


def main():
    inc = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    d = prep(inc)

    body = ["Odds ratios from logistic regression; **White** is the race reference, "
            "male the gender reference. Age is per +10 years. All models include "
            "year fixed effects. *Cases only — conditional on having been shot.*\n"]

    summary = {}
    for outcome, title in OUTCOMES.items():
        res, n, base = fit_logit(d, outcome)
        body.append(f"\n### {title} (N={n:,}, base rate {base:.1%})\n")
        body.append("| Factor | OR | 95% CI | p |\n|---|---|---|---|")
        for grp in RACES[1:]:
            r = or_row(res, f"C(race_label, Treatment('White'))[T.{grp}]",
                       f"{grp} (vs White)")
            if r:
                body.append(r)
        for term, lbl in [("age10", "Age (+10 yrs)"), ("female", "Female"),
                           ("is_south", "South region")]:
            r = or_row(res, term, lbl)
            if r:
                body.append(r)
        summary[outcome] = (n, base)

    section("7. Within-incident models (cases only)", "\n".join(body))
    log(f"Incident models fit for {list(OUTCOMES)} on N≈{summary['unarmed'][0]}.")


if __name__ == "__main__":
    main()
