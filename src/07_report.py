"""07 — Figures + synthesis: render charts to figures/ and append a key-findings
and limitations section to findings.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import PROCESSED, FIGURES, section, log

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def fig_trend(panel):
    yr = panel.groupby("year")["shootings"].sum()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(yr.index, yr.values, marker="o")
    ax.set(title="Fatal police shootings per year (2015–2024)",
           xlabel="Year", ylabel="Shootings")
    ax.set_ylim(0, yr.max() * 1.15)
    fig.tight_layout(); fig.savefig(FIGURES / "trend.png"); plt.close(fig)


def fig_state_rate(cs):
    d = cs.sort_values("shootings_per_100k_yr")
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(d["state"], d["shootings_per_100k_yr"], color="#b03a2e")
    ax.set(title="Annual fatal police shootings per 100k, by state (2015–2024 avg)",
           xlabel="Per 100k per year")
    fig.tight_layout(); fig.savefig(FIGURES / "state_rates.png"); plt.close(fig)


def fig_forest(coefs):
    keep = [i for i in coefs.index if not i.startswith("C(year)")
            and i not in ("Intercept", "const")]
    d = coefs.loc[keep].sort_values("irr")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(d))
    ax.errorbar(d["irr"], y, xerr=[d["irr"] - d["ci_low"], d["ci_high"] - d["irr"]],
                fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(1.0, color="grey", ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=8)
    ax.set(title="Rate-model incidence rate ratios (per +1 SD)", xlabel="IRR")
    fig.tight_layout(); fig.savefig(FIGURES / "rate_forest.png"); plt.close(fig)


def fig_density(cs):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(cs["log_density"], cs["shootings_per_100k_yr"], color="#1f4e79")
    for _, r in cs.iterrows():
        ax.annotate(r["state"], (r["log_density"], r["shootings_per_100k_yr"]),
                    fontsize=6, alpha=.7)
    ax.set(title="Population density vs shooting rate",
           xlabel="log(population density)", ylabel="Shootings per 100k/yr")
    fig.tight_layout(); fig.savefig(FIGURES / "density_scatter.png"); plt.close(fig)


def main():
    panel = pd.read_parquet(PROCESSED / "state_year_panel.parquet")
    cs = pd.read_parquet(PROCESSED / "state_cross_section.parquet")
    coefs = pd.read_csv(PROCESSED / "rate_model_coefs.csv", index_col=0)

    fig_trend(panel)
    fig_state_rate(cs)
    fig_forest(coefs)
    fig_density(cs)

    synthesis = """\
### Conclusions (plain language)
Read these as adjusted *associations* across places and cases, not individual-level
causation; the WaPo data is fatal-only and has no exposure denominator.

1. **Race matters, but the gap is denominator-dependent and not in excess of offending.**
   Black Americans are fatally shot at ~2.6× the White per-resident rate, and that gap
   *survives* every state confounder we tried (crime, poverty, income, density, guns →
   2.8×). But it shrinks to ~1.3× per *arrest*, and against homicide involvement (6.7×) it
   is *smaller* than proportional. So the disparity is real and large per resident, yet
   below what involvement in serious violence would predict — population over-states it by
   ignoring contact exposure; arrest/homicide under-state it by absorbing any upstream bias.
2. **Rural rates are higher, and distance to trauma care is a likely contributor — not a
   proven sole cause.** Sparse states have ~2.5× the rate on *similar* crime: a
   lethality-per-encounter gap, not a crime gap. A placebo test is consistent with
   medical-access case-fatality (car/fall deaths track the rate; the overdose control does
   not), but officer isolation and thin de-escalation resources plausibly contribute too,
   and per-capita accident deaths blend exposure with case-fatality. The "rural sheriff"
   explanation does **not** hold (sparse shootings are mostly municipal police).
3. **Women's shootings differ in character — this is *not* a statement that women are more
   likely to be shot.** These are cases-only models (women are ~5% of victims, and there is
   no risk denominator). *When* women are shot, the encounter is disproportionately a
   mental-health crisis (OR 1.87) and unarmed (1.73), and much less often a fleeing suspect
   (0.71).
4. **Mental health is a through-line, not a footnote.** ~20% of all shootings are flagged
   mental-illness-related; at the *state* level, population mental-distress independently
   predicts a higher rate (IRR ~1.15). A meaningful share of fatal police violence is
   effectively a mental-health-system outcome — concentrated among older, White and female
   victims.
5. **Different *types* of fatal encounter cluster by demographic.** Age is the strongest
   within-incident factor: younger victims are far more often unarmed and fleeing; older
   ones, mental-health-related. Black victims' shootings are more often unarmed (1.34) and
   fleeing (1.21); White/Hispanic/Native victims' more often mental-illness-flagged.
6. **The body-camera gap is about departments, and the South lags on transparency.** The
   raw Black/White body-camera OR (~1.9) is almost entirely *which agency was involved*
   (collapses to ~1.3× within the same department); cameras are present in only ~13% of
   Southern shootings (OR 0.64) vs elsewhere.
7. **The toll is rising and under-discussed groups matter.** Deaths climbed ~995→~1,175/yr
   over the decade (~18%, steady — not a post-2020 spike), and Native Americans are
   massively overrepresented in the sparse West (4.7% of victims there vs 0.2% in dense
   states) — a disparity the Black/White framing misses.

### Figures
- `figures/trend.png` — shootings per year (a clear upward drift, ~995→1,175).
- `figures/state_rates.png` — per-capita rate by state.
- `figures/rate_forest.png` — rate-model IRRs with 95% CIs.
- `figures/density_scatter.png` — density vs rate.
- `figures/rural_lethality.png` — fatal shootings per 1,000 violent crimes vs density.
- `figures/rural_medaccess.png` — shooting-rate correlation by accident type (medical-access test).

### Key findings
1. **Geography/urbanicity dominates the per-capita rate.** Population density is the
   strongest correlate: denser (more urban) states have *lower* per-capita shooting
   rates (IRR ≈ 0.67 per +1 SD). The highest-rate states are sparse Western ones
   (NM, AK, OK, CO, AZ); the lowest are dense Northeastern ones (RI, MA, CT, NY, NJ).
   This rural excess is **counterintuitive but resolvable** (section 7): across states,
   sparse ≠ low-crime (violent crime is flat-to-higher in sparse states), so the gap is a
   *lethality-per-encounter* one — sparse states fatally shoot ~2.3× more per violent crime
   and ~20% more per arrest. Density survives controls while gun prevalence washes out;
   incident composition and a "rural sheriff" effect explain little (sparse shootings are
   mostly municipal police). Consistent with officer isolation / thinner de-escalation
   resources, plus a fatal-only artifact: a placebo test (§7f) finds the shooting rate
   tracks trauma-access-sensitive accident deaths (car crashes r≈0.47, falls) but *not* the
   overdose negative control (r≈−0.11) — pointing to distance-to-medical-care case-fatality.
2. **Violent crime and population mental-distress** are positively associated with the
   state shooting rate (IRR ≈ 1.16 and 1.15 per +1 SD).
3. **A state's Black population share does not predict its overall rate** once crime and
   density are controlled — the rate story is regional, not racial-composition driven.
4. **The Black/White disparity is entirely denominator-dependent.** Per resident it is
   ~2.6× (unchanged by adjusting for crime/poverty/income/density/guns, ≈2.8×); per
   *arrest* ~1.3×; and against *homicide victimization* (a police-independent offending
   proxy) Black involvement is ~6.7× — larger than the shooting gap. Most of the
   per-resident gap reflects higher arrest/contact exposure, and relative to serious-
   violence involvement the shooting rate is not inflated. This brackets rather than
   settles the question: population over-states the gap by ignoring exposure, while
   arrest/homicide denominators under-state it by baking in any upstream enforcement
   bias, and homicide is the wrong exposure for many (traffic, mental-health) shootings.
5. **Gun background checks and binge-drinking** show no robust independent association
   at the state level; poverty/income wash out once crime and density are included.
6. **Within incidents:** Black victims are modestly more likely to be unarmed
   (OR 1.34) and fleeing (OR 1.21); White victims' shootings are far more often flagged
   mental-illness-related (Black/Hispanic OR ≈ 0.56). The large raw body-camera gap
   (Black OR ≈ 1.9) is mostly *agency-level* confounding — large urban departments
   adopted cameras earliest and handle more Black victims; holding agency fixed it
   falls to ≈1.3×.

### Limitations (read before citing)
- **Numerator-only + ecological.** WaPo records only fatal shootings. State-level
  associations can suffer ecological fallacy and cannot establish individual-level
  causation; they describe *places*, not encounters.
- **No exposure denominator.** We model shootings per resident, not per police–civilian
  encounter or per arrest. Differences in contact rates are unmeasured confounders.
- **FBI 2021 coverage gap.** NIBRS-transition under-reporting forced us to null
  low-coverage crime cells; year fixed effects absorb the national dip.
- **Proxy covariates.** Gun availability uses NICS checks (a flow proxy, 2015–2023,
  carried forward to 2024); alcohol/mental-health from BRFSS have year gaps filled by
  state means. These attenuate toward null rather than inflate effects.
- **Missing race (11% of incidents)** and self-reported survey covariates add noise.
- **Associational, not causal.** No claims of causation; these are adjusted associations.
"""
    section("9. Synthesis, figures & limitations", synthesis)
    log("Report complete; figures written to figures/.")


if __name__ == "__main__":
    main()
