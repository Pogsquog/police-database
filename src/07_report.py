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
### Figures
- `figures/trend.png` — shootings per year (a clear upward drift, ~995→1,175).
- `figures/state_rates.png` — per-capita rate by state.
- `figures/rate_forest.png` — rate-model IRRs with 95% CIs.
- `figures/density_scatter.png` — density vs rate.

### Key findings
1. **Geography/urbanicity dominates the per-capita rate.** Population density is the
   strongest correlate: denser (more urban) states have *lower* per-capita shooting
   rates (IRR ≈ 0.67 per +1 SD). The highest-rate states are sparse Western ones
   (NM, AK, OK, CO, AZ); the lowest are dense Northeastern ones (RI, MA, CT, NY, NJ).
2. **Violent crime and population mental-distress** are positively associated with the
   state shooting rate (IRR ≈ 1.16 and 1.15 per +1 SD).
3. **A state's Black population share does not predict its overall rate** once crime and
   density are controlled — the rate story is regional, not racial-composition driven.
4. **The Black/White disparity is real and not explained by state confounders, but
   most of it tracks police-contact exposure.** Black Americans are shot at ~2.6× the
   White per-capita rate, unchanged by adjusting for crime/poverty/income/density/guns
   (≈2.8×). Benchmarked against *arrests* instead of residents it falls to ~1.3×: most
   of the per-resident gap reflects higher arrest/contact rates, with a ~1.3× residual
   remaining even per arrest (arrests are themselves a policing output — a contested
   benchmark, not a truer one).
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
    section("8. Synthesis, figures & limitations", synthesis)
    log("Report complete; figures written to figures/.")


if __name__ == "__main__":
    main()
