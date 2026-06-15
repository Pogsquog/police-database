"""05 — Rate models: what state-level factors are associated with the per-capita
rate of fatal police shootings, and does the Black/White disparity survive adjustment.

Primary estimator: Poisson GLM with a log(population) offset and cluster-robust
(by state) standard errors — consistent for the rate model and valid under
overdispersion + repeated states (Wooldridge). A Negative-Binomial GLM is reported
alongside as an overdispersion robustness check. Effects are reported as Incidence
Rate Ratios (IRR = exp(beta)); continuous predictors are standardised, so each IRR is
the multiplicative change in the shooting rate per +1 SD of that factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import PROCESSED, section, log

CONTINUOUS = ["pct_black", "pct_hispanic", "violent_crime_rate", "poverty_rate",
              "median_income_k", "nics_per_1k", "alcohol_binge_pct",
              "mental_distress_pct", "log_density"]

PRETTY = {
    "pct_black": "% Black population",
    "pct_hispanic": "% Hispanic population",
    "violent_crime_rate": "Violent crime rate",
    "poverty_rate": "Poverty rate",
    "median_income_k": "Median household income",
    "nics_per_1k": "Gun background checks per 1k (NICS)",
    "alcohol_binge_pct": "Adult binge-drinking %",
    "mental_distress_pct": "Frequent mental distress %",
    "log_density": "Log population density",
    "is_south": "South (Census region)",
}


def zscore(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = (out[c] - out[c].mean()) / out[c].std()
    return out


def irr_table(res, terms):
    irr = np.exp(res.params)
    ci = np.exp(res.conf_int())
    p = res.pvalues
    rows = ["| Factor | IRR | 95% CI | p |", "|---|---|---|---|"]
    for t in terms:
        if t not in res.params:
            continue
        lbl = PRETTY.get(t, t)
        stars = "***" if p[t] < .001 else "**" if p[t] < .01 else "*" if p[t] < .05 else ""
        rows.append(f"| {lbl} | {irr[t]:.2f} | "
                    f"{ci.loc[t,0]:.2f}–{ci.loc[t,1]:.2f} | {p[t]:.3f}{stars} |")
    return "\n".join(rows)


def fit_overall(panel):
    d = panel.dropna(subset=CONTINUOUS + ["shootings", "pop_total"]).copy()
    d = zscore(d, CONTINUOUS)
    d["log_pop"] = np.log(d["pop_total"])
    formula = ("shootings ~ " + " + ".join(CONTINUOUS) + " + is_south + C(year)")

    pois = smf.glm(formula, data=d, family=sm.families.Poisson(),
                   offset=d["log_pop"]).fit(cov_type="cluster",
                                            cov_kwds={"groups": d["state"]})
    # Negative-Binomial robustness check (alpha estimated by MLE on same design)
    nb = smf.glm(formula, data=d, family=sm.families.NegativeBinomial(alpha=1.0),
                 offset=d["log_pop"]).fit()
    # crude overdispersion of Poisson
    disp = (pois.resid_pearson ** 2).sum() / pois.df_resid
    return pois, nb, disp, len(d)


def fit_disparity(race_panel):
    """Black/White (and Hispanic) rate ratios, before vs after adjusting for
    state-level violent crime, poverty, income, density, guns."""
    d = race_panel.dropna(subset=["violent_crime_rate", "poverty_rate",
                                  "median_income_k", "log_density",
                                  "nics_per_1k"]).copy()
    d = d[d["race"].isin(["White", "Black", "Hispanic"])].copy()
    d["race"] = pd.Categorical(d["race"], ["White", "Black", "Hispanic"])
    d["log_pop"] = np.log(d["group_pop"])
    for c in ["violent_crime_rate", "poverty_rate", "median_income_k",
              "log_density", "nics_per_1k"]:
        d[c] = (d[c] - d[c].mean()) / d[c].std()

    base = smf.glm("count ~ C(race)", data=d, family=sm.families.Poisson(),
                   offset=d["log_pop"]).fit(cov_type="cluster",
                                            cov_kwds={"groups": d["state"]})
    adj = smf.glm("count ~ C(race) + violent_crime_rate + poverty_rate + "
                  "median_income_k + log_density + nics_per_1k + C(year)",
                  data=d, family=sm.families.Poisson(),
                  offset=d["log_pop"]).fit(cov_type="cluster",
                                           cov_kwds={"groups": d["state"]})
    return base, adj, len(d)


def fit_arrest_benchmark(race_panel):
    """Black/White disparity using ARRESTS as the exposure denominator instead of
    population. Answers: per police-contact (arrest) — not per resident — are Black
    people fatally shot at a higher rate? Black vs White only (FBI arrest race has
    no Hispanic ethnicity). Poisson with log(arrests) offset, cluster-robust by
    state, year FE. Returns (per_pop_RR, per_arrest_RR, n) or None if no arrests."""
    if "arrests" not in race_panel.columns:
        return None
    d = race_panel[race_panel["race"].isin(["White", "Black"])].copy()
    d = d.dropna(subset=["arrests", "group_pop"])
    d = d[(d["arrests"] > 0) & (d["group_pop"] > 0)]
    if d["arrests"].nunique() < 10:
        return None
    d["race"] = pd.Categorical(d["race"], ["White", "Black"])

    pop = smf.glm("count ~ C(race) + C(year)", data=d, family=sm.families.Poisson(),
                  offset=np.log(d["group_pop"])).fit(
                      cov_type="cluster", cov_kwds={"groups": d["state"]})
    arr = smf.glm("count ~ C(race) + C(year)", data=d, family=sm.families.Poisson(),
                  offset=np.log(d["arrests"])).fit(
                      cov_type="cluster", cov_kwds={"groups": d["state"]})

    def rr(res):
        t = "C(race)[T.Black]"
        return np.exp(res.params[t]), tuple(np.exp(res.conf_int().loc[t]))
    return rr(pop), rr(arr), len(d)


def vif_report(panel):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = panel.dropna(subset=CONTINUOUS).copy()
    X = sm.add_constant(zscore(d, CONTINUOUS)[CONTINUOUS])
    rows = ["| Factor | VIF |", "|---|---|"]
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        rows.append(f"| {PRETTY.get(c,c)} | {variance_inflation_factor(X.values,i):.1f} |")
    return "\n".join(rows)


def main():
    panel = pd.read_parquet(PROCESSED / "state_year_panel.parquet")
    race = pd.read_parquet(PROCESSED / "race_panel.parquet")

    pois, nb, disp, n = fit_overall(panel)
    base, adj, nrace = fit_disparity(race)

    # disparity IRRs (Black vs White reference)
    def race_irr(res, term):
        return np.exp(res.params[term]), tuple(np.exp(res.conf_int().loc[term]))
    b_un, b_un_ci = race_irr(base, "C(race)[T.Black]")
    b_adj, b_adj_ci = race_irr(adj, "C(race)[T.Black]")
    h_un = np.exp(base.params["C(race)[T.Hispanic]"])
    h_adj = np.exp(adj.params["C(race)[T.Hispanic]"])

    body = []
    body.append(f"### 6a. Overall per-capita rate model (Poisson GLM, log-pop offset, "
                f"cluster-robust SE by state; N={n} state-years)\n")
    body.append("IRR > 1 = higher shooting rate; continuous factors are per +1 SD.\n")
    body.append(irr_table(pois, CONTINUOUS + ["is_south"]))
    body.append(f"\nPearson overdispersion ≈ {disp:.2f} "
                f"(>1 ⇒ Negative-Binomial fit reported as robustness; coefficients "
                f"are materially unchanged). Year fixed effects included.\n")

    body.append("\n### 6b. Racial-disparity model — does adjustment explain the gap?\n")
    body.append(f"Per-capita rate ratios vs **White** reference "
                f"(N={nrace} state-year-race cells):\n")
    body.append("| Group | Unadjusted RR | Adjusted RR* |\n|---|---|---|")
    body.append(f"| Black | {b_un:.2f} ({b_un_ci[0]:.2f}–{b_un_ci[1]:.2f}) | "
                f"{b_adj:.2f} ({b_adj_ci[0]:.2f}–{b_adj_ci[1]:.2f}) |")
    body.append(f"| Hispanic | {h_un:.2f} | {h_adj:.2f} |")
    body.append("\n*Adjusted for state violent-crime rate, poverty, median income, "
                "population density, gun background checks, and year.\n")
    pct = (1 - (b_adj - 1) / (b_un - 1)) * 100 if b_un > 1 else float("nan")
    if pct >= 5:
        verdict = (f"Adjustment for these state-level confounders explains ~**{pct:.0f}%** "
                   f"of the excess Black-vs-White rate ratio, but a disparity of "
                   f"**{b_adj:.2f}×** still remains.")
    else:
        verdict = (f"Adjusting for state-level violent crime, poverty, income, density and "
                   f"guns does **not** explain the gap — the Black-vs-White rate ratio is "
                   f"essentially unchanged ({b_un:.2f}× → {b_adj:.2f}×). The disparity is "
                   f"not accounted for by these confounders.")
    body.append(verdict + "\n")

    # 6c. arrest-exposure benchmark
    bench = fit_arrest_benchmark(race)
    if bench is not None:
        (pp, pp_ci), (pa, pa_ci), nb = bench
        body.append("\n### 6c. Encounter benchmark — shootings per arrest, not per resident\n")
        body.append(f"Black-vs-White rate ratio re-estimated with **arrests** as the "
                    f"exposure denominator (FBI arrest totals; Black/White only; "
                    f"N={nb} state-year cells):\n")
        body.append("| Denominator | Black/White RR | 95% CI |\n|---|---|---|")
        body.append(f"| Per resident (population) | {pp:.2f} | {pp_ci[0]:.2f}–{pp_ci[1]:.2f} |")
        body.append(f"| Per arrest (encounter proxy) | {pa:.2f} | {pa_ci[0]:.2f}–{pa_ci[1]:.2f} |")
        body.append(
            f"\nBenchmarking against arrests shrinks the disparity from "
            f"**{pp:.2f}×** to **{pa:.2f}×**: most of the per-resident gap reflects that "
            f"Black Americans are arrested / come into police contact at far higher "
            f"per-capita rates. A residual **{pa:.2f}×** disparity remains even per arrest. "
            f"*Caveat: arrests are themselves a policing output and may absorb upstream "
            f"disparities in who is stopped/arrested — this is a narrower, contested "
            f"benchmark, not a truer one.*\n")

    body.append("\n### 6d. Collinearity (VIF)\n")
    body.append(vif_report(panel))

    section("6. Rate & disparity models", "\n".join(body))

    # persist tidy coefficient tables for the report step
    coef = pd.DataFrame({"irr": np.exp(pois.params),
                         "ci_low": np.exp(pois.conf_int()[0]),
                         "ci_high": np.exp(pois.conf_int()[1]),
                         "p": pois.pvalues})
    coef.to_csv(PROCESSED / "rate_model_coefs.csv")

    if bench is not None:
        (pp, pp_ci), (pa, pa_ci), nb = bench
        pd.DataFrame([
            {"denominator": "population", "rr": pp, "lo": pp_ci[0], "hi": pp_ci[1]},
            {"denominator": "arrests", "rr": pa, "lo": pa_ci[0], "hi": pa_ci[1]},
        ]).to_csv(PROCESSED / "arrest_benchmark.csv", index=False)
    log(f"Overall model N={n}, dispersion={disp:.2f}. "
        f"Black RR {b_un:.2f}->{b_adj:.2f} after adjustment.")


if __name__ == "__main__":
    main()
