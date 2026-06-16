"""05b — The rural paradox: why do sparse states have HIGHER per-capita fatal
shooting rates, when within any state it is cities that carry the violent crime?

This decomposes the negative density gradient (rate IRR ≈ 0.67 per +1 SD of
log-density) to locate where the rural excess actually sits:

  rate = (contacts per resident)  ×  (shootings per contact / lethality)

We benchmark shootings against two exposure denominators — violent crimes and
arrests — to show the gap is a *lethality-per-encounter* gap, not a crime gap;
test incident composition (weapon, flee, mental-health, Native American share);
and directly test the "rural sheriff" hypothesis via WaPo agency types. Outputs a
figure and a findings section. All associations are ecological; fatal-only data
carries a medical-access caveat (rural wounds are likelier to prove fatal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, PROCESSED, RAW, FIGURES, section, log

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def build(cs, panel):
    """State cross-section augmented with exposure-benchmarked lethality measures."""
    arr = (pd.read_csv(RAW / "fbi_arrests_state_year.csv")
           .groupby("state")["arrests"].sum().rename("arrests_bw"))
    pop = panel.groupby("state")["pop_total"].mean().rename("pop_mean")
    vc = panel.groupby("state")["violent_crime_rate"].mean().rename("vc_mean")

    d = cs.merge(arr, on="state", how="left")
    d["pop_mean"] = d["state"].map(pop)
    d["vc_mean"] = d["state"].map(vc)
    # total violent crimes over the 10-year window, and lethality benchmarks
    d["vcrimes_10y"] = d["vc_mean"] * d["pop_mean"] / 1e5 * 10
    d["shoot_per_1k_vc"] = d["shootings"] / d["vcrimes_10y"] * 1000
    d["shoot_per_10k_arr"] = d["shootings"] / d["arrests_bw"] * 1e4
    d["dens_tertile"] = pd.qcut(d["log_density"], 3, labels=["Sparse", "Mid", "Dense"])
    return d


def composition(inc, ag, cs):
    """Incident-level composition and agency type by state density tertile."""
    id2type = dict(zip(ag["id"].astype(str), ag["type"]))

    def first_type(s):
        if pd.isna(s):
            return None
        return id2type.get(str(s).split(";")[0].strip())

    m = inc.copy()
    m["ag_type"] = m["agency_ids"].apply(first_type)
    m = m.merge(cs[["state", "dens_tertile"]], on="state", how="inner")

    def to_bool(s):  # nullable Int64/boolean/object -> plain bool, NaN->False
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).astype(bool)

    m["gun"] = m["armed_with"].eq("gun")
    m["unarmed_f"] = to_bool(m["unarmed"])
    m["flee_any"] = m["flee_status"].notna() & ~m["flee_status"].isin(["not"])
    m["mental"] = to_bool(m["was_mental_illness_related"])
    lbl = m["race_label"].astype(str)
    m["native"] = lbl.str.contains("Native|American Indian", case=False, na=False)
    m["sheriff"] = m["ag_type"].eq("sheriff")

    comp = (m.groupby("dens_tertile", observed=True).agg(
        n=("id", "count"),
        pct_gun=("gun", "mean"),
        pct_unarmed=("unarmed_f", "mean"),
        pct_flee=("flee_any", "mean"),
        pct_mental=("mental", "mean"),
        pct_native=("native", "mean"),
        pct_sheriff=("sheriff", "mean")) )
    for c in [c for c in comp.columns if c.startswith("pct_")]:
        comp[c] = comp[c] * 100
    # per-state sheriff share for correlation with rate
    sher = m.groupby("state")["sheriff"].mean().rename("sheriff_share")
    return comp, sher


def fig_lethality(d):
    sub = d.dropna(subset=["shoot_per_1k_vc", "log_density"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(sub["log_density"], sub["shoot_per_1k_vc"], color="#b03a2e")
    for _, r in sub.iterrows():
        ax.annotate(r["state"], (r["log_density"], r["shoot_per_1k_vc"]),
                    fontsize=6, alpha=.7)
    # fit line
    b = np.polyfit(sub["log_density"], np.log(sub["shoot_per_1k_vc"]), 1)
    xs = np.linspace(sub["log_density"].min(), sub["log_density"].max(), 50)
    ax.plot(xs, np.exp(np.polyval(b, xs)), color="#1f4e79", ls="--", lw=1)
    ax.set(title="Lethality per crime falls with density",
           xlabel="log(population density)",
           ylabel="Fatal shootings per 1,000 violent crimes")
    fig.tight_layout(); fig.savefig(FIGURES / "rural_lethality.png"); plt.close(fig)


def main():
    cs = pd.read_parquet(PROCESSED / "state_cross_section.parquet")
    panel = pd.read_parquet(PROCESSED / "state_year_panel.parquet")
    inc = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    ag = pd.read_csv(ROOT / "05b_WaPo_Police_Agencies.csv")

    d = build(cs, panel)
    comp, sher = composition(inc, ag, d)
    d = d.merge(sher, on="state", how="left")
    fig_lethality(d)

    # tertile rate/exposure table
    tert = d.groupby("dens_tertile", observed=True).agg(
        n=("state", "count"),
        rate=("shootings_per_100k_yr", "mean"),
        vcr=("violent_crime_rate", "mean"),
        per_1k_vc=("shoot_per_1k_vc", "mean"),
        per_10k_arr=("shoot_per_10k_arr", "mean"))

    # regressions: does density survive controls, on the rate and on lethality-per-crime?
    for c in ["log_density", "nics_per_1k", "poverty_rate", "median_income_k"]:
        d["z_" + c] = (d[c] - d[c].mean()) / d[c].std()
    d["ly"] = np.log(d["shootings_per_100k_yr"])
    m_rate = smf.ols("ly ~ z_log_density + z_nics_per_1k + z_poverty_rate "
                     "+ z_median_income_k", d).fit()
    dv = d.dropna(subset=["shoot_per_1k_vc"]).copy()
    dv["lyv"] = np.log(dv["shoot_per_1k_vc"])
    m_leth = smf.ols("lyv ~ z_log_density + z_nics_per_1k", dv).fit()

    # sheriff hypothesis
    ds = d.dropna(subset=["sheriff_share"])
    r_sher_rate = np.corrcoef(ds["sheriff_share"], ds["shootings_per_100k_yr"])[0, 1]

    top = d.dropna(subset=["shoot_per_1k_vc"]).nlargest(6, "shoot_per_1k_vc")

    # persist tidy outputs for the HTML report (step 08)
    disp = tert.join(comp[["pct_gun", "pct_unarmed", "pct_flee", "pct_mental",
                           "pct_native", "pct_sheriff"]])
    disp.reset_index().rename(columns={"dens_tertile": "tertile"}).to_csv(
        PROCESSED / "rural_tertiles.csv", index=False)
    pd.DataFrame([{
        "density_coef_rate": m_rate.params["z_log_density"],
        "nics_coef_rate": m_rate.params["z_nics_per_1k"],
        "density_coef_leth": m_leth.params["z_log_density"],
        "leth_r2": m_leth.rsquared,
        "sheriff_r": r_sher_rate,
        "top_leth_states": ", ".join(top["state"]),
    }]).to_csv(PROCESSED / "rural_stats.csv", index=False)

    # ---- findings markdown ----
    b = []
    b.append("**The puzzle:** denser (more urban) states have *lower* per-capita fatal "
             "shooting rates (rate IRR ≈ 0.67 per +1 SD of log-density), even though, "
             "*within* any state, cities carry the violent crime. The resolution is that "
             "the gradient is a **lethality-per-encounter** gap, not a crime gap.\n")

    b.append("\n### 7a. Across states, sparse ≠ low-crime\n")
    b.append("Density tertiles (17 states each); the rate gap is ~2.5× on *similar* crime:\n")
    b.append("| Density tertile | Shootings /100k/yr | Violent crime rate | "
             "Per 1,000 violent crimes | Per 10k arrests |\n|---|---|---|---|---|")
    for t, row in tert.iterrows():
        b.append(f"| {t} | {row['rate']:.2f} | {row['vcr']:.0f} | "
                 f"{row['per_1k_vc']:.1f} | {row['per_10k_arr']:.2f} |")
    b.append("\nViolent crime is flat-to-higher in sparse states, yet they fatally shoot "
             "~2.3× more **per violent crime** and ~20% more **per arrest**: the excess is "
             "in how encounters end, not in how many crimes occur. The states topping "
             "shootings-per-crime are rural and *lower*-crime — "
             + ", ".join(top["state"]) + ".\n")

    b.append("\n### 7b. Density survives controls; guns wash out\n")
    b.append("OLS on log(rate) per +1 SD (N=51): density holds, gun prevalence (NICS) "
             "drops to ~0 once density/poverty/income are included.\n")
    b.append("| Predictor | log(rate) coef | log(shootings per crime) coef |\n|---|---|---|")
    for z, lbl in [("z_log_density", "Log density"), ("z_nics_per_1k", "Gun checks (NICS)")]:
        b.append(f"| {lbl} | {m_rate.params[z]:+.2f} | "
                 f"{m_leth.params.get(z, float('nan')):+.2f} |")
    b.append(f"| Poverty rate | {m_rate.params['z_poverty_rate']:+.2f} | — |")
    b.append(f"| Median income | {m_rate.params['z_median_income_k']:+.2f} | — |")
    b.append(f"\nThe lethality-per-crime model (R²={m_leth.rsquared:.2f}) keeps a density "
             "coefficient of "
             f"{m_leth.params['z_log_density']:+.2f}. So the rural excess is **not** simply "
             "'more guns in the country'.\n")

    b.append("\n### 7c. Incident composition explains little\n")
    b.append("Composition shifts modestly across the gradient (% of incidents):\n")
    b.append("| Density tertile | % gun-armed | % unarmed | % fleeing | % mental-health | "
             "% Native American |\n|---|---|---|---|---|---|")
    for t, row in comp.iterrows():
        b.append(f"| {t} | {row['pct_gun']:.0f} | {row['pct_unarmed']:.0f} | "
                 f"{row['pct_flee']:.0f} | {row['pct_mental']:.0f} | {row['pct_native']:.1f} |")
    b.append("\nSparse-state victims are only slightly more often gun-armed and fleeing, and "
             "*less* often unarmed or mental-health-flagged — composition cannot account for a "
             "2.5× gap. One real, under-discussed contributor: **Native Americans** are "
             f"{comp.loc['Sparse','pct_native']:.1f}% of sparse-state victims vs "
             f"{comp.loc['Dense','pct_native']:.1f}% in dense states, concentrated in the "
             "sparse West (NM, AK, AZ, OK).\n")

    b.append("\n### 7d. The 'rural sheriff' hypothesis is not supported\n")
    b.append("Linking each incident to its WaPo agency type, sparse-state shootings are "
             f"predominantly **municipal police**; the *sheriff* share is **lowest** in "
             f"sparse states ({comp.loc['Sparse','pct_sheriff']:.0f}%) vs mid-density "
             f"({comp.loc['Mid','pct_sheriff']:.0f}%), and sheriff-share by state correlates "
             f"only r={r_sher_rate:.2f} with the shooting rate. The gradient is not carried "
             "by sheriffs.\n")

    b.append("\n### 7e. What fits — and a fatal-only caveat\n")
    b.append("The signature (a lethality-per-encounter gap, not a crime or composition gap) "
             "is consistent with **how rural encounters are handled**: officer isolation and "
             "slow/absent backup pushing faster lethal-force decisions, fewer crisis-"
             "intervention/de-escalation resources, and higher gun-carry. These are "
             "plausible but **unmeasured** here. Crucially, WaPo records *fatal* shootings "
             "only: rural gunshot victims are far from trauma care, so a shooting is more "
             "likely to **prove fatal** — part of the rural excess in *fatal* shootings may "
             "be higher case-fatality rather than more shootings. Numerator-only data cannot "
             "separate the two. See `figures/rural_lethality.png`.\n")

    section("7. The rural paradox — lethality, not crime", "\n".join(b))
    log(f"Rural analysis: density coef on rate {m_rate.params['z_log_density']:+.2f}, "
        f"on lethality/crime {m_leth.params['z_log_density']:+.2f}; "
        f"sheriff-share r={r_sher_rate:.2f}.")


if __name__ == "__main__":
    main()
