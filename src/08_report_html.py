"""08 — Self-contained HTML report.

Renders a single portable `report.html` at the repo root from the processed
parquet/CSV outputs and the figures in `figures/`. Figures are embedded as
base64 so the file stands alone (email it, open it offline). No new data is
computed here — it presents the results produced by scripts 00–07.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import ROOT, PROCESSED, FIGURES, section, log

OUT = ROOT / "report.html"

# Human-readable labels for rate-model terms.
TERM_LABELS = {
    "pct_black": "% Black population",
    "pct_hispanic": "% Hispanic population",
    "violent_crime_rate": "Violent crime rate",
    "poverty_rate": "Poverty rate",
    "median_income_k": "Median household income",
    "nics_per_1k": "Gun background checks / 1k (NICS)",
    "alcohol_binge_pct": "Adult binge-drinking %",
    "mental_distress_pct": "Frequent mental distress %",
    "log_density": "Log population density",
    "is_south": "South (Census region)",
}


def b64_img(name: str) -> str:
    p = FIGURES / name
    if not p.exists():
        return ""
    enc = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{enc}"


def fig_block(name: str, caption: str) -> str:
    src = b64_img(name)
    if not src:
        return ""
    return (f'<figure><img src="{src}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def df_table(df: pd.DataFrame, *, classes: str = "") -> str:
    return df.to_html(index=False, border=0, classes=classes,
                      escape=False, justify="left")


def stat_cards(panel: pd.DataFrame, cs: pd.DataFrame) -> str:
    total = int(panel["shootings"].sum())
    n_states = panel["state"].nunique()
    yrs = f"{int(panel['year'].min())}–{int(panel['year'].max())}"
    top = cs.sort_values("shootings_per_100k_yr", ascending=False).iloc[0]
    cards = [
        ("Fatal shootings", f"{total:,}", f"WaPo, {yrs}"),
        ("State-years modelled", f"{len(panel):,}", f"{n_states} jurisdictions × 10 yrs"),
        ("Highest per-capita state", top["state"],
         f"{top['shootings_per_100k_yr']:.2f} per 100k/yr"),
        ("Black vs White rate ratio", "2.8×", "after full confounder adjustment"),
    ]
    items = "".join(
        f'<div class="card"><div class="card-val">{v}</div>'
        f'<div class="card-lab">{lab}</div><div class="card-sub">{sub}</div></div>'
        for lab, v, sub in cards
    )
    return f'<div class="cards">{items}</div>'


def rate_table(coefs: pd.DataFrame) -> str:
    rows = []
    for term, lab in TERM_LABELS.items():
        if term not in coefs.index:
            continue
        r = coefs.loc[term]
        sig = ("***" if r["p"] < .001 else "**" if r["p"] < .01
               else "*" if r["p"] < .05 else "")
        cls = "up" if r["irr"] > 1 else "down"
        bar = irr_bar(r["irr"], r["ci_low"], r["ci_high"])
        rows.append(
            f"<tr><td>{lab}</td>"
            f"<td class='num {cls}'>{r['irr']:.2f}</td>"
            f"<td class='num'>{r['ci_low']:.2f}–{r['ci_high']:.2f}</td>"
            f"<td class='num'>{r['p']:.3f}{sig}</td>"
            f"<td class='barcell'>{bar}</td></tr>"
        )
    return ("<table class='data irr'><thead><tr><th>Factor (per +1 SD)</th>"
            "<th>IRR</th><th>95% CI</th><th>p</th>"
            "<th>effect (IRR vs 1.0)</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def irr_bar(irr: float, lo: float, hi: float) -> str:
    """Tiny inline SVG forest bar; axis 0.5–1.6 (log), 1.0 marked."""
    lo_ax, hi_ax = 0.5, 1.6
    def x(v):
        v = min(max(v, lo_ax), hi_ax)
        return (np.log(v) - np.log(lo_ax)) / (np.log(hi_ax) - np.log(lo_ax)) * 100
    x1, xc, x2 = x(lo), x(irr), x(hi)
    one = x(1.0)
    col = "#b03a2e" if irr > 1 else "#1f4e79"
    return (
        f'<svg viewBox="0 0 100 14" preserveAspectRatio="none" class="bar">'
        f'<line x1="{one:.1f}" y1="0" x2="{one:.1f}" y2="14" stroke="#bbb" '
        f'stroke-dasharray="2,2"/>'
        f'<line x1="{x1:.1f}" y1="7" x2="{x2:.1f}" y2="7" stroke="{col}" '
        f'stroke-width="2"/>'
        f'<circle cx="{xc:.1f}" cy="7" r="2.6" fill="{col}"/></svg>'
    )


def arrest_benchmark_block() -> str:
    p = PROCESSED / "arrest_benchmark.csv"
    if not p.exists():
        return ""
    b = pd.read_csv(p).set_index("denominator")
    pop, arr = b.loc["population"], b.loc["arrests"]
    rows = "".join(
        f"<tr><td>{lab}</td><td class='num'>{r['rr']:.2f}×</td>"
        f"<td class='num'>{r['lo']:.2f}–{r['hi']:.2f}</td></tr>"
        for lab, r in [("Per resident (population)", pop),
                       ("Per arrest (encounter proxy)", arr)]
    )
    return f"""
  <h3 style="margin-top:1.4rem">Encounter benchmark — per arrest, not per resident</h3>
  <p>The per-resident rate cannot tell "more shootings" apart from "more police
  contact." Using FBI <strong>arrest totals</strong> as an exposure denominator
  (Black/White only — FBI arrest race carries no Hispanic ethnicity) re-frames the
  question as <em>shootings per police contact</em>.</p>
  <table class='data'><thead><tr><th>Denominator</th><th>Black/White RR</th>
  <th>95% CI</th></tr></thead><tbody>{rows}</tbody></table>
  <div class="panel callout">
    Benchmarking against arrests shrinks the disparity from
    <strong>{pop['rr']:.1f}×</strong> to <strong>{arr['rr']:.1f}×</strong>. Most of
    the per-resident gap reflects that Black Americans are arrested / come into
    police contact at far higher per-capita rates; a residual
    <strong>{arr['rr']:.1f}×</strong> remains even per arrest.
    <br/><span class="muted">Caveat: arrests are themselves a policing output and
    can absorb upstream disparities in who is stopped or arrested. This is a
    narrower, contested benchmark — not a "truer" number than the per-resident
    rate. The two answer different questions.</span>
  </div>"""


def ladder_block() -> str:
    p = PROCESSED / "offending_ladder.csv"
    if not p.exists():
        return ""
    d = pd.read_csv(p)
    mx = d["black_white_ratio"].max()
    rows = ""
    for _, r in d.iterrows():
        ratio = r["black_white_ratio"]
        w = ratio / mx * 100
        col = "#7d3c98" if "Homicide" in r["benchmark"] else (
            "#b03a2e" if "shooting" in r["benchmark"].lower() else "#1f4e79")
        rows += (
            f"<tr><td>{r['benchmark']}</td>"
            f"<td class='barcell'><div class='hbar' style='width:{w:.0f}%;"
            f"background:{col}'>{ratio:.1f}×</div></td>"
            f"<td class='muted' style='font-size:.78rem'>{r['note']}</td></tr>")
    return f"""
  <h3 style="margin-top:1.6rem">Does the arrest gap reflect offending or enforcement?</h3>
  <p>The arrest benchmark can't tell higher offending apart from heavier policing —
  arrests are a policing output. <strong>Homicide</strong> is the least discretionary
  crime (almost always recorded; ~80–90% intra-racial), so the Black/White
  homicide-<em>victimization</em> ratio proxies involvement in lethal violence
  <em>without</em> passing through police discretion. The same disparity, measured
  against progressively more exposure-aware (but more policing-entangled)
  denominators:</p>
  <table class='data ladder'><tbody>{rows}</tbody></table>
  <div class="panel callout">
    The involvement proxy (<strong>6.7×</strong>) is <em>larger</em> than the
    per-resident shooting disparity (<strong>2.6×</strong>) or the arrest disparity
    (<strong>2.0×</strong>). So relative to involvement in serious violence, Black
    arrest and shooting rates are <strong>not inflated</strong> — if anything they sit
    below it. That argues against "police/states target Black people <em>regardless of
    crime</em>" as a description of <em>serious</em> violence.
    <br/><br/><span class="muted"><strong>Three caveats keep this open.</strong>
    (1) Homicide is the wrong exposure for many shootings (traffic stops, mental-health
    calls, low-level offenses), so benchmarking all shootings against it overstates
    "justified" exposure. (2) Total arrests blend serious crime (which tracks
    involvement) with low-level/drug offenses, where enforcement disparities are well
    documented to exceed offending — the ~2× average hides that. (3) These are
    ecological aggregates describing the country, not any single encounter; the choice
    of denominator <em>brackets</em> the disparity rather than pinning it — population
    over-states it by ignoring exposure, arrests/homicide under-state it by baking in
    any upstream bias.</span>
  </div>"""


def rural_block() -> str:
    """The rural paradox: sparse states have HIGHER rates — a lethality-per-
    encounter gap, not a crime gap. Reads tidy outputs from step 05b."""
    tp = PROCESSED / "rural_tertiles.csv"
    sp = PROCESSED / "rural_stats.csv"
    if not (tp.exists() and sp.exists()):
        return ""
    t = pd.read_csv(tp)
    s = pd.read_csv(sp).iloc[0]

    exp_rows = "".join(
        f"<tr><td>{r['tertile']}</td>"
        f"<td class='num'>{r['rate']:.2f}</td>"
        f"<td class='num'>{r['vcr']:.0f}</td>"
        f"<td class='num'>{r['per_1k_vc']:.1f}</td>"
        f"<td class='num'>{r['per_10k_arr']:.2f}</td></tr>"
        for _, r in t.iterrows())
    comp_rows = "".join(
        f"<tr><td>{r['tertile']}</td>"
        f"<td class='num'>{r['pct_gun']:.0f}</td>"
        f"<td class='num'>{r['pct_unarmed']:.0f}</td>"
        f"<td class='num'>{r['pct_flee']:.0f}</td>"
        f"<td class='num'>{r['pct_mental']:.0f}</td>"
        f"<td class='num'>{r['pct_native']:.1f}</td>"
        f"<td class='num'>{r['pct_sheriff']:.0f}</td></tr>"
        for _, r in t.iterrows())
    native_sparse = t.loc[t["tertile"] == "Sparse", "pct_native"].iloc[0]
    native_dense = t.loc[t["tertile"] == "Dense", "pct_native"].iloc[0]
    sher_sparse = t.loc[t["tertile"] == "Sparse", "pct_sheriff"].iloc[0]
    sher_mid = t.loc[t["tertile"] == "Mid", "pct_sheriff"].iloc[0]

    return f"""
  <h3 style="margin-top:1.6rem">The rural paradox — why sparse states rank highest</h3>
  <p>The negative density gradient is counterintuitive: shouldn't cities, with more
  violent crime, see more police shootings? They do <em>within</em> a state — but
  <strong>across</strong> states, sparse ones are not low-crime, so the gap is a
  <strong>lethality-per-encounter</strong> gap, not a crime gap.</p>
  <div class="grid2">
    {fig_block('rural_lethality.png', 'Fatal shootings per 1,000 violent crimes fall steeply with density.')}
    <div class="panel">
      <h4>Rate vs exposure, by density tertile</h4>
      <table class='data'><thead><tr><th>Tertile</th><th>Per 100k/yr</th>
      <th>Violent crime</th><th>Per 1k crimes</th><th>Per 10k arrests</th></tr></thead>
      <tbody>{exp_rows}</tbody></table>
      <p class="muted" style="margin:.5rem 0 0">Violent crime is flat-to-higher in
      sparse states, yet they fatally shoot ~2.3× more per violent crime and ~20%
      more per arrest.</p>
    </div>
  </div>
  <h4 style="margin-top:1.2rem">Incident composition explains little (% of incidents)</h4>
  <table class='data'><thead><tr><th>Tertile</th><th>% gun-armed</th><th>% unarmed</th>
  <th>% fleeing</th><th>% mental-health</th><th>% Native Am.</th><th>% sheriff</th>
  </tr></thead><tbody>{comp_rows}</tbody></table>
  <div class="panel callout">
    Density survives every control (log-rate coef {s['density_coef_rate']:+.2f}; gun
    prevalence washes out to ~0), and the lethality-per-crime model
    (R²={s['leth_r2']:.2f}) keeps a density coef of {s['density_coef_leth']:+.2f} — so it
    is <strong>not</strong> simply "more guns in the country." The
    <strong>"rural sheriff" hypothesis fails</strong>: sparse-state shootings are mostly
    municipal police, sheriff share is <em>lowest</em> there ({sher_sparse:.0f}% vs
    {sher_mid:.0f}% mid-density), correlating only r={s['sheriff_r']:.2f} with the rate.
    <strong>Native Americans</strong> ({native_sparse:.1f}% of sparse-state victims vs
    {native_dense:.1f}% dense) are a real, concentrated contributor.
    <br/><br/><span class="muted">What fits the signature: officer isolation / slow
    backup and thinner de-escalation resources — plausible but unmeasured here. And a
    <strong>fatal-only artifact</strong>: rural gunshot wounds are far from trauma care,
    so a shooting is likelier to <em>prove fatal</em>; part of the rural excess in
    <em>fatal</em> shootings may be higher case-fatality, not more shootings.</span>
  </div>"""


def medaccess_block() -> str:
    """The distance-to-medical-care test: trauma-access-sensitive accident deaths
    (car, falls) track the shooting rate; the overdose negative control does not."""
    p = PROCESSED / "rural_medaccess.csv"
    if not p.exists():
        return ""
    c = pd.read_csv(p)
    rows = ""
    for _, r in c.iterrows():
        cls = "up" if r["r_rate"] > 0.2 else ("down" if r["r_rate"] < 0 else "")
        rows += (f"<tr><td>{r['measure']}</td>"
                 f"<td class='num {cls}'>{r['r_rate']:+.2f}</td>"
                 f"<td class='num'>{r['r_leth']:+.2f}</td>"
                 f"<td class='num'>{r['r_density']:+.2f}</td>"
                 f"<td class='muted' style='font-size:.78rem'>{r['kind']}</td></tr>")
    mv = c.set_index("measure").loc["Motor vehicle (car)", "r_rate"]
    od = c.set_index("measure").loc["Drug overdose (negative control)", "r_rate"]
    return f"""
  <h3 style="margin-top:1.6rem">Is it distance to medical care? A placebo test</h3>
  <p>WaPo records <em>fatal</em> shootings only, and a gunshot is likelier to
  <strong>prove fatal</strong> where trauma care is far. We can't measure shooting
  case-fatality directly, but we can test the mechanism with other injuries: if medical
  access matters, accident deaths that are <strong>trauma-access-sensitive</strong> (car
  crashes, falls) should track the shooting rate, while <strong>drug overdose</strong> —
  whose lethality is about addiction, not trauma-centre distance — should not.</p>
  <div class="grid2">
    {fig_block('rural_medaccess.png', 'Correlation with the state shooting rate, by accident type — trauma-sensitive deaths track it; overdose (placebo) does not.')}
    <div class="panel">
      <h4>Cross-state correlations (N≈51)</h4>
      <table class='data'><thead><tr><th>Accident-death rate</th><th>vs rate</th>
      <th>vs lethality</th><th>vs density</th><th>Trauma-access?</th></tr></thead>
      <tbody>{rows}</tbody></table>
    </div>
  </div>
  <div class="panel callout">
    The pattern is what the medical-access theory predicts: car-crash mortality
    correlates <strong>{mv:+.2f}</strong> with the shooting rate, but the non-trauma
    negative control — overdose — is <strong>{od:+.2f}</strong> (null/negative), despite
    overdose also being high in poor areas. So it is not a generic "deadly places"
    signal; it is specific to injury-deaths that <em>fast trauma care</em> governs.
    <br/><br/><span class="muted">Consistent-with, not proof: per-capita accident deaths
    blend more accidents (rural driving exposure) with more-lethal accidents
    (case-fatality), and these are ecological state-level correlations. But the overdose
    placebo rules out a purely generic rural-risk explanation.</span>
  </div>"""


def disparity_table() -> str:
    rows = [
        ("Black", "2.65 (2.30–3.06)", "2.82 (2.43–3.28)", "Not explained"),
        ("Hispanic", "1.24", "1.21", "Roughly unchanged"),
    ]
    body = "".join(
        f"<tr><td>{g}</td><td class='num'>{u}</td><td class='num'>{a}</td>"
        f"<td>{note}</td></tr>" for g, u, a, note in rows
    )
    return ("<table class='data'><thead><tr><th>Group (vs White)</th>"
            "<th>Unadjusted RR</th><th>Adjusted RR</th><th>Verdict</th>"
            "</tr></thead><tbody>" + body + "</tbody></table>")


def incident_tables() -> str:
    """Re-fit the four incident logits so the HTML carries the same ORs."""
    import statsmodels.formula.api as smf
    inc = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    races = ["White", "Black", "Hispanic", "Asian", "Native American"]
    d = inc[inc["race_label"].isin(races)].copy()
    d = d[d["gender"].isin(["male", "female"])]
    d = d[d["age"].notna() & d["year"].notna()]
    d["race_label"] = pd.Categorical(d["race_label"], races)
    d["age10"] = d["age"] / 10.0
    d["female"] = (d["gender"] == "female").astype(int)
    d["body_camera"] = d["body_camera"].astype(int)
    d["was_mental_illness_related"] = d["was_mental_illness_related"].astype(int)

    outcomes = {
        "unarmed": "Victim unarmed",
        "body_camera": "Body camera present",
        "was_mental_illness_related": "Mental-illness-related",
        "fleeing": "Was fleeing",
    }
    labels = {
        "C(race_label, Treatment('White'))[T.Black]": "Black (vs White)",
        "C(race_label, Treatment('White'))[T.Hispanic]": "Hispanic (vs White)",
        "C(race_label, Treatment('White'))[T.Asian]": "Asian (vs White)",
        "C(race_label, Treatment('White'))[T.Native American]": "Native Am. (vs White)",
        "age10": "Age (+10 yrs)", "female": "Female", "is_south": "South region",
    }
    blocks = []
    for outcome, title in outcomes.items():
        sub = d[d[outcome].notna()].copy()
        sub[outcome] = sub[outcome].astype(int)
        f = (f"{outcome} ~ C(race_label, Treatment('White')) + age10 + female + "
             f"is_south + C(year)")
        res = smf.logit(f, data=sub).fit(disp=0)
        rows = []
        for term, lab in labels.items():
            if term not in res.params:
                continue
            orr = np.exp(res.params[term])
            lo, hi = np.exp(res.conf_int().loc[term])
            p = res.pvalues[term]
            sig = ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "")
            cls = "up" if orr > 1 else "down"
            rows.append(
                f"<tr><td>{lab}</td><td class='num {cls}'>{orr:.2f}</td>"
                f"<td class='num'>{lo:.2f}–{hi:.2f}</td>"
                f"<td class='num'>{p:.3f}{sig}</td></tr>")
        blocks.append(
            f"<div class='panel'><h4>{title} "
            f"<span class='muted'>N={len(sub):,} · base {sub[outcome].mean():.1%}</span></h4>"
            "<table class='data'><thead><tr><th>Factor</th><th>OR</th>"
            "<th>95% CI</th><th>p</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")
    return "<div class='grid2'>" + "".join(blocks) + "</div>"


def bodycam_confound_block() -> str:
    """Re-fit body-camera logit with state then agency FE to show the Black OR
    is mostly between-agency confounding."""
    import statsmodels.formula.api as smf
    inc = pd.read_parquet(PROCESSED / "incidents_clean.parquet")
    races = ["White", "Black", "Hispanic", "Asian", "Native American"]
    d = inc[inc["race_label"].isin(races)].copy()
    d = d[d["gender"].isin(["male", "female"])]
    d = d[d["age"].notna() & d["year"].notna()]
    d["race_label"] = pd.Categorical(d["race_label"], races)
    d["age10"] = d["age"] / 10.0
    d["female"] = (d["gender"] == "female").astype(int)
    d["body_camera"] = d["body_camera"].astype(int)
    d["agency"] = d["agency_ids"].astype(str).str.split(";").str[0]
    t = "C(race_label, Treatment('White'))[T.Black]"
    base = "body_camera ~ C(race_label, Treatment('White')) + age10 + female + C(year)"
    # agency FE only identified on agencies with within-agency camera variation
    var_ag = d.groupby("agency")["body_camera"].transform("nunique") > 1
    agency_sub = d[var_ag]
    specs = [("Base (race + age + sex + year)", base, d),
             ("+ State fixed effects", base + " + C(state)", d),
             ("…same agencies, no agency FE", base, agency_sub),
             ("+ Agency fixed effects", base + " + C(agency)", agency_sub)]
    rows = []
    for label, f, data in specs:
        try:
            res = smf.logit(f, data=data).fit(disp=0, maxiter=200)
            orr = np.exp(res.params[t]); lo, hi = np.exp(res.conf_int().loc[t])
            cls = "up" if orr > 1 else "down"
            rows.append(f"<tr><td>{label}</td><td class='num {cls}'>{orr:.2f}</td>"
                        f"<td class='num'>{lo:.2f}–{hi:.2f}</td><td class='num'>"
                        f"{int(res.nobs):,}</td></tr>")
        except Exception:
            continue
    return f"""
  <h3 style="margin-top:1.4rem">Why the body-camera gap is large — it's the department</h3>
  <p>The raw Black-vs-White body-camera OR (~1.9) looks alarming, but body-worn
  cameras are an <strong>agency-level program</strong>: large urban departments
  adopted them earliest and most, and Black victims are disproportionately shot by
  those same departments. Adding fixed effects isolates within-unit comparisons.</p>
  <table class='data'><thead><tr><th>Adjustment</th><th>Black/White OR</th>
  <th>95% CI</th><th>N</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  <div class="panel callout">
    State fixed effects barely move the OR (it is <em>not</em> a between-state
    effect), but <strong>agency fixed effects collapse it from ~1.9 to ~1.3</strong>.
    Most of the apparent gap is <em>which department was involved</em>, not
    differential camera use against individuals within a department. A modest
    residual (~1.3×) remains within the same agency.
  </div>"""


def state_table(cs: pd.DataFrame) -> str:
    d = cs.sort_values("shootings_per_100k_yr", ascending=False).copy()
    d = d[["state", "shootings", "shootings_per_100k_yr", "violent_crime_rate",
           "pct_black", "log_density", "is_south"]]
    d.columns = ["State", "Shootings (10y)", "Per 100k/yr", "Violent crime/100k",
                 "% Black", "log(density)", "South"]
    d["Per 100k/yr"] = d["Per 100k/yr"].map("{:.2f}".format)
    d["Violent crime/100k"] = d["Violent crime/100k"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.0f}")
    d["% Black"] = (d["% Black"] * 100).map("{:.1f}".format)
    d["log(density)"] = d["log(density)"].map("{:.2f}".format)
    d["South"] = d["South"].map({1: "Yes", 0: "—"})
    return df_table(d, classes="data sortable")


def build_html() -> str:
    panel = pd.read_parquet(PROCESSED / "state_year_panel.parquet")
    cs = pd.read_parquet(PROCESSED / "state_cross_section.parquet")
    coefs = pd.read_csv(PROCESSED / "rate_model_coefs.csv", index_col=0)

    yearly = panel.groupby("year")["shootings"].sum().reset_index()
    yearly.columns = ["Year", "Shootings"]
    yearly["Year"] = yearly["Year"].astype(int)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Factors Influencing US Police Shootings — WaPo 2015–2024</title>
<style>{CSS}</style></head>
<body>
<header class="hero">
  <div class="wrap">
    <p class="eyebrow">Reproducible analysis · generated {ts}</p>
    <h1>Factors Influencing US Fatal Police Shootings</h1>
    <p class="lede">A state-year panel and within-incident analysis of the
    <em>Washington Post</em> Fatal Police Shootings database (2015–2024),
    adjusted for crime, poverty, income, gun availability, alcohol, mental
    health, density and region.</p>
    {stat_cards(panel, cs)}
  </div>
</header>

<nav class="toc"><div class="wrap">
  <a href="#data">Data</a><a href="#trend">Trend</a>
  <a href="#geography">Geography</a><a href="#rate">Rate model</a>
  <a href="#disparity">Disparity</a><a href="#incident">Within-incident</a>
  <a href="#findings">Findings</a><a href="#limits">Limitations</a>
</div></nav>

<main class="wrap">

<section id="data">
  <h2>1 · The data</h2>
  <p>The WaPo file is <strong>numerator-only</strong>: it records
  {int(panel['shootings'].sum()):,} fatal shootings but no population at risk.
  To estimate <em>rates</em> and adjust for confounders we built a state-year
  panel ({len(panel):,} rows = 51 jurisdictions × 10 years) joining external
  denominators and covariates: Census ACS5 demographics, FBI violent crime,
  NICS gun background checks, CDC binge-drinking and mental-distress prevalence,
  and Census land area for population density.</p>
  <div class="grid2">
    {fig_block('trend.png', 'Fatal police shootings per year — a steady rise from ~995 to ~1,175.')}
    <div class="panel">
      <h4>Shootings by year</h4>
      {df_table(yearly, classes='data')}
    </div>
  </div>
</section>

<section id="geography">
  <h2>2 · Geography dominates the per-capita rate</h2>
  <p>Per-capita rates vary roughly <strong>10-fold</strong> across states.
  The highest are sparse Western states (NM, AK, OK, CO, AZ); the lowest are
  dense Northeastern ones (RI, MA, CT, NY, NJ). Population density is the single
  strongest correlate — denser, more urban states have <em>lower</em> rates.</p>
  <div class="grid2">
    {fig_block('state_rates.png', 'Annual fatal shootings per 100k by state (2015–2024 average).')}
    {fig_block('density_scatter.png', 'Population density vs shooting rate — a strong negative gradient.')}
  </div>
  {rural_block()}
  {medaccess_block()}
  <details><summary>Full state table ({len(cs)} jurisdictions)</summary>
  {state_table(cs)}
  </details>
</section>

<section id="rate">
  <h2>3 · Rate model — what predicts a state's shooting rate?</h2>
  <p>Poisson GLM of shooting counts with a <code>log(population)</code> offset,
  year fixed effects and cluster-robust standard errors by state (N=478
  state-years). Continuous factors are scaled per +1 SD, so incidence-rate
  ratios (IRRs) are directly comparable. IRR &gt; 1 means a higher rate.</p>
  {rate_table(coefs)}
  <div class="grid2" style="margin-top:1.2rem">
    {fig_block('rate_forest.png', 'Rate-model IRRs with 95% confidence intervals.')}
    <div class="panel callout">
      <h4>Reading the model</h4>
      <ul>
        <li><strong>Density (IRR 0.67)</strong> — the dominant, highly
        significant effect: urbanicity, not racial composition, drives the rate.</li>
        <li><strong>Violent crime (1.16)</strong> and <strong>mental distress
        (1.15)</strong> are positively associated.</li>
        <li><strong>% Black population (0.96, n.s.)</strong> — a state's racial
        composition does <em>not</em> predict its overall rate once crime and
        density are controlled.</li>
        <li>Guns (NICS), poverty, income and binge-drinking show no robust
        independent association at the state level.</li>
      </ul>
    </div>
  </div>
  <p class="muted">Pearson overdispersion ≈ 2.84; a Negative-Binomial refit
  leaves coefficients materially unchanged. VIFs all &lt; 6.</p>
</section>

<section id="disparity">
  <h2>4 · The racial disparity survives adjustment</h2>
  <p>Stacking counts by state × race with group-specific population offsets, we
  test whether the Black-vs-White rate ratio shrinks after adjusting for state
  violent crime, poverty, income, density and guns. <strong>It does not.</strong></p>
  {disparity_table()}
  <div class="panel callout warn">
    Black Americans are fatally shot at about <strong>2.6×</strong> the White
    per-capita rate. Adjusting for the measured state-level confounders does not
    explain the gap — the adjusted ratio is <strong>2.8×</strong>. This is an
    ecological, place-level association, not an individual-level causal estimate.
  </div>
  {arrest_benchmark_block()}
  {ladder_block()}
</section>

<section id="incident">
  <h2>5 · Within-incident models <span class="muted">(cases only)</span></h2>
  <p>Among people fatally shot, what is associated with an incident's
  characteristics? Logistic regression with <strong>White</strong> as the race
  reference, male the gender reference, age per +10 years, and year fixed
  effects. These are <em>conditional on having been shot</em> — they describe the
  character of shootings, not the risk of being shot.</p>
  {incident_tables()}
  {bodycam_confound_block()}
</section>

<section id="findings">
  <h2>6 · Key findings</h2>
  <ol class="findings">
    <li><strong>Geography/urbanicity dominates the per-capita rate.</strong>
    Density has IRR ≈ 0.67 per +1 SD; sparse Western states rank highest. The
    counterintuitive rural excess is a <em>lethality-per-encounter</em> gap, not a
    crime gap — sparse states aren't low-crime, yet they fatally shoot ~2.3× more per
    violent crime; the "rural sheriff" explanation does not hold. A placebo test points
  partly to distance-to-medical-care: the rate tracks trauma-access-sensitive accident
  deaths (car crashes, falls) but not the overdose negative control.</li>
    <li><strong>Violent crime and population mental-distress</strong> are
    positively associated with the state rate (IRR ≈ 1.16 and 1.15).</li>
    <li><strong>A state's Black population share does not predict its overall
    rate</strong> once crime and density are controlled — the rate story is
    regional, not racial-composition driven.</li>
    <li><strong>The Black/White disparity depends entirely on the denominator.</strong>
    Per resident it is ≈2.6× (unchanged by state confounders); per arrest ≈1.3×; and
    against homicide-victimization (an offending proxy) Black involvement is ≈6.7× —
    <em>larger</em> than the shooting gap. So the per-resident gap mostly tracks
    police-contact exposure, and relative to serious-violence involvement the shooting
    rate is not inflated — though that conclusion is bracketed, not settled (see §4).</li>
    <li><strong>Guns and binge-drinking</strong> show no robust independent
    state-level association; poverty/income wash out once crime and density
    are included.</li>
    <li><strong>Within incidents:</strong> Black victims are modestly more often
    unarmed (OR 1.34) and fleeing (OR 1.21); White victims' shootings are far
    more often flagged mental-illness-related (Black/Hispanic OR ≈ 0.56). The large
    raw body-camera gap (OR ~1.9) is mostly <em>agency</em> confounding — it falls
    to ~1.3× once department is held fixed.</li>
  </ol>
</section>

<section id="limits">
  <h2>7 · Limitations <span class="muted">(read before citing)</span></h2>
  <ul class="limits">
    <li><strong>Numerator-only + ecological.</strong> State-level associations
    can suffer ecological fallacy and describe <em>places</em>, not encounters.</li>
    <li><strong>Exposure denominator is imperfect.</strong> The main model is per
    resident; the arrest benchmark proxies police contact but arrests are
    themselves a policing output and may absorb upstream disparities — neither
    denominator is definitive.</li>
    <li><strong>FBI 2021 coverage gap.</strong> NIBRS-transition under-reporting
    forced nulling of low-coverage crime cells; year FE absorb the national dip.</li>
    <li><strong>Proxy covariates.</strong> Gun availability uses NICS checks
    (2015–2023, carried forward); alcohol/mental-health from BRFSS have gaps
    filled by state means. These attenuate toward null.</li>
    <li><strong>Missing race</strong> (11% of incidents) and self-reported
    survey covariates add noise.</li>
    <li><strong>Associational, not causal.</strong> No causal claims are made.</li>
  </ul>
</section>

</main>
<footer><div class="wrap">
  <p>Generated by <code>src/08_report_html.py</code> from the reproducible
  pipeline (<code>./run_all.sh</code>). Source: Washington Post Fatal Police
  Shootings database; Census ACS5; FBI Crime Data Explorer (crime + arrests);
  FBI/NICS; CDC BRFSS; CDC WONDER (homicide mortality); CDC NCHS
  (accident/motor-vehicle/overdose mortality).
  Full activity log in <code>findings.md</code>.</p>
</div></footer>
</body></html>
"""


CSS = """
:root{--ink:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--blue:#1f4e79;
--red:#b03a2e;--bg:#f7f7f5;--card:#fff}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:980px;margin:0 auto;padding:0 1.2rem}
.hero{background:linear-gradient(160deg,#11243a,#1f4e79);color:#fff;
padding:3rem 0 2rem}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:.72rem;
opacity:.75;margin:0 0 .4rem}
.hero h1{font-size:2.1rem;line-height:1.15;margin:.1rem 0 .6rem}
.lede{font-size:1.05rem;max-width:62ch;opacity:.92;margin:0 0 1.6rem}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem}
.card{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.18);
border-radius:10px;padding:.9rem}
.card-val{font-size:1.5rem;font-weight:700;line-height:1.1}
.card-lab{font-size:.82rem;opacity:.9;margin-top:.2rem}
.card-sub{font-size:.72rem;opacity:.65}
.toc{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);
z-index:10;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.toc .wrap{display:flex;flex-wrap:wrap;gap:.2rem}
.toc a{padding:.7rem .7rem;color:var(--muted);text-decoration:none;
font-size:.85rem;font-weight:600}
.toc a:hover{color:var(--blue)}
main{padding:1rem 0 3rem}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:1.4rem 1.6rem;margin:1.4rem 0}
h2{font-size:1.45rem;margin:.2rem 0 .8rem;border-bottom:2px solid var(--line);
padding-bottom:.4rem}
h4{margin:.2rem 0 .6rem;font-size:.95rem}
.muted{color:var(--muted);font-weight:400;font-size:.9em}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;align-items:start}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:8px;
padding:.6rem;overflow:hidden}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:.8rem;color:var(--muted);margin-top:.5rem}
.panel{background:#fff;border:1px solid var(--line);border-radius:8px;
padding:.9rem 1rem}
.callout{background:#f3f7fc;border-color:#cfe0f2}
.callout.warn{background:#fcf3f1;border-color:#f0cfc8}
.callout ul{margin:.3rem 0 0;padding-left:1.1rem}
.callout li{margin:.35rem 0}
table.data{border-collapse:collapse;width:100%;font-size:.86rem;margin:.3rem 0}
table.data th,table.data td{text-align:left;padding:.4rem .55rem;
border-bottom:1px solid var(--line)}
table.data thead th{background:#f3f4f6;font-size:.78rem;text-transform:uppercase;
letter-spacing:.03em;color:#374151}
table.data td.num{text-align:right;font-variant-numeric:tabular-nums}
td.up{color:var(--red);font-weight:700}
td.down{color:var(--blue);font-weight:700}
table.irr .barcell{width:170px}
table.ladder td{vertical-align:middle}
table.ladder td:first-child{font-weight:600;width:38%}
.hbar{color:#fff;font-weight:700;font-size:.8rem;padding:.25rem .5rem;
border-radius:4px;white-space:nowrap;min-width:2.6rem;text-align:right}
svg.bar{width:160px;height:14px}
details{margin-top:1rem}
summary{cursor:pointer;font-weight:600;color:var(--blue);padding:.4rem 0}
.findings{padding-left:1.2rem}.findings li{margin:.5rem 0}
.limits{padding-left:1.1rem}.limits li{margin:.4rem 0}
footer{border-top:1px solid var(--line);background:#fff;padding:1.4rem 0;
color:var(--muted);font-size:.82rem}
code{background:#eef0f2;padding:.05rem .3rem;border-radius:4px;font-size:.85em}
@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}
.grid2{grid-template-columns:1fr}.hero h1{font-size:1.6rem}}
"""


def main():
    html = build_html()
    OUT.write_text(html)
    size_kb = OUT.stat().st_size / 1024
    log(f"Wrote {OUT} ({size_kb:.0f} KB, self-contained).")
    section("10. HTML report",
            f"Rendered self-contained `report.html` ({size_kb:.0f} KB) with "
            f"embedded figures and result tables. Open it in any browser; "
            f"regenerate with `uv run python src/08_report_html.py`.")


if __name__ == "__main__":
    main()
