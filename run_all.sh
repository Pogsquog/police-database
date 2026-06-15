#!/usr/bin/env bash
# Reproduce the full analysis end to end.
# Requires: uv, and a .env with CENSUS_API_KEY and DATA_GOV_API_KEY.
# External pulls are cached under data/raw/, so re-runs work offline.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 00 profile & clean WaPo data"
uv run python src/00_profile_clean.py
echo "==> 01 fetch Census demographics (ACS5)"
uv run python src/01_fetch_census.py
echo "==> 02 fetch FBI violent crime"
uv run python src/02_fetch_fbi.py
echo "==> 02b fetch FBI arrests by race (encounter proxy)"
uv run python src/02b_fetch_arrests.py
echo "==> 03 fetch contextual confounders (guns, alcohol, mental health, density)"
uv run python src/03_fetch_context.py
echo "==> 04 build state-year panel"
uv run python src/04_build_panel.py
echo "==> 05 rate & disparity models"
uv run python src/05_model_rates.py
echo "==> 06 within-incident models"
uv run python src/06_model_incident.py
echo "==> 07 figures, synthesis & limitations"
uv run python src/07_report.py
echo "==> 08 self-contained HTML report"
uv run python src/08_report_html.py

echo "Done. See findings.md, report.html and figures/."
