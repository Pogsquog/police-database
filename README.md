# Police Shootings Database

Analysis of the [Washington Post Fatal Police Shootings dataset (2015–2024)](https://www.kaggle.com/datasets/ibrahimqasimi/wapo-fatal-police-shootings-2015-2024), covering fatal shootings by on-duty police officers in the United States.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- A [Kaggle account](https://www.kaggle.com) with an API token

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configure Kaggle credentials

1. Go to your [Kaggle account settings](https://www.kaggle.com/settings) and click **Create New Token**.
2. Move the downloaded `kaggle.json` to `~/.config/kaggle/kaggle.json`:

```bash
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Download the dataset

```bash
./download.sh
```

This downloads and extracts the dataset ZIP into the current directory.

### 5. Configure external-data API keys

The analysis joins external state-level confounders. Create a `.env` file in the
project root:

```bash
CENSUS_API_KEY=your_census_key   # free: https://api.census.gov/data/key_signup.html
DATA_GOV_API_KEY=your_datagov_key # free: https://api.data.gov/signup/ (or DEMO_KEY)
```

## Usage

Run the whole reproducible pipeline (clean → fetch → model → report):

```bash
./run_all.sh
```

Outputs:
- **`findings.md`** — running activity log + results (data profile, models, limitations).
- **`figures/`** — charts (trend, per-capita rates, IRR forest plot, density scatter).
- **`data/processed/`** — cleaned incidents and the assembled state-year panel.

External API pulls are cached under `data/raw/`, so re-runs are offline and
reproducible. Individual stages can be run on their own, e.g.:

```bash
uv run python src/05_model_rates.py
```

### What the analysis does

1. **Within-incident models** (cases only) — logistic regressions for whether a
   shooting involved an unarmed/fleeing/mentally-ill person or a body camera.
2. **Rate + confounder models** — a state-year panel (51 states × 10 years) joining
   Census demographics/poverty/income, FBI violent crime, gun background checks,
   alcohol & mental-distress prevalence, and population density. A Poisson rate model
   (population offset, cluster-robust SE) estimates which factors drive the per-capita
   shooting rate, and a disparity model tests whether the Black/White gap survives
   adjustment.
