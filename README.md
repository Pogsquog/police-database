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

## Usage

```bash
uv run python main.py
```
