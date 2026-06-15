"""Entry point. The analysis lives in src/ and is orchestrated by run_all.sh.

Run the full reproducible pipeline with:

    ./run_all.sh

or individual stages, e.g.:

    uv run python src/05_model_rates.py
"""


def main():
    print(__doc__)


if __name__ == "__main__":
    main()
