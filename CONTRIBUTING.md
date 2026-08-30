# Contributing to evalstats

Thanks for looking. This is an early project and feedback — especially "your
adapter choked on my real eval logs" — is the most useful thing right now.

## Development setup

```bash
git clone https://github.com/clavis-systems/evalstats
cd evalstats
python -m pip install -e ".[dev,report]"
python scripts/make_synthetic.py     # writes examples/results.jsonl
python -m pytest
```

## Before opening a PR

```bash
ruff check .
ruff format --check .
python -m pytest --cov=evalstats
```

CI runs exactly these on Python 3.10–3.13.

## Ground rules for statistical code

1. **A new estimator or test ships with a calibration test.** Not just "it runs"
   — a Monte-Carlo check that it behaves correctly under a known truth (interval
   coverage near the nominal level, or false-positive rate near α under the
   null). See `tests/test_stats.py` for the pattern.
2. **Cite the method.** A one-line reference in the docstring and, if it is a
   headline method, an entry in `docs/METHODS.md`.
3. **Keep the core dependency-light.** `numpy` / `scipy` / `pandas` / `typer`
   only. Anything heavier goes behind an optional extra.
4. **Record non-obvious design choices** in `docs/DECISIONS.md`.

## Adding an input adapter

Adapters live in `loading.py` and must return the tidy frame
(`model, task, item_id, score`) via `from_records`. Add a test with a small
fixture that mimics the real format, including its quirks.

## Commit messages

Short imperative subject ("add lighteval adapter"), body explains *why* if it is
not obvious. No strict convention beyond that.
