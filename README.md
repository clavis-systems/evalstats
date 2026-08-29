# evalstats

**Significance-aware comparison of language-model evaluation results.**

Most eval reports show a single number per model. `evalstats` takes your
per-item results and answers the question that number can't:

> Is model A *actually* better than model B, or is the gap within noise?

It reads the result formats people already produce, applies standard statistics
(bootstrap confidence intervals, paired randomization tests, cluster-robust
standard errors, multiple-comparison correction), and gives you a
publication-ready table plus a one-line verdict.

```text
$ evalstats compare examples/results.jsonl --a model_a --b model_c
model_a vs model_c: difference -0.1133 (95% CI [-0.1408, -0.0817]), p=0.0001 < 0.05. model_c is significantly better.
  paired items      : 1200
  discordant pairs  : 396
  McNemar exact p   : 7.33e-12
  bootstrap method  : cluster-bootstrap

$ evalstats compare examples/results.jsonl --a model_a --b model_b
model_a vs model_b: difference +0.0008 (95% CI [-0.0233, +0.0242]), p=1 >= 0.05. Within noise -- no significant difference.
```

> **Status:** pre-1.0, built in the open. The statistics are textbook; the
> contribution is packaging them into one command that plugs into real eval
> pipelines. Bug reports and format requests very welcome.

## Install

```bash
pip install -e ".[dev,report]"   # from a clone
```

Core needs only `numpy`, `scipy`, `pandas`, `typer`. The HTML report adds
`matplotlib` (the `report` extra).

## Input format

A CSV or JSON Lines file with one row per (model, item):

| column    | meaning                                             |
|-----------|----------------------------------------------------|
| `model`   | system under test                                  |
| `task`    | benchmark / sub-task the item belongs to (cluster) |
| `item_id` | stable id, shared across models                     |
| `score`   | `0`/`1` for pass-fail, or any real number           |

Common aliases are recognised (`system`, `benchmark`, `doc_id`, `exact_match`,
booleans, `"correct"`/`"incorrect"`, ...). lm-evaluation-harness `--log_samples`
output is read directly with `--lm-eval`.

```bash
python scripts/make_synthetic.py        # writes examples/results.jsonl
```

## Commands

| command | what you get |
|---------|--------------|
| `evalstats summary R`                | per-(model, task) mean, SE, CI, n; plus a cluster-robust per-model overall |
| `evalstats compare R --a M1 --b M2`  | paired difference, bootstrap CI, permutation p-value, exact McNemar, verdict |
| `evalstats leaderboard R`            | models ranked by cluster-robust score + pairwise significance matrix (Holm / BH) |
| `evalstats power ci --p 0.7 --half-width 0.02`     | items needed for a target CI width |
| `evalstats power paired --mde 0.02 --p-pooled 0.7` | items needed to detect a given gap at 80% power |
| `evalstats report R -o report.html`  | self-contained HTML report (needs `report` extra) |

`R` is your results file. Add `--lm-eval` (and optionally `--metric acc`) for
lm-evaluation-harness sample logs.

## Methods

* **Confidence intervals** — CLT normal approximation by default; Wilson score
  for binary scores near 0 or 1.
* **Clustered uncertainty** — when items come in related groups (sub-tasks,
  documents), a cluster bootstrap (default) or analytic CR1 standard error, so
  the interval isn't falsely narrow.
* **Comparing two models** — inference on per-item paired differences: a
  two-sided sign-flip randomization test, a cluster-aware bootstrap CI, and an
  exact McNemar p-value for binary scores.
* **Many models** — every pair compared, then Holm (family-wise) or
  Benjamini-Hochberg (false discovery rate) correction.
* **Planning** — sample-size formulas for a target CI width or a target
  detectable gap.

Full write-up with formulas and validity conditions: [`docs/METHODS.md`](docs/METHODS.md).
Design rationale: [`docs/DECISIONS.md`](docs/DECISIONS.md).

References: Miller (2024), *Adding Error Bars to Evals*, `arXiv:2411.00640`;
Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical Significance
in NLP*.

## Roadmap

- [ ] v0.2: Bradley-Terry / Elo for pairwise-preference arenas
- [ ] v0.2: LLM-as-judge with annotator variance (cf. `arXiv:2511.21140`)
- [ ] adapters for `lighteval` and OpenAI `evals` output
- [ ] `--format md` for drop-in paper tables

## Development

```bash
python -m pip install -e ".[dev,report]"
python -m pytest --cov=evalstats   # calibration + unit tests
ruff check . && ruff format --check .
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — new estimators must ship with a
calibration test.

## License

MIT

