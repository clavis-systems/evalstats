# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to
follow [Semantic Versioning](https://semver.org/) from 1.0.0 onward.

## [Unreleased]

### Changed
- Table / Markdown / HTML output shows a small nonzero value (a tiny p-value or
  CI bound) in scientific notation instead of rounding it to `0.0000`; an exact
  zero still prints `0.0000`.
- CLI input selection is now `--source auto | lm-eval | lighteval` (replaces the
  `--lm-eval` flag) on `summary` / `compare` / `leaderboard` / `report`.

### Added
- `from_lighteval` adapter for HuggingFace lighteval `details/` output (parquet
  via the new `lighteval` extra / `pyarrow`, or json/jsonl). Reads the per-row
  `metric` (older: `metrics`) dict, keeps `|` in task names, disambiguates
  repeated row ids. Exposed as `evalstats ... --source lighteval`.
- `from_openai_evals` adapter for OpenAI `evals` run logs (the JSONL event
  stream): model/task from the `spec` line, score from each sample's
  `match` / `metrics` event (`data.correct`, else `data.score` / a `metric`
  key), first scoring event per `sample_id` wins. `--source openai-evals`.
- Pairwise-preference (arena) support: `evalstats.preference` with
  `bradley_terry` (MM fit, bootstrap CIs on ratings and on every pairwise gap,
  Holm/BH correction), `elo` (order-dependent, provided but not recommended),
  and `load_pairwise`. New `evalstats arena` CLI command and
  `scripts/make_arena.py` / `examples/arena.csv`.

### Fixed
- Real-data validation against a live lm-evaluation-harness `samples.jsonl`
  (see `docs/VALIDATION.md`):
  - `from_lm_eval_harness` disambiguates repeated `doc_id`s (`7`, `7#2`, ...)
    when a log concatenates several runs, so paired comparison no longer merges
    them; `load_results` warns on duplicate `(model, task, item_id)` keys.
  - single-cluster inputs no longer collapse the interval to a point:
    `summarize` / `leaderboard` fall back to a plain CLT estimate,
    `paired_difference` to an item-level bootstrap, `clustered_mean_estimate`
    warns and returns the CLT interval — each with a warning.

## [0.1.0] - 2026-08-29

Initial release.

### Library
- `stats` primitives: `mean_estimate` (CLT / Wilson), `clustered_mean_estimate`
  (cluster bootstrap / CR1), `paired_difference` (sign-flip randomization test,
  cluster bootstrap CI, exact McNemar), `holm`, `benjamini_hochberg`,
  `sample_size_for_ci_halfwidth`, `sample_size_for_paired_detection`.
- `loading`: `load_results` (CSV / JSONL / JSON with column aliases and boolean
  coercion) and `from_lm_eval_harness` — reads `--log_samples` output including
  `metric,filter` keys (`acc,none`, `exact_match,strict-match`); a `,none`
  filter wins ties, an otherwise-ambiguous bare metric name raises with guidance.
- `analysis`: `summarize`, `leaderboard`, `pairwise_significance`.
- `report`: self-contained HTML report (`evalstats[report]` extra).

### CLI
- `summary`, `compare`, `leaderboard`, `power`, `report`, `version`.
- `--format table | csv | json | md` for `summary` and `leaderboard`.

### Project
- Calibration test suite (Monte-Carlo checks that the tests are honest);
  44 tests, ~92% coverage.
- GitHub Actions CI on Python 3.10-3.13 (ruff + pytest).
- `docs/METHODS.md`, `docs/DECISIONS.md`, `CONTRIBUTING.md`.
- `py.typed` marker; `.gitattributes` normalising line endings to LF.

[Unreleased]: https://github.com/clavis-systems/evalstats/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/clavis-systems/evalstats/releases/tag/v0.1.0
