# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims to
follow [Semantic Versioning](https://semver.org/) from 1.0.0 onward.

## [Unreleased]

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

[Unreleased]: https://github.com/your-username/evalstats/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/evalstats/releases/tag/v0.1.0
