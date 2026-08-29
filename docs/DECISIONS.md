# Design decisions

Short architecture-decision records. Newest first. Each entry: the choice, why,
and what would make us revisit it.

## ADR-007 — Calibration tests are a release gate
The credibility of the whole package is "the p-values are honest". So
`tests/test_stats.py` includes Monte-Carlo checks that, under the null, the
permutation test rejects at ~α and the CLT interval covers at ~95%. These run in
CI on every push. A method without a calibration test does not ship.

## ADR-006 — v0.1 scope stops at question-level, correctness-based evals
No Bradley-Terry/Elo (preference arenas), no LLM-as-judge variance, no
training-seed variance. Each needs a different noise model and would dilute a
first release. Tracked as v0.2 in `README.md`. Revisit once the core is adopted.

## ADR-005 — MIT, dependency-light core, matplotlib behind an extra
Core install is `numpy`, `scipy`, `pandas`, `typer` only — these are already in
any eval environment. Plotting pulls `matplotlib`, which is heavier and not
needed for the numbers, so it sits behind `pip install evalstats[report]`.
MIT (not copyleft) to maximise adoption inside closed eval stacks.

## ADR-004 — Collapse the lm-eval-harness filter suffix on metric keys
Real `samples_*.jsonl` keys look like `acc,none` or
`exact_match,strict-match`. `_pick_metric` compares on the part before the
comma. When one metric is exposed under several filters, a `,none` filter wins
the tie; otherwise a bare metric name is ambiguous and raises, asking for the
full key. Keeps the common case zero-config without silently picking a filter.

## ADR-003 — Four-column tidy input contract
Everything downstream consumes `model, task, item_id, score`. `task` doubles as
the cluster id. Narrow and boring on purpose: any eval format can be adapted to
it in a few lines, and it is the natural upload schema if a hosted layer ever
happens. Column-name aliases are accepted at load time.

## ADR-002 — Cluster bootstrap is the default interval, CR1 is opt-in
Both are offered. The cluster bootstrap makes no assumption about the
within-cluster correlation structure and degrades gracefully; CR1 is a tidy
closed form but leans on `G` being not-tiny. Default to the robust one; let
users ask for CR1 when they want speed or a reported formula.

## ADR-001 — Sign-flip randomization test is the primary p-value
Chosen over (a) an unpaired t-test — ignores pairing, wastes power; (b) a paired
t-test — distributional assumption on per-item differences; (c) bootstrap-only —
no clean null. The sign-flip test is assumption-light, exact in expectation, and
matches McNemar for binary data (which we also report as a cross-check).
