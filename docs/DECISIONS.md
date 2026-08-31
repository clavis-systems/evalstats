# Design decisions

Short architecture-decision records. Newest first. Each entry: the choice, why,
and what would make us revisit it.

## ADR-012 — Vectorised resampling, identical numbers
The bootstrap / permutation / MM-fit inner loops are vectorised over the
resample axis (cluster-sum lookups instead of `concatenate`; a broadcast MM
update instead of a per-model Python loop; `bincount` on integer-coded rows for
the arena bootstrap instead of re-scanning the DataFrame). The RNG is drawn in
the same order as before, so results are **bit-identical for a given seed** —
this is a speed/memory change, not a statistical one, and the calibration tests
are unchanged. Large temp arrays are chunked to stay under ~8M cells.
`scripts/bench.py` times the hot paths.

## ADR-011 — Rao-Kupper ties: opt-in, MLE, bootstrap for theta
`bradley_terry(tie="split")` stays the default (fast MM fit, matches Chatbot
Arena's basic BT). `tie="rao-kupper"` adds the Rao-Kupper (1967) tie model,
fitted by ML with `scipy.optimize` (already a dependency) — no new closed-form
SEs, `theta` gets a bootstrap CI like everything else. It is opt-in because it
runs an optimiser per bootstrap resample (slower) and because `theta` is
unidentified without ties (in that case it warns and falls back to `"split"`).
Chose Rao-Kupper over Davidson (1970) as the more commonly cited of the two;
Davidson could be a later `tie=` value.

## ADR-010 — `--source` flag, one metric-picker, pyarrow behind an extra
Input adapters (`lm-eval`, `lighteval`, `openai-evals`) are selected with
`--source` rather than a flag per runner, so adding the next one is a new enum
value, not a new option on four commands. The adapters share `_pick_metric`,
which reads a metric map keyed either plainly (`exact_match`) or with a filter
suffix (`acc,none`), prefers a `,none` filter on ties, and raises on a genuinely
ambiguous bare name. `lighteval` details are parquet, so `pyarrow` is an
optional `lighteval` extra (like `matplotlib` for `report`); the JSON details
variant needs nothing. Task names keep their `|` from the lighteval filename;
repeated row ids are disambiguated like the lm-eval adapter (ADR-... / the
repeated-doc_id fix).

## ADR-009 — Bradley-Terry via MM, uncertainty via row bootstrap
For arena data, fit BT with Hunter's (2004) minorize-maximize iteration rather
than pulling in an optimiser or a new dependency (`choix`, `scipy.optimize`):
it is ~10 lines, monotonic, and always converges. Ties are split 50/50 for v0.2
(a Davidson/Rao-Kupper tie model is noted as future work). Uncertainty is a
nonparametric bootstrap over the comparison rows — consistent with how the rest
of the tool gets its CIs, and it sidesteps BT's fragile closed-form standard
errors near a separated dataset. A small pseudo-count keeps perfect records
finite. `elo()` is included because people ask for it, but flagged
order-dependent and not the thing to report.

## ADR-008 — Degrade gracefully when there is only one cluster
A single `task` (one cluster) gives the cluster bootstrap nothing to resample,
so it reported `SE = 0` / a point CI — actively misleading. Found during
real-data validation (`docs/VALIDATION.md`). Now: `summarize` / `leaderboard`
fall back to a plain CLT estimate, `paired_difference` falls back to an
item-level bootstrap, and `clustered_mean_estimate` warns and returns the CLT
interval. A warning fires so the fallback is never silent. Revisit if we add a dedicated
single-cluster method (e.g. subsampling within the cluster).

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
