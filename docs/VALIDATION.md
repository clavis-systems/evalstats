# Real-data validation

The synthetic example (`scripts/make_synthetic.py`) exercises the happy path.
This note records checks against **real** eval logs and what they turned up.

## Run 1 — 2026-08-29 — lm-evaluation-harness gsm8k samples

**Source:** `kuleshov-group/_dev_e2d2_lm_eval_gsm8k_zeroshot_cot_samples` on the
Hugging Face Hub (ungated), `samples.jsonl`, 24 lines. A real
`--log_samples` file for a GSM8K zero-shot-CoT run; the file concatenates three
runs over the same 8 documents.

**What the adapter had to handle in the real schema:**
- metric as a plain key `exact_match` (float), with the filter carried in a
  separate `"filter": "boxed-match"` field and a `"metrics": ["exact_match"]`
  list — *not* the `exact_match,boxed-match` comma form. Both forms are now
  covered.
- non-metric numeric fields (`doc_id`) and a numeric-looking `target` string —
  the allowlist in `_pick_metric` keeps these out of the score.
- `doc`, `arguments`, `resps` as nested dict/list — ignored correctly.

**Defects found and fixed:**

1. **Repeated `doc_id`s.** The file replays the same 8 docs three times.
   `from_lm_eval_harness` keyed `item_id` on `doc_id` alone, so the pivot in
   `pairwise` / `compare` would have silently averaged the repeats. Fixed: a
   per-`(model, task, doc_id)` counter disambiguates repeats as `7`, `7#2`,
   `7#3`. `load_results` now also warns on duplicate `(model, task, item_id)`.

2. **Single-cluster bootstrap collapsed the interval.** With only one `task`,
   the cluster bootstrap resamples the one cluster and every resample is
   identical, so `summarize` / `leaderboard` reported `SE = 0`, `CI = [x, x]`,
   and `pairwise` collapsed the paired CI to a point. Fixed: with fewer than two
   clusters, `summarize` / `leaderboard` fall back to a plain CLT estimate and
   `paired_difference` falls back to an item-level bootstrap, each with a
   warning. `clustered_mean_estimate("cluster-bootstrap")` now warns and returns
   the CLT interval in that case instead of a misleading zero.

**After the fixes:** treating the three runs as three models, `evalstats`
correctly reports run 1 == run 3 (diff 0, p = 1) and run 1 vs run 2
(3/8 vs 0/8) as *not* significant — diff 0.375, 95% CI [0.125, 0.750],
p = 0.25, 3 discordant pairs. Exactly the "big-looking gap, n far too small"
call the tool exists to make.

## Still wanted

- A clean **two-model, multi-task** real comparison (e.g. two Open LLM
  Leaderboard "details" datasets on the same tasks). Those repos are gated on
  the Hub, so this needs a Hugging Face token or a user-supplied log set.
- A generative task with genuine `metric,filter` comma keys and multiple
  filters, to exercise the ambiguity path end to end.
