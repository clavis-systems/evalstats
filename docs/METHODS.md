# Statistical methods in evalstats

This is the reference for what every number `evalstats` prints actually means.
None of the methods are new; the goal is that they are applied correctly and
consistently. The framing follows Miller (2024).

## The model: questions as a sample

An eval benchmark is treated as a random sample of `n` questions drawn from an
unseen *super-population* of questions we could have asked. A model's score on
the benchmark, `x̄ = (1/n) Σ xᵢ`, is therefore an estimate of its unknown
super-population mean `μ`, and it carries sampling error. Every interval below
answers: *if we had drawn a different sample of questions, how much would the
score move?*

This is why a bare leaderboard number is not enough, and why "model A beat model
B by 0.4 points on 500 questions" may or may not be a real difference.

## 1. One model, one number

### CLT interval (default)

For `n` scored items with sample standard deviation `s`,

```
SE(x̄) = s / sqrt(n)
CI     = x̄ ± z(1 - α/2) · SE
```

`z` is the standard-normal quantile. Valid by the Central Limit Theorem for `n`
larger than ~30; for pass/fail items `s² = p̂(1 - p̂)`. When the raw scores lie
in `[0, 1]` the interval is clipped to `[0, 1]`.

### Wilson interval (binary, small `n` or extreme `p̂`)

The CLT interval degrades when `p̂` is near 0 or 1 (it can even leave `[0, 1]`).
For binary scores `evalstats` offers the Wilson score interval, which stays
inside `[0, 1]` and has better coverage in those regimes:

```
center = (p̂ + z²/2n) / (1 + z²/n)
half   = (z / (1 + z²/n)) · sqrt( p̂(1-p̂)/n + z²/4n² )
```

## 2. Clustered uncertainty

Benchmark items rarely arrive independently. A suite has sub-tasks; a
reading-comprehension set has several questions per passage; a generated test
set has templates. Items within such a group are correlated, so treating all
`n` items as independent makes the interval **too narrow** — sometimes by a
large factor.

`evalstats` treats each `task` value as a cluster and offers two estimators.

### Cluster bootstrap (default)

Resample whole clusters with replacement `B` times; recompute the pooled
(micro-averaged) mean each time; take the `α/2` and `1 - α/2` quantiles of the
bootstrap distribution as the interval, and its standard deviation as the SE.
With `G` clusters, each resample draws `G` clusters. This makes no parametric
assumption about the correlation structure.

*Caveat:* with very few clusters (`G < ~8`) the cluster bootstrap is noisy and
conservative. That is a correct reflection of how little independent
information few clusters carry, but it means small suites give wide bars.

### CR1 analytic standard error

The cluster-robust ("sandwich") variance for the mean, with `eᵢ = xᵢ - x̄`:

```
Var(x̄) = ( G / (G-1) ) · ( 1 / n² ) · Σ_g ( Σ_{i in g} eᵢ )²
SE      = sqrt(Var(x̄))
```

`G / (G-1)` is the usual small-sample correction. Needs `G ≥ 2`.

## 3. Comparing two models

The two models are scored on the **same** items, so the comparison is paired.
Work with the per-item difference `dᵢ = aᵢ - bᵢ` and estimate `d̄`.

### Confidence interval

Bootstrap the mean of `d`: resample items (or clusters, when a cluster column is
available) with replacement `B` times and take the quantiles of `d̄*`. If the
interval contains 0, the gap is within noise.

### p-value: sign-flip randomization test

Under the null hypothesis "the two models are exchangeable on every item", the
sign of each `dᵢ` is equally likely to be `+` or `-`. So:

1. draw `sᵢ ∈ {-1, +1}` uniformly, `i = 1..n`;
2. compute the permuted statistic `t* = mean(sᵢ · dᵢ)`;
3. repeat `P` times;
4. `p = (1 + #{ |t*| ≥ |d̄| }) / (P + 1)` (two-sided).

Concordant pairs have `dᵢ = 0` and contribute nothing, as they should. The test
is *exact in expectation* — no distributional assumption — and the `+1`/`+1` in
the ratio keeps it valid for small `P`. This is the approximate-randomization
test discussed by Dror et al. (2018).

Validity condition: it assumes each `dᵢ` is, under the null, symmetric about 0.
For paired pass/fail scoring this is exactly the McNemar exchangeability
assumption. For graded scores it is a mild assumption but not free — if you
expect strongly skewed per-item differences, lean on the bootstrap interval.

### Exact McNemar (binary scores)

For pass/fail scores `evalstats` also reports the exact McNemar p-value. With
`b₁₀` items that `a` gets right and `b` wrong, and `b₀₁` the reverse, the
discordant count is `m = b₁₀ + b₀₁` and

```
p = 2 · P( Binomial(m, 1/2) ≤ min(b₁₀, b₀₁) )   (capped at 1)
```

It should track the sign-flip p-value closely; a large divergence is a signal to
look at the data.

## 4. Many models

Every unordered pair is compared with the paired test above, producing one raw
p-value per pair. Comparing `k` models gives `k(k-1)/2` tests, so `evalstats`
corrects:

* **Holm** (default) — controls the family-wise error rate (probability of *any*
  false positive). Sort p-values ascending; the `j`-th smallest is scaled by
  `(k_tests - j + 1)` and made monotone.
* **Benjamini-Hochberg** — controls the false discovery rate (expected
  *fraction* of false positives among the flagged pairs). Less strict; use it
  when you are screening many models and can tolerate some false flags.

## 5. Planning an experiment

### Target CI width

Items needed for a single-model accuracy CI of half-width `h`:

```
n ≈ z(1-α/2)² · p̂(1 - p̂) / h²
```

### Target detectable gap

Items needed to detect a paired gap of at least `Δ` (the minimum
detectable effect) with power `1 - β`:

```
n ≈ ( z(1-α/2) + z(1-β) )² · σ_d² / Δ²
```

`σ_d` is the standard deviation of the per-item difference. If you do not have
it, pass a rough shared accuracy `p` and `evalstats` uses the independence
approximation `σ_d = sqrt(2 p (1 - p))`; with nothing, it assumes a
conservative `σ_d = 0.5`.

## 6. Pairwise preference (arena)

When the data is head-to-head votes ("A beat B", "tie") rather than per-item
scores, `evalstats arena` fits a **Bradley-Terry** model:

```
P(i beats j) = sigmoid(r_i - r_j)
```

The ratings `r` (log-strength, centred to sum to zero) are fitted by the
minorize-maximize iteration of Hunter (2004) — a short, monotonically
convergent update that needs no optimiser. Ties are split as half a win to each
side. A small pseudo-count (`prior`, default 0.1) is added to every matchup that
actually occurred, so a model with a perfect or empty record still gets a finite
rating.

**Uncertainty** comes from resampling: the comparison rows are drawn with
replacement `n_boot` times and the model is refitted each time. Per-model
rating CIs are percentiles of that bootstrap distribution; each pairwise
rating gap gets a CI and a two-sided bootstrap p-value
`2·min(frac(Δ* > 0), frac(Δ* < 0))`, then Holm / BH correction across all
pairs. A gap whose CI spans 0 is within noise — the same reading as everywhere
else in the tool.

`evalstats.preference.elo` is also available but is order-dependent (it walks
the rows in sequence); Bradley-Terry is the one to report.

*Not modelled yet:* a Rao-Kupper / Davidson tie model (ties carry information
beyond "half a win"), position bias, and judge identity as a covariate.

## What evalstats does not do (yet)

* **LLM-as-judge variance** — when the "votes" come from a judge model, that
  judge adds its own variance (judge sampling, judge bias) on top of the
  comparison sampling that `arena` already handles. See Gao et al. (2025),
  `arXiv:2511.21140`. Planned for v0.2.
* **Multiple checkpoints / training seeds** — `evalstats` quantifies
  *evaluation* noise, not *training* noise. If you have several trained
  replicas, that variance must be added separately.
* **Non-random benchmarks** — if the benchmark is the entire population of
  interest (not a sample), sampling-based inference does not apply.

## References

* Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations.* `arXiv:2411.00640`.
* Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). *The Hitchhiker's
  Guide to Testing Statistical Significance in Natural Language Processing.* ACL.
* Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms.* Neural Computation.
* Brown, L. D., Cai, T. T., & DasGupta, A. (2001). *Interval Estimation for a
  Binomial Proportion.* Statistical Science. (Wilson interval.)
* Cameron, A. C., & Miller, D. L. (2015). *A Practitioner's Guide to
  Cluster-Robust Inference.* Journal of Human Resources.
* Bradley, R. A., & Terry, M. E. (1952). *Rank Analysis of Incomplete Block
  Designs: I. The Method of Paired Comparisons.* Biometrika.
* Hunter, D. R. (2004). *MM Algorithms for Generalized Bradley-Terry Models.*
  Annals of Statistics.
