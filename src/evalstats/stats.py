"""Core statistical primitives.

Everything in this module takes plain array-likes and returns small dataclasses
or numpy arrays. No pandas, no I/O -- so it is easy to test in isolation and
easy to reuse.

Conventions
-----------
* ``level`` is the confidence level for intervals (0.95 -> 95% CI).
* ``alpha`` for a hypothesis test is ``1 - level``.
* ``seed`` makes every randomized procedure reproducible.
* "score" is per item: 0/1 for pass-fail evals, or any real number for graded
  metrics. Binary-only procedures check and raise if given non-binary input.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps

__all__ = [
    "MeanEstimate",
    "PairedResult",
    "benjamini_hochberg",
    "clustered_mean_estimate",
    "holm",
    "mean_estimate",
    "paired_difference",
    "sample_size_for_ci_halfwidth",
    "sample_size_for_paired_detection",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    a = a[~np.isnan(a)]
    if a.size == 0:
        raise ValueError("no (non-NaN) values")
    return a


def _is_binary(a: np.ndarray) -> bool:
    return bool(np.all(np.isin(a, (0.0, 1.0))))


_MAX_RESAMPLE_CELLS = 8_000_000  # cap the temp (rows x n) array in resampling loops


def _row_chunks(n_rows: int, n_cols: int):
    """Yield chunk sizes that keep each temp array under ~_MAX_RESAMPLE_CELLS."""
    step = max(1, _MAX_RESAMPLE_CELLS // max(n_cols, 1))
    for start in range(0, n_rows, step):
        yield min(step, n_rows - start)


def _z(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    return float(_sps.norm.ppf(0.5 + level / 2.0))


def _maybe_clip(lo: float, hi: float, a: np.ndarray) -> tuple[float, float]:
    """Clip an interval to [0, 1] when the data itself lives in [0, 1]."""
    if a.min() >= 0.0 and a.max() <= 1.0:
        return max(lo, 0.0), min(hi, 1.0)
    return lo, hi


# --------------------------------------------------------------------------- #
# one-sample: mean of a set of scores
# --------------------------------------------------------------------------- #
@dataclass
class MeanEstimate:
    mean: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    level: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0


def mean_estimate(scores, *, level: float = 0.95, method: str = "clt") -> MeanEstimate:
    """Point estimate and confidence interval for the mean score.

    method
    ------
    ``"clt"``    normal approximation, ``se = s / sqrt(n)`` (Miller 2024, sec. 3).
    ``"wilson"`` Wilson score interval; binary scores only, better near 0 or 1.
    """
    a = _clean(scores)
    n = a.size
    mean = float(a.mean())
    z = _z(level)

    if method == "clt":
        se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        lo, hi = mean - z * se, mean + z * se
        lo, hi = _maybe_clip(lo, hi, a)
    elif method == "wilson":
        if not _is_binary(a):
            raise ValueError("method='wilson' requires binary 0/1 scores")
        p = mean
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lo, hi = center - half, center + half
        se = float(math.sqrt(p * (1 - p) / n))
    else:
        raise ValueError(f"unknown method {method!r}")

    return MeanEstimate(mean, se, float(lo), float(hi), n, method, level)


# --------------------------------------------------------------------------- #
# clustered: items arrive in related groups (e.g. sub-tasks, documents)
# --------------------------------------------------------------------------- #
def clustered_mean_estimate(
    scores,
    clusters,
    *,
    level: float = 0.95,
    method: str = "cluster-bootstrap",
    n_boot: int = 10_000,
    seed: int | None = None,
) -> MeanEstimate:
    """Mean score with a cluster-robust interval.

    When benchmark questions come in related groups, treating every question as
    independent understates the uncertainty (Miller 2024, sec. 4). Both methods
    here account for within-cluster correlation.

    method
    ------
    ``"cluster-bootstrap"``  resample whole clusters with replacement.
    ``"cr1"``                analytic CR1 cluster-robust standard error.
    """
    a = np.asarray(scores, dtype=float).ravel()
    g = np.asarray(clusters).ravel()
    if a.shape != g.shape:
        raise ValueError("scores and clusters must have the same length")
    keep = ~np.isnan(a)
    a, g = a[keep], g[keep]
    if a.size == 0:
        raise ValueError("no (non-NaN) values")

    uniq = np.unique(g)
    n_clusters = uniq.size
    n = a.size
    mean = float(a.mean())
    z = _z(level)

    if method == "cr1":
        if n_clusters < 2:
            raise ValueError("need >= 2 clusters for cr1")
        resid = a - mean
        ssq = 0.0
        for cid in uniq:
            ssq += float(resid[g == cid].sum()) ** 2
        var = (n_clusters / (n_clusters - 1)) * ssq / (n * n)
        se = math.sqrt(var)
        lo, hi = mean - z * se, mean + z * se
        lo, hi = _maybe_clip(lo, hi, a)
    elif method == "cluster-bootstrap":
        if n_clusters < 2:
            warnings.warn(
                "clustered_mean_estimate got fewer than 2 clusters; a single "
                "cluster carries no clustering information, so returning a plain "
                "CLT interval instead.",
                stacklevel=2,
            )
            return mean_estimate(a, level=level)
        rng = np.random.default_rng(seed)
        # mean of a cluster resample = sum of picked cluster sums / sum of sizes
        c_sum = np.array([a[g == cid].sum() for cid in uniq])
        c_size = np.array([np.count_nonzero(g == cid) for cid in uniq], dtype=float)
        picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        boot = c_sum[picks].sum(axis=1) / c_size[picks].sum(axis=1)
        se = float(boot.std(ddof=1))
        lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
        lo, hi = _maybe_clip(float(lo), float(hi), a)
    else:
        raise ValueError(f"unknown method {method!r}")

    return MeanEstimate(mean, float(se), float(lo), float(hi), n, f"clustered:{method}", level)


# --------------------------------------------------------------------------- #
# two-sample paired: same items, two models
# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    diff: float  # mean(a) - mean(b), positive => a better
    ci_low: float
    ci_high: float
    p_permutation: float
    p_mcnemar: float | None  # binary scores only, else None
    n: int
    n_discordant: int | None  # binary scores only
    level: float
    method: str

    @property
    def significant(self) -> bool:
        """Significant at alpha = 1 - level, by the primary (permutation) test."""
        return self.p_permutation < (1.0 - self.level)

    def verdict(self, model_a: str = "A", model_b: str = "B") -> str:
        alpha = 1.0 - self.level
        gap = f"{self.diff:+.4f} (95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}])"
        if self.significant:
            better = model_a if self.diff > 0 else model_b
            return (
                f"{model_a} vs {model_b}: difference {gap}, p={self.p_permutation:.3g} "
                f"< {alpha:g}. {better} is significantly better."
            )
        return (
            f"{model_a} vs {model_b}: difference {gap}, p={self.p_permutation:.3g} "
            f">= {alpha:g}. Within noise -- no significant difference."
        )


def paired_difference(
    a,
    b,
    *,
    clusters=None,
    level: float = 0.95,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int | None = None,
) -> PairedResult:
    """Compare two models scored on the *same* items.

    * Confidence interval: bootstrap over items (or over clusters, if given) of
      the per-item difference ``a_i - b_i``.
    * p-value: two-sided sign-flip randomization test -- valid whenever, under
      the null, swapping which model produced which score leaves the joint
      distribution unchanged (Dror et al. 2018). Concordant pairs contribute a
      zero difference and drop out automatically.
    * For binary scores an exact McNemar p-value is added for free.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length (paired)")
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = a.size
    if n == 0:
        raise ValueError("no paired observations")

    d = a - b
    obs = float(d.mean())
    rng = np.random.default_rng(seed)

    use_clusters = clusters is not None
    if use_clusters:
        g = np.asarray(clusters).ravel()[keep]
        if np.unique(g).size < 2:
            warnings.warn(
                "paired_difference got fewer than 2 distinct clusters; "
                "falling back to an item-level bootstrap for the interval.",
                stacklevel=2,
            )
            use_clusters = False

    # --- bootstrap CI -----------------------------------------------------
    if use_clusters:
        uniq = np.unique(g)
        n_clusters = uniq.size
        c_sum = np.array([d[g == cid].sum() for cid in uniq])
        c_size = np.array([np.count_nonzero(g == cid) for cid in uniq], dtype=float)
        picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        boot = c_sum[picks].sum(axis=1) / c_size[picks].sum(axis=1)
    else:
        boot = np.empty(n_boot, dtype=float)
        filled = 0
        for size in _row_chunks(n_boot, n):
            idx = rng.integers(0, n, size=(size, n))
            boot[filled : filled + size] = d[idx].mean(axis=1)
            filled += size
    ci_low, ci_high = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])

    # --- sign-flip randomization p-value --------------------------------
    threshold = abs(obs) - 1e-12
    exceed = 0
    for size in _row_chunks(n_perm, n):
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        exceed += int(np.count_nonzero(np.abs((signs * d).mean(axis=1)) >= threshold))
    p_perm = (1.0 + exceed) / (n_perm + 1.0)

    # --- McNemar (binary only) -----------------------------------------
    p_mcnemar: float | None = None
    n_disc: int | None = None
    if _is_binary(a) and _is_binary(b):
        b10 = int(np.count_nonzero((a == 1) & (b == 0)))
        b01 = int(np.count_nonzero((a == 0) & (b == 1)))
        n_disc = b10 + b01
        if n_disc == 0:
            p_mcnemar = 1.0
        else:
            p_mcnemar = float(_sps.binomtest(min(b10, b01), n_disc, 0.5).pvalue)

    return PairedResult(
        diff=obs,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_permutation=float(p_perm),
        p_mcnemar=p_mcnemar,
        n=n,
        n_discordant=n_disc,
        level=level,
        method="cluster-bootstrap" if use_clusters else "bootstrap",
    )


# --------------------------------------------------------------------------- #
# multiple-comparison correction
# --------------------------------------------------------------------------- #
def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls false discovery rate)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------- #
# experiment planning (Miller 2024, sec. 5)
# --------------------------------------------------------------------------- #
def sample_size_for_ci_halfwidth(p_est: float, half_width: float, *, level: float = 0.95) -> int:
    """Items needed so a single-model accuracy CI has the given half-width."""
    if not 0.0 <= p_est <= 1.0:
        raise ValueError("p_est must be in [0, 1]")
    if half_width <= 0.0:
        raise ValueError("half_width must be > 0")
    z = _z(level)
    return math.ceil(z * z * p_est * (1.0 - p_est) / (half_width * half_width))


def sample_size_for_paired_detection(
    mde: float,
    *,
    sd_diff: float | None = None,
    p_pooled: float | None = None,
    level: float = 0.95,
    power: float = 0.80,
) -> int:
    """Items needed to detect a paired accuracy gap of ``mde`` with given power.

    Provide either ``sd_diff`` (std. dev. of the per-item difference a_i - b_i)
    or ``p_pooled`` (both models' rough accuracy, used as an independence-based
    estimate ``sd_diff = sqrt(2 p (1 - p))``). With neither, a conservative
    ``sd_diff = 0.5`` is assumed.
    """
    if mde <= 0.0:
        raise ValueError("mde must be > 0")
    if sd_diff is None:
        if p_pooled is not None:
            if not 0.0 <= p_pooled <= 1.0:
                raise ValueError("p_pooled must be in [0, 1]")
            sd_diff = math.sqrt(max(2.0 * p_pooled * (1.0 - p_pooled), 1e-12))
        else:
            sd_diff = 0.5
    z_a = _z(level)
    z_b = float(_sps.norm.ppf(power))
    return math.ceil(((z_a + z_b) ** 2) * (sd_diff**2) / (mde * mde))
