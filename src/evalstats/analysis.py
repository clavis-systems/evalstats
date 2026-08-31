"""DataFrame-level analyses built on top of :mod:`evalstats.stats`.

These functions take the tidy 4-column frame from :mod:`evalstats.loading`
(``model, task, item_id, score``) and return DataFrames ready to print or plot.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from evalstats.stats import (
    _row_chunks,
    benjamini_hochberg,
    clustered_mean_estimate,
    holm,
    mean_estimate,
    paired_difference,
)

__all__ = ["leaderboard", "pairwise_significance", "rank_probabilities", "summarize"]

OVERALL = "(overall)"


def _overall_estimate(mdf: pd.DataFrame, *, level: float, seed: int):
    """Per-model overall score across tasks.

    Cluster-robust (tasks as clusters) when there are >= 2 tasks; a plain CLT
    estimate when there is only one, because a single cluster carries no
    between-cluster information and the cluster bootstrap would then report a
    misleadingly tiny interval.
    """
    scores = mdf["score"].to_numpy()
    if mdf["task"].nunique() >= 2:
        return clustered_mean_estimate(scores, mdf["task"].to_numpy(), level=level, seed=seed)
    return mean_estimate(scores, level=level)


def summarize(df: pd.DataFrame, *, level: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Per-(model, task) means plus a per-model overall row (cluster-robust when
    the model has more than one task)."""
    rows: list[dict] = []
    for model, mdf in df.groupby("model", sort=True):
        for task, tdf in mdf.groupby("task", sort=True):
            est = mean_estimate(tdf["score"].to_numpy(), level=level)
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "n": est.n,
                    "mean": est.mean,
                    "se": est.se,
                    "ci_low": est.ci_low,
                    "ci_high": est.ci_high,
                }
            )
        overall = _overall_estimate(mdf, level=level, seed=seed)
        rows.append(
            {
                "model": model,
                "task": OVERALL,
                "n": overall.n,
                "mean": overall.mean,
                "se": overall.se,
                "ci_low": overall.ci_low,
                "ci_high": overall.ci_high,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["model", "task"]).reset_index(drop=True)


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """Items x models matrix, indexed by (task, item_id)."""
    return df.pivot_table(
        index=["task", "item_id"], columns="model", values="score", aggfunc="mean"
    )


def rank_probabilities(
    df: pd.DataFrame,
    *,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """For each model, the probability it holds each leaderboard rank once
    evaluation noise is accounted for.

    Cluster bootstrap over tasks: resample tasks with replacement, recompute
    every model's overall score on the pooled items, rank them, repeat. Returns
    one row per model with the observed ``mean`` and ``rank``, ``p_rank1``
    (probability it is truly best), ``p_top3``, the mean bootstrap rank and a
    rank CI. Models are ranked on the items they all share; anything outside
    that intersection is dropped with a warning.
    """
    full = _wide(df)
    wide = full.dropna()
    dropped = len(full) - len(wide)
    if dropped:
        warnings.warn(
            f"{dropped} of {len(full)} items are not shared by all models and are "
            "excluded from the ranking.",
            stacklevel=2,
        )
    models = list(wide.columns)
    if len(models) < 2 or wide.empty:
        raise ValueError("need >= 2 models evaluated on shared items")

    values = wide.to_numpy()
    tasks = wide.index.get_level_values("task").to_numpy()
    uniq = np.unique(tasks)
    n_tasks = uniq.size
    n_models = len(models)
    t_sum = np.stack([values[tasks == u].sum(axis=0) for u in uniq])  # (n_tasks, n_models)
    t_cnt = np.array([np.count_nonzero(tasks == u) for u in uniq], dtype=float)  # (n_tasks,)

    obs_mean = values.mean(axis=0)
    obs_rank = np.empty(n_models, dtype=int)
    obs_rank[(-obs_mean).argsort()] = np.arange(1, n_models + 1)

    rng = np.random.default_rng(seed)
    rank_counts = np.zeros((n_models, n_models))  # rank_counts[m, r-1]
    boot_ranks = np.empty((n_boot, n_models))
    filled = 0
    for size in _row_chunks(n_boot, n_tasks * n_models):
        picks = rng.integers(0, n_tasks, size=(size, n_tasks))
        means = t_sum[picks].sum(axis=1) / t_cnt[picks].sum(axis=1)[:, None]
        order = (-means).argsort(axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(1, n_models + 1), axis=1)
        boot_ranks[filled : filled + size] = ranks
        for r in range(1, n_models + 1):
            rank_counts[:, r - 1] += (ranks == r).sum(axis=0)
        filled += size

    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = pd.DataFrame(
        {
            "model": models,
            "mean": obs_mean,
            "rank": obs_rank,
            "p_rank1": rank_counts[:, 0] / n_boot,
            "p_top3": rank_counts[:, : min(3, n_models)].sum(axis=1) / n_boot,
            "mean_rank": boot_ranks.mean(axis=0),
            "rank_lo": np.quantile(boot_ranks, lo_q, axis=0),
            "rank_hi": np.quantile(boot_ranks, hi_q, axis=0),
        }
    )
    return out.sort_values(["p_rank1", "mean"], ascending=False).reset_index(drop=True)


def pairwise_significance(
    df: pd.DataFrame,
    *,
    level: float = 0.95,
    correction: str = "holm",
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparison of every unordered pair of models on their common items.

    correction : "holm" (family-wise), "bh" (false discovery rate) or "none".
    """
    wide = _wide(df)
    models = list(wide.columns)
    if len(models) < 2:
        raise ValueError("need >= 2 models to compare")

    recs: list[dict] = []
    for a, b in itertools.combinations(models, 2):
        pair = wide[[a, b]].dropna()
        if pair.empty:
            continue
        tasks = pair.index.get_level_values("task")
        # Clustering only means something with >= 2 tasks in common.
        clusters = tasks.to_numpy() if tasks.nunique() >= 2 else None
        res = paired_difference(
            pair[a].to_numpy(),
            pair[b].to_numpy(),
            clusters=clusters,
            level=level,
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
        )
        recs.append(
            {
                "model_a": a,
                "model_b": b,
                "n": res.n,
                "diff": res.diff,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_raw": res.p_permutation,
                "p_mcnemar": res.p_mcnemar,
                "n_discordant": res.n_discordant,
            }
        )
    out = pd.DataFrame(recs)
    if out.empty:
        return out

    if correction == "holm":
        out["p_adj"] = holm(out["p_raw"].to_numpy())
    elif correction == "bh":
        out["p_adj"] = benjamini_hochberg(out["p_raw"].to_numpy())
    elif correction == "none":
        out["p_adj"] = out["p_raw"]
    else:
        raise ValueError(f"unknown correction {correction!r}")

    alpha = 1.0 - level
    out["significant"] = out["p_adj"] < alpha
    return out.sort_values("p_adj").reset_index(drop=True)


def leaderboard(
    df: pd.DataFrame,
    *,
    level: float = 0.95,
    correction: str = "holm",
    seed: int = 0,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranked table (overall score, cluster-robust across tasks) and the
    pairwise matrix.

    Returns ``(table, pairs)``. ``table`` has one row per model with rank, mean,
    CI and n. ``pairs`` is :func:`pairwise_significance` output.
    """
    rows: list[dict] = []
    for model, mdf in df.groupby("model", sort=True):
        est = _overall_estimate(mdf, level=level, seed=seed)
        rows.append(
            {
                "model": model,
                "mean": est.mean,
                "se": est.se,
                "ci_low": est.ci_low,
                "ci_high": est.ci_high,
                "n": est.n,
            }
        )
    table = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    pairs = pairwise_significance(
        df,
        level=level,
        correction=correction,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    return table, pairs
