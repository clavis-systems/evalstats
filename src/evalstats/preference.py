"""Pairwise-preference ("arena") evaluation.

Given records of head-to-head comparisons between models -- "A beat B", "B beat
A", "tie" -- estimate a skill rating per model with uncertainty, and say which
rating gaps are real.

The rating model is Bradley-Terry: ``P(i beats j) = sigmoid(r_i - r_j)`` where
``r`` are the (log-strength) ratings, centred to sum to zero. It is fitted by
the minorize-maximize iteration of Hunter (2004); ties are split as half a win
to each side; a small pseudo-count regularises models with a perfect or empty
record. Uncertainty comes from resampling the comparisons.

An order-dependent :func:`elo` is also provided for people who want it, but
Bradley-Terry is the one to report -- Elo depends on the order rows happen to be
in.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evalstats.stats import benjamini_hochberg, holm

__all__ = ["BTResult", "bradley_terry", "elo", "load_pairwise"]

_A_ALIASES = ("model_a", "model_1", "a", "left", "player_a")
_B_ALIASES = ("model_b", "model_2", "b", "right", "player_b")
_OUT_ALIASES = ("outcome", "winner", "result", "label", "vote")


def load_pairwise(path: str, *, fmt: str | None = None) -> pd.DataFrame:
    """Load head-to-head records into columns ``model_a, model_b, outcome``.

    ``outcome`` is normalised to ``"a"`` / ``"b"`` / ``"tie"``. Accepts a
    ``winner`` column holding the winning model's name, or an outcome column
    already using ``a``/``b``/``tie`` (plus ``left``/``right``/``draw`` and
    ``0``/``1``/``0.5`` synonyms).
    """
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        fmt = "csv" if ext in (".csv", ".tsv") else "json" if ext == ".json" else "jsonl"
    if fmt == "csv":
        sep = "\t" if path.lower().endswith(".tsv") else ","
        raw = pd.read_csv(path, sep=sep)
    elif fmt == "jsonl":
        raw = pd.read_json(path, lines=True)
    else:
        raw = pd.read_json(path)
    return _normalize_pairwise(raw)


def _pick(cols: dict[str, str], names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _normalize_pairwise(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    ca, cb, co = _pick(cols, _A_ALIASES), _pick(cols, _B_ALIASES), _pick(cols, _OUT_ALIASES)
    if ca is None or cb is None or co is None:
        raise ValueError(f"need columns for model A, model B and outcome; got {list(raw.columns)}")
    a = raw[ca].astype(str).str.strip()
    b = raw[cb].astype(str).str.strip()
    out_raw = raw[co].astype(str).str.strip()

    outcome = pd.Series(index=raw.index, dtype="object")
    low = out_raw.str.lower()
    outcome[low.isin(["a", "left", "1", "1.0", "win_a", "model_a"])] = "a"
    outcome[low.isin(["b", "right", "0", "0.0", "win_b", "model_b"])] = "b"
    outcome[low.isin(["tie", "draw", "0.5", "both", "equal"])] = "tie"
    # a bare model name in the outcome column means that model won
    name_is_a = out_raw.eq(a) & outcome.isna()
    name_is_b = out_raw.eq(b) & outcome.isna()
    outcome[name_is_a] = "a"
    outcome[name_is_b] = "b"

    if outcome.isna().any():
        bad = out_raw[outcome.isna()].unique()[:5]
        raise ValueError(f"could not read outcome value(s): {list(bad)}")

    df = pd.DataFrame({"model_a": a, "model_b": b, "outcome": outcome})
    same = df["model_a"] == df["model_b"]
    if same.any():
        df = df[~same].reset_index(drop=True)
    if df.empty:
        raise ValueError("no usable comparisons")
    return df


def _wins_matrix(df: pd.DataFrame, models: list[str]) -> np.ndarray:
    """``w[i, j]`` = credited wins of model i over model j (ties split)."""
    idx = {m: k for k, m in enumerate(models)}
    n = len(models)
    w = np.zeros((n, n), dtype=float)
    for a, b, o in zip(df["model_a"], df["model_b"], df["outcome"], strict=True):
        i, j = idx[a], idx[b]
        if o == "a":
            w[i, j] += 1.0
        elif o == "b":
            w[j, i] += 1.0
        else:  # tie
            w[i, j] += 0.5
            w[j, i] += 0.5
    return w


def _fit_bt(
    w: np.ndarray, *, prior: float = 0.1, max_iter: int = 1000, tol: float = 1e-9
) -> np.ndarray:
    """Minorize-maximize fit (Hunter 2004). Returns centred log-strength ratings."""
    n = w.shape[0]
    pair_seen = (w + w.T) > 0
    reg = prior * pair_seen  # pseudo-counts only between models that actually met
    wr = w + reg
    games = wr + wr.T
    wins = wr.sum(axis=1)

    p = np.ones(n, dtype=float)
    for _ in range(max_iter):
        denom = np.zeros(n, dtype=float)
        for i in range(n):
            mask = np.arange(n) != i
            denom[i] = np.sum(games[i, mask] / (p[i] + p[mask]))
        p_new = wins / np.where(denom > 0, denom, 1.0)
        p_new = p_new / p_new.sum() * n  # keep scale pinned
        if np.max(np.abs(np.log(p_new) - np.log(p))) < tol:
            p = p_new
            break
        p = p_new
    r = np.log(p)
    return r - r.mean()


@dataclass
class BTResult:
    ranking: pd.DataFrame  # model, rating, ci_low, ci_high, n_games, win_rate
    pairwise: (
        pd.DataFrame
    )  # model_a, model_b, rating_diff, ci_low, ci_high, p_raw, p_adj, significant
    level: float

    def verdict(self) -> str:
        top = self.ranking.iloc[0]
        sig = int(self.pairwise["significant"].sum()) if "significant" in self.pairwise else 0
        return (
            f"{len(self.ranking)} models, {int(self.ranking['n_games'].sum() // 2)} comparisons. "
            f"Top: {top['model']} (rating {top['rating']:+.3f}). "
            f"{sig}/{len(self.pairwise)} rating gaps are significant."
        )


def bradley_terry(
    df: pd.DataFrame,
    *,
    level: float = 0.95,
    prior: float = 0.1,
    n_boot: int = 2000,
    correction: str = "holm",
    seed: int = 0,
) -> BTResult:
    """Fit Bradley-Terry ratings with resampling-based uncertainty.

    ``df`` has columns ``model_a, model_b, outcome`` (``outcome`` in
    ``{"a", "b", "tie"}``) -- see :func:`load_pairwise`.
    """
    df = _normalize_pairwise(df)
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    if len(models) < 2:
        raise ValueError("need >= 2 models")
    idx = {m: k for k, m in enumerate(models)}

    w = _wins_matrix(df, models)
    ratings = _fit_bt(w, prior=prior)
    games_per_model = (w + w.T).sum(axis=1)
    wins_per_model = w.sum(axis=1)
    win_rate = np.divide(
        wins_per_model, games_per_model, out=np.full(len(models), np.nan), where=games_per_model > 0
    )

    rng = np.random.default_rng(seed)
    n = len(df)
    boot = np.full((n_boot, len(models)), np.nan)
    for b in range(n_boot):
        sample = df.iloc[rng.integers(0, n, size=n)]
        present = set(sample["model_a"]) | set(sample["model_b"])
        if len(present) < len(models):
            # keep replicate but only fill models that appear
            sub = sorted(present)
            wb = _wins_matrix(sample, sub)
            rb = _fit_bt(wb, prior=prior)
            for m, val in zip(sub, rb, strict=True):
                boot[b, idx[m]] = val
        else:
            boot[b] = _fit_bt(_wins_matrix(sample, models), prior=prior)

    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    ci_low = np.nanquantile(boot, lo_q, axis=0)
    ci_high = np.nanquantile(boot, hi_q, axis=0)

    ranking = (
        pd.DataFrame(
            {
                "model": models,
                "rating": ratings,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_games": games_per_model,
                "win_rate": win_rate,
            }
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    recs: list[dict] = []
    for a, bmod in ((x, y) for i, x in enumerate(models) for y in models[i + 1 :]):
        ia, ib = idx[a], idx[bmod]
        diff = ratings[ia] - ratings[ib]
        bd = boot[:, ia] - boot[:, ib]
        bd = bd[~np.isnan(bd)]
        if bd.size == 0:
            continue
        frac_pos = float(np.mean(bd > 0))
        p_raw = 2.0 * min(frac_pos, 1.0 - frac_pos)
        recs.append(
            {
                "model_a": a,
                "model_b": bmod,
                "rating_diff": diff,
                "ci_low": float(np.quantile(bd, lo_q)),
                "ci_high": float(np.quantile(bd, hi_q)),
                "p_raw": max(p_raw, 1.0 / (bd.size + 1)),
            }
        )
    pairwise = pd.DataFrame(recs)
    if not pairwise.empty:
        if correction == "holm":
            pairwise["p_adj"] = holm(pairwise["p_raw"].to_numpy())
        elif correction == "bh":
            pairwise["p_adj"] = benjamini_hochberg(pairwise["p_raw"].to_numpy())
        elif correction == "none":
            pairwise["p_adj"] = pairwise["p_raw"]
        else:
            raise ValueError(f"unknown correction {correction!r}")
        pairwise["significant"] = pairwise["p_adj"] < (1.0 - level)
        pairwise = pairwise.sort_values("p_adj").reset_index(drop=True)

    return BTResult(ranking=ranking, pairwise=pairwise, level=level)


def elo(
    df: pd.DataFrame, *, k: float = 32.0, initial: float = 1000.0, scale: float = 400.0
) -> pd.DataFrame:
    """Sequential Elo ratings, in row order. Order-dependent -- prefer
    :func:`bradley_terry` for anything you report."""
    df = _normalize_pairwise(df)
    r: dict[str, float] = {}
    games: dict[str, int] = {}
    for a, b, o in zip(df["model_a"], df["model_b"], df["outcome"], strict=True):
        ra, rb = r.get(a, initial), r.get(b, initial)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))
        sa = 1.0 if o == "a" else 0.0 if o == "b" else 0.5
        r[a] = ra + k * (sa - ea)
        r[b] = rb + k * ((1.0 - sa) - (1.0 - ea))
        games[a] = games.get(a, 0) + 1
        games[b] = games.get(b, 0) + 1
    out = pd.DataFrame(
        {"model": list(r), "elo": [r[m] for m in r], "n_games": [games[m] for m in r]}
    )
    return out.sort_values("elo", ascending=False).reset_index(drop=True)


def _bt_scale_to_elo(rating: float, *, anchor: float = 1000.0, scale: float = 400.0) -> float:
    """Convert a natural-log Bradley-Terry rating to an Elo-like number."""
    return anchor + rating * scale / math.log(10.0)
