"""Rough timing of the hot paths on a realistic-size synthetic dataset.

    python scripts/bench.py

Not a test -- just a sanity check that the resampling paths stay fast.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from evalstats import bradley_terry, leaderboard, paired_difference


def _timed(label: str, fn, *, repeat: int = 3) -> None:
    best = min(_run_once(fn) for _ in range(repeat))
    print(f"{label:<42} {best * 1e3:8.1f} ms")


def _run_once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _scores_frame(n_models: int, n_tasks: int, n_items: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.45, 0.8, n_models)
    rows = []
    for t in range(n_tasks):
        off = rng.normal(0, 0.1)
        diff = rng.normal(0, 0.15, n_items)
        for i in range(n_items):
            for m in range(n_models):
                p = np.clip(base[m] + off - diff[i], 0.02, 0.98)
                rows.append((f"m{m}", f"t{t}", f"t{t}-{i}", int(rng.random() < p)))
    return pd.DataFrame(rows, columns=["model", "task", "item_id", "score"])


def _arena_frame(n_models: int, n_per_pair: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.7, n_models)
    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            pa = 1 / (1 + np.exp(-(skill[i] - skill[j])))
            for _ in range(n_per_pair):
                r = rng.random()
                out = "tie" if r < 0.12 else ("a" if rng.random() < pa else "b")
                rows.append((f"m{i}", f"m{j}", out))
    return pd.DataFrame(rows, columns=["model_a", "model_b", "outcome"])


def main() -> None:
    print("scores: 8 models x 12 tasks x 200 items")
    df = _scores_frame(8, 12, 200)
    a = df[df["model"] == "m0"].set_index(["task", "item_id"])["score"]
    b = df[df["model"] == "m1"].set_index(["task", "item_id"])["score"]
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    _timed(
        "paired_difference (n=2400, 10k boot/perm)",
        lambda: paired_difference(
            pair["a"].to_numpy(),
            pair["b"].to_numpy(),
            clusters=pair.index.get_level_values("task").to_numpy(),
        ),
    )
    _timed("leaderboard (8 models, full pairwise)", lambda: leaderboard(df))

    print("\narena: 12 models x 200 comparisons/pair")
    ar = _arena_frame(12, 200)
    _timed("bradley_terry split (2k boot)", lambda: bradley_terry(ar, n_boot=2000))
    _timed(
        "bradley_terry rao-kupper (300 boot)",
        lambda: bradley_terry(ar, tie="rao-kupper", n_boot=300),
    )


if __name__ == "__main__":
    main()
