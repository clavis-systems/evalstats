"""Bradley-Terry / arena tests, including recovery of a known truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evalstats.preference import bradley_terry, elo, load_pairwise


def _simulate(skill: dict[str, float], n_per_pair: int, tie_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = list(skill)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            p_a = 1.0 / (1.0 + np.exp(-(skill[a] - skill[b])))
            for _ in range(n_per_pair):
                if rng.random() < tie_rate:
                    rows.append((a, b, "tie"))
                else:
                    rows.append((a, b, "a" if rng.random() < p_a else "b"))
    return pd.DataFrame(rows, columns=["model_a", "model_b", "outcome"])


@pytest.fixture
def arena_df():
    return _simulate({"top": 1.0, "mid": 0.0, "low": -1.0}, n_per_pair=120, tie_rate=0.1, seed=0)


def test_ranking_recovers_true_order(arena_df):
    res = bradley_terry(arena_df, n_boot=300)
    assert list(res.ranking["model"]) == ["top", "mid", "low"]
    assert list(res.ranking["rank"]) == [1, 2, 3]


def test_ratings_are_centred_and_ordered(arena_df):
    res = bradley_terry(arena_df, n_boot=300)
    assert abs(res.ranking["rating"].sum()) < 1e-6
    assert res.ranking["rating"].is_monotonic_decreasing


def test_true_gap_lies_in_bootstrap_ci(arena_df):
    res = bradley_terry(arena_df, n_boot=800, seed=1)
    idx = res.pairwise.set_index(["model_a", "model_b"])
    row = idx.loc[("low", "top")] if ("low", "top") in idx.index else idx.loc[("top", "low")]
    lo, hi = sorted((abs(row["ci_low"]), abs(row["ci_high"])))
    assert lo < 2.0 < hi  # true |top - low| gap is 2.0
    assert bool(row["significant"])


def test_near_tie_is_not_flagged_significant():
    df = _simulate({"a": 0.02, "b": 0.0}, n_per_pair=150, tie_rate=0.1, seed=2)
    res = bradley_terry(df, n_boot=800, seed=0)
    assert not bool(res.pairwise.iloc[0]["significant"])


def test_load_pairwise_winner_column(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame(
        {"model_a": ["x", "x", "y"], "model_b": ["y", "y", "x"], "winner": ["x", "y", "y"]}
    ).to_csv(p, index=False)
    df = load_pairwise(str(p))
    assert list(df["outcome"]) == ["a", "b", "a"]


def test_load_pairwise_rejects_unreadable_outcome(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame({"model_a": ["x"], "model_b": ["y"], "outcome": ["???"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="could not read outcome"):
        load_pairwise(str(p))


def test_elo_runs_and_orders(arena_df):
    out = elo(arena_df)
    assert out.iloc[0]["model"] == "top"
    assert set(out["model"]) == {"top", "mid", "low"}


def test_bradley_terry_needs_two_models():
    df = pd.DataFrame({"model_a": ["x"], "model_b": ["x"], "outcome": ["tie"]})
    with pytest.raises(ValueError):
        bradley_terry(df)


def _simulate_rk(skill, theta, n_per_pair, seed):
    rng = np.random.default_rng(seed)
    models = list(skill)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pa, pb = np.exp(skill[a]), np.exp(skill[b])
            p_a = pa / (pa + theta * pb)
            p_b = pb / (theta * pa + pb)
            probs = [p_a, p_b, max(1.0 - p_a - p_b, 0.0)]
            picks = rng.choice(["a", "b", "tie"], size=n_per_pair, p=probs)
            rows.extend((a, b, o) for o in picks)
    return pd.DataFrame(rows, columns=["model_a", "model_b", "outcome"])


def test_rao_kupper_recovers_ratings_and_theta():
    df = _simulate_rk({"top": 0.8, "mid": 0.0, "low": -0.8}, theta=2.0, n_per_pair=300, seed=0)
    res = bradley_terry(df, tie="rao-kupper", n_boot=120, seed=1)
    assert list(res.ranking["model"]) == ["top", "mid", "low"]
    assert res.tie_param is not None
    assert 1.4 < res.tie_param < 3.0  # true theta is 2.0
    assert res.tie_param_ci[0] <= res.tie_param <= res.tie_param_ci[1]


def test_rao_kupper_without_ties_warns_and_falls_back():
    df = _simulate({"a": 0.5, "b": 0.0}, n_per_pair=80, tie_rate=0.0, seed=0)
    with pytest.warns(UserWarning, match="no ties"):
        res = bradley_terry(df, tie="rao-kupper", n_boot=50)
    assert res.tie_param is None


def test_unknown_tie_model_raises():
    df = _simulate({"a": 0.3, "b": 0.0}, n_per_pair=40, tie_rate=0.1, seed=0)
    with pytest.raises(ValueError, match="tie model"):
        bradley_terry(df, tie="bogus", n_boot=10)


def test_split_tie_result_has_no_theta(arena_df):
    res = bradley_terry(arena_df, n_boot=200)
    assert res.tie == "split"
    assert res.tie_param is None
    assert "theta" not in res.verdict()
