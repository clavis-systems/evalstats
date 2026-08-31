"""Tests for rank_probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evalstats.analysis import rank_probabilities


def test_clear_winner_has_p_rank1_near_one(toy_frame):
    out = rank_probabilities(toy_frame, n_boot=3000)
    top = out.iloc[0]
    assert top["model"] == "c"
    assert top["p_rank1"] > 0.95
    assert top["rank"] == 1


def test_probabilities_are_a_distribution(toy_frame):
    out = rank_probabilities(toy_frame, n_boot=3000)
    # each model's rank probabilities sum to 1 -> p_rank1 columns across models sum to 1
    assert abs(out["p_rank1"].sum() - 1.0) < 1e-9
    assert (out["p_rank1"].between(0, 1)).all()
    assert (out["mean_rank"] >= 1).all() and (out["mean_rank"] <= 3).all()


def test_noise_pair_splits_the_rank(toy_frame):
    out = rank_probabilities(toy_frame, n_boot=4000).set_index("model")
    # a and b are within noise of each other -> neither dominates rank 1
    assert out.loc["a", "p_rank1"] < 0.2
    assert out.loc["b", "p_rank1"] < 0.2


def test_needs_two_models():
    df = pd.DataFrame(
        {"model": ["a", "a"], "task": ["t", "t"], "item_id": ["1", "2"], "score": [1.0, 0.0]}
    )
    with pytest.raises(ValueError):
        rank_probabilities(df)


def test_warns_when_items_not_shared():
    df = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "task": ["t", "t", "t", "t"],
            "item_id": ["1", "2", "1", "3"],  # item 2 only for a, item 3 only for b
            "score": [1.0, 0.0, 1.0, 0.0],
        }
    )
    with pytest.warns(UserWarning, match="not shared"):
        out = rank_probabilities(df, n_boot=500)
    assert set(out["model"]) == {"a", "b"}


def test_reproducible(toy_frame):
    a = rank_probabilities(toy_frame, n_boot=1000, seed=7)
    b = rank_probabilities(toy_frame, n_boot=1000, seed=7)
    assert np.allclose(a["p_rank1"], b["p_rank1"])
