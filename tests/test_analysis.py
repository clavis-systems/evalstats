from __future__ import annotations

import numpy as np

from evalstats.analysis import OVERALL, leaderboard, pairwise_significance, summarize
from evalstats.stats import mean_estimate


def test_summarize_has_overall_row_per_model(toy_frame):
    summ = summarize(toy_frame)
    per_model_overall = summ[summ["task"] == OVERALL]
    assert set(per_model_overall["model"]) == {"a", "b", "c"}
    assert (summ["ci_low"] <= summ["mean"]).all()
    assert (summ["mean"] <= summ["ci_high"]).all()


def test_leaderboard_ranks_best_model_first(toy_frame):
    table, _pairs = leaderboard(toy_frame, n_boot=1500, n_perm=1500)
    assert table.iloc[0]["model"] == "c"
    assert list(table["rank"]) == [1, 2, 3]


def test_single_task_overall_is_plain_not_degenerate_cluster(toy_frame):
    """With one task the overall row must fall back to a plain CLT estimate,
    not the single-cluster bootstrap that reports SE == 0."""
    one = toy_frame[toy_frame["task"] == "t1"]
    summ = summarize(one)
    overall = summ[summ["task"] == OVERALL]
    assert (overall["se"] > 0).all()
    for model, mdf in one.groupby("model"):
        plain = mean_estimate(mdf["score"].to_numpy())
        got = overall[overall["model"] == model].iloc[0]
        assert np.isclose(got["se"], plain.se)
        assert np.isclose(got["ci_high"], plain.ci_high)


def test_single_task_pairwise_ci_is_not_degenerate(toy_frame):
    """One task -> item bootstrap, so the CI must have real width (regression:
    the single-cluster bootstrap used to collapse it to a point)."""
    one = toy_frame[toy_frame["task"] == "t1"]
    pairs = pairwise_significance(one, n_boot=2000, n_perm=2000)
    assert (pairs["ci_high"] - pairs["ci_low"] > 1e-6).all()


def test_pairwise_flags_clear_winner_and_not_the_noise_pair(toy_frame):
    pairs = pairwise_significance(toy_frame, n_boot=1500, n_perm=1500)
    row = pairs.set_index(["model_a", "model_b"])

    def get(x, y):
        return row.loc[(x, y)] if (x, y) in row.index else row.loc[(y, x)]

    assert bool(get("a", "c")["significant"])  # real gap
    assert not bool(get("a", "b")["significant"])  # within noise
