from __future__ import annotations

from evalstats.analysis import OVERALL, leaderboard, pairwise_significance, summarize


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


def test_pairwise_flags_clear_winner_and_not_the_noise_pair(toy_frame):
    pairs = pairwise_significance(toy_frame, n_boot=1500, n_perm=1500)
    row = pairs.set_index(["model_a", "model_b"])

    def get(x, y):
        return row.loc[(x, y)] if (x, y) in row.index else row.loc[(y, x)]

    assert bool(get("a", "c")["significant"])  # real gap
    assert not bool(get("a", "b")["significant"])  # within noise
