"""Degenerate inputs should give sane results, not crashes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalstats.stats import (
    benjamini_hochberg,
    clustered_mean_estimate,
    holm,
    mean_estimate,
    paired_difference,
)


def test_mean_estimate_single_observation():
    est = mean_estimate([1.0])
    assert est.mean == 1.0
    assert math.isnan(est.se)


def test_mean_estimate_rejects_empty():
    with pytest.raises(ValueError):
        mean_estimate([])


def test_paired_all_concordant_is_a_tie():
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0] * 8)
    res = paired_difference(x, x.copy(), seed=0)
    assert res.diff == 0.0
    assert res.n_discordant == 0
    assert res.p_mcnemar == 1.0
    assert res.p_permutation == 1.0
    assert not res.significant


def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_difference([1, 0, 1], [1, 0])


def test_clustered_cr1_needs_two_clusters():
    with pytest.raises(ValueError):
        clustered_mean_estimate([1.0, 0.0, 1.0], ["only", "only", "only"], method="cr1")


def test_clustered_bootstrap_single_cluster_warns_and_falls_back():
    with pytest.warns(UserWarning, match="fewer than 2 clusters"):
        est = clustered_mean_estimate(
            [1.0, 0.0, 1.0, 0.0], ["c", "c", "c", "c"], method="cluster-bootstrap", seed=0
        )
    assert est.se > 0.0  # plain CLT interval, not a degenerate zero
    assert est.method == "clt"


def test_correction_on_single_pvalue():
    assert holm([0.2]).tolist() == [0.2]
    assert benjamini_hochberg([0.2]).tolist() == [0.2]


def test_correction_clips_to_one():
    assert np.all(holm([0.9, 0.8, 0.7]) <= 1.0)


def test_mean_estimate_ci_clipped_to_unit_interval_for_rates():
    est = mean_estimate([1.0] * 30, level=0.99)  # mean 1.0, would overshoot
    assert est.ci_high <= 1.0
