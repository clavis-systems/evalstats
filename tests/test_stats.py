"""Calibration tests -- the point of the package is that these hold."""

from __future__ import annotations

import numpy as np
import pytest

from evalstats.stats import (
    benjamini_hochberg,
    clustered_mean_estimate,
    holm,
    mean_estimate,
    paired_difference,
    sample_size_for_ci_halfwidth,
    sample_size_for_paired_detection,
)


def test_clt_ci_coverage_is_about_95_percent():
    true_p, n, reps = 0.6, 200, 600
    hits = 0
    for r in range(reps):
        x = (np.random.default_rng(r).random(n) < true_p).astype(float)
        est = mean_estimate(x, level=0.95)
        hits += est.ci_low <= true_p <= est.ci_high
    assert 0.92 <= hits / reps <= 0.975


def test_permutation_test_is_calibrated_under_the_null():
    """Two identical distributions -> P(p < 0.05) should be about 0.05."""
    n, reps = 160, 300
    below = 0
    for r in range(reps):
        g = np.random.default_rng(r)
        a = (g.random(n) < 0.6).astype(float)
        b = (g.random(n) < 0.6).astype(float)
        res = paired_difference(a, b, n_boot=400, n_perm=800, seed=r)
        below += res.p_permutation < 0.05
    assert 0.02 <= below / reps <= 0.10


def test_permutation_test_detects_a_real_gap():
    g = np.random.default_rng(7)
    a = (g.random(400) < 0.72).astype(float)
    b = (g.random(400) < 0.55).astype(float)
    res = paired_difference(a, b, seed=0)
    assert res.diff > 0
    assert res.p_permutation < 0.01
    assert res.ci_low > 0
    assert res.significant


def test_clustered_ci_is_wider_when_clusters_are_correlated():
    g = np.random.default_rng(3)
    scores, clusters = [], []
    for c in range(12):
        level = g.uniform(0.3, 0.9)  # strong between-cluster variation
        s = (g.random(50) < level).astype(float)
        scores.extend(s)
        clusters.extend([c] * 50)
    scores = np.array(scores)
    naive = mean_estimate(scores, level=0.95)
    clust = clustered_mean_estimate(scores, clusters, level=0.95, seed=0)
    assert clust.se > naive.se


@pytest.mark.parametrize("method", ["cluster-bootstrap", "cr1"])
def test_clustered_methods_run_and_agree_roughly(method):
    g = np.random.default_rng(5)
    scores = (g.random(600) < 0.65).astype(float)
    clusters = np.repeat(np.arange(20), 30)
    est = clustered_mean_estimate(scores, clusters, method=method, seed=0)
    assert 0.55 < est.mean < 0.75
    assert est.ci_low < est.mean < est.ci_high


def test_wilson_interval_stays_in_unit_range_at_extremes():
    est = mean_estimate(np.zeros(50), method="wilson")
    assert est.ci_low >= 0.0
    assert est.ci_high > 0.0
    est = mean_estimate(np.ones(50), method="wilson")
    assert est.ci_high <= 1.0


def test_mcnemar_symmetric_case_is_not_significant():
    a = np.array([1, 0] * 25 + [1, 1] * 10, dtype=float)
    b = np.array([0, 1] * 25 + [1, 1] * 10, dtype=float)
    res = paired_difference(a, b, seed=0)
    assert res.p_mcnemar is not None
    assert res.p_mcnemar > 0.5


def test_holm_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    adj = holm(p)
    assert np.all(adj >= p)
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_bh_matches_hand_computation():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    # BH with m=5: adj_k = min over j>=k of p_j * 5 / j  -> all 0.05 here
    assert np.allclose(benjamini_hochberg(p), 0.05)


def test_bh_is_never_more_conservative_than_holm():
    p = np.array([0.001, 0.008, 0.03, 0.04, 0.2, 0.7])
    assert np.all(benjamini_hochberg(p) <= holm(p) + 1e-12)


def test_sample_size_helpers_are_monotone():
    assert sample_size_for_ci_halfwidth(0.5, 0.02) > sample_size_for_ci_halfwidth(0.5, 0.04)
    assert sample_size_for_paired_detection(0.01) > sample_size_for_paired_detection(0.02)
    assert sample_size_for_paired_detection(0.02, p_pooled=0.9) < sample_size_for_paired_detection(
        0.02, p_pooled=0.5
    )
