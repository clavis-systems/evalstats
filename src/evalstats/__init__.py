"""evalstats -- significance-aware comparison of language-model evaluation results.

The statistics here are standard (bootstrap confidence intervals, paired
randomization tests, cluster-robust standard errors, multiple-comparison
correction). What this package adds is the "last mile": it reads the per-item
result formats people actually produce and turns those methods into one command
with a plain-language verdict.

Method references:
  Miller (2024), "Adding Error Bars to Evals", arXiv:2411.00640
  Dror et al. (2018), "The Hitchhiker's Guide to Testing Statistical
      Significance in Natural Language Processing"
"""

from evalstats.analysis import leaderboard, pairwise_significance, summarize
from evalstats.loading import from_lighteval, from_lm_eval_harness, load_results
from evalstats.preference import BTResult, bradley_terry, elo, load_pairwise
from evalstats.stats import (
    MeanEstimate,
    PairedResult,
    benjamini_hochberg,
    clustered_mean_estimate,
    holm,
    mean_estimate,
    paired_difference,
    sample_size_for_ci_halfwidth,
    sample_size_for_paired_detection,
)

__version__ = "0.1.0"

__all__ = [
    "BTResult",
    "MeanEstimate",
    "PairedResult",
    "__version__",
    "benjamini_hochberg",
    "bradley_terry",
    "clustered_mean_estimate",
    "elo",
    "from_lighteval",
    "from_lm_eval_harness",
    "holm",
    "leaderboard",
    "load_pairwise",
    "load_results",
    "mean_estimate",
    "paired_difference",
    "pairwise_significance",
    "sample_size_for_ci_halfwidth",
    "sample_size_for_paired_detection",
    "summarize",
]
