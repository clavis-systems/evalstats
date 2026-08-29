"""Tests for the lm-evaluation-harness --log_samples adapter."""

from __future__ import annotations

import json

import pytest

from evalstats.loading import from_lm_eval_harness


def _write(path, lines):
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


@pytest.fixture
def sample_dir(tmp_path):
    """A realistic layout: one model dir, two task files, comma-filtered keys."""
    mdir = tmp_path / "meta-llama__Model-7b"
    mdir.mkdir()

    arc = [
        {
            "doc_id": i,
            "doc": {"question": f"q{i}", "choices": ["a", "b"]},
            "target": i % 2,
            "resps": [[["", False]]],
            "filtered_resps": ["a"],
            "acc,none": float(i % 2 == 0),
            "acc_norm,none": float(i % 3 == 0),
        }
        for i in range(20)
    ]
    _write(mdir / "samples_arc_easy_2024-01-02T03-04-05.000000.jsonl", arc)

    gsm = [
        {
            "doc_id": i,
            "doc": {"question": f"g{i}"},
            "target": "42",
            "exact_match,strict-match": float(i % 4 == 0),
            "exact_match,flexible-extract": float(i % 2 == 0),
        }
        for i in range(16)
    ]
    _write(mdir / "samples_gsm8k_2024-01-02T03-04-05.000000.jsonl", gsm)
    return tmp_path, mdir


def test_reads_model_and_task_from_paths(sample_dir):
    root, _ = sample_dir
    df = from_lm_eval_harness(str(root))
    assert set(df["model"]) == {"meta-llama__Model-7b"}
    assert set(df["task"]) == {"arc_easy", "gsm8k"}


def test_auto_pick_prefers_acc_over_acc_norm(sample_dir):
    _, mdir = sample_dir
    df = from_lm_eval_harness(str(mdir / "samples_arc_easy_2024-01-02T03-04-05.000000.jsonl"))
    # acc,none is 1.0 on even doc_id -> 10 of 20
    assert df["score"].sum() == 10


def test_explicit_full_key(sample_dir):
    _, mdir = sample_dir
    f = str(mdir / "samples_arc_easy_2024-01-02T03-04-05.000000.jsonl")
    df = from_lm_eval_harness(f, metric="acc_norm,none")
    assert df["score"].sum() == 7  # i % 3 == 0 for i in 0..19


def test_bare_metric_ambiguous_across_filters_raises(sample_dir):
    _, mdir = sample_dir
    f = str(mdir / "samples_gsm8k_2024-01-02T03-04-05.000000.jsonl")
    with pytest.raises(ValueError, match="ambiguous"):
        from_lm_eval_harness(f, metric="exact_match")


def test_helpful_error_when_no_known_metric(tmp_path):
    f = tmp_path / "samples_weird_2024-01-02T03-04-05.000000.jsonl"
    _write(f, [{"doc_id": 0, "target": 3, "my_metric": 1.0}])
    with pytest.raises(ValueError, match="numeric keys present"):
        from_lm_eval_harness(str(f))


def test_glob_input(sample_dir):
    _, mdir = sample_dir
    df = from_lm_eval_harness(str(mdir / "samples_*.jsonl"))
    assert len(df) == 36  # 20 arc + 16 gsm


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_lm_eval_harness(str(tmp_path / "nope"))


def test_repeated_doc_ids_are_disambiguated(tmp_path):
    """Some published logs concatenate several runs of the same docs."""
    f = tmp_path / "samples_gsm8k_2024-01-02T03-04-05.000000.jsonl"
    rows = [{"doc_id": i % 4, "target": "x", "exact_match": float(i % 2)} for i in range(8)]
    _write(f, rows)
    df = from_lm_eval_harness(str(f))
    assert len(df) == 8
    assert df["item_id"].is_unique
    assert set(df["item_id"]) == {"0", "1", "2", "3", "0#2", "1#2", "2#2", "3#2"}
