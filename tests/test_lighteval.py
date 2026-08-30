"""Tests for the HuggingFace lighteval `details` adapter."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from evalstats.loading import _lighteval_model, _lighteval_task, from_lighteval


def _details_frame(n: int, metric_key: str = "acc") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc": [{"query": f"q{i}"} for i in range(n)],
            "model_response": [{"text": f"r{i}"} for i in range(n)],
            "metric": [{metric_key: float(i % 2), "acc_norm": float(i % 3 == 0)} for i in range(n)],
        }
    )


@pytest.fixture
def details_dir(tmp_path):
    """.../details/<model>/<date_id>/details_<task>_<date_id>.parquet"""
    date = "2024-06-01T12-00-00.000000"
    d = tmp_path / "details" / "Org__Model-7B" / date
    d.mkdir(parents=True)
    _details_frame(20, "acc").to_parquet(d / f"details_gsm8k_{date}.parquet")
    _details_frame(12, "acc").to_parquet(d / f"details_arc_easy_{date}.parquet")
    return tmp_path, d, date


def test_reads_model_task_and_score_from_parquet(details_dir):
    root, _, _ = details_dir
    df = from_lighteval(str(root))
    assert set(df["model"]) == {"Org__Model-7B"}
    assert set(df["task"]) == {"gsm8k", "arc_easy"}
    assert len(df) == 32
    assert set(df["score"].unique()) <= {0.0, 1.0}


def test_auto_pick_is_acc_over_acc_norm(details_dir):
    _, d, date = details_dir
    df = from_lighteval(str(d / f"details_gsm8k_{date}.parquet"))
    assert df["score"].sum() == 10  # acc == i % 2 over 20 rows


def test_comma_filter_metric_keys(tmp_path):
    date = "2024-06-01T12-00-00.000000"
    f = tmp_path / f"details_gsm8k_{date}.parquet"
    pd.DataFrame({"metric": [{"acc,none": 1.0}, {"acc,none": 0.0}]}).to_parquet(f)
    df = from_lighteval(str(f))
    assert df["score"].tolist() == [1.0, 0.0]


def test_explicit_metric(details_dir):
    _, d, date = details_dir
    df = from_lighteval(str(d / f"details_gsm8k_{date}.parquet"), metric="acc_norm")
    assert df["score"].sum() == 7  # i % 3 == 0 for i in 0..19


def test_json_details_variant(tmp_path):
    date = "2024-06-01T12-00-00.000000"
    rows = [{"metric": {"exact_match": 1.0}}, {"metric": {"exact_match": 0.0}}]
    f = tmp_path / f"details_mytask_{date}.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    df = from_lighteval(str(f))
    assert df["task"].unique().tolist() == ["mytask"]
    assert df["score"].tolist() == [1.0, 0.0]


def test_helpful_error_when_no_known_metric(tmp_path):
    date = "2024-06-01T12-00-00.000000"
    f = tmp_path / f"details_weird_{date}.json"
    f.write_text(json.dumps([{"metric": {"my_metric": 1.0}}]), encoding="utf-8")
    with pytest.raises(ValueError, match="metric keys seen"):
        from_lighteval(str(f))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_lighteval(str(tmp_path / "nope"))


def test_task_name_keeps_pipes():
    assert (
        _lighteval_task("details_lighteval|gsm8k|0_2024-06-01T12-00-00.000000.parquet")
        == "lighteval|gsm8k|0"
    )
    assert _lighteval_task("details_plain.parquet") == "plain"


def test_model_from_path_layouts(tmp_path):
    nested = tmp_path / "details" / "MyModel" / "2024-06-01T00-00-00.0" / "details_x_2024.parquet"
    assert _lighteval_model(str(nested), None) == "MyModel"
    flat = tmp_path / "MyModel" / "details_x.parquet"
    assert _lighteval_model(str(flat), None) == "MyModel"
    assert _lighteval_model(str(flat), "Override") == "Override"
