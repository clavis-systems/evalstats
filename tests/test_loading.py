from __future__ import annotations

import json

import pandas as pd
import pytest

from evalstats.loading import REQUIRED_COLUMNS, from_records, load_results


def test_csv_round_trip(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame(
        {
            "model": ["a", "a", "b"],
            "task": ["t1", "t2", "t1"],
            "item_id": [1, 2, 1],
            "score": [1, 0, 1],
        }
    ).to_csv(p, index=False)
    df = load_results(str(p))
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert pd.api.types.is_string_dtype(df["item_id"])
    assert df["item_id"].tolist() == ["1", "2", "1"]
    assert df["score"].tolist() == [1.0, 0.0, 1.0]


def test_jsonl_with_aliased_columns(tmp_path):
    p = tmp_path / "r.jsonl"
    lines = [
        {"system": "m1", "benchmark": "gsm8k", "doc_id": 0, "exact_match": True},
        {"system": "m1", "benchmark": "gsm8k", "doc_id": 1, "exact_match": False},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    df = load_results(str(p))
    assert set(df["model"]) == {"m1"}
    assert df["score"].tolist() == [1.0, 0.0]


def test_string_boolean_scores_are_coerced():
    df = from_records(
        [
            {"model": "a", "task": "t", "item_id": "1", "score": "correct"},
            {"model": "a", "task": "t", "item_id": "2", "score": "incorrect"},
        ]
    )
    assert df["score"].tolist() == [1.0, 0.0]


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        from_records([{"model": "a", "task": "t", "score": 1}])
