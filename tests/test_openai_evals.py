"""Tests for the OpenAI evals run-log adapter."""

from __future__ import annotations

import json

import pytest

from evalstats.loading import from_openai_evals


def _write(path, objs):
    path.write_text("\n".join(json.dumps(o) for o in objs), encoding="utf-8")


def _run(sample_scores, *, eval_name="coqa.dev.v0", fns=("gpt-3.5-turbo",), score_key="correct"):
    objs = [{"spec": {"completion_fns": list(fns), "eval_name": eval_name, "base_eval": "coqa"}}]
    for i, s in enumerate(sample_scores):
        sid = f"coqa.dev.{i}"
        objs.append(
            {"event_id": 2 * i, "sample_id": sid, "type": "sampling", "data": {"sampled": "x"}}
        )
        val = bool(s) if score_key == "correct" else s
        objs.append(
            {"event_id": 2 * i + 1, "sample_id": sid, "type": "match", "data": {score_key: val}}
        )
    objs.append({"final_report": {"accuracy": sum(sample_scores) / len(sample_scores)}})
    return objs


def test_reads_model_task_and_correctness(tmp_path):
    f = tmp_path / "run.jsonl"
    _write(f, _run([1, 0, 1, 1]))
    df = from_openai_evals(str(f))
    assert set(df["model"]) == {"gpt-3.5-turbo"}
    assert set(df["task"]) == {"coqa.dev.v0"}
    assert df["score"].tolist() == [1.0, 0.0, 1.0, 1.0]
    assert df["item_id"].tolist() == ["coqa.dev.0", "coqa.dev.1", "coqa.dev.2", "coqa.dev.3"]


def test_ignores_sampling_and_final_report(tmp_path):
    f = tmp_path / "run.jsonl"
    _write(f, _run([1, 0]))
    df = from_openai_evals(str(f))
    assert len(df) == 2  # not counting the 2 sampling events or final_report


def test_model_graded_score_key(tmp_path):
    f = tmp_path / "run.jsonl"
    _write(f, _run([0.9, 0.2, 0.7], score_key="score"))
    df = from_openai_evals(str(f))
    assert df["score"].tolist() == [0.9, 0.2, 0.7]


def test_explicit_metric(tmp_path):
    f = tmp_path / "run.jsonl"
    objs = [{"spec": {"completion_fns": ["m"], "eval_name": "t"}}]
    objs.append({"sample_id": "t.0", "type": "metrics", "data": {"correct": True, "bleu": 0.4}})
    _write(f, objs)
    df = from_openai_evals(str(f), metric="bleu")
    assert df["score"].tolist() == [0.4]


def test_model_override_and_first_event_wins(tmp_path):
    f = tmp_path / "run.jsonl"
    objs = [
        {"spec": {"completion_fns": ["ignored"], "eval_name": "t"}},
        {"sample_id": "t.0", "type": "match", "data": {"correct": True}},
        {"sample_id": "t.0", "type": "match", "data": {"correct": False}},
    ]
    _write(f, objs)
    df = from_openai_evals(str(f), model="my-model")
    assert df["model"].tolist() == ["my-model"]
    assert df["score"].tolist() == [1.0]  # first scoring event wins


def test_no_spec_uses_filename(tmp_path):
    f = tmp_path / "gsm8k_20240601123456.jsonl"
    _write(f, [{"sample_id": "s.0", "type": "match", "data": {"correct": True}}])
    df = from_openai_evals(str(f))
    assert df["task"].tolist() == ["gsm8k"]
    assert df["model"].tolist() == ["gsm8k"]


def test_helpful_error_without_scores(tmp_path):
    f = tmp_path / "run.jsonl"
    _write(
        f,
        [
            {"spec": {"completion_fns": ["m"], "eval_name": "t"}},
            {"sample_id": "t.0", "type": "metrics", "data": {"weird": 3}},
        ],
    )
    with pytest.raises(ValueError, match="numeric data keys"):
        from_openai_evals(str(f))


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_openai_evals(str(tmp_path / "nope.jsonl"))
