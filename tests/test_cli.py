"""End-to-end CLI tests via typer's CliRunner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evalstats.cli import app

runner = CliRunner()


@pytest.fixture
def results_csv(tmp_path, toy_frame):
    p = tmp_path / "results.csv"
    toy_frame.to_csv(p, index=False)
    return str(p)


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert res.stdout.strip() == "0.1.0"


def test_summary_has_overall(results_csv):
    res = runner.invoke(app, ["summary", results_csv])
    assert res.exit_code == 0
    assert "(overall)" in res.stdout


def test_compare_clear_winner(results_csv):
    res = runner.invoke(
        app,
        ["compare", results_csv, "--a", "a", "--b", "c", "--n-boot", "1500", "--n-perm", "1500"],
    )
    assert res.exit_code == 0
    assert "significantly better" in res.stdout


def test_compare_noise_pair(results_csv):
    res = runner.invoke(
        app,
        ["compare", results_csv, "--a", "a", "--b", "b", "--n-boot", "1500", "--n-perm", "1500"],
    )
    assert res.exit_code == 0
    assert "within noise" in res.stdout.lower()


def test_compare_unknown_model_exits_2(results_csv):
    res = runner.invoke(app, ["compare", results_csv, "--a", "a", "--b", "nope"])
    assert res.exit_code == 2


def test_leaderboard_csv_is_parseable(results_csv, tmp_path):
    res = runner.invoke(app, ["leaderboard", results_csv, "--format", "csv"])
    assert res.exit_code == 0
    assert "rank,model" in res.stdout


def test_summary_markdown_format(results_csv):
    res = runner.invoke(app, ["summary", results_csv, "--format", "md"])
    assert res.exit_code == 0
    assert "| model | task |" in res.stdout
    assert "| --- |" in res.stdout


def test_power_modes():
    r1 = runner.invoke(app, ["power", "ci", "--p", "0.7", "--half-width", "0.02"])
    r2 = runner.invoke(app, ["power", "paired", "--mde", "0.02", "--p-pooled", "0.7"])
    assert r1.exit_code == 0 and "items" in r1.stdout
    assert r2.exit_code == 0 and "paired items" in r2.stdout
    assert runner.invoke(app, ["power", "bogus"]).exit_code == 2


def test_report_writes_html(results_csv, tmp_path):
    out = tmp_path / "r.html"
    res = runner.invoke(app, ["report", results_csv, "-o", str(out)])
    assert res.exit_code == 0
    assert out.exists() and out.stat().st_size > 1000
    assert "<title>evalstats report</title>" in out.read_text(encoding="utf-8")


def test_bad_file_exits_2(tmp_path):
    res = runner.invoke(app, ["summary", str(tmp_path / "missing.csv")])
    assert res.exit_code == 2


def test_arena_command(tmp_path):
    import pandas as pd

    rows = []
    for _ in range(60):
        rows += [("strong", "weak", "a"), ("strong", "weak", "a"), ("strong", "weak", "b")]
    p = tmp_path / "arena.csv"
    pd.DataFrame(rows, columns=["model_a", "model_b", "outcome"]).to_csv(p, index=False)
    res = runner.invoke(app, ["arena", str(p), "--n-boot", "200"])
    assert res.exit_code == 0
    assert "Bradley-Terry" in res.stdout
    assert "strong" in res.stdout


def test_arena_bad_outcome_exits_2(tmp_path):
    import pandas as pd

    p = tmp_path / "arena.csv"
    pd.DataFrame({"model_a": ["x"], "model_b": ["y"], "outcome": ["???"]}).to_csv(p, index=False)
    res = runner.invoke(app, ["arena", str(p)])
    assert res.exit_code == 2
