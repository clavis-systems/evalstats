"""Command-line interface: ``evalstats <command> ...``."""

from __future__ import annotations

import sys

import pandas as pd
import typer

from evalstats import __version__
from evalstats.analysis import leaderboard as _leaderboard
from evalstats.analysis import summarize
from evalstats.loading import from_lm_eval_harness, load_results
from evalstats.stats import (
    paired_difference,
    sample_size_for_ci_halfwidth,
    sample_size_for_paired_detection,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Significance-aware comparison of language-model evaluation results.",
)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def _load(path: str, lm_eval: bool, metric: str | None) -> pd.DataFrame:
    try:
        if lm_eval:
            return from_lm_eval_harness(path, metric=metric)
        return load_results(path)
    except (OSError, ValueError) as exc:  # pragma: no cover - user input errors
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _to_markdown(df: pd.DataFrame) -> str:
    def cell(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _emit(df: pd.DataFrame, fmt: str) -> None:
    if fmt == "csv":
        typer.echo(df.to_csv(index=False))
    elif fmt == "json":
        typer.echo(df.to_json(orient="records", indent=2))
    elif fmt == "md":
        typer.echo(_to_markdown(df))
    else:
        typer.echo(df.to_string(index=False))


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def summary(
    results: str = typer.Argument(..., help="CSV / JSONL results file."),
    level: float = typer.Option(0.95, help="Confidence level."),
    fmt: str = typer.Option("table", "--format", help="table | csv | json | md."),
    lm_eval: bool = typer.Option(False, "--lm-eval", help="Input is lm-eval-harness samples."),
    metric: str | None = typer.Option(None, help="Metric key for --lm-eval input."),
) -> None:
    """Per-(model, task) mean scores with confidence intervals."""
    df = _load(results, lm_eval, metric)
    _emit(summarize(df, level=level), fmt)


@app.command()
def compare(
    results: str = typer.Argument(..., help="CSV / JSONL results file."),
    a: str = typer.Option(..., "--a", help="First model name."),
    b: str = typer.Option(..., "--b", help="Second model name."),
    level: float = typer.Option(0.95, help="Confidence level."),
    seed: int = typer.Option(0, help="RNG seed."),
    n_boot: int = typer.Option(10_000, help="Bootstrap resamples."),
    n_perm: int = typer.Option(10_000, help="Permutation resamples."),
    lm_eval: bool = typer.Option(False, "--lm-eval", help="Input is lm-eval-harness samples."),
    metric: str | None = typer.Option(None, help="Metric key for --lm-eval input."),
) -> None:
    """Paired comparison of two models on their common items."""
    df = _load(results, lm_eval, metric)
    wide = df.pivot_table(
        index=["task", "item_id"], columns="model", values="score", aggfunc="mean"
    )
    for name in (a, b):
        if name not in wide.columns:
            typer.secho(
                f"error: model {name!r} not in results (have: {list(wide.columns)})",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2)
    pair = wide[[a, b]].dropna()
    if pair.empty:
        typer.secho("error: no items shared by both models", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    res = paired_difference(
        pair[a].to_numpy(),
        pair[b].to_numpy(),
        clusters=pair.index.get_level_values("task").to_numpy(),
        level=level,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    typer.echo(res.verdict(a, b))
    typer.echo(f"  paired items      : {res.n}")
    if res.n_discordant is not None:
        typer.echo(f"  discordant pairs  : {res.n_discordant}")
        typer.echo(f"  McNemar exact p   : {res.p_mcnemar:.3g}")
    typer.echo(f"  bootstrap method  : {res.method}")


@app.command()
def leaderboard(
    results: str = typer.Argument(..., help="CSV / JSONL results file."),
    level: float = typer.Option(0.95, help="Confidence level."),
    correction: str = typer.Option("holm", help="holm | bh | none."),
    seed: int = typer.Option(0, help="RNG seed."),
    fmt: str = typer.Option("table", "--format", help="table | csv | json | md."),
    lm_eval: bool = typer.Option(False, "--lm-eval", help="Input is lm-eval-harness samples."),
    metric: str | None = typer.Option(None, help="Metric key for --lm-eval input."),
) -> None:
    """Rank models by cluster-robust overall score, with a pairwise matrix."""
    df = _load(results, lm_eval, metric)
    table, pairs = _leaderboard(df, level=level, correction=correction, seed=seed)
    typer.echo("# ranking")
    _emit(table, fmt)
    typer.echo(f"\n# pairwise (paired randomization test, {correction} corrected)")
    _emit(pairs, fmt)


@app.command()
def power(
    mode: str = typer.Argument(..., help="ci | paired."),
    p: float = typer.Option(0.5, help="[ci] expected accuracy."),
    half_width: float = typer.Option(0.03, help="[ci] target CI half-width."),
    mde: float = typer.Option(0.02, help="[paired] smallest gap to detect."),
    sd_diff: float | None = typer.Option(None, help="[paired] std dev of per-item diff."),
    p_pooled: float | None = typer.Option(None, help="[paired] rough shared accuracy."),
    level: float = typer.Option(0.95, help="Confidence level."),
    stat_power: float = typer.Option(0.80, "--power", help="[paired] desired power."),
) -> None:
    """How many items an experiment needs (Miller 2024, sec. 5)."""
    if mode == "ci":
        n = sample_size_for_ci_halfwidth(p, half_width, level=level)
        typer.echo(f"{n} items for a {level:.0%} CI of half-width {half_width:g} at p={p:g}")
    elif mode == "paired":
        n = sample_size_for_paired_detection(
            mde, sd_diff=sd_diff, p_pooled=p_pooled, level=level, power=stat_power
        )
        typer.echo(
            f"{n} paired items to detect a gap of {mde:g} "
            f"with power {stat_power:.0%} at level {level:.0%}"
        )
    else:
        typer.secho("error: mode must be 'ci' or 'paired'", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


@app.command()
def report(
    results: str = typer.Argument(..., help="CSV / JSONL results file."),
    out: str = typer.Option("report.html", "-o", "--out", help="Output HTML path."),
    level: float = typer.Option(0.95, help="Confidence level."),
    correction: str = typer.Option("holm", help="holm | bh | none."),
    seed: int = typer.Option(0, help="RNG seed."),
    lm_eval: bool = typer.Option(False, "--lm-eval", help="Input is lm-eval-harness samples."),
    metric: str | None = typer.Option(None, help="Metric key for --lm-eval input."),
) -> None:
    """Write a self-contained HTML report (needs the 'report' extra)."""
    try:
        from evalstats.report import build_report
    except ImportError:
        typer.secho(
            "error: reporting needs extra deps -- pip install 'evalstats[report]'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(3)
    df = _load(results, lm_eval, metric)
    path = build_report(df, out, level=level, correction=correction, seed=seed)
    typer.echo(f"wrote {path}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
