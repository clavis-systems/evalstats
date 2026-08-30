"""Read per-item evaluation results into a tidy DataFrame.

The rest of evalstats works on a DataFrame with four columns::

    model     str   -- the system under test
    task      str   -- benchmark / sub-task the item belongs to (the cluster)
    item_id   str   -- stable id of the item, shared across models
    score     float -- 0/1 for pass-fail, or any real number for graded metrics

``load_results`` accepts CSV or JSON Lines with flexible column names, plus the
sample logs written by EleutherAI's lm-evaluation-harness (``--log_samples``).
"""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "from_lighteval",
    "from_lm_eval_harness",
    "from_records",
    "load_results",
]


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))


REQUIRED_COLUMNS = ("model", "task", "item_id", "score")

_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("model", "system", "run", "model_name", "system_name"),
    "task": ("task", "benchmark", "dataset", "subtask", "group", "category"),
    "item_id": ("item_id", "id", "doc_id", "example_id", "qid", "question_id", "idx"),
    "score": (
        "score",
        "correct",
        "is_correct",
        "acc",
        "accuracy",
        "exact_match",
        "em",
        "pass",
        "passed",
        "reward",
        "value",
    ),
}

# Metric names we recognise in lm-eval-harness sample lines, in auto-pick
# priority order. Real sample logs key metrics as "<name>,<filter>" (e.g.
# "acc,none", "exact_match,strict-match"), so we always compare on the part
# before the comma.
_LM_EVAL_METRICS = (
    "acc",
    "exact_match",
    "acc_norm",
    "em",
    "f1",
    "mc1",
    "mc2",
    "pass@1",
    "score",
)


_BOOL_WORDS = {
    "true": 1.0,
    "false": 0.0,
    "yes": 1.0,
    "no": 0.0,
    "correct": 1.0,
    "incorrect": 0.0,
    "pass": 1.0,
    "fail": 0.0,
}


def _coerce_score(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    # string-like (object, or the pandas >=3 str dtype)
    lowered = series.astype("string").str.strip().str.lower()
    mapped = lowered.map(_BOOL_WORDS)
    if mapped.notna().all():
        return mapped.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canon, names in _ALIASES.items():
        if canon in df.columns:
            continue
        for cand in names:
            if cand in lower:
                rename[lower[cand]] = canon
                break
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing required column(s) {missing}; got {list(df.columns)}. "
            "Rename your columns or pass a column mapping."
        )
    out = df[list(REQUIRED_COLUMNS)].copy()
    out["model"] = out["model"].astype(str)
    out["task"] = out["task"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["score"] = _coerce_score(out["score"])
    bad = int(out["score"].isna().sum())
    if bad:
        raise ValueError(f"{bad} row(s) have a non-numeric score after coercion")

    dup = int(out.duplicated(subset=["model", "task", "item_id"]).sum())
    if dup:
        warnings.warn(
            f"{dup} row(s) share a (model, task, item_id) key; model comparison "
            "averages such rows together. De-duplicate or make item_id unique.",
            stacklevel=2,
        )
    return out.reset_index(drop=True)


def from_records(records: Iterable[dict]) -> pd.DataFrame:
    """Build the tidy frame from an iterable of dict rows."""
    return _normalize_columns(pd.DataFrame(list(records)))


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results(path: str, *, fmt: str | None = None) -> pd.DataFrame:
    """Load results from ``path`` and return the tidy 4-column frame.

    fmt : "csv" | "jsonl" | "json" | None (inferred from the extension)
    """
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        fmt = {
            ".csv": "csv",
            ".tsv": "csv",
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
            ".json": "json",
        }.get(ext, "jsonl")

    if fmt == "csv":
        sep = "\t" if path.lower().endswith(".tsv") else ","
        return _normalize_columns(pd.read_csv(path, sep=sep))
    if fmt == "jsonl":
        return from_records(_read_jsonl(path))
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        return from_records(data)
    raise ValueError(f"unknown fmt {fmt!r}")


def _normalize_metric_key(key: str) -> str:
    """``"acc,none"`` -> ``"acc"``; ``"exact_match,strict-match"`` -> ``"exact_match"``."""
    return key.split(",", 1)[0].strip()


def _pick_metric(record: dict, metric: str | None) -> tuple[str, float] | None:
    """Return ``(raw_key, value)`` for the chosen metric in one sample record.

    Works on an lm-eval sample line or a lighteval ``metric`` dict alike.
    ``metric`` may be a bare name (``"acc"``) or a full key (``"acc,none"``).
    When several filters expose the same metric, a ``,none`` filter wins;
    otherwise a bare name is ambiguous and raises.
    """
    numeric = {k: float(v) for k, v in record.items() if _is_number(v)}
    by_norm: dict[str, list[str]] = {}
    for raw in numeric:
        by_norm.setdefault(_normalize_metric_key(raw), []).append(raw)

    def _resolve(raws: list[str]) -> str:
        if len(raws) == 1:
            return raws[0]
        none_filtered = [r for r in raws if r.split(",", 1)[-1] == "none"]
        if len(none_filtered) == 1:
            return none_filtered[0]
        return min(raws)

    if metric is not None:
        if metric in numeric:  # caller passed a full key
            return metric, numeric[metric]
        want = _normalize_metric_key(metric)
        if want not in by_norm:
            return None
        raws = by_norm[want]
        if len(raws) > 1 and not [r for r in raws if r.split(",", 1)[-1] == "none"]:
            raise ValueError(
                f"metric {metric!r} is ambiguous across filters {sorted(raws)}; "
                "pass the full key instead"
            )
        chosen = _resolve(raws)
        return chosen, numeric[chosen]

    for name in _LM_EVAL_METRICS:
        if name in by_norm:
            chosen = _resolve(by_norm[name])
            return chosen, numeric[chosen]
    return None


def from_lm_eval_harness(
    path_or_glob: str,
    *,
    metric: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Load lm-evaluation-harness ``--log_samples`` output.

    ``path_or_glob`` may point at a single ``samples_*.jsonl`` file, a directory
    containing them, or a glob. The model name is taken from ``model`` if given,
    otherwise from the containing directory; the task from the ``samples_<task>``
    filename; the score from ``metric`` (or the first recognised metric key).
    """
    if os.path.isdir(path_or_glob):
        files = sorted(
            glob.glob(os.path.join(path_or_glob, "**", "samples_*.jsonl"), recursive=True)
        )
    else:
        files = sorted(glob.glob(path_or_glob))
    if not files:
        raise FileNotFoundError(f"no samples_*.jsonl found for {path_or_glob!r}")

    records: list[dict] = []
    seen_numeric_keys: set[str] = set()
    id_counts: dict[tuple[str, str, str], int] = {}
    for fp in files:
        fname = os.path.basename(fp)
        m = re.match(r"samples_(?P<task>.+?)_\d{4}-\d{2}-\d{2}.*\.jsonl$", fname) or re.match(
            r"samples_(?P<task>.+)\.jsonl$", fname
        )
        task = m.group("task") if m else fname
        this_model = model or os.path.basename(os.path.dirname(fp)) or "model"
        for line in _read_jsonl(fp):
            picked = _pick_metric(line, metric)
            if picked is None:
                seen_numeric_keys.update(k for k, v in line.items() if _is_number(v))
                continue
            raw_id = str(line.get("doc_id", line.get("id", len(records))))
            key = (this_model, task, raw_id)
            id_counts[key] = id_counts.get(key, 0) + 1
            # Some published sample logs concatenate several runs of the same
            # docs; disambiguate repeats so downstream pairing does not merge them.
            item_id = raw_id if id_counts[key] == 1 else f"{raw_id}#{id_counts[key]}"
            records.append(
                {"model": this_model, "task": task, "item_id": item_id, "score": picked[1]}
            )
    if not records:
        hint = ", ".join(sorted(seen_numeric_keys)) or "none"
        raise ValueError(
            f"no usable metric found; numeric keys present were: {hint}. "
            "Pass metric=<one of those>."
        )
    return from_records(records)


# --------------------------------------------------------------------------- #
# HuggingFace lighteval `details` output (parquet, or json/jsonl)
# --------------------------------------------------------------------------- #
_DETAIL_DATE_RE = re.compile(
    r"^details_(?P<task>.+)_\d{4}-\d{2}-\d{2}T[^/\\]*\.(?:parquet|jsonl?|json)$"
)


def _lighteval_task(fname: str) -> str:
    m = _DETAIL_DATE_RE.match(fname)
    if m:
        return m.group("task")
    stem = re.sub(r"\.(?:parquet|jsonl?|json)$", "", fname)
    return stem.removeprefix("details_")


def _lighteval_model(fp: str, override: str | None) -> str:
    if override:
        return override
    parent = os.path.basename(os.path.dirname(fp))
    if re.match(r"\d{4}-\d{2}-\d{2}T", parent):  # .../details/<model>/<date_id>/file
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        return grandparent or parent
    return parent or "model"


def _read_detail_rows(fp: str) -> list[dict]:
    low = fp.lower()
    if low.endswith(".parquet"):
        try:
            frame = pd.read_parquet(fp)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "reading lighteval .parquet details needs pyarrow -- "
                "pip install 'evalstats[lighteval]'"
            ) from exc
        return frame.to_dict("records")
    if low.endswith(".jsonl"):
        return _read_jsonl(fp)
    with open(fp, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data) if isinstance(data, list) else [data]


def _detail_metric_map(row: dict) -> dict | None:
    for key in ("metric", "metrics"):
        val = row.get(key)
        if isinstance(val, dict):
            return val
    return None


def _detail_row_id(row: dict, i: int) -> str:
    for k in ("id", "example_index", "doc_id", "index"):
        v = row.get(k)
        if v is not None and not isinstance(v, (list, dict)):
            return str(v)
    return str(i)


def from_lighteval(
    path_or_glob: str,
    *,
    metric: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Load HuggingFace lighteval ``details`` output.

    ``path_or_glob`` may point at a details directory
    (``.../details/<model>/<date_id>/``), a single ``details_*.parquet`` file, a
    ``details_*.json`` / ``.jsonl`` file, or a glob. Each row carries a
    ``metric`` (older: ``metrics``) dict of per-sample scores; the score is
    chosen from ``metric`` or the first recognised metric name. The model comes
    from ``model`` or the ``details/<model>/<date_id>`` path; the task from the
    ``details_<task>_<date>`` filename (lighteval task names keep their ``|``).

    Reading ``.parquet`` needs the ``lighteval`` extra (``pyarrow``); JSON does
    not.
    """
    if os.path.isdir(path_or_glob):
        files = sorted(
            f
            for pat in ("details_*.parquet", "details_*.jsonl", "details_*.json")
            for f in glob.glob(os.path.join(path_or_glob, "**", pat), recursive=True)
        )
    else:
        files = sorted(glob.glob(path_or_glob))
    files = [f for f in files if "results_" not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"no details_*.parquet/json found for {path_or_glob!r}")

    records: list[dict] = []
    seen_keys: set[str] = set()
    id_counts: dict[tuple[str, str, str], int] = {}
    for fp in files:
        task = _lighteval_task(os.path.basename(fp))
        this_model = _lighteval_model(fp, model)
        for i, row in enumerate(_read_detail_rows(fp)):
            mmap = _detail_metric_map(row)
            if mmap is None:
                continue
            picked = _pick_metric(mmap, metric)
            if picked is None:
                seen_keys.update(k for k, v in mmap.items() if _is_number(v))
                continue
            raw_id = _detail_row_id(row, i)
            key = (this_model, task, raw_id)
            id_counts[key] = id_counts.get(key, 0) + 1
            item_id = raw_id if id_counts[key] == 1 else f"{raw_id}#{id_counts[key]}"
            records.append(
                {"model": this_model, "task": task, "item_id": item_id, "score": picked[1]}
            )
    if not records:
        hint = ", ".join(sorted(seen_keys)) or "none"
        raise ValueError(
            f"no usable metric found in lighteval details; metric keys seen: {hint}. "
            "Pass metric=<one of those>."
        )
    return from_records(records)
