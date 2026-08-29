from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """Small tidy frame: model_c clearly best, model_a ~ model_b."""
    gen = np.random.default_rng(0)
    tasks = ["t1", "t2", "t3"]
    base = {"a": 0.60, "b": 0.61, "c": 0.72}
    rows = []
    for task in tasks:
        diff = gen.normal(0, 0.15, 80)
        off = gen.uniform(-0.1, 0.1)
        for i in range(80):
            for m, p0 in base.items():
                p = np.clip(p0 + off - diff[i], 0.02, 0.98)
                rows.append(
                    {
                        "model": m,
                        "task": task,
                        "item_id": f"{task}-{i}",
                        "score": int(gen.random() < p),
                    }
                )
    return pd.DataFrame(rows)
