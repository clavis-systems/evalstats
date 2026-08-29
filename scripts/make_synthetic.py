"""Generate examples/results.jsonl -- a toy 3-model / 5-task eval log.

Ground truth built in:
  * model_a and model_b are within noise of each other (true gap ~0.01)
  * model_c is clearly best (true gap ~0.12 over model_a)
  * per-task difficulty varies, so items are genuinely clustered
Run:  python scripts/make_synthetic.py
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "results.jsonl")
TASKS = ["arithmetic", "reading", "code", "trivia", "logic", "science", "history", "grammar"]
ITEMS_PER_TASK = 150
BASE = {"model_a": 0.70, "model_b": 0.71, "model_c": 0.82}


def main() -> None:
    rng = np.random.default_rng(0)
    task_offset = {t: o for t, o in zip(TASKS, rng.uniform(-0.12, 0.12, len(TASKS)))}

    rows = []
    for task in TASKS:
        # a shared per-item "difficulty" makes the models correlated within a task
        difficulty = rng.normal(0.0, 0.15, ITEMS_PER_TASK)
        for i in range(ITEMS_PER_TASK):
            for model, base in BASE.items():
                p = np.clip(base + task_offset[task] - difficulty[i], 0.02, 0.98)
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "item_id": f"{task}-{i:03d}",
                        "score": int(rng.random() < p),
                    }
                )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows to {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
