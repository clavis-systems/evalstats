"""Generate examples/arena.csv -- a toy head-to-head tournament.

True (log-strength) skills are built in so you can check the fit recovers them:
  gpt-ish   +0.9   (clearly best)
  claude-ish +0.8
  llama-ish  0.0
  mistral-ish -0.7
  small-ish  -1.0  (clearly worst)
gpt-ish vs claude-ish is a near tie; everything else is separable.
Run:  python scripts/make_arena.py
"""

from __future__ import annotations

import csv
import itertools
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "arena.csv")
SKILL = {
    "gpt-ish": 0.9,
    "claude-ish": 0.8,
    "llama-ish": 0.0,
    "mistral-ish": -0.7,
    "small-ish": -1.0,
}
N_PER_PAIR = 60
TIE_RATE = 0.15


def main() -> None:
    rng = np.random.default_rng(0)
    models = list(SKILL)
    rows = [("model_a", "model_b", "outcome")]
    for a, b in itertools.combinations(models, 2):
        p_a = 1.0 / (1.0 + np.exp(-(SKILL[a] - SKILL[b])))
        for _ in range(N_PER_PAIR):
            if rng.random() < TIE_RATE:
                rows.append((a, b, "tie"))
            else:
                rows.append((a, b, "a" if rng.random() < p_a else "b"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"wrote {len(rows) - 1} comparisons to {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
