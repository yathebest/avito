"""Recalculate the reported HierText validation scores."""

from __future__ import annotations

import csv
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROBABILITY_LIMIT = 1e-6


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def logit(value: float) -> float:
    p = min(1 - PROBABILITY_LIMIT, max(PROBABILITY_LIMIT, value))
    return math.log(p / (1 - p))


def score(probabilities: list[float], labels: list[int]) -> tuple[float, float]:
    n = len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n
    accuracy = sum((p >= 0.5) == bool(y)
                   for p, y in zip(probabilities, labels)) / n
    return 1 - brier, accuracy


def main() -> None:
    rows = read(HERE / "val.csv")
    predicted = read(HERE / "paddle_predictions.csv")
    if len(rows) != len(predicted) or len(rows) != 6000:
        raise ValueError("Expected 6000 aligned predictions")
    for expected, actual in zip(rows, predicted):
        for key in ("image_relpath", "rotation_deg", "y_180"):
            if expected[key] != actual[key]:
                raise ValueError(f"Prediction order differs at {key}")
    for i in range(0, len(rows), 2):
        if (rows[i]["image_relpath"] != rows[i + 1]["image_relpath"]
                or rows[i]["rotation_deg"] != "0"
                or rows[i + 1]["rotation_deg"] != "180"):
            raise ValueError(f"Invalid rotation pair at row {i}")
    labels = [int(row["y_180"]) for row in rows]
    raw = [float(row["p_180"]) for row in predicted]
    paired = []
    for index, value in enumerate(raw):
        other = raw[index ^ 1]
        strength = (logit(value) - logit(other)) / 2
        paired.append(1 / (1 + math.exp(-strength)))
    for name, values in (("One pass", raw), ("Two rotations", paired)):
        brier, accuracy = score(values, labels)
        print(f"{name}: n={len(labels)}, 1-Brier={brier:.8f}, accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()
