"""Recompute Brier score from the saved checkpoint on synthetic validation.

Run: python evaluate_synthetic_orientation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from train_synthetic_orientation import (DATA_DIR, OUTPUT_DIR, OrientationDataset,
                                         BATCH_SIZE, NUM_WORKERS)


def main() -> None:
    checkpoint_path = OUTPUT_DIR / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data = OrientationDataset(DATA_DIR / "val.csv", augment=False)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=device.type == "cuda")
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            probs = model(images.to(device, non_blocking=True)).flatten().sigmoid()
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    n = len(all_probs)
    brier = sum((p - y) ** 2 for p, y in zip(all_probs, all_labels)) / n
    accuracy = sum((p >= 0.5) == (y >= 0.5)
                   for p, y in zip(all_probs, all_labels)) / n
    result = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
              "validation_csv": str(DATA_DIR / "val.csv"), "n": n,
              "brier": brier, "score_1_minus_brier": 1.0 - brier,
              "accuracy": accuracy}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT_DIR / "val_predictions.csv").open(
            "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_path", "y_180", "p_180"))
        for row, probability in zip(data.rows, all_probs):
            writer.writerow((row["image_path"], row["y_180"],
                             f"{probability:.10f}"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
