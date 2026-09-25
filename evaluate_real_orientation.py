"""Evaluate the saved compact model on disjoint HierText validation crops.

Labels are based on word geometry in HierText, not manually verified. The
official Avito test set is neither read nor labelled here.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from train_real_orientation import (BATCH_SIZE, PROJECT_DIR, RealPairsDataset,
                                    SEED, VAL_CSV, make_loader, seed_everything)


CHECKPOINT = PROJECT_DIR / "models" / "real_hiertext_mobilenetv3_small" / "best.pt"
OUTPUT_PATH = PROJECT_DIR / "models" / "real_hiertext_mobilenetv3_small" / "final_validation.json"
PREDICTIONS_PATH = PROJECT_DIR / "models" / "real_hiertext_mobilenetv3_small" / "validation_predictions.csv"


@torch.inference_mode()
def main() -> None:
    seed_everything(SEED)
    torch.set_num_threads(4)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = RealPairsDataset(VAL_CSV, augment=False)
    loader = make_loader(VAL_CSV, augment=False, seed=SEED + 1)
    predictions = []
    started = time.perf_counter()
    for images, _ in loader:
        probabilities = model(images.to(device, non_blocking=True)).flatten().sigmoid()
        predictions.extend(probabilities.cpu().tolist())
    elapsed = time.perf_counter() - started
    if len(predictions) != len(dataset):
        raise RuntimeError("Prediction count does not match validation rows")
    rows = dataset.rows
    brier = sum((p - int(row["y_180"])) ** 2
                for p, row in zip(predictions, rows)) / len(rows)
    correct = sum((p >= 0.5) == bool(int(row["y_180"]))
                  for p, row in zip(predictions, rows))
    output = {"model": "mobilenet_v3_small fine-tuned on real HierText pairs",
              "validation": "official HierText validation; geometry-inferred base labels",
              "checkpoint_epoch": checkpoint["epoch"],
              "n": len(rows), "brier": brier,
              "score_1_minus_brier": 1.0 - brier,
              "accuracy": correct / len(rows),
              "seconds": elapsed, "device": str(device),
              "batch_size": BATCH_SIZE}
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_relpath", "rotation_deg", "y_180", "p_180"))
        writer.writerows((row["image_relpath"], row["rotation_deg"],
                          row["y_180"], f"{p:.10f}")
                         for row, p in zip(rows, predictions))
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
