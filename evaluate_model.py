"""Run the local model on the official HierText validation split."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import TextLineOrientationClassification

from predict_submission import probability_180


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("HIERTEXT_ROOT", str(PROJECT_DIR)))
VAL_CSV = PROJECT_DIR / "validation" / "val.csv"
OUTPUT_CSV = PROJECT_DIR / "validation" / "paddle_predictions.csv"
MODEL_DIR = PROJECT_DIR / "models" / "PP-LCNet_x1_0_textline_ori_infer"
BATCH_SIZE = 32


def main() -> None:
    with VAL_CSV.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    model = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=str(MODEL_DIR), device="cpu")
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_relpath", "rotation_deg", "y_180", "p_180"))
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            images = []
            for row in batch:
                path = DATA_ROOT / row["image_relpath"]
                with Image.open(path) as source:
                    image = source.convert("RGB")
                if row["rotation_deg"] == "180":
                    image = image.transpose(Image.Transpose.ROTATE_180)
                images.append(np.asarray(image))
            results = model.predict(images, batch_size=BATCH_SIZE)
            if len(results) != len(batch):
                raise RuntimeError("Prediction count differs from batch size")
            for row, result in zip(batch, results):
                writer.writerow((row["image_relpath"], row["rotation_deg"],
                                 row["y_180"], probability_180(result)))
    print(f"Saved {len(rows)} validation predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
