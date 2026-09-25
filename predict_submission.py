"""Predict 0°/180° text orientation with local PaddleOCR weights."""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from paddleocr import TextLineOrientationClassification

from test_input import TestData, find_test_data


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "PP-LCNet_x1_0_textline_ori_infer"
OUTPUT_CSV = PROJECT_DIR / "submission.csv"
MODEL_NAME = "PP-LCNet_x1_0_textline_ori"
DEVICE = "cpu"
BATCH_SIZE = 32  # 16 crops and their 180° copies
PROBABILITY_LIMIT = 1e-6


def probability_180(result) -> float:
    label = result["label_names"][0]
    score = float(result["scores"][0])
    if label not in ("0_degree", "180_degree") or not math.isfinite(score):
        raise ValueError(f"Unexpected model output: {label}, {score}")
    if not 0 <= score <= 1:
        raise ValueError(f"Invalid confidence: {score}")
    return score if label == "180_degree" else 1 - score


def logit(probability: float) -> float:
    p = min(1 - PROBABILITY_LIMIT, max(PROBABILITY_LIMIT, probability))
    return math.log(p / (1 - p))


def combine(original: float, rotated: float) -> float:
    """Average two orientation estimates in log-odds space."""
    value = (logit(original) - logit(rotated)) / 2
    return 1 / (1 + math.exp(-value))


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    with TestData(find_test_data(PROJECT_DIR)) as test:
        test.check_images()
        model = TextLineOrientationClassification(
            model_name=MODEL_NAME, model_dir=str(MODEL_DIR), device=DEVICE)
        temporary = OUTPUT_CSV.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("image_id", "p_180"))
            for start in range(0, len(test.image_ids), BATCH_SIZE // 2):
                ids = test.image_ids[start:start + BATCH_SIZE // 2]
                images = []
                for image_id in ids:
                    with Image.open(io.BytesIO(test.read_image(image_id))) as source:
                        image = source.convert("RGB")
                    images.append(np.asarray(image))
                    images.append(np.asarray(image.transpose(Image.Transpose.ROTATE_180)))
                results = model.predict(images, batch_size=BATCH_SIZE)
                if len(results) != len(images):
                    raise RuntimeError("Model returned an unexpected number of predictions")
                for index, image_id in enumerate(ids):
                    original = probability_180(results[2 * index])
                    rotated = probability_180(results[2 * index + 1])
                    writer.writerow((image_id, f"{combine(original, rotated):.8f}"))
                done = start + len(ids)
                if done % 1024 == 0 or done == len(test.image_ids):
                    print(f"Predicted {done}/{len(test.image_ids)}", flush=True)
        os.replace(temporary, OUTPUT_CSV)
    print(f"Saved {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
