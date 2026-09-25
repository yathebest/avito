"""Prepare conservative HierText 0/180 pairs with a disjoint validation set.

Uses HierText's ordered words to select likely upright real scene-text lines.
For each selected original crop, its exactly rotated counterpart is described
in the CSV and created on demand by the training Dataset. Labels are inferred,
not human-verified orientation ground truth.

Run from the project folder: python prepare_real_orientation.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image

from build_train_orientation_csv import label_line
from make_hiertext_crops import iter_annotations, line_rectangle


PROJECT_DIR = Path(r"C:\Users\User\Desktop\avito_stazh")
TRAIN_DIR = PROJECT_DIR / "old_train_data"
VAL_DIR = PROJECT_DIR / "old_validation_data"
VAL_JSON = VAL_DIR / "validation.jsonl.gz"
VAL_IMAGE_DIR = VAL_DIR / "validation"
VAL_OUTPUT_DIR = VAL_DIR / "orientation_val"
SEED = 20260925
TRAIN_BASE_CROPS = 40_000
VAL_BASE_CROPS = 3_000
MIN_WORDS = 3
MAX_ABS_ANGLE = 10.0


def add_pair(writer: csv.writer, rel_path: str, source_id: str) -> None:
    writer.writerow((rel_path, 0, 0, source_id))
    writer.writerow((rel_path, 180, 1, source_id))


def make_train_csv() -> None:
    source_csv = TRAIN_DIR / "orientation_labels.csv"
    candidates = []
    with source_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (row["base_p180"] == "0"
                    and int(row["word_count"]) >= MIN_WORDS
                    and abs(float(row["reading_angle_deg"])) <= MAX_ABS_ANGLE):
                candidates.append((row["crop_file"], row["source_image_id"]))
    if len(candidates) < TRAIN_BASE_CROPS:
        raise ValueError(f"Only {len(candidates)} train candidates")
    chosen = random.Random(SEED).sample(candidates, TRAIN_BASE_CROPS)
    destination = TRAIN_DIR / "real_orientation_train.csv"
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_relpath", "rotation_deg", "y_180", "source_image_id"))
        for filename, image_id in chosen:
            add_pair(writer, f"old_train_data/images/{filename}", image_id)
    print(f"Train: {len(chosen)} source crops, {len(chosen)*2} labelled views: {destination}", flush=True)


def make_val_csv() -> None:
    candidates = []
    for image_number, annotation in enumerate(iter_annotations(VAL_JSON), 1):
        image_id = annotation["image_id"]
        index = 0
        for paragraph in annotation.get("paragraphs", []):
            for line in paragraph.get("lines", []):
                base, angle, _, words = label_line(line)
                if (base == 0 and words >= MIN_WORDS
                        and abs(float(angle)) <= MAX_ABS_ANGLE):
                    candidates.append((image_id, index, line["vertices"]))
                index += 1
        if image_number % 500 == 0:
            print(f"Parsed {image_number} validation annotations", flush=True)
    if len(candidates) < VAL_BASE_CROPS:
        raise ValueError(f"Only {len(candidates)} validation candidates")
    chosen = random.Random(SEED + 1).sample(candidates, VAL_BASE_CROPS)
    VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_images = VAL_OUTPUT_DIR / "images"
    output_images.mkdir(exist_ok=True)
    destination = VAL_OUTPUT_DIR / "val.csv"
    made = 0
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_relpath", "rotation_deg", "y_180", "source_image_id"))
        for image_id, index, vertices in chosen:
            source = VAL_IMAGE_DIR / f"{image_id}.jpg"
            if not source.is_file():
                continue
            with Image.open(source) as image:
                image = image.convert("RGB")
                box = line_rectangle(vertices, *image.size, 0.05)
                if box is None:
                    continue
                filename = f"{image_id}_line_{index:04d}.png"
                image.crop(box).save(output_images / filename)
            add_pair(writer, f"old_validation_data/orientation_val/images/{filename}",
                     image_id)
            made += 1
    print(f"Validation: {made} source crops, {made*2} labelled views: {destination}", flush=True)


def main() -> None:
    if not VAL_JSON.is_file():
        raise FileNotFoundError(VAL_JSON)
    make_train_csv()
    make_val_csv()


if __name__ == "__main__":
    main()
