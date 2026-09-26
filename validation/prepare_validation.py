"""Build 0°/180° validation pairs from the official HierText validation split."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

from PIL import Image

from hiertext import iter_annotations, label_line, line_rectangle


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HIERTEXT_ROOT", str(PROJECT_DIR)))
DATA_DIR = DATA_ROOT / "old_validation_data"
ANNOTATIONS = DATA_DIR / "validation.jsonl.gz"
SOURCE_IMAGES = DATA_DIR / "validation"
OUTPUT_IMAGES = DATA_DIR / "orientation_val" / "images"
OUTPUT_CSV = PROJECT_DIR / "validation" / "val.csv"
SEED = 20260926
BASE_CROPS = 3000
MIN_WORDS = 3
MAX_ABS_ANGLE = 10.0


def main() -> None:
    candidates = []
    for annotation in iter_annotations(ANNOTATIONS):
        image_id = annotation["image_id"]
        line_index = 0
        for paragraph in annotation.get("paragraphs", []):
            for line in paragraph.get("lines", []):
                orientation, angle, _, words = label_line(line)
                if (orientation == 0 and words >= MIN_WORDS
                        and abs(float(angle)) <= MAX_ABS_ANGLE):
                    candidates.append((image_id, line_index, line["vertices"]))
                line_index += 1
    if len(candidates) < BASE_CROPS:
        raise ValueError(f"Only {len(candidates)} suitable validation lines")
    selected = random.Random(SEED).sample(candidates, BASE_CROPS)
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_relpath", "rotation_deg", "y_180", "source_image_id"))
        for image_id, line_index, vertices in selected:
            source_path = SOURCE_IMAGES / f"{image_id}.jpg"
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                box = line_rectangle(vertices, *image.size, 0.05)
                if box is None:
                    continue
                filename = f"{image_id}_line_{line_index:04d}.png"
                image.crop(box).save(OUTPUT_IMAGES / filename)
            relpath = f"old_validation_data/orientation_val/images/{filename}"
            writer.writerow((relpath, 0, 0, image_id))
            writer.writerow((relpath, 180, 1, image_id))
    print(f"Saved validation pairs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
