"""Create text-line crops from HierText training images.

Run: python make_hiertext_crops.py

Expected source layout:
    old_train_data/json/train.jsonl
    old_train_data/<image_id>.jpg

Output:
    old_train_data/images/<image_id>_line_<index>.png
    old_train_data/images/manifest.csv

The output crops do not yet have trustworthy 0/180 orientation labels.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from itertools import chain
from pathlib import Path
from typing import Iterator

from PIL import Image


# Paths match the extracted HierText files in the current project.
TRAIN_DIR = Path(__file__).resolve().parent / "old_train_data"
ANNOTATIONS_PATH = TRAIN_DIR / "json" / "train.jsonl"
SOURCE_IMAGES_DIR = TRAIN_DIR
OUTPUT_IMAGES_DIR = TRAIN_DIR / "images"
MANIFEST_PATH = OUTPUT_IMAGES_DIR / "manifest.csv"
PADDING_FRACTION = 0.05
READ_BLOCK_SIZE = 64 * 1024


def iter_annotations(path: Path) -> Iterator[dict]:
    """Read one image annotation at a time from HierText's large JSON object."""
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while True:
            block = stream.read(READ_BLOCK_SIZE)
            if not block:
                raise ValueError("Could not find the annotations array")
            buffer += block
            match = re.search(r'"annotations"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > 1_000_000:
                raise ValueError("Could not find the annotations array near the file start")

        while True:
            buffer = buffer.lstrip()
            if not buffer:
                block = stream.read(READ_BLOCK_SIZE)
                if not block:
                    raise ValueError("Unexpected end of annotations array")
                buffer = block
                continue
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            try:
                annotation, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = stream.read(READ_BLOCK_SIZE)
                if not block:
                    raise ValueError("Incomplete or invalid image annotation") from None
                buffer += block
                continue
            if not isinstance(annotation, dict) or "image_id" not in annotation:
                raise ValueError("Unexpected HierText annotation format")
            yield annotation
            buffer = buffer[end:]


def line_rectangle(vertices: list[list[float]], width: int, height: int,
                   padding_fraction: float) -> tuple[int, int, int, int] | None:
    """Bounding rectangle plus a small margin around one annotated text line."""
    if len(vertices) != 4:
        return None
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    line_height = max(ys) - min(ys)
    margin = max(1, round(line_height * padding_fraction))
    left = max(0, math.floor(min(xs) - margin))
    top = max(0, math.floor(min(ys) - margin))
    right = min(width, math.ceil(max(xs) + margin))
    bottom = min(height, math.ceil(max(ys) + margin))
    if right - left < 8 or bottom - top < 4:
        return None
    return left, top, right, bottom


def main() -> None:
    if not ANNOTATIONS_PATH.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATIONS_PATH}")
    if not SOURCE_IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {SOURCE_IMAGES_DIR}")

    annotations = iter_annotations(ANNOTATIONS_PATH)
    first = next(annotations, None)
    if first is None:
        raise ValueError("No annotations found")

    first_source = SOURCE_IMAGES_DIR / f"{first['image_id']}.jpg"
    if not first_source.is_file():
        raise FileNotFoundError(
            f"Original training JPGs are missing. Expected: {first_source}\n"
            "Download HierText train.tgz and extract its train/*.jpg files into "
            f"{SOURCE_IMAGES_DIR}. The JSON annotations alone contain no pixels."
        )

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    missing = []
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=(
            "crop_file", "source_image_id", "line_index", "text", "x1", "y1", "x2", "y2"))
        writer.writeheader()

        for image_number, annotation in enumerate(chain((first,), annotations), 1):
            image_id = annotation["image_id"]
            source = SOURCE_IMAGES_DIR / f"{image_id}.jpg"
            if not source.is_file():
                missing.append(str(source))
                continue

            with Image.open(source) as image:
                image = image.convert("RGB")
                line_index = 0
                for paragraph in annotation.get("paragraphs", []):
                    for line in paragraph.get("lines", []):
                        current_index = line_index
                        line_index += 1
                        if not line.get("legible", False) or line.get("vertical", False):
                            skipped += 1
                            continue

                        box = line_rectangle(line.get("vertices", []),
                                             *image.size, PADDING_FRACTION)
                        if box is None:
                            skipped += 1
                            continue

                        crop_file = f"{image_id}_line_{current_index:04d}.png"
                        image.crop(box).save(OUTPUT_IMAGES_DIR / crop_file)
                        writer.writerow({
                            "crop_file": crop_file,
                            "source_image_id": image_id,
                            "line_index": current_index,
                            "text": line.get("text", ""),
                            "x1": box[0], "y1": box[1],
                            "x2": box[2], "y2": box[3],
                        })
                        written += 1

            if image_number % 1000 == 0:
                print(f"Processed {image_number} source images; saved {written} crops", flush=True)

    print(f"Saved {written} crops; skipped {skipped} lines; manifest: {MANIFEST_PATH}")
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} source JPGs; first missing: {missing[0]}"
        )


if __name__ == "__main__":
    main()
