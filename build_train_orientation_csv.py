"""Build reliable 0/180 labels from HierText's ordered words.

Run from the project directory with:
    python build_train_orientation_csv.py

HierText does not directly provide orientation labels. This script labels only
horizontal lines with at least two readable words whose centres consistently
move in the documented reading order. Other crops remain explicitly unlabeled.
The paired CSV describes deterministic 180-degree augmentation without saving
another 260k image files. The separate HierText validation split should be
used for model selection, rather than splitting these rows at random.
"""

from __future__ import annotations

import csv
import math
import os
import unicodedata
from collections import Counter
from pathlib import Path

from make_hiertext_crops import iter_annotations


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "train_data"
ANNOTATIONS_PATH = TRAIN_DIR / "json" / "train.jsonl"
MANIFEST_PATH = TRAIN_DIR / "images" / "manifest.csv"
LABELS_PATH = TRAIN_DIR / "orientation_labels.csv"
PAIRS_PATH = TRAIN_DIR / "orientation_pairs.csv"

MIN_HORIZONTAL_SPAN_PX = 25.0
MAX_ABS_SLOPE = 0.5  # Reading direction must stay within about 27 degrees of horizontal.
MAX_BACKTRACK_FRACTION = 0.10


def word_centre(word: dict) -> tuple[float, float] | None:
    vertices = word.get("vertices") or []
    if not vertices or any(len(point) != 2 for point in vertices):
        return None
    return (
        sum(point[0] for point in vertices) / len(vertices),
        sum(point[1] for point in vertices) / len(vertices),
    )


def label_line(line: dict) -> tuple[int | None, str, str, int]:
    """Return (base label, angle, reason, word count) from reading order."""
    if not line.get("legible") or line.get("vertical"):
        return None, "", "illegible_or_vertical", 0
    text = line.get("text", "")
    if any(unicodedata.bidirectional(ch) in {"R", "AL"} for ch in text):
        return None, "", "right_to_left_script", 0

    words = [word for word in line.get("words", [])
             if word.get("legible") and word.get("text", "").strip()]
    centres = [word_centre(word) for word in words]
    if len(centres) < 2 or any(centre is None for centre in centres):
        return None, "", "fewer_than_two_readable_words", len(centres)

    dx = centres[-1][0] - centres[0][0]
    dy = centres[-1][1] - centres[0][1]
    if abs(dx) < MIN_HORIZONTAL_SPAN_PX:
        return None, "", "short_horizontal_span", len(centres)
    if abs(dy) > MAX_ABS_SLOPE * abs(dx):
        return None, "", "near_vertical", len(centres)

    direction = 1 if dx > 0 else -1
    backtrack = sum(max(0.0, -direction * (b[0] - a[0]))
                    for a, b in zip(centres, centres[1:]))
    if backtrack > MAX_BACKTRACK_FRACTION * abs(dx):
        return None, "", "word_order_backtracks", len(centres)

    angle = math.degrees(math.atan2(dy, dx))
    return (0 if dx > 0 else 1), f"{angle:.2f}", "ordered_words", len(centres)


def annotation_labels() -> dict[str, tuple[int | None, str, str, int]]:
    """Match the exact line indices used when the existing crops were made."""
    labels = {}
    for image_number, annotation in enumerate(iter_annotations(ANNOTATIONS_PATH), 1):
        image_id = annotation["image_id"]
        line_index = 0
        for paragraph in annotation.get("paragraphs", []):
            for line in paragraph.get("lines", []):
                crop_file = f"{image_id}_line_{line_index:04d}.png"
                labels[crop_file] = label_line(line)
                line_index += 1
        if image_number % 2000 == 0:
            print(f"Parsed {image_number} annotated source images", flush=True)
    return labels


def main() -> None:
    if not ANNOTATIONS_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Expected train_data/json/train.jsonl and train_data/images/manifest.csv")

    labels = annotation_labels()
    counts = Counter()
    labels_temp = LABELS_PATH.with_suffix(".csv.tmp")
    pairs_temp = PAIRS_PATH.with_suffix(".csv.tmp")
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as manifest_file, \
         labels_temp.open("w", encoding="utf-8", newline="") as label_file, \
         pairs_temp.open("w", encoding="utf-8", newline="") as pair_file:
        reader = csv.DictReader(manifest_file)
        label_writer = csv.writer(label_file)
        pair_writer = csv.writer(pair_file)
        label_writer.writerow(("crop_file", "source_image_id", "line_index", "text",
                               "base_p180", "reading_angle_deg", "label_source", "word_count"))
        pair_writer.writerow(("crop_file", "source_image_id", "rotation_deg", "y_180"))

        for row in reader:
            crop_file = row["crop_file"]
            if crop_file not in labels:
                raise ValueError(f"Crop missing from annotations: {crop_file}")
            base, angle, source, nwords = labels[crop_file]
            label_writer.writerow((crop_file, row["source_image_id"], row["line_index"],
                                   row["text"], "" if base is None else base,
                                   angle, source, nwords))
            counts[source] += 1
            if base is not None:
                pair_writer.writerow((crop_file, row["source_image_id"], 0, base))
                pair_writer.writerow((crop_file, row["source_image_id"], 180, 1 - base))
                counts[f"base_{base}"] += 1

    os.replace(labels_temp, LABELS_PATH)
    os.replace(pairs_temp, PAIRS_PATH)
    print(f"Wrote {sum(counts.values()) - counts['base_0'] - counts['base_1']} crop rows: {LABELS_PATH}")
    print(f"Reliable base labels: {counts['base_0']} upright, {counts['base_1']} upside down")
    print(f"Balanced augmented pairs: {2 * (counts['base_0'] + counts['base_1'])} rows: {PAIRS_PATH}")
    print("Unlabeled reasons:", {key: value for key, value in counts.items()
                                  if key not in ("ordered_words", "base_0", "base_1")})


if __name__ == "__main__":
    main()
