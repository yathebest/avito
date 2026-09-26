"""Small helpers for reading HierText annotations and selecting text lines."""

from __future__ import annotations

import gzip
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterator


READ_BLOCK_SIZE = 64 * 1024
MIN_HORIZONTAL_SPAN_PX = 25.0
MAX_ABS_SLOPE = 0.5
MAX_BACKTRACK_FRACTION = 0.10


def iter_annotations(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while True:
            block = stream.read(READ_BLOCK_SIZE)
            if not block:
                raise ValueError("Annotations array not found")
            buffer += block
            match = re.search(r'"annotations"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > 1_000_000:
                raise ValueError("Annotations array not found near the file start")
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
                    raise ValueError("Incomplete image annotation") from None
                buffer += block
                continue
            if not isinstance(annotation, dict) or "image_id" not in annotation:
                raise ValueError("Unexpected annotation format")
            yield annotation
            buffer = buffer[end:]


def word_centre(word: dict) -> tuple[float, float] | None:
    vertices = word.get("vertices") or []
    if not vertices or any(len(point) != 2 for point in vertices):
        return None
    return (sum(point[0] for point in vertices) / len(vertices),
            sum(point[1] for point in vertices) / len(vertices))


def label_line(line: dict) -> tuple[int | None, str, str, int]:
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


def line_rectangle(vertices: list[list[float]], width: int, height: int,
                   padding_fraction: float) -> tuple[int, int, int, int] | None:
    if len(vertices) != 4:
        return None
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    margin = max(1, round((max(ys) - min(ys)) * padding_fraction))
    left = max(0, math.floor(min(xs) - margin))
    top = max(0, math.floor(min(ys) - margin))
    right = min(width, math.ceil(max(xs) + margin))
    bottom = min(height, math.ceil(max(ys) + margin))
    if right - left < 8 or bottom - top < 4:
        return None
    return left, top, right, bottom
