"""Read the competition test from its ZIP or an extracted directory."""

from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path


EXPECTED_ROWS = 20_000


def find_test_data(project_dir: Path) -> Path:
    explicit = os.environ.get("AVITO_TEST_PATH") or os.environ.get("AVITO_TEST_ZIP")
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = (
        project_dir / "test.zip",
        project_dir / "test_data",
        Path.home() / "Downloads" / "test.zip" / "test",
        Path.home() / "Downloads" / "test.zip",
        Path.home() / "Downloads" / "test",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("test.zip or extracted test directory not found")


class TestData:
    def __init__(self, path: Path):
        self.path = path
        self.archive: zipfile.ZipFile | None = None
        self.image_dir: Path | None = None
        if path.is_file():
            self.archive = zipfile.ZipFile(path)
            sample = self.archive.read("sample_submission.csv")
        elif path.is_dir():
            if (path / "test" / "images").is_dir():
                self.image_dir = path / "test" / "images"
                sample_path = path / "sample_submission.csv"
            elif (path / "images").is_dir():
                self.image_dir = path / "images"
                sample_path = path / "sample_submission.csv"
                if not sample_path.is_file():
                    sample_path = path.parent / "sample_submission.csv"
            else:
                raise FileNotFoundError(f"images directory not found in {path}")
            sample = sample_path.read_bytes()
        else:
            raise FileNotFoundError(path)
        reader = csv.DictReader(io.StringIO(sample.decode("utf-8-sig")))
        if reader.fieldnames != ["image_id", "p_180"]:
            raise ValueError(f"Unexpected sample columns: {reader.fieldnames}")
        self.image_ids = [row["image_id"] for row in reader]
        if len(self.image_ids) != EXPECTED_ROWS or len(set(self.image_ids)) != EXPECTED_ROWS:
            raise ValueError("Expected 20,000 unique image IDs")

    def __enter__(self) -> "TestData":
        return self

    def __exit__(self, *_: object) -> None:
        if self.archive is not None:
            self.archive.close()

    def read_image(self, image_id: str) -> bytes:
        name = f"{image_id}.png"
        if self.archive is not None:
            return self.archive.read(f"test/images/{name}")
        assert self.image_dir is not None
        return (self.image_dir / name).read_bytes()

    def check_images(self) -> None:
        if self.archive is not None:
            missing = [image_id for image_id in self.image_ids
                       if f"test/images/{image_id}.png" not in self.archive.NameToInfo]
        else:
            assert self.image_dir is not None
            missing = [image_id for image_id in self.image_ids
                       if not (self.image_dir / f"{image_id}.png").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing test image: {missing[0]}")
