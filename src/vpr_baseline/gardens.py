"""Prepare the day-right/night-right Gardens Point Walking protocol."""

import csv
from io import StringIO
from pathlib import Path

from .data import EVAL_SPLITS, IMAGE_COLUMNS, POSITIVE_COLUMNS

# Leave ten unused frames between adjacent parts of the route.
FRAME_RANGES = {"train": range(0, 80), "val": range(90, 140), "test": range(150, 200)}
GALLERY = "day_right"
QUERY = "night_right"


def _csv_text(columns: tuple[str, ...], rows) -> str:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue()


def prepare_gardens_point(root: Path) -> None:
    """Build manifests alongside two manually acquired image traversals."""
    for traversal in (GALLERY, QUERY):
        for frame in range(200):
            image = root / traversal / f"Image{frame:03d}.jpg"
            if not image.is_file():
                raise FileNotFoundError(f"Missing Gardens Point image: {image}")

    files = {}
    for split, frames in FRAME_RANGES.items():
        for role, traversal in (("gallery", GALLERY), ("query", QUERY)):
            files[root / split / f"{role}.csv"] = _csv_text(
                IMAGE_COLUMNS,
                ((f"{traversal}-{frame:03d}", f"{traversal}/Image{frame:03d}.jpg") for frame in frames),
            )
        if split in EVAL_SPLITS:
            files[root / split / "positives.csv"] = _csv_text(
                POSITIVE_COLUMNS,
                ((f"{QUERY}-{frame:03d}", f"{GALLERY}-{frame:03d}") for frame in frames),
            )

    pending = {}
    for path, contents in files.items():
        if not path.exists():
            pending[path] = contents
        elif path.read_text(encoding="utf-8") != contents:
            raise ValueError(f"Existing manifest differs from the Gardens Point protocol: {path}")
    for path, contents in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(contents)
