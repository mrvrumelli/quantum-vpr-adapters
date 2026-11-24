"""CSV manifests and split-local relevance labels."""

import csv
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")
EVAL_SPLITS = ("val", "test")
ROLES = ("gallery", "query")
IMAGE_COLUMNS = ("image_id", "path")
POSITIVE_COLUMNS = ("query_id", "gallery_id")


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str


def _read_rows(path: Path, columns: tuple[str, ...]) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != list(columns):
            raise ValueError(f"{path}: expected columns {', '.join(columns)}")
        rows = []
        for row in reader:
            if len(row) != len(columns) or any(not value.strip() for value in row):
                raise ValueError(f"{path}:{reader.line_num}: incomplete or invalid row")
            rows.append(row)
    return rows


def load_manifests(root: Path) -> dict[str, dict[str, tuple[ImageRecord, ...]]]:
    """Read image metadata only and reject paths or IDs shared across roles or splits."""
    root = root.resolve()
    manifests = {}
    seen_ids = set()
    seen_paths = set()
    for split in SPLITS:
        manifests[split] = {}
        for role in ROLES:
            manifest = root / split / f"{role}.csv"
            records = []
            for image_id, path in _read_rows(manifest, IMAGE_COLUMNS):
                relative = Path(path)
                resolved = (root / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or root not in resolved.parents:
                    raise ValueError(f"{manifest}: path must be relative to the dataset root: {path}")
                if image_id in seen_ids or resolved in seen_paths:
                    raise ValueError(f"{manifest}: reused image ID or path: {image_id}, {path}")
                seen_ids.add(image_id)
                seen_paths.add(resolved)
                records.append(ImageRecord(image_id, path))
            if not records:
                raise ValueError(f"{manifest}: expected at least one image")
            manifests[split][role] = tuple(records)
    return manifests


def load_positives(
    root: Path,
    split: str,
    gallery: tuple[ImageRecord, ...],
    queries: tuple[ImageRecord, ...],
) -> list[set[int]]:
    """Map evaluation positives to gallery indices in query-manifest order."""
    if split not in EVAL_SPLITS:
        raise ValueError("positive labels are used only for val and test evaluation")
    path = root / split / "positives.csv"
    gallery_indices = {record.image_id: index for index, record in enumerate(gallery)}
    query_positives: dict[str, set[int]] = {record.image_id: set() for record in queries}
    for query_id, gallery_id in _read_rows(path, POSITIVE_COLUMNS):
        if query_id not in query_positives or gallery_id not in gallery_indices:
            raise ValueError(f"{path}: unknown query or gallery ID: {query_id}, {gallery_id}")
        query_positives[query_id].add(gallery_indices[gallery_id])
    for query_id, positives in query_positives.items():
        if not positives:
            raise ValueError(f"{path}: missing positive for query {query_id}")
    return list(query_positives.values())
