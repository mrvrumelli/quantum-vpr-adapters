"""Portable descriptor archives with separate gallery and query row metadata."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import Tensor

from .data import ROLES, SPLITS, ImageRecord

CACHE_VERSION = 1


@dataclass(frozen=True)
class DescriptorSet:
    images: tuple[ImageRecord, ...]
    descriptors: Tensor


@dataclass(frozen=True)
class SplitCache:
    split: str
    provenance: dict
    gallery: DescriptorSet
    query: DescriptorSet


def save_cache(path: Path, cache: SplitCache) -> None:
    """Publish both roles in one replacement, with neither exposed if extraction fails."""
    header = {"version": CACHE_VERSION, "split": cache.split, "provenance": cache.provenance}
    arrays = {"metadata": np.asarray(json.dumps(header, sort_keys=True))}
    for role in ROLES:
        group = getattr(cache, role)
        arrays[f"{role}_descriptors"] = group.descriptors.numpy()
        arrays[f"{role}_ids"] = np.asarray([image.image_id for image in group.images], dtype=str)
        arrays[f"{role}_paths"] = np.asarray([image.path for image in group.images], dtype=str)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, **arrays)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_cache(path: Path) -> SplitCache:
    with np.load(path, allow_pickle=False) as archive:
        try:
            header = json.loads(archive["metadata"].item())
            if header["version"] != CACHE_VERSION or header["split"] not in SPLITS:
                raise ValueError(f"{path}: unsupported cache version or split")
            dim = header["provenance"]["descriptor_dim"]
            if type(dim) is not int or dim <= 0:
                raise ValueError(f"{path}: invalid cache descriptor dimension")
            groups = {}
            for role in ROLES:
                vectors = archive[f"{role}_descriptors"]
                ids = archive[f"{role}_ids"]
                paths = archive[f"{role}_paths"]
                if (
                    ids.ndim != 1 or ids.size == 0 or ids.shape != paths.shape
                    or ids.dtype.kind != "U" or paths.dtype.kind != "U"
                    or vectors.dtype != np.float32 or vectors.shape != (ids.size, dim)
                ):
                    raise ValueError(f"{path}: invalid {role} descriptor rows or image metadata")
                images = tuple(ImageRecord(*row) for row in zip(ids.tolist(), paths.tolist(), strict=True))
                groups[role] = DescriptorSet(images, torch.from_numpy(vectors))
            return SplitCache(header["split"], header["provenance"], **groups)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: invalid cache metadata") from exc
