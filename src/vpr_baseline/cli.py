"""Command-line entry point for preparing data, extracting descriptors and evaluating."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import BadZipFile

import torch
import torchvision
from PIL import __version__ as pillow_version

from .cache import DescriptorSet, SplitCache, load_cache, save_cache
from .data import EVAL_SPLITS, ROLES, SPLITS, load_manifests, load_positives
from .extract import DEFAULT_IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, extract_descriptors
from .gardens import prepare_gardens_point
from .model import CHECKPOINT_SHA256, CHECKPOINT_URL, DESCRIPTOR_DIM, load_cosplace
from .retrieval import RECALL_KS, rank_cosine, recall_at_k


def _extract(args: argparse.Namespace) -> dict:
    manifests = load_manifests(args.dataset_root)[args.split]
    model = load_cosplace(args.checkpoint)
    image_size = (args.height, args.width)
    groups = {}
    for role, images in manifests.items():
        descriptors = extract_descriptors(
            model, args.dataset_root, images, DESCRIPTOR_DIM,
            image_size=image_size, batch_size=args.batch_size, workers=args.workers, device=args.device,
        )
        groups[role] = DescriptorSet(images, descriptors)

    provenance = {
        "model": "CosPlace ResNet18",
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "descriptor_dim": DESCRIPTOR_DIM,
        "image_size_hw": image_size,
        "resize": "bilinear_antialias",
        "rgb_mean": IMAGE_MEAN,
        "rgb_std": IMAGE_STD,
        "l2_normalized": True,
        "device": args.device,
        "batch_size": args.batch_size,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": pillow_version,
    }
    path = args.cache_dir / f"{args.split}.npz"
    save_cache(path, SplitCache(args.split, provenance, **groups))
    return {
        "split": args.split,
        "cache": str(path),
        "descriptor_dim": DESCRIPTOR_DIM,
        "gallery_count": len(groups["gallery"].images),
        "query_count": len(groups["query"].images),
    }


def _evaluate(args: argparse.Namespace) -> dict:
    manifests = load_manifests(args.dataset_root)[args.split]
    path = args.cache_dir / f"{args.split}.npz"
    cache = load_cache(path)
    if cache.split != args.split or any(getattr(cache, role).images != manifests[role] for role in ROLES):
        raise ValueError(f"{path}: cached split or image rows differ from the manifests. Re-extract descriptors.")
    positives = load_positives(args.dataset_root, args.split, cache.gallery.images, cache.query.images)
    rankings = rank_cosine(
        cache.query.descriptors, cache.gallery.descriptors, max(RECALL_KS), args.query_batch_size
    )
    recalls = recall_at_k(rankings, positives)
    return {
        "split": args.split,
        "gallery_count": len(cache.gallery.images),
        "query_count": len(cache.query.images),
        **{f"Recall@{k}": recall for k, recall in recalls.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-gardens-point", help="write manifests for local Gardens Point images")
    extract = commands.add_parser("extract", help="extract a split using the verified local CosPlace checkpoint")
    evaluate = commands.add_parser("evaluate", help="evaluate a split using cached descriptors")
    for command in (prepare, extract, evaluate):
        command.add_argument("--dataset-root", type=Path, required=True)
    for command, splits in ((extract, SPLITS), (evaluate, EVAL_SPLITS)):
        command.add_argument("--split", choices=splits, required=True)
        command.add_argument("--cache-dir", type=Path, default=Path("artifacts"))

    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    extract.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--workers", type=int, default=0)
    extract.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    evaluate.add_argument("--query-batch-size", type=int, default=256)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-gardens-point":
            prepare_gardens_point(args.dataset_root)
            result = {"dataset_root": str(args.dataset_root), "splits": SPLITS}
        elif args.command == "extract":
            result = _extract(args)
        else:
            result = _evaluate(args)
    except (OSError, ValueError, RuntimeError, csv.Error, BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
