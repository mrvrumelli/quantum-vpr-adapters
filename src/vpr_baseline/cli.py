"""Prepare VPR data, extract descriptors, train adapters and evaluate retrieval."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import BadZipFile

import torch
import torchvision
from PIL import __version__ as pillow_version

from .adapters import ADAPTER_KINDS, BOTTLENECK_ACTIVATIONS, AdapterConfig, evaluate_adapter, load_adapter
from .cache import DescriptorSet, SplitCache, load_split_cache, save_cache
from .data import EVAL_SPLITS, SPLITS, load_manifests, load_positives, load_training_positives
from .extract import DEFAULT_IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, extract_descriptors
from .gardens import prepare_gardens_point
from .model import CHECKPOINT_SHA256, CHECKPOINT_URL, DESCRIPTOR_DIM, load_cosplace
from .retrieval import evaluate_recall
from .training import TrainConfig, train_adapter


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


def _load_evaluation(args: argparse.Namespace) -> tuple[SplitCache, list[set[int]]]:
    manifests = load_manifests(args.dataset_root)[args.split]
    cache = load_split_cache(args.cache_dir, args.split, manifests)
    positives = load_positives(args.dataset_root, args.split, cache.gallery.images, cache.query.images)
    return cache, positives


def _split_summary(cache: SplitCache) -> dict:
    return {
        "split": cache.split,
        "gallery_count": len(cache.gallery.images),
        "query_count": len(cache.query.images),
    }


def _evaluate(args: argparse.Namespace) -> dict:
    cache, positives = _load_evaluation(args)
    result = _split_summary(cache)
    if args.adapter_checkpoint is None:
        return {**result, **evaluate_recall(
            cache.query.descriptors, cache.gallery.descriptors, positives, args.query_batch_size
        )}
    model, epoch = load_adapter(args.adapter_checkpoint, cache)
    return {
        **result, "adapter": model.config.kind, "checkpoint": str(args.adapter_checkpoint), "epoch": epoch,
        **evaluate_adapter(model, cache, positives, args.query_batch_size),
    }


def _compare_adapters(args: argparse.Namespace) -> dict:
    cache, positives = _load_evaluation(args)
    scores = {"cosplace": evaluate_recall(
        cache.query.descriptors, cache.gallery.descriptors, positives, args.query_batch_size
    )}
    for kind, path in (("direct", args.direct_checkpoint), ("residual", args.residual_checkpoint)):
        model, epoch = load_adapter(path, cache)
        if model.config.kind != kind:
            raise ValueError(f"{path}: expected a {kind} adapter, got {model.config.kind}")
        scores[kind] = {
            "checkpoint": str(path), "epoch": epoch,
            **evaluate_adapter(model, cache, positives, args.query_batch_size),
        }
    return {**_split_summary(cache), "models": scores}


def _train_adapter(args: argparse.Namespace) -> dict:
    manifests = load_manifests(args.dataset_root)
    train = load_split_cache(args.cache_dir, "train", manifests["train"])
    val = load_split_cache(args.cache_dir, "val", manifests["val"])
    train_positives = load_training_positives(args.dataset_root, train.gallery.images, train.query.images)
    val_positives = load_positives(args.dataset_root, "val", val.gallery.images, val.query.images)
    adapter = AdapterConfig(
        args.adapter, args.down_dim, args.hidden_dim, args.alpha, up_bias=args.up_bias, activation=args.activation
    )
    config = TrainConfig(
        epochs=args.epochs, learning_rate=args.learning_rate, margin=args.margin, batch_size=args.batch_size,
        negative_window=args.negative_window, patience=args.patience, seed=args.seed,
    )
    report = train_adapter(adapter, config, train, train_positives, val, val_positives, args.output_dir)
    return {
        "adapter": args.adapter, "epochs_run": report["epochs_run"], "best_epoch": report["best_epoch"],
        "best_validation_recall": report["best_validation_recall"],
        "checkpoint": str(args.output_dir / "best.pt"), "history": str(args.output_dir / "history.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-gardens-point", help="write manifests for local Gardens Point images")
    extract = commands.add_parser("extract", help="extract a split using the verified local CosPlace checkpoint")
    evaluate = commands.add_parser("evaluate", help="evaluate a split using cached descriptors")
    train = commands.add_parser("train-adapter", help="train one adapter on cached train descriptors")
    compare = commands.add_parser("compare-adapters", help="compare CosPlace and both saved adapters")
    for command in (prepare, extract, evaluate, train, compare):
        command.add_argument("--dataset-root", type=Path, required=True)
    for command in (extract, evaluate, train, compare):
        command.add_argument("--cache-dir", type=Path, default=Path("artifacts"))
    extract.add_argument("--split", choices=SPLITS, required=True)
    for command in (evaluate, compare):
        command.add_argument("--split", choices=EVAL_SPLITS, required=True)
        command.add_argument("--query-batch-size", type=int, default=256)
        command.add_argument("--output", type=Path, help="also save the results to this JSON file")

    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    extract.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--workers", type=int, default=0)
    extract.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    evaluate.add_argument("--adapter-checkpoint", type=Path)
    compare.add_argument("--direct-checkpoint", type=Path, required=True)
    compare.add_argument("--residual-checkpoint", type=Path, required=True)

    train.add_argument("--adapter", choices=ADAPTER_KINDS, required=True)
    train.add_argument("--output-dir", type=Path, required=True, help="new directory for checkpoint and history")
    train.add_argument("--down-dim", type=int, default=AdapterConfig.down_dim)
    train.add_argument("--hidden-dim", type=int, default=AdapterConfig.hidden_dim)
    train.add_argument("--alpha", type=float, default=AdapterConfig.alpha, help="fixed residual scale")
    train.add_argument("--no-up-bias", dest="up_bias", action="store_false",
                       help="omit the Direct adapter's final projection bias")
    train.add_argument("--activation", choices=BOTTLENECK_ACTIVATIONS, default=AdapterConfig.activation,
                       help="bottleneck activation (Direct LeakyReLU slope: 0.01)")
    train.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    train.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    train.add_argument("--margin", type=float, default=TrainConfig.margin)
    train.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    train.add_argument("--negative-window", type=int, default=TrainConfig.negative_window)
    train.add_argument("--patience", type=int, default=TrainConfig.patience)
    train.add_argument("--seed", type=int, default=TrainConfig.seed)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-gardens-point":
            prepare_gardens_point(args.dataset_root)
            result = {"dataset_root": str(args.dataset_root), "splits": SPLITS}
        elif args.command == "extract":
            result = _extract(args)
        elif args.command == "evaluate":
            result = _evaluate(args)
        elif args.command == "train-adapter":
            result = _train_adapter(args)
        else:
            result = _compare_adapters(args)
        payload = json.dumps(result, indent=2)
        if output := getattr(args, "output", None):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, csv.Error, BadZipFile) as exc:
        parser.error(str(exc))
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
