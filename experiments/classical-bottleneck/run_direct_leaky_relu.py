"""Train the LeakyReLU arm and compare it with the archived ReLU arms."""

import argparse
import csv
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from vpr_baseline.adapter_experiment import FIXED_INTERMEDIATE, SEEDS, TRAINING, WIDTHS, _measure, _write_csv
from vpr_baseline.adapters import LEAKY_RELU_SLOPE, AdapterConfig, MLPAdapter, load_adapter
from vpr_baseline.cache import SplitCache, load_split_cache
from vpr_baseline.data import load_manifests, load_positives, load_training_positives
from vpr_baseline.diagnostics import first_positive_ranks
from vpr_baseline.retrieval import RECALL_KS, ZeroDescriptorError, rank_cosine
from vpr_baseline.training import TrainConfig, build_triplet_pool, train_adapter

ARMS = ("A", "B", "C")
SPLITS = ("train", "val")
METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "pair_cosine_mean", "total_variance", "effective_rank")

# First failing forward passes recorded in direct-bias-results.md.
ARCHIVED_B_FAILURES = {
    (4, 42): ("train", 3, 1, 32),
    (4, 123): ("train", 43, 1, 16),
    (4, 2026): ("train", 1, 1, 32),
    (8, 2026): ("val", 33, 1, 50),
}


@torch.inference_mode()
def _snapshot(model: MLPAdapter, cache: SplitCache, prefix: str) -> dict:
    zeros, activations, zero_rows = 0, 0, 0
    for group in (cache.query, cache.gallery):
        hidden = model.mlp(model.down(group.descriptors))
        norms = torch.linalg.vector_norm(model.up(hidden), dim=1)
        if not torch.isfinite(norms).all():
            raise ValueError(f"{cache.split}: nonfinite descriptor at {prefix}")
        zeros += int((hidden == 0).sum().item())
        activations += hidden.numel()
        zero_rows += int((norms == 0).sum().item())
    return {
        f"{prefix}_zero_activation_fraction": zeros / activations,
        f"{prefix}_zero_norm_descriptors": zero_rows,
    }


def _seeded(config: AdapterConfig, seed: int) -> MLPAdapter:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return MLPAdapter(config).eval()


def _failure(split: str, epoch: int, zero_rows: int, batch_rows: int) -> dict:
    return {"status": "zero_descriptor", "failure_split": split, "failure_epoch": epoch,
            "failure_zero_rows": zero_rows, "failure_batch_rows": batch_rows}


def _archived_rows(bias_dir: Path) -> dict:
    with (bias_dir / "direct_bias_runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = {(int(row["bottleneck_dim"]), int(row["seed"]), row["split"]): row
                for row in csv.DictReader(stream) if row["family"] == "fixed_intermediate"}
    if set(rows) != set(product(WIDTHS, SEEDS, SPLITS)):
        raise ValueError("the archived bias study must contain the full width-64 train/val grid")
    return rows


def _compare_history(history: dict, config: AdapterConfig, training: TrainConfig, val_row: dict, prefix: str) -> None:
    if (
        AdapterConfig(**history["adapter"]) != config
        or history["training"] != vars(training)
        or history["epochs_run"] != training.epochs
        or history["best_epoch"] != int(val_row[f"{prefix}_selected_epoch"])
        or any(history["best_validation_recall"][f"Recall@{k}"] != float(val_row[f"{prefix}_recall_at_{k}"])
               for k in RECALL_KS)
    ):
        raise ValueError("archived checkpoint history does not match the requested experimental protocol")


def _summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for arm, width, split in product(ARMS, WIDTHS, SPLITS):
        group = [row for row in rows if (row["configuration"], row["bottleneck_dim"], row["split"]) == (arm, width, split)]
        completed = [row for row in group if row["status"] == "completed"]
        entry = {"configuration": arm, "bottleneck_dim": width, "split": split, "attempted": len(group),
                 "successful": len(completed), "failures": len(group) - len(completed)}
        if completed:
            for metric in (*METRICS, "selected_zero_activation_fraction", "selected_zero_norm_descriptors"):
                values = [row[metric] for row in completed]
                entry[f"{metric}_mean"] = mean(values)
                entry[f"{metric}_std"] = stdev(values) if len(values) > 1 else None
        summary.append(entry)
    return summary


def run(dataset: Path, cache_dir: Path, bias_dir: Path, output: Path) -> None:
    manifests = load_manifests(dataset)
    caches = {split: load_split_cache(cache_dir, split, manifests[split]) for split in SPLITS}
    train, val = caches.values()
    labels = {"train": load_training_positives(dataset, train.gallery.images, train.query.images),
              "val": load_positives(dataset, "val", val.gallery.images, val.query.images)}
    negatives, ranks = {}, {}
    for split, cache in caches.items():
        negatives[split] = build_triplet_pool(
            cache.gallery.images, cache.query.images, labels[split], TRAINING.negative_window
        ).negatives
        ranks[split] = first_positive_ranks(
            rank_cosine(cache.query.descriptors, cache.gallery.descriptors, len(cache.gallery.images)), labels[split]
        )
    archived = _archived_rows(bias_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows, pair_scores = [], {}
    for width, seed in product(WIDTHS, SEEDS):
        training = replace(TRAINING, seed=seed)
        a = AdapterConfig("direct", FIXED_INTERMEDIATE, width)
        configs = {"A": a, "B": replace(a, up_bias=False), "C": replace(a, up_bias=False, activation="leaky_relu")}
        initials = {arm: _seeded(config, seed) for arm, config in configs.items()}
        reference = initials["A"].state_dict()
        for arm in ("B", "C"):
            for name, parameter in initials[arm].state_dict().items():
                torch.testing.assert_close(parameter, reference[name], atol=0, rtol=0)
        for arm, config in configs.items():
            prefix = {"A": "bias_on", "B": "bias_off"}.get(arm)
            slug = f"direct-i{FIXED_INTERMEDIATE}-b{width}-s{seed}"
            identity = {"configuration": arm, "intermediate_dim": FIXED_INTERMEDIATE, "bottleneck_dim": width,
                        "seed": seed, "source": "trained" if arm == "C" else "archived",
                        "activation": config.activation, "negative_slope": LEAKY_RELU_SLOPE if arm == "C" else 0,
                        "up_bias": config.up_bias, "parameter_count": sum(p.numel() for p in initials[arm].parameters())}
            split_rows = {split: {**identity, "split": split, "query_count": len(cache.query.images),
                                  "gallery_count": len(cache.gallery.images), **_snapshot(initials[arm], cache, "initial")}
                          for split, cache in caches.items()}
            if prefix is not None:
                previous = archived[width, seed, "val"]
                if any(
                    archived[width, seed, split]["control_history_matches_archive"] != "True"
                    or split_rows[split]["initial_zero_norm_descriptors"] != int(
                        archived[width, seed, split][f"{prefix}_initial_zero_rows"]
                    ) for split in SPLITS
                ):
                    raise ValueError(f"{slug}: the archive does not match initialized {arm} descriptors")
                if arm == "B" and previous[f"{prefix}_status"] == "zero_descriptor":
                    result = _failure(*ARCHIVED_B_FAILURES[width, seed])
                    rows.extend({**row, **result} for row in split_rows.values())
                    continue
                if previous[f"{prefix}_status"] != "completed":
                    raise ValueError(f"{slug}: unrecognized status for archived {arm}")
                folder = bias_dir / "runs" / f"{slug}-{prefix}"
                history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
                _compare_history(history, config, training, previous, prefix)
            else:
                folder = output / "runs" / slug
                try:
                    history = train_adapter(config, training, train, labels["train"], val, labels["val"], folder)
                except ZeroDescriptorError as error:
                    rows.extend({**row, **_failure(error.split, error.epoch, error.zero_rows, error.batch_rows)}
                                for row in split_rows.values())
                    print(f"C width {width} seed {seed}: zero descriptor in {error.split} epoch {error.epoch}", flush=True)
                    continue
                if history["epochs_run"] != training.epochs:
                    raise ValueError(f"{slug}: C used a different training budget")

            model, epoch = load_adapter(folder / "best.pt", val)
            if model.config != config or epoch != history["best_epoch"]:
                raise ValueError(f"{folder}: checkpoint and history disagree")
            for split, cache in caches.items():
                split_rows[split].update(_snapshot(model, cache, "selected"))
            if any(row["selected_zero_norm_descriptors"] for row in split_rows.values()):
                if prefix is not None:
                    raise ValueError(f"{folder}: unexpected zeros in archived selected descriptors")
                rows.extend({**row, "status": "zero_descriptor_at_selected", "selected_epoch": epoch,
                             "epochs_run": history["epochs_run"]} for row in split_rows.values())
                print(f"C width {width} seed {seed}: zero descriptor at selected checkpoint", flush=True)
                continue
            for split, cache in caches.items():
                metrics, _, distributions = _measure(cache, labels[split], negatives[split], ranks[split], {}, model)
                row = split_rows[split]
                row.update({"status": "completed", "selected_epoch": epoch, "epochs_run": history["epochs_run"], **metrics})
                if prefix is not None and any(
                    metrics[key] != float(archived[width, seed, split][f"{prefix}_{key}"]) for key in METRICS
                ):
                    raise ValueError(f"{folder}: reconstructed metrics differ from archived {split} results")
                for label, scores in zip(("positive", "negative"), distributions, strict=True):
                    pair_scores[f"{split}_{arm}_{width}_{seed}_{label}"] = scores.numpy()
                    for percentile, value in zip(("p05", "p50", "p95"), scores.quantile(
                        scores.new_tensor((0.05, 0.5, 0.95))
                    ).tolist(), strict=True):
                        row[f"{label}_cosine_{percentile}"] = value
                rows.append(row)
            if arm == "C":
                print(f"C width {width} seed {seed}: completed, selected epoch {epoch}", flush=True)

    _write_csv(output / "comparison.csv", rows)
    _write_csv(output / "summary.csv", _summarize(rows))
    np.savez_compressed(output / "pair_scores.npz", **pair_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "cache-dir", "bias-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.cache_dir, args.bias_dir, args.output_dir)
