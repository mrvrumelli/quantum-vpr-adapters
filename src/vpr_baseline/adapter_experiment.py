"""Run the validation-only classical bottleneck ablation using Phase 1/2 artifacts."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch

from .adapters import ADAPTER_KINDS, AdapterConfig, MLPAdapter, load_adapter
from .cache import SplitCache, load_split_cache
from .data import load_manifests, load_positives, load_training_positives
from .diagnostics import descriptor_geometry, first_positive_ranks, pair_similarities, rank_changes, residual_stats
from .retrieval import RECALL_KS, rank_cosine, recall_at_k
from .training import TrainConfig, build_triplet_pool, train_adapter

SEEDS = (42, 123, 2026)
WIDTHS = (4, 8, 16, 32)
FIXED_INTERMEDIATE = 64
TRAINING = TrainConfig(epochs=50, patience=50)


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _measure(
    cache: SplitCache, positives: list[set[int]], negatives: tuple[tuple[int, ...], ...],
    baseline_ranks: torch.Tensor, identity: dict, model: MLPAdapter | None,
) -> tuple[dict, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    source = torch.cat((cache.query.descriptors, cache.gallery.descriptors))
    adapted = source
    if model is not None:
        adapted = torch.cat((model(cache.query.descriptors), model(cache.gallery.descriptors)))
    q, g = adapted.split((len(cache.query.images), len(cache.gallery.images)))
    geometry = descriptor_geometry(adapted)
    ranking = rank_cosine(q, g, len(g))
    ranks = first_positive_ranks(ranking, positives)
    pair_scores = pair_similarities(q, g, positives, negatives)
    row = {
        **identity, "split": cache.split, "query_count": len(q), "gallery_count": len(g),
        **{f"recall_at_{k}": value for k, value in recall_at_k(ranking, positives).items()},
        "pair_cosine_mean": geometry.cosine_mean, "pair_cosine_std": geometry.cosine_std,
        "total_variance": geometry.total_variance, "numerical_rank": geometry.numerical_rank,
        "effective_rank": geometry.effective_rank, "mean_first_positive_rank": ranks.double().mean().item(),
        **rank_changes(baseline_ranks, ranks),
    }
    for label, scores in zip(("positive", "negative"), pair_scores, strict=True):
        row.update({f"{label}_pairs": scores.numel(), f"{label}_cosine_mean": scores.mean().item(),
                    f"{label}_cosine_std": scores.std(correction=0).item()})
    if model is not None:
        hidden = model.mlp(model.down(source))
        row["inactive_bottleneck_units"] = int((hidden == 0).all(dim=0).sum().item())
        if model.config.kind == "residual":
            row.update(residual_stats(source, adapted, model.up(hidden), model.config.alpha))
    return row, geometry.singular_values, pair_scores


def _identity(family: str, seed: int, model: MLPAdapter, epoch: int, epochs_run: int) -> dict:
    config = model.config
    return {
        "family": family, "model": config.kind, "seed": seed, "selected_epoch": epoch, "epochs_run": epochs_run,
        "intermediate_dim": config.down_dim, "bottleneck_dim": config.hidden_dim,
        "trainable_parameters": sum(param.numel() for param in model.parameters()),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        if row["split"] != "val" or row["family"] not in ("reference_repeat", "fixed_intermediate"):
            continue
        key = (row["family"], row["model"], row["intermediate_dim"], row["bottleneck_dim"], row["trainable_parameters"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (family, kind, intermediate, bottleneck, parameters), runs in sorted(groups.items()):
        record = {"family": family, "model": kind, "intermediate_dim": intermediate, "bottleneck_dim": bottleneck,
                  "trainable_parameters": parameters, "seeds": " ".join(str(row["seed"]) for row in runs),
                  "run_count": len(runs)}
        for metric in (*[f"recall_at_{k}" for k in RECALL_KS], "pair_cosine_mean", "total_variance", "effective_rank"):
            values = [row[metric] for row in runs]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        summary.append(record)
    return summary


def run(dataset_root: Path, cache_dir: Path, reference_dir: Path, output_dir: Path) -> None:
    manifests = load_manifests(dataset_root)
    train = load_split_cache(cache_dir, "train", manifests["train"])
    val = load_split_cache(cache_dir, "val", manifests["val"])
    positives = {
        "train": load_training_positives(dataset_root, train.gallery.images, train.query.images),
        "val": load_positives(dataset_root, "val", val.gallery.images, val.query.images),
    }
    splits = (train, val)
    baseline_ranks, negatives = {}, {}
    for cache in splits:
        split = cache.split
        negatives[split] = build_triplet_pool(
            cache.gallery.images, cache.query.images, positives[split], TRAINING.negative_window
        ).negatives
        baseline_ranks[split] = first_positive_ranks(
            rank_cosine(cache.query.descriptors, cache.gallery.descriptors, len(cache.gallery.images)), positives[split]
        )

    references = []
    for kind in ADAPTER_KINDS:
        folder = reference_dir / kind
        model, epoch = load_adapter(folder / "best.pt", val)
        history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        if model.config != AdapterConfig(kind) or history["training"]["seed"] != 0:
            raise ValueError(f"{folder}: expected the original seed-0 Phase 2 architecture and history")
        references.append((_identity("phase2", 0, model, epoch, history["epochs_run"]), model))

    output_dir.mkdir(parents=True, exist_ok=False)
    rows, spectra, distributions = [], [], []

    def record(identity: dict, model: MLPAdapter | None, save_distributions: bool = False) -> None:
        for cache in splits:
            split = cache.split
            row, spectrum, pair_scores = _measure(
                cache, positives[split], negatives[split], baseline_ranks[split], identity, model
            )
            rows.append(row)
            if save_distributions:
                key = {"model": identity["model"], "split": split}
                spectra.extend({**key, "component": i, "singular_value": value}
                               for i, value in enumerate(spectrum.tolist(), start=1))
                for label, scores in zip(("positive", "negative"), pair_scores, strict=True):
                    distributions.extend({**key, "label": label, "cosine": value} for value in scores.tolist())

    record({"family": "phase2", "model": "cosplace", "trainable_parameters": 0}, None, True)
    for identity, model in references:
        record(identity, model, True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        initial = MLPAdapter(AdapterConfig("residual")).eval()
    record(_identity("residual_initialization", 0, initial, 0, 0), initial)

    designs = [("reference_repeat", 8, 4), *(("fixed_intermediate", FIXED_INTERMEDIATE, d) for d in WIDTHS)]
    for family, intermediate, bottleneck in designs:
        for kind in ADAPTER_KINDS:
            for seed in SEEDS:
                config = TrainConfig(**{**asdict(TRAINING), "seed": seed})
                adapter = AdapterConfig(kind, intermediate, bottleneck)
                folder = output_dir / "runs" / f"{kind}-i{intermediate}-b{bottleneck}-s{seed}"
                history = train_adapter(adapter, config, train, positives["train"], val, positives["val"], folder)
                if history["epochs_run"] != TRAINING.epochs:
                    raise RuntimeError("ablation runs must use the same number of epochs")
                model, epoch = load_adapter(folder / "best.pt", val)
                record(_identity(family, seed, model, epoch, history["epochs_run"]), model)
                print(f"{kind} {intermediate}->{bottleneck}, seed {seed}, selected epoch {epoch}", flush=True)

    _write_csv(output_dir / "diagnostics.csv", rows)
    _write_csv(output_dir / "seed_summary.csv", _summarize(rows))
    _write_csv(output_dir / "singular_values.csv", spectra)
    _write_csv(output_dir / "pair_scores.csv", distributions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset-root", "cache-dir", "reference-dir", "output-dir"):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    args = parser.parse_args()
    run(args.dataset_root, args.cache_dir, args.reference_dir, args.output_dir)


if __name__ == "__main__":
    main()
