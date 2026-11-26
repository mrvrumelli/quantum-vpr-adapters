"""Compare Direct adapters with and without the up-projection bias on train and val."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vpr_baseline.adapter_experiment import FIXED_INTERMEDIATE, SEEDS, WIDTHS, _measure, _write_csv
from vpr_baseline.adapters import AdapterConfig, MLPAdapter, load_adapter
from vpr_baseline.cache import load_split_cache
from vpr_baseline.data import load_manifests, load_positives, load_training_positives
from vpr_baseline.diagnostics import descriptor_geometry, first_positive_ranks
from vpr_baseline.retrieval import ZeroDescriptorError, rank_cosine
from vpr_baseline.training import TrainConfig, build_triplet_pool, train_adapter

GEOMETRY = ("cosine_mean", "total_variance", "effective_rank", "numerical_rank")
SELECTED = ("recall_at_1", "recall_at_5", "recall_at_10", "pair_cosine_mean", "pair_cosine_std",
            "total_variance", "effective_rank", "numerical_rank", "positive_cosine_mean",
            "negative_cosine_mean", "inactive_bottleneck_units")


def _seeded(config: AdapterConfig, seed: int) -> MLPAdapter:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return MLPAdapter(config).eval()


@torch.inference_mode()
def _initial_geometry(models: dict[str, MLPAdapter], source: torch.Tensor) -> dict:
    raw = {name: model.up(model.mlp(model.down(source))) for name, model in models.items()}
    norms = {name: outputs.norm(dim=1) for name, outputs in raw.items()}
    if any(not torch.isfinite(values).all() for values in norms.values()):
        raise ValueError("nonfinite initial descriptor norm")
    valid = {name: values > 0 for name, values in norms.items()}
    shared = valid["bias_on"] & valid["bias_off"]
    result = {"initial_matched_valid_rows": int(shared.sum().item())}
    for name, output in raw.items():
        result[f"{name}_initial_zero_rows"] = int((~valid[name]).sum().item())
        if result["initial_matched_valid_rows"] >= 2:
            geometry = descriptor_geometry(output[shared])
            result.update({f"{name}_initial_{key}": getattr(geometry, key) for key in GEOMETRY})
    return result


def run(dataset: Path, cache_dir: Path, controls: Path, phase2: Path, output: Path) -> None:
    manifests = load_manifests(dataset)
    train, val = (load_split_cache(cache_dir, name, manifests[name]) for name in ("train", "val"))
    positives = {
        "train": load_training_positives(dataset, train.gallery.images, train.query.images),
        "val": load_positives(dataset, "val", val.gallery.images, val.query.images),
    }
    ranks = {
        cache.split: first_positive_ranks(
            rank_cosine(cache.query.descriptors, cache.gallery.descriptors, len(cache.gallery.images)),
            positives[cache.split],
        ) for cache in (train, val)
    }
    layouts = [("reference_repeat", 8, 4), *(("fixed_intermediate", FIXED_INTERMEDIATE, d) for d in WIDTHS)]
    designs = [("phase2", 8, 4, 0, phase2 / "direct")]
    for family, intermediate, bottleneck in layouts:
        for seed in SEEDS:
            designs.append((family, intermediate, bottleneck, seed,
                            controls / f"direct-i{intermediate}-b{bottleneck}-s{seed}"))

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for family, intermediate, bottleneck, seed, reference in designs:
        archived = json.loads((reference / "history.json").read_text(encoding="utf-8"))
        training = TrainConfig(**archived["training"])
        biased = AdapterConfig(**archived["adapter"])
        if biased != AdapterConfig("direct", intermediate, bottleneck) or training.seed != seed:
            raise ValueError(f"{reference}: controls must match the intended architecture and seed")
        configs = {"bias_on": biased, "bias_off": replace(biased, up_bias=False)}
        initial = {name: _seeded(config, seed) for name, config in configs.items()}
        biased_weights = initial["bias_on"].state_dict()
        for parameter, value in initial["bias_off"].state_dict().items():
            torch.testing.assert_close(value, biased_weights[parameter], rtol=0, atol=0)

        identity = {"family": family, "intermediate_dim": intermediate, "bottleneck_dim": bottleneck, "seed": seed}
        split_rows = {}
        for cache in (train, val):
            source = torch.cat((cache.query.descriptors, cache.gallery.descriptors))
            split_rows[cache.split] = {**identity, "split": cache.split, "descriptor_count": len(source),
                                       **_initial_geometry(initial, source)}

        for name, config in configs.items():
            destination = output / "runs" / f"direct-i{intermediate}-b{bottleneck}-s{seed}-{name}"
            for row in split_rows.values():
                row[f"{name}_parameter_count"] = sum(p.numel() for p in initial[name].parameters())
            try:
                history = train_adapter(config, training, train, positives["train"], val, positives["val"], destination)
            except ZeroDescriptorError:
                if name == "bias_on":
                    raise
                for row in split_rows.values():
                    row[f"{name}_status"] = "zero_descriptor"
                continue
            model, epoch = load_adapter(destination / "best.pt", val)
            for cache in (train, val):
                split = cache.split
                negatives = build_triplet_pool(
                    cache.gallery.images, cache.query.images, positives[split], training.negative_window
                ).negatives
                row = split_rows[split]
                row.update({f"{name}_status": "completed", f"{name}_selected_epoch": epoch,
                            f"{name}_epochs_run": history["epochs_run"]})
                try:
                    metrics, _, _ = _measure(cache, positives[split], negatives, ranks[split], {}, model)
                except ZeroDescriptorError:
                    if name == "bias_on":
                        raise
                    row[f"{name}_status"] = "zero_descriptor_at_selected"
                    continue
                row.update({f"{name}_{key}": metrics[key] for key in SELECTED})
                if name == "bias_on":
                    row["control_history_matches_archive"] = history["history"] == archived["history"]
        rows.extend(split_rows.values())
        print(f"{intermediate}->{bottleneck}, seed {seed}: {split_rows['val']['bias_off_status']}", flush=True)

    _write_csv(output / "direct_bias_runs.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "cache-dir", "control-dir", "phase2-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.cache_dir, args.control_dir, args.phase2_dir, args.output_dir)
