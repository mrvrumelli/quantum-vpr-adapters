"""Train and compare Stage 3.2 residual controls on the frozen Gardens Point caches."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np
import torch

from vpr_baseline.adapter_experiment import TRAINING, _write_csv
from vpr_baseline.adapters import AdapterConfig, MLPAdapter, load_adapter
from vpr_baseline.cache import load_split_cache
from vpr_baseline.data import load_manifests, load_positives, load_training_positives
from vpr_baseline.diagnostics import rank_changes
from vpr_baseline.quantum import QuantumMixer
from vpr_baseline.residual_study import (
    RETRIEVAL_METRICS, distribution, measure_snapshot, paired_block_bootstrap, retrieval_values,
)
from vpr_baseline.training import build_triplet_pool, train_adapter

SEEDS = (42, 123, 2026, 7, 31415)
SPLITS = ("train", "val")
QUANTUM = AdapterConfig("residual", 64, up_bias=False, activation="leaky_relu", transform="quantum")
SCALES = {"005": 0.05, "010": 0.1, "020": 0.2, "050": 0.5}
CONTROLS = {
    "pqc_frozen": {"train_quantum": False},
    "pqc_separable": {"entangle": False},
    "classical_matched": {"transform": "surrogate"},
    "classical_identity": {"transform": "identity"},
}
ACTIVE = {f"pqc_{suffix}": replace(QUANTUM, alpha=scale) for suffix, scale in SCALES.items()}
ACTIVE.update({f"{control}_{suffix}": replace(QUANTUM, alpha=scale, **options)
               for suffix, scale in SCALES.items() for control, options in CONTROLS.items()})
ANGLE_RESETS = {f"pqc_{suffix}": f"pqc_reset_{suffix}" for suffix in SCALES}
PAIRS = (
    *((f"pqc_{suffix}", f"{control}_{suffix}") for suffix in SCALES for control in (*CONTROLS, "pqc_reset")),
    *((arm, "cosplace") for arm in (*ACTIVE, *ANGLE_RESETS.values())),
    *((arm, "pqc_010") for arm in ("pqc_005", "pqc_020", "pqc_050")),
)


def seeded_model(config: AdapterConfig, seed: int) -> MLPAdapter:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return MLPAdapter(config).eval()


def verify_initialization(initial, seed: int) -> None:
    reference = seeded_model(QUANTUM, seed).state_dict()
    for key, value in initial.state_dict().items():
        if key.startswith(("down.", "mlp.", "up.")) or key == "mixer.angles":
            torch.testing.assert_close(value, reference[key], atol=0, rtol=0)


def parameter_counts(model) -> dict:
    angles = model.mixer.angles if model is not None and isinstance(model.mixer, QuantumMixer) else None
    active_angles = angles.numel() if angles is not None and angles.requires_grad else 0
    learned = sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0
    return {"classical_trainable": learned - active_angles, "quantum_trainable": active_angles,
            "quantum_frozen": angles.numel() - active_angles if angles is not None else 0}


def frame(record) -> int:
    return int(re.fullmatch(r"Image(\d+)", Path(record.path).stem, re.IGNORECASE)[1])


def similarity_effects(snapshot, baseline, positives, negatives) -> dict:
    effects = snapshot.scores - baseline.scores
    metrics = {}
    for label, pairs in (("positive", positives), ("negative", negatives)):
        # Pool shifts across all eligible pairs in the split.
        rows = [effects[q, sorted(refs)] for q, refs in enumerate(pairs)]
        changes = torch.cat(rows)
        values = torch.cat([snapshot.scores[q, sorted(refs)] for q, refs in enumerate(pairs)])
        metrics.update({**distribution(changes, f"{label}_cosine_change"),
                        f"{label}_cosine_mean": values.mean().item(),
                        f"{label}_cosine_increased": (changes > 0).sum().item(),
                        f"{label}_cosine_decreased": (changes < 0).sum().item(),
                        f"{label}_cosine_tied": (changes == 0).sum().item()})
    return metrics


def record_snapshots(tables, ranks, caches, labels, negatives, baseline, identity, snapshot_name, model):
    for split in SPLITS:
        cache = caches[split]
        stats = measure_snapshot(cache, labels[split], model) if model is not None else baseline[split]
        original = baseline[split]
        row = {**identity, "split": split, "snapshot": snapshot_name}
        changes = rank_changes(original.ranks, stats.ranks)
        rank_shift = original.ranks - stats.ranks
        rr_shift = stats.ranks.double().reciprocal() - original.ranks.double().reciprocal()
        tables["runs"].append({**row, **stats.metrics, **changes, **distribution(rank_shift, "rank_gain"),
                              **distribution(rr_shift, "rr_gain"),
                              **similarity_effects(stats, original, labels[split], negatives[split])})
        if stats.covariance is not None:
            for feature, row_values in enumerate(stats.covariance.tolist()):
                tables["feature_covariance"].append({**row, "feature": feature,
                                                     **{f"cov_to_{i}": v for i, v in enumerate(row_values)}})
        if snapshot_name != "selected":
            continue
        ranks[identity["arm"], identity["seed"], split] = stats.ranks
        if identity["arm"] == "pqc_010":
            for role, records, offset in (("query", cache.query.images, 0),
                                          ("gallery", cache.gallery.images, len(cache.query.images))):
                for i, record in enumerate(records, offset):
                    tables["residual_norms"].append({**row, "role": role, "image_id": record.image_id,
                        "original_norm": stats.descriptor_norms[i].item(),
                        "unscaled_correction": stats.unscaled_correction[i].item(),
                        "scaled_correction": stats.scaled_correction[i].item(),
                        "cosine_to_original": stats.descriptor_cosines[i].item()})
        for i, record in enumerate(cache.query.images):
            similarities = stats.scores[i] - original.scores[i]
            base_rank, rank = original.ranks[i].item(), stats.ranks[i].item()
            tables["query_effects"].append({"arm": identity["arm"], "seed": identity["seed"], "split": split,
                "query_id": record.image_id, "frame": frame(record), "cosplace_rank": base_rank, "rank": rank,
                "rank_gain": base_rank - rank, "rr_gain": rr_shift[i].item(),
                "positive_cosine_change": similarities[sorted(labels[split][i])].mean().item(),
                "negative_cosine_change": similarities[list(negatives[split][i])].mean().item()})


def reset_angles(selected, initial, seed, caches, tables):
    reset = seeded_model(QUANTUM, seed)
    reset.load_state_dict(selected.state_dict())
    with torch.no_grad():
        reset.mixer.angles.copy_(initial.mixer.angles)
    with torch.inference_mode():
        for split, cache in caches.items():
            inputs = torch.cat([selected.components(group)[0]
                                for group in (cache.query.descriptors, cache.gallery.descriptors)])
            original_output = reset.mixer(inputs)
            learned_output = selected.mixer(inputs)
            change = learned_output.double() - original_output.double()
            centered_change = change - change.mean(dim=0)
            projected = selected.config.alpha * selected.up(change.to(inputs.dtype)).double()
            tables["angle_reset_features"].append({"seed": seed, "split": split, "lambda": selected.config.alpha,
                "readout_rmse": change.square().mean().sqrt().item(),
                "centered_readout_change_rms": centered_change.square().mean().sqrt().item(),
                "learned_centered_readout_rms": (learned_output.double() - learned_output.double().mean(0))
                .square().mean().sqrt().item(),
                **distribution(projected.norm(dim=1), "scaled_correction_difference"),
                **{f"wire_{i}_mean_change": value for i, value in enumerate(change.mean(0).tolist())}})
    return reset


def summarize(tables, ranks, queries) -> None:
    selected = [row for row in tables["runs"] if row["snapshot"] == "selected"]
    for arm in ("cosplace", *ACTIVE, *ANGLE_RESETS.values()):
        for split in SPLITS:
            rows = [row for row in selected if row["arm"] == arm and row["split"] == split]
            metrics = {"arm": arm, "split": split, "lambda": rows[0]["lambda"], "evaluated_seeds": len(rows)}
            for key in (*RETRIEVAL_METRICS, "effective_rank", "descriptor_variance", "distinct_cosine_mean",
                        "feature_effective_rank", "feature_total_variance", "original_norm_mean",
                        "unscaled_correction_mean", "scaled_correction_mean", "relative_correction_mean",
                        "cosine_to_original_mean", "positive_cosine_change_mean", "negative_cosine_change_mean",
                        "rank_improved", "rank_worsened", "rank_unchanged", "mixer_gradient_norm",
                        "mixer_input_gradient_norm", "input_projection_gradient_norm",
                        "quantum_layer_1_gradient_norm", "quantum_layer_2_gradient_norm"):
                values = [row[key] for row in rows if row.get(key) is not None]
                if values:
                    metrics[f"mean_{key}"] = mean(values)
                    metrics[f"sd_{key}"] = stdev(values) if len(values) > 1 else None
            tables["summary"].append(metrics)

    frames = tuple(frame(record) for record in queries)
    for model, control in PAIRS:
        effects = {key: [] for key in RETRIEVAL_METRICS}
        for seed in SEEDS:
            result = ranks[model, seed, "val"]
            reference = ranks[control, None if control == "cosplace" else seed, "val"]
            values, base = retrieval_values(result), retrieval_values(reference)
            differences = {key: values[key] - base[key] for key in RETRIEVAL_METRICS}
            pair = {"arm": model, "control": control, "seed": seed}
            tables["paired_seeds"].append({**pair, **rank_changes(reference, result),
                                           **{f"delta_{k}": v.mean().item() for k, v in differences.items()}})
            for key, delta in differences.items():
                effects[key].append(delta.numpy())
            for i, record in enumerate(queries):
                tables["paired_queries"].append({**pair, "query_id": record.image_id, "frame": frames[i],
                    "arm_rank": result[i].item(), "control_rank": reference[i].item(),
                    **{f"delta_{k}": v[i].item() for k, v in differences.items()}})
        for key, queries_by_seed in effects.items():
            matrix = np.stack(queries_by_seed)
            row = {"arm": model, "control": control, "metric": key, "seeds": len(SEEDS),
                   "queries": len(queries), "mean_delta": matrix.mean().item(),
                   "sd_seed_delta": matrix.mean(axis=1).std(ddof=1).item()}
            for block in (5, 10):
                low, high = paired_block_bootstrap(matrix, frames, block_length=block)
                row.update({f"ci95_low_block_{block}": low, f"ci95_high_block_{block}": high})
            tables["paired_bootstrap"].append(row)


def run(dataset: Path, cache_dir: Path, output: Path, results: Path, stage31: Path | None) -> None:
    manifests = load_manifests(dataset)
    caches = {split: load_split_cache(cache_dir, split, manifests[split]) for split in SPLITS}
    train, val = (caches[split] for split in SPLITS)
    labels = {"train": load_training_positives(dataset, train.gallery.images, train.query.images),
              "val": load_positives(dataset, "val", val.gallery.images, val.query.images)}
    negatives = {split: build_triplet_pool(cache.gallery.images, cache.query.images, labels[split],
                                           TRAINING.negative_window).negatives for split, cache in caches.items()}
    output.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    tables = {table: [] for table in ("runs", "summary", "paired_seeds", "paired_queries", "paired_bootstrap",
              "query_effects", "residual_norms", "feature_covariance", "angle_updates", "angle_reset_features",
              "training_trace", "stage_3_1_checkpoints")}
    ranks = {}
    baseline = {split: measure_snapshot(cache, labels[split], None) for split, cache in caches.items()}
    caches_and_labels = (caches, labels, negatives, baseline)
    record_snapshots(tables, ranks, *caches_and_labels, {"arm": "cosplace", "seed": None, "lambda": 0,
                     "epochs_run": 0, **parameter_counts(None)}, "selected", None)
    for seed in SEEDS:
        for arm, config in ACTIVE.items():
            initial = seeded_model(config, seed)
            verify_initialization(initial, seed)
            identity = {"arm": arm, "seed": seed, "lambda": config.alpha, **parameter_counts(initial)}
            record_snapshots(tables, ranks, *caches_and_labels, identity, "initial", initial)
            folder = output / "runs" / f"{arm}-s{seed}"
            started = perf_counter()
            history = train_adapter(config, replace(TRAINING, seed=seed), train, labels["train"], val, labels["val"],
                                    folder, detailed_mixer_diagnostics=True)
            seconds = perf_counter() - started
            selected, epoch = load_adapter(folder / "best.pt", val)
            diagnostics = history["history"][epoch - 1]["mixer_diagnostics"]
            chosen = {**identity, "selected_epoch": epoch, "epochs_run": history["epochs_run"],
                      "training_wall_seconds": seconds,
                      **{k: v for k, v in diagnostics.items() if k.endswith("gradient_norm")}}
            record_snapshots(tables, ranks, *caches_and_labels, chosen, "selected", selected)
            for step in history["history"]:
                monitor = step["mixer_diagnostics"].copy()
                monitor["mixer_parameters"] = json.dumps(monitor["mixer_parameters"], separators=(",", ":"))
                tables["training_trace"].append({**identity, "epoch": step["epoch"], "train_loss": step["train_loss"],
                                                 "val_loss": step["val_loss"], **step["val_recall"], **monitor})
            if isinstance(selected.mixer, QuantumMixer):
                before = initial.mixer.angles.detach().flatten().tolist()
                after = selected.mixer.angles.detach().flatten().tolist()
                final = history["history"][-1]["mixer_diagnostics"]["mixer_parameters"]
                for index, (start, end, last) in enumerate(zip(before, after, final, strict=True)):
                    tables["angle_updates"].append({**identity, "selected_epoch": epoch, "layer": index // 8 + 1,
                        "wire": index % 8 // 2, "gate": ("RZ", "RY")[index % 2], "initial_angle": start,
                        "selected_angle": end, "last_angle": last, "selected_update": end - start,
                        "last_update": last - start})
            if arm in ANGLE_RESETS:
                reset = reset_angles(selected, initial, seed, caches, tables)
                record_snapshots(tables, ranks, *caches_and_labels, {"arm": ANGLE_RESETS[arm], "seed": seed,
                                 "lambda": config.alpha, "source_epoch": epoch}, "selected", reset)
                if stage31 is not None and arm == "pqc_010" and seed in (42, 123, 2026):
                    path = stage31 / "benchmark" / "runs" / f"quantum_residual-s{seed}" / "best.pt"
                    archived, archive_epoch = load_adapter(path, val)
                    if archived.config != config or archive_epoch != epoch:
                        raise ValueError(f"archived Stage 3.1 configuration or epoch differs: {path}")
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    for split in SPLITS:
                        old = measure_snapshot(caches[split], labels[split], archived)
                        torch.testing.assert_close(old.ranks, ranks[arm, seed, split], atol=0, rtol=0)
                        tables["stage_3_1_checkpoints"].append({"seed": seed, "split": split, "sha256": digest,
                            "selected_epoch": archive_epoch, "rank_matches": len(old.ranks), **old.metrics})
            score = ranks[arm, seed, "val"].double().reciprocal().mean().item()
            print(f"{arm} seed {seed}, epoch {epoch}, validation MRR {score:.6f}, {seconds:.2f}s", flush=True)

    summarize(tables, ranks, val.query.images)
    for name, rows in tables.items():
        if rows:
            _write_csv(results / f"{name}.csv", rows)
    (results / "protocol.json").write_text(json.dumps({"seeds": SEEDS, "training": asdict(TRAINING),
        "models": {arm: asdict(config) for arm, config in ACTIVE.items()}, "torch": str(torch.__version__),
        "numpy": np.__version__, "torch_threads": torch.get_num_threads(), "splits": SPLITS,
        "feature_spec": history["feature_spec"], "bootstrap_draws": 20000, "bootstrap_seed": 3202,
        "bootstrap_block_lengths": [5, 10]}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "cache-dir", "output-dir", "results-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage-31-dir", type=Path, help="verify the archived primary Quantum Residual checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.dataset_root, args.cache_dir, args.output_dir, args.results_dir, args.stage_31_dir)
