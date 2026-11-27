"""Profile, smoke-test, and benchmark the Stage 3.1 adapter controls."""

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import torch

from vpr_baseline.adapter_experiment import SEEDS, TRAINING, _write_csv
from vpr_baseline.adapters import AdapterConfig, MLPAdapter, load_adapter
from vpr_baseline.cache import load_split_cache
from vpr_baseline.data import load_manifests, load_positives, load_training_positives
from vpr_baseline.quantum import STATE_SIZE, QuantumMixer
from vpr_baseline.quantum_diagnostics import evaluate_descriptors
from vpr_baseline.retrieval import ZeroDescriptorError
from vpr_baseline.training import _hinge, build_triplet_pool, train_adapter

SPLITS = ("train", "val")
MEASURES = ("Recall@1", "Recall@5", "Recall@10", "MRR", "pair_cosine_mean", "total_variance",
            "effective_rank", "relative_correction_mean", "train_wall_seconds", "mixer_gradient_norm_mean",
            "mixer_gradient_variance_mean", "input_projection_gradient_norm_mean")
SMOKE_EPOCHS = 5


def _seed_model(config, seed: int) -> MLPAdapter:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return MLPAdapter(config)


def _models() -> dict[str, AdapterConfig]:
    return {f"{transform}_{kind}": AdapterConfig(kind, 64, up_bias=False, activation="leaky_relu", transform=transform)
            for transform, kind in product(("quantum", "surrogate"), ("direct", "residual"))}


def _counts(model: MLPAdapter | None) -> dict:
    classical, quantum, frozen = 0, 0, 0
    if model is not None:
        for name, param in model.named_parameters():
            is_quantum = isinstance(model.mixer, QuantumMixer) and name.startswith("mixer.")
            if is_quantum and not param.requires_grad:
                frozen += param.numel()
            elif is_quantum:
                quantum += param.numel()
            elif param.requires_grad:
                classical += param.numel()
    return {"classical_trainable": classical, "quantum_trainable": quantum, "quantum_frozen": frozen}


def _archived(artifacts: Path) -> dict:
    source = Path(__file__).resolve().parents[1] / "classical-bottleneck/final-benchmark/experiments.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = {(row["model"], int(row["seed"]), row["split"]): row for row in csv.DictReader(stream)
                if row["scope"] == "primary" and row["bottleneck_dim"] == "4" and row["model"] in ("A", "C", "D")}
    resolved = {}
    for arm, seed in product(("A", "C", "D"), SEEDS):
        matches = {split: rows[arm, seed, split] for split in SPLITS}
        entry = matches["val"]
        path = artifacts / Path(entry["checkpoint"]).relative_to("artifacts")
        if entry["evaluation_status"] != "evaluated" or not path.is_file():
            raise ValueError(f"missing Stage 2.3 checkpoint for {arm}, seed {seed}: {path}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["checkpoint_sha256"]:
                raise ValueError(f"Stage 2.3 checkpoint digest differs: {path}")
        history_path = path.with_name("history.json")
        if not history_path.is_file():
            raise ValueError(f"missing Stage 2.3 training history: {history_path}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history["training"] != asdict(replace(TRAINING, seed=seed)) or history["epochs_run"] != TRAINING.epochs:
            raise ValueError(f"Stage 2.3 training protocol differs for {arm}, seed {seed}")
        resolved[arm, seed] = (path, matches)
    return resolved


def _profile(models, train, labels, repetitions: int = 12) -> list[dict]:
    pool = build_triplet_pool(train.gallery.images, train.query.images, labels, TRAINING.negative_window)
    q, p, n = pool.sample(torch.Generator().manual_seed(SEEDS[0]))[:TRAINING.batch_size].unbind(1)
    queries = train.query.descriptors[q]
    gallery = train.gallery.descriptors[torch.cat((p, n))]
    measurements = []
    for name, model in models.items():
        times, inference_times = [], []
        for _ in range(3 + repetitions):
            model.zero_grad(set_to_none=True)
            started = perf_counter()
            qz = model(queries)
            pz, nz = model(gallery).chunk(2)
            _hinge((qz * pz).sum(1), (qz * nz).sum(1), TRAINING.margin).backward()
            times.append(perf_counter() - started)
            with torch.inference_mode():
                started = perf_counter()
                model(queries)
                model(gallery)
                inference_times.append(perf_counter() - started)
        measurements.append({"model": name, "descriptors_per_iteration": len(queries) + len(gallery),
                             "forward_backward_ms_median": 1000 * median(times[3:]),
                             "inference_ms_median": 1000 * median(inference_times[3:]),
                             "repetitions": repetitions, "warmups": 3, **_counts(model)})
    return measurements


def _summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["snapshot"] == "selected":
            groups[row["model"], row["split"]].append(row)
    summaries = []
    for (model, split), group in groups.items():
        evaluated = [row for row in group if row["evaluation_status"] == "evaluated"]
        entry = {"model": model, "split": split, "attempted": len(group), "evaluated": len(evaluated),
                 **{key: group[0][key] for key in ("classical_trainable", "quantum_trainable", "quantum_frozen")}}
        for metric in MEASURES:
            values = [row[metric] for row in evaluated if row.get(metric) is not None]
            if values:
                entry[f"{metric}_mean"] = mean(values)
                entry[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
        summaries.append(entry)
    return summaries


def _monitor_summary(history: dict) -> dict:
    entries = history["history"]
    monitors = [entry["mixer_diagnostics"] for entry in entries]
    summary = {"first_train_loss": entries[0]["train_loss"], "last_train_loss": entries[-1]["train_loss"],
               "last_mixer_parameter_drift": monitors[-1]["mixer_parameter_drift"],
               "last_residual_scale": monitors[-1]["residual_scale"]}
    for name in ("mixer_gradient_norm", "mixer_gradient_variance", "input_projection_gradient_norm"):
        values = [monitor[name] for monitor in monitors if monitor[name] is not None]
        summary[f"{name}_mean"] = mean(values) if values else None
        summary[f"{name}_first_epoch"] = monitors[0][name]
    return summary


def _check_projections(models: dict, seed: int) -> None:
    for kind in ("direct", "residual"):
        quantum, surrogate = (_seed_model(models[f"{transform}_{kind}"], seed) for transform in ("quantum", "surrogate"))
        control = surrogate.state_dict()
        for name, weight in quantum.state_dict().items():
            if not name.startswith("mixer."):
                torch.testing.assert_close(weight, control[name], atol=0, rtol=0)


def run(stage: str, dataset: Path, artifacts: Path, output: Path) -> None:
    manifests = load_manifests(dataset)
    caches = {split: load_split_cache(artifacts / "gardens-point", split, manifests[split]) for split in SPLITS}
    train, val = caches.values()
    labels = {"train": load_training_positives(dataset, train.gallery.images, train.query.images),
              "val": load_positives(dataset, "val", val.gallery.images, val.query.images)}
    negatives = {split: build_triplet_pool(cache.gallery.images, cache.query.images, labels[split],
                                           TRAINING.negative_window).negatives for split, cache in caches.items()}
    models = _models()
    output.mkdir(parents=True, exist_ok=False)
    for seed in SEEDS if stage == "benchmark" else (SEEDS[0],):
        _check_projections(models, seed)

    if stage == "profile":
        profiled = {name: _seed_model(config, SEEDS[0]) for name, config in models.items()}
        for (arm, seed), (path, _) in _archived(artifacts).items():
            if seed == SEEDS[0]:
                profiled[arm] = load_adapter(path, val)[0]
        _write_csv(output / "profile.csv", _profile(profiled, train, labels["train"]))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mib = peak / (1024**2 if sys.platform == "darwin" else 1024)
        (output / "memory.json").write_text(json.dumps({"profile_process_peak_rss_mib": rss_mib,
            "complex64_state_bytes_per_max_train_forward": 2 * TRAINING.batch_size * STATE_SIZE * 8}, indent=2) + "\n")
        return

    rows, traces = [], []

    def evaluate(identity, snapshot, model):
        for split, cache in caches.items():
            scores = evaluate_descriptors(cache, labels[split], negatives[split], model)
            rows.append({**identity, "snapshot": snapshot, "split": split, **scores})

    if stage == "benchmark":
        evaluate({"model": "CosPlace", "seed": "", "source": "frozen_cache", **_counts(None)}, "selected", None)
        for (arm, seed), (path, archive) in _archived(artifacts).items():
            model, epoch = load_adapter(path, val)
            identity = {"model": arm, "seed": seed, "source": "stage_2_3_checkpoint", "selected_epoch": epoch, **_counts(model)}
            evaluate(identity, "selected", model)
            for row in rows[-2:]:
                old = archive[row["split"]]
                for field, key in (("Recall@1", "recall_at_1"), ("Recall@5", "recall_at_5"),
                                   ("Recall@10", "recall_at_10"), ("MRR", "mrr")):
                    if not math.isclose(row[field], float(old[key]), abs_tol=1e-12):
                        raise ValueError(f"Stage 2.3 score changed for {arm}, seed {seed}: {field}")

    if stage == "ablations":
        residual = models["quantum_residual"]
        models = {"quantum_residual_depth1": replace(residual, circuit_depth=1),
                  "quantum_residual_separable": replace(residual, entangle=False),
                  "quantum_residual_frozen": replace(residual, train_quantum=False)}
    seeds = SEEDS if stage == "benchmark" else (SEEDS[0],)
    for (name, config), seed in product(models.items(), seeds):
        initial = _seed_model(config, seed)
        identity = {"model": name, "seed": seed, "source": stage, "depth": config.circuit_depth if config.transform == "quantum" else "",
                    "entangle": config.entangle if config.transform == "quantum" else "", **_counts(initial)}
        evaluate(identity, "initial", initial)
        folder = output / "runs" / f"{name}-s{seed}"
        training = replace(TRAINING, seed=seed, epochs=SMOKE_EPOCHS if stage == "smoke" else TRAINING.epochs)
        started = perf_counter()
        try:
            history = train_adapter(config, training, train, labels["train"], val, labels["val"], folder)
        except ZeroDescriptorError as error:
            rows.append({**identity, "snapshot": "selected", "split": error.split, "evaluation_status": "training_failed",
                         "failure_epoch": error.epoch, "zero_norm_rows": error.zero_rows, "error": str(error),
                         "train_wall_seconds": perf_counter() - started})
            print(f"{name} seed {seed}: zero descriptor in {error.split} epoch {error.epoch}", flush=True)
            continue
        seconds = perf_counter() - started
        model, epoch = load_adapter(folder / "best.pt", val)
        epochs = history["epochs_run"]
        extra = {"selected_epoch": epoch, "epochs_run": epochs, "train_wall_seconds": seconds,
                 "train_statevector_evaluations": 3 * len(train.query.images) * epochs if config.transform == "quantum" else 0,
                 "validation_statevector_evaluations": (len(val.query.images) + len(val.gallery.images)) * epochs
                 if config.transform == "quantum" else 0, "measurement_shots": 0, **_monitor_summary(history)}
        evaluate({**identity, **extra}, "selected", model)
        for entry in history["history"]:
            monitor = entry["mixer_diagnostics"]
            traces.append({**identity, "epoch": entry["epoch"], "train_loss": entry["train_loss"],
                           "val_loss": entry["val_loss"], **entry["val_recall"], **monitor,
                           "mixer_parameters": json.dumps(monitor["mixer_parameters"], separators=(",", ":"))})
        print(f"{name} seed {seed}: epoch {epoch}, val MRR {rows[-1].get('MRR', 'invalid')}, {seconds:.2f}s", flush=True)
    _write_csv(output / "runs.csv", rows)
    _write_csv(output / "summary.csv", _summaries(rows))
    _write_csv(output / "training_trace.csv", traces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "smoke", "benchmark", "ablations"), required=True)
    for option in ("dataset-root", "artifacts-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.dataset_root, args.artifacts_dir, args.output_dir)
