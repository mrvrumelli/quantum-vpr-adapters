"""Consolidate saved classical runs using full-gallery train and validation rankings."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean, stdev

import torch

from vpr_baseline.adapter_experiment import FIXED_INTERMEDIATE, SEEDS, TRAINING, WIDTHS, _write_csv
from vpr_baseline.adapters import AdapterConfig, MLPAdapter, load_adapter
from vpr_baseline.cache import SplitCache, load_split_cache
from vpr_baseline.data import load_manifests, load_positives, load_training_positives
from vpr_baseline.diagnostics import descriptor_geometry, pair_similarities, rank_changes, residual_stats
from vpr_baseline.retrieval import RECALL_KS, first_positive_ranks, mean_reciprocal_rank, rank_cosine, recall_at_k
from vpr_baseline.training import build_triplet_pool

SPLITS = ("train", "val")
ARMS = ("A", "B", "C", "D")
RECALL_FIELDS = tuple(f"recall_at_{k}" for k in RECALL_KS)
GEOMETRY_FIELDS = ("pair_cosine_mean", "total_variance", "effective_rank", "positive_cosine_mean",
                   "negative_cosine_mean", "positive_negative_gap")
RESIDUAL_FIELDS = ("cosine_to_original_mean", "relative_correction_mean", "alpha",
                   "rank_improved", "rank_worsened", "rank_unchanged")
SUMMARY_FIELDS = (*RECALL_FIELDS, "mrr", *GEOMETRY_FIELDS, *RESIDUAL_FIELDS)


@dataclass(frozen=True)
class SavedRun:
    model: str
    intermediate: int
    bottleneck: int
    seed: int
    folder: Path
    archived: dict[str, dict]

    @property
    def identity(self) -> dict:
        scope = "primary" if self.intermediate == FIXED_INTERMEDIATE else "historical"
        if self.seed == 0:
            scope = "phase2"
        return {"scope": scope, "model": self.model, "intermediate_dim": self.intermediate,
                "bottleneck_dim": self.bottleneck, "seed": self.seed}

    @property
    def config(self) -> AdapterConfig:
        return AdapterConfig(
            "residual" if self.model == "D" else "direct", self.intermediate, self.bottleneck,
            up_bias=self.model in ("A", "D"), activation="leaky_relu" if self.model == "C" else "relu",
        )


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _catalog(artifacts: Path) -> list[SavedRun]:
    root = Path(__file__).resolve().parent
    relu = {(r["model"], int(r["intermediate_dim"]), int(r["bottleneck_dim"]), int(r["seed"]), r["split"]): r
            for r in _read(root / "diagnostics.csv") if r["model"] != "cosplace" and r["family"] != "residual_initialization"}
    bias = {(int(r["intermediate_dim"]), int(r["bottleneck_dim"]), int(r["seed"]), r["split"]): r
            for r in _read(root / "direct_bias_runs.csv")}
    activation = {(r["configuration"], int(r["bottleneck_dim"]), int(r["seed"]), r["split"]): r
                  for r in _read(root / "leaky-relu" / "comparison.csv")}
    grid = [(arm, FIXED_INTERMEDIATE, width, seed) for arm, width, seed in product(ARMS, WIDTHS, SEEDS)]
    grid.extend((arm, 8, 4, seed) for arm, seed in product(("A", "B", "D"), (0, *SEEDS)))
    out = []
    for arm, intermediate, width, seed in grid:
        kind = "residual" if arm == "D" else "direct"
        slug = f"{kind}-i{intermediate}-b{width}-s{seed}"
        if arm in ("A", "D"):
            folder = artifacts / "classical-bottleneck" / "runs" / slug
            if seed == 0:
                folder = artifacts / "gardens-point-adapters" / kind
            archived = {split: {**relu[kind, intermediate, width, seed, split], "status": "completed",
                                "parameter_count": relu[kind, intermediate, width, seed, split]["trainable_parameters"]}
                        for split in SPLITS}
        else:
            folder = artifacts / ("direct-up-bias" if arm == "B" else "direct-leaky-relu") / "runs" / slug
            if arm == "B":
                folder = folder.with_name(f"{slug}-bias_off")
            if intermediate == FIXED_INTERMEDIATE:
                archived = {split: activation[arm, width, seed, split] for split in SPLITS}
            else:
                archived = {split: {key.removeprefix("bias_off_"): value for key, value in
                                    bias[intermediate, width, seed, split].items() if key.startswith("bias_off_")}
                            for split in SPLITS}
        out.append(SavedRun(arm, intermediate, width, seed, folder, archived))
    return out


def _archived_metrics(row: dict) -> dict:
    fields = (*RECALL_FIELDS, "pair_cosine_mean", "total_variance", "effective_rank", "positive_cosine_mean",
              "negative_cosine_mean", *RESIDUAL_FIELDS)
    values = {key: float(row[key]) for key in fields if row.get(key, "") != ""}
    if "positive_cosine_mean" in values and "negative_cosine_mean" in values:
        values["positive_negative_gap"] = values["positive_cosine_mean"] - values["negative_cosine_mean"]
    return values


@torch.inference_mode()
def _measure(
    cache: SplitCache, positives: list[set[int]], negatives: tuple[tuple[int, ...], ...],
    model: MLPAdapter | None = None, baseline_ranks: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor]:
    queries, gallery = cache.query.descriptors, cache.gallery.descriptors
    original = torch.cat((queries, gallery))
    if model is not None:
        queries, gallery = model(queries), model(gallery)
    adapted = torch.cat((queries, gallery))
    ranking = rank_cosine(queries, gallery, len(gallery))
    ranks = first_positive_ranks(ranking, positives)
    geometry = descriptor_geometry(adapted)
    positive, negative = pair_similarities(queries, gallery, positives, negatives)
    positive_mean, negative_mean = positive.mean().item(), negative.mean().item()
    metrics = {
        **{f"recall_at_{k}": value for k, value in recall_at_k(ranking, positives).items()},
        "mrr": mean_reciprocal_rank(ranking, positives), "pair_cosine_mean": geometry.cosine_mean,
        "total_variance": geometry.total_variance, "effective_rank": geometry.effective_rank,
        "positive_cosine_mean": positive_mean, "negative_cosine_mean": negative_mean,
        "positive_negative_gap": positive_mean - negative_mean,
    }
    if model is not None and model.config.kind == "residual":
        metrics.update(residual_stats(original, adapted, model.up(model.mlp(model.down(original))), model.config.alpha))
        metrics.update(rank_changes(baseline_ranks, ranks))
    return metrics, ranks


def _summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scope"], row["model"], row["intermediate_dim"], row["bottleneck_dim"], row["split"]].append(row)
    summary = []
    for (scope, model, intermediate, bottleneck, split), group in groups.items():
        archived_or_evaluated = [row for row in group if row["training_status"] in ("fixed", "completed")]
        included = [row for row in archived_or_evaluated if row["evaluation_status"] == "evaluated"]
        summary_row = {"scope": scope, "model": model, "intermediate_dim": intermediate, "bottleneck_dim": bottleneck,
                       "split": split, "attempted_runs": len(group) if model != "CosPlace" else 0,
                       "successful_runs": len(archived_or_evaluated) if model != "CosPlace" else 0,
                       "zero_descriptor_failures": sum(r["training_status"].startswith("zero_descriptor") for r in group),
                       "evaluated_runs": len(included), "seeds": " ".join(str(r["seed"]) for r in included if r["seed"] != "")}
        for metric in SUMMARY_FIELDS:
            values = [row[metric] for row in included if metric in row]
            if values:
                summary_row[f"{metric}_mean"] = mean(values)
                summary_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else None
        summary.append(summary_row)
    return summary


def _paired(identity: dict, cache: SplitCache, baseline: torch.Tensor, adapted: torch.Tensor) -> tuple[dict, list[dict]]:
    gain = adapted.double().reciprocal() - baseline.double().reciprocal()
    summary = {**identity, "query_count": len(baseline), **rank_changes(baseline, adapted),
               "mean_change_reciprocal_rank": gain.mean().item()}
    for k in (1, 5):
        before, after = baseline <= k, adapted <= k
        summary[f"recall_at_{k}_gains"] = int((~before & after).sum().item())
        summary[f"recall_at_{k}_losses"] = int((before & ~after).sum().item())
    queries = [{**identity, "query_id": record.image_id, "baseline_first_positive_rank": start,
                "adapter_first_positive_rank": finish, "delta_reciprocal_rank": change}
               for record, start, finish, change in zip(
                   cache.query.images, baseline.tolist(), adapted.tolist(), gain.tolist(), strict=True
               )]
    return summary, queries


def run(dataset: Path, artifacts: Path, output: Path) -> None:
    manifests = load_manifests(dataset)
    cache_dir = artifacts / "gardens-point"
    caches = {split: load_split_cache(cache_dir, split, manifests[split]) for split in SPLITS}
    train, val = caches.values()
    positives = {"train": load_training_positives(dataset, train.gallery.images, train.query.images),
                 "val": load_positives(dataset, "val", val.gallery.images, val.query.images)}
    negatives = {split: build_triplet_pool(cache.gallery.images, cache.query.images, positives[split],
                                           TRAINING.negative_window).negatives for split, cache in caches.items()}
    output.mkdir(parents=True, exist_ok=False)
    rows, baseline, paired_rows, paired_queries = [], {}, [], []
    for split, cache in caches.items():
        metrics, baseline[split] = _measure(cache, positives[split], negatives[split])
        rows.append({"scope": "baseline", "model": "CosPlace", "intermediate_dim": "", "bottleneck_dim": "",
                     "seed": "", "split": split, "query_count": len(cache.query.images),
                     "gallery_count": len(cache.gallery.images), "training_status": "fixed",
                     "evaluation_status": "evaluated", "parameter_count": 0, **metrics})
    for run in _catalog(artifacts):
        checkpoint = run.folder / "best.pt"
        history_path = run.folder / "history.json"
        model, epoch, checksum = None, "", ""
        if run.archived["val"]["status"] == "completed" and checkpoint.is_file():
            model, epoch = load_adapter(checkpoint, val)
            if model.config != run.config or epoch != int(run.archived["val"]["selected_epoch"]):
                raise ValueError(f"{checkpoint}: configuration or selected epoch differs from the archived result")
            with checkpoint.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if history_path.is_file():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history["training"]["seed"] != run.seed or history["best_epoch"] != epoch:
                    raise ValueError(f"{history_path}: seed or epoch differs from the archived result")
        for split, cache in caches.items():
            archived = run.archived[split]
            status = archived["status"]
            row = {**run.identity, "split": split, "query_count": len(cache.query.images),
                   "gallery_count": len(cache.gallery.images), "training_status": status,
                   "checkpoint": str(checkpoint) if status == "completed" else "", "checkpoint_sha256": checksum,
                   "selected_epoch": archived.get("selected_epoch", ""), "epochs_run": archived.get("epochs_run", ""),
                   "parameter_count": archived["parameter_count"], "activation": run.config.activation,
                   "up_bias": run.config.up_bias, "failure_split": archived.get("failure_split", ""),
                   "failure_epoch": archived.get("failure_epoch", ""),
                   "failure_zero_rows": archived.get("failure_zero_rows", "")}
            if status != "completed":
                row["evaluation_status"] = "training_failed"
            elif model is None:
                row.update({"evaluation_status": "missing_checkpoint", **_archived_metrics(archived)})
            else:
                if sum(param.numel() for param in model.parameters()) != int(archived["parameter_count"]):
                    raise ValueError(f"{checkpoint}: parameter count differs from the archived result")
                metrics, ranks = _measure(cache, positives[split], negatives[split], model, baseline[split])
                if any(metrics[key] != float(archived[key]) for key in RECALL_FIELDS):
                    raise ValueError(f"{checkpoint}: {split} recall differs from the archived result")
                row.update({"evaluation_status": "evaluated", "history_available": history_path.is_file(), **metrics})
                if split == "val" and run.intermediate == FIXED_INTERMEDIATE and run.bottleneck == 4:
                    comparison, queries = _paired(run.identity, cache, baseline[split], ranks)
                    paired_rows.append(comparison)
                    paired_queries.extend(queries)
            rows.append(row)
    _write_csv(output / "experiments.csv", rows)
    _write_csv(output / "multi_seed_summary.csv", _summary(rows))
    _write_csv(output / "paired_d4_summary.csv", paired_rows)
    _write_csv(output / "paired_d4_query_ranks.csv", paired_queries)
    print(f"Exported {len(rows)} train/val records and {len(paired_queries)} paired validation query ranks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "artifacts-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_root, args.artifacts_dir, args.output_dir)
