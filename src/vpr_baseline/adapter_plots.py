"""Render the adapter experiment figures from the saved CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {"cosplace": "CosPlace", "direct": "Direct", "residual": "Residual"}
SPLITS = {"train": "Training", "val": "Validation"}
COLORS = {"cosplace": "#4b5563", "direct": "#b45309", "residual": "#096c70"}
PAIR_COLORS = {"positive": "#096c70", "negative": "#b45309"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _ecdf(axis, values: list[float], label: str) -> None:
    scores = np.sort(values)
    fraction = np.arange(1, len(scores) + 1) / len(scores)
    axis.step(np.r_[-1, scores, 1.02], np.r_[0, fraction, 1], where="post", label=label.title(),
              color=PAIR_COLORS[label], linewidth=1.7)


def _plot_pairs(rows: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 5.8), sharex=True, sharey=True, layout="constrained")
    left_edge = max(-1, min(float(row["cosine"]) for row in rows) - 0.03)
    for i, (split, split_label) in enumerate(SPLITS.items()):
        for j, (model, model_label) in enumerate(MODELS.items()):
            axis = axes[i, j]
            zoom = axis.inset_axes([0.13, 0.38, 0.55, 0.50]) if model == "direct" else None
            model_scores = []
            for label in PAIR_COLORS:
                values = [float(row["cosine"]) for row in rows
                          if row["split"] == split and row["model"] == model and row["label"] == label]
                _ecdf(axis, values, label)
                model_scores.extend(values)
                if zoom is not None:
                    _ecdf(zoom, values, label)
            if zoom is not None:
                zoom.set_xlim(min(model_scores), 1.0005)
                zoom.set_title("Direct score range", fontsize=8)
                zoom.set_yticks((0, 0.5, 1), labels=("0", "0.5", "1"))
                zoom.tick_params(labelsize=7)
            axis.set_title(f"{split_label} · {model_label}")
            axis.set_xlim(left_edge, 1.02)
            axis.set_ylim(-0.02, 1.03)
            axis.grid(alpha=0.22)
    fig.supxlabel("Cosine similarity")
    fig.supylabel("Cumulative fraction of pairs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(out / "similarity-distributions.png", dpi=170)
    plt.close(fig)


def _plot_spectra(rows: list[dict[str, str]], diagnostics: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True, layout="constrained")
    for axis, (split, split_label) in zip(axes, SPLITS.items(), strict=True):
        for model, label in MODELS.items():
            points = [(int(row["component"]), float(row["singular_value"])) for row in rows
                      if row["split"] == split and row["model"] == model and float(row["singular_value"]) > 1e-12]
            rank = next(float(row["effective_rank"]) for row in diagnostics
                        if row["family"] == "phase2" and row["split"] == split and row["model"] == model)
            axis.semilogy(*zip(*points), color=COLORS[model], label=f"{label} (effective rank {rank:.1f})")
        axis.set_title(split_label)
        axis.grid(alpha=0.22)
        axis.legend(loc="center right", frameon=False, fontsize=9)
    fig.supxlabel("Component index")
    fig.supylabel("Centered singular value")
    fig.savefig(out / "singular-spectrum.png", dpi=170)
    plt.close(fig)


def _plot_recall(rows: list[dict[str, str]], diagnostics: list[dict[str, str]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.4), sharey=True, layout="constrained")
    baseline = next(row for row in diagnostics if row["model"] == "cosplace" and row["split"] == "val")
    for axis, k in zip(axes, (1, 5, 10), strict=True):
        metric = f"recall_at_{k}"
        for model, label in (("direct", "Direct"), ("residual", "Residual")):
            data = sorted((row for row in rows if row["family"] == "fixed_intermediate" and row["model"] == model),
                          key=lambda row: int(row["bottleneck_dim"]))
            axis.errorbar([int(row["bottleneck_dim"]) for row in data],
                          [100 * float(row[f"{metric}_mean"]) for row in data],
                          yerr=[100 * float(row[f"{metric}_std"]) for row in data],
                          label=label, color=COLORS[model], marker="o", capsize=3)
        axis.axhline(100 * float(baseline[metric]), color=COLORS["cosplace"], linestyle="--", label="CosPlace")
        axis.set_title(f"Recall@{k}")
        axis.set_xscale("log", base=2)
        axis.set_xticks((4, 8, 16, 32), labels=("4", "8", "16", "32"))
        axis.set_ylim(0, 107)
        axis.grid(alpha=0.22)
    fig.supxlabel("Bottleneck output dimension · intermediate width fixed at 64")
    fig.supylabel("Validation recall (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(out / "bottleneck-recall.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    source = args.run_dir
    out = source / "figures"
    out.mkdir(exist_ok=True)
    diagnostics = _read(source / "diagnostics.csv")
    _plot_pairs(_read(source / "pair_scores.csv"), out)
    _plot_spectra(_read(source / "singular_values.csv"), diagnostics, out)
    _plot_recall(_read(source / "seed_summary.csv"), diagnostics, out)


if __name__ == "__main__":
    main()
