"""Render the Direct activation ablation plots from the experiment exports."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_direct_leaky_relu import ARMS, SEEDS, SPLITS, WIDTHS

LABELS = {"A": "A · ReLU + bias", "B": "B · ReLU, bias-free", "C": "C · LeakyReLU, bias-free"}
COLORS = {"A": "#4b5563", "B": "#b45309", "C": "#096c70"}
SPLIT_LABELS = {"train": "Training", "val": "Validation"}
PAIR_COLORS = {"positive": "#096c70", "negative": "#b45309"}


def _geometry(summary: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 5.9), sharex=True, layout="constrained")
    for i, split in enumerate(SPLITS):
        fields = (("pair_cosine_mean", "Distinct-image cosine"), ("total_variance", "Total variance"),
                  ("effective_rank", "Effective rank"))
        for axis, (metric, name) in zip(axes[i], fields, strict=True):
            for arm in ARMS:
                rows = {int(row["bottleneck_dim"]): row for row in summary
                        if row["split"] == split and row["configuration"] == arm}
                averages = [float(rows[d][f"{metric}_mean"] or "nan") for d in WIDTHS]
                deviations = [float(rows[d][f"{metric}_std"] or "nan") for d in WIDTHS]
                axis.errorbar(WIDTHS, averages, yerr=deviations, color=COLORS[arm], label=LABELS[arm],
                              marker="o", capsize=3, linewidth=1.6)
            axis.set_title(f"{SPLIT_LABELS[split]} · {name}")
            axis.set_xscale("log", base=2)
            axis.set_xticks(WIDTHS, labels=tuple(map(str, WIDTHS)))
            axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    figure.supxlabel("Bottleneck output dimension")
    figure.savefig(output / "descriptor-geometry.png", dpi=155)
    plt.close(figure)


def _plot_pairs(scores, split: str, output: Path) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(10.4, 8.8), sharex=True, sharey=True, layout="constrained")
    for i, width in enumerate(WIDTHS):
        for j, arm in enumerate(ARMS):
            axis = axes[i, j]
            keys = [f"{split}_{arm}_{width}_{seed}" for seed in SEEDS
                    if f"{split}_{arm}_{width}_{seed}_positive" in scores]
            if keys:
                for label, color in PAIR_COLORS.items():
                    values = np.sort(np.concatenate([scores[f"{key}_{label}"] for key in keys]))
                    fraction = np.arange(1, len(values) + 1) / len(values)
                    axis.step(np.r_[-1, values, 1], np.r_[0, fraction, 1], where="post",
                              color=color, label=label.title(), linewidth=1.7)
            else:
                axis.text(0.5, 0.5, "0 successful seeds", transform=axis.transAxes, ha="center", va="center")
            axis.set_title(f"{LABELS[arm]} · d={width} ({len(keys)}/3)", fontsize=10)
            axis.set_xlim(-1.02, 1.02)
            axis.set_ylim(-0.02, 1.03)
            axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False,
                  title=f"{SPLIT_LABELS[split]} pairs")
    figure.supxlabel("Cosine similarity")
    figure.supylabel("Cumulative fraction of pairs")
    figure.savefig(output / f"pair-distributions-{split}.png", dpi=155)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    root = parser.parse_args().run_dir
    output = root / "figures"
    output.mkdir(exist_ok=True)
    with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    _geometry(summary, output)
    with np.load(root / "pair_scores.npz", allow_pickle=False) as scores:
        for split in SPLITS:
            _plot_pairs(scores, split, output)


if __name__ == "__main__":
    main()
