"""Plot final classical Recall@K and full-gallery MRR from the benchmark CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_final_benchmark import ARMS, WIDTHS

LABELS = {"A": "A · ReLU + bias", "B": "B · Bias-free ReLU", "C": "C · Bias-free LeakyReLU", "D": "D · Residual"}
COLORS = {"A": "#526578", "B": "#b45309", "C": "#096c70", "D": "#7845a2"}


def _panel(axis, rows: list[dict], split: str, metric: str, scale: int) -> None:
    for arm in ARMS:
        entries = {int(row["bottleneck_dim"]): row for row in rows
                   if row["scope"] == "primary" and row["model"] == arm and row["split"] == split}
        means = [scale * float(entries[width][f"{metric}_mean"] or "nan") for width in WIDTHS]
        deviations = [scale * float(entries[width][f"{metric}_std"] or "nan") for width in WIDTHS]
        axis.errorbar(WIDTHS, means, yerr=deviations, label=LABELS[arm], color=COLORS[arm],
                      marker="o", linewidth=1.6, capsize=3)
    baseline = next(row for row in rows if row["model"] == "CosPlace" and row["split"] == split)
    axis.axhline(scale * float(baseline[f"{metric}_mean"]), color="#24262a", linestyle="--", label="CosPlace")
    axis.set_xscale("log", base=2)
    axis.set_xticks(WIDTHS, labels=tuple(map(str, WIDTHS)))
    axis.grid(alpha=0.22)


def _finish(figure, axes, name: str, out: Path) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=5, frameon=False, fontsize=8.5)
    figure.supxlabel("Bottleneck output dimension · intermediate width 64")
    figure.savefig(out / name, dpi=165)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    root = parser.parse_args().run_dir
    with (root / "multi_seed_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = root / "figures"
    output.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(10.7, 3.6), sharey=True, layout="constrained")
    for axis, k in zip(axes, (1, 5, 10), strict=True):
        _panel(axis, rows, "val", f"recall_at_{k}", 100)
        axis.set_title(f"Recall@{k}")
        axis.set_ylim(0, 104)
    figure.supylabel("Validation recall (%)")
    _finish(figure, axes, "recall-comparison.png", output)

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True, layout="constrained")
    for axis, split, label in zip(axes, ("train", "val"), ("Training", "Validation"), strict=True):
        _panel(axis, rows, split, "mrr", 1)
        axis.set_title(label)
        axis.set_ylim(0, 1)
    figure.supylabel("Mean reciprocal rank")
    _finish(figure, axes, "mrr-comparison.png", output)


if __name__ == "__main__":
    main()
