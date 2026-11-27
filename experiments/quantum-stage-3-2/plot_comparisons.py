"""Plot paired validation MRR differences from the Stage 3.2 bootstrap table."""

import argparse
import csv
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(table: Path, output: Path) -> None:
    scales = {"005": "0.05", "010": "0.10", "020": "0.20", "050": "0.50"}
    controls = {"pqc_frozen": "Frozen PQC", "pqc_separable": "PQC without CNOTs",
                "classical_matched": "Matched classical", "classical_identity": "Identity mixer",
                "pqc_reset": "Angles reset"}
    with table.open(newline="", encoding="utf-8") as stream:
        rows = {(row["arm"], row["control"]): row for row in csv.DictReader(stream) if row["metric"] == "MRR"}

    fig, panels = plt.subplots(2, 2, figsize=(10.2, 5.2), sharex=True, sharey=True, layout="constrained")
    for panel, (suffix, scale) in zip(panels.flat, scales.items(), strict=True):
        for i, control in enumerate(controls):
            row = rows[f"pqc_{suffix}", f"{control}_{suffix}"]
            effect, low, high = (float(row[field]) for field in
                                 ("mean_delta", "ci95_low_block_5", "ci95_high_block_5"))
            panel.plot([low, high], [i, i], color="#266d88", linewidth=2, solid_capstyle="round")
            panel.scatter(effect, i, color="#16465a", s=24, zorder=3)
        panel.axvline(0, color="#707070", linestyle="--", linewidth=1)
        panel.set_title(f"λ = {scale}")
        panel.set_yticks(range(len(controls)), labels=list(controls.values()))
        panel.spines[["top", "right"]].set_visible(False)
    panels[0, 0].invert_yaxis()
    fig.suptitle("Trainable PQC versus same-scale residual controls")
    fig.supxlabel("Validation MRR difference (PQC − control), 95% paired 5-frame block intervals")
    svg = StringIO()
    with plt.rc_context({"svg.hashsalt": "quantum-stage-3-2"}):
        fig.savefig(svg, format="svg", metadata={"Date": None})
    output.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.table, args.output)
