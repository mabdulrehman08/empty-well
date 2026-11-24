"""Plot benchmark means and 95% intervals from results/summary.csv."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SPLITS = ("random", "held-out additive")
METHODS = {
    "residual": ("OOF residual", "#00746f", "o"),
    "random": ("Random flags", "#805c2c", "s"),
    "lowest": ("Lowest yield", "#6651a8", "^"),
}
PANELS = (
    ("corruption_cost", "Error added by corruption", "MAE change (yield points)"),
    ("precision", "Flag precision", "Selected wells among flags"),
    ("dropping_gain", "Error recovered after dropping", "MAE improvement (yield points)"),
)


def plot(folder):
    with (folder / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": "#152738",
        "axes.labelcolor": "#33485d",
        "xtick.color": "#33485d",
        "ytick.color": "#33485d",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.3, 7.7), sharex=True, sharey="col")
    fig.patch.set_facecolor("#f5f7fa")
    fig.subplots_adjust(left=0.08, right=0.975, top=0.776, bottom=0.13, hspace=0.37, wspace=0.36)
    fig.text(0.08, 0.942, "Dispense failure or real negative?", fontsize=23, weight="bold")
    fig.text(0.08, 0.902, "Simulated training-yield failures  /  Buchwald-Hartwig HTE  /  10% of wells flagged",
             fontsize=11, color="#33485d")
    fig.text(0.08, 0.027, "Means and 95% t intervals across runs. "
             "Positive recovery means lower test MAE after dropping flags.", color="#33485d")

    for i, split in enumerate(SPLITS):
        for j, (metric, title, y_label) in enumerate(PANELS):
            ax = axes[i, j]
            ax.set_facecolor("#ffffff")
            ax.set_title(title, loc="left", weight="bold", pad=10)
            ax.set_ylabel(y_label)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#dee5ed")
            ax.set_xticks([5, 10, 20], ["5%", "10%", "20%"])
            ax.set_xlim(3, 22)
            ax.tick_params(length=0, labelbottom=True, labelleft=True)
            if metric == "precision":
                ax.set_ylim(0, 1.02)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.axhline(0, color="#65778a", linewidth=0.9)
            if i == 1:
                ax.set_xlabel("Training wells selected for zeroing")

            methods = ("residual",) if metric == "corruption_cost" else METHODS
            for k, method in enumerate(methods):
                label, color, marker = METHODS[method]
                rows = [
                    row for row in summary
                    if (row["split"], row["method"], row["metric"]) == (split, method, metric)
                ]
                rates = [100 * float(row["rate"]) for row in rows]
                mean = [float(row["mean"]) for row in rows]
                lo = [m - float(row["ci_low"]) for m, row in zip(mean, rows)]
                hi = [float(row["ci_high"]) - m for m, row in zip(mean, rows)]
                offset = 0 if len(methods) == 1 else (k - 1) * 0.40
                ax.errorbar(
                    [rate + offset for rate in rates], mean, yerr=(lo, hi),
                    color="#273e56" if len(methods) == 1 else color,
                    marker=marker, label=label, capsize=3, linewidth=1.5,
                )
        y = axes[i, 0].get_position().y1 + 0.048
        fig.text(0.08, y, split.upper(), fontsize=10, weight="bold", color="#425b74")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, frameon=False, loc="upper right", bbox_to_anchor=(0.977, 0.866))
    fig.savefig(folder / "results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    plot(parser.parse_args().output)
