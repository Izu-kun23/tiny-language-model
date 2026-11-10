"""Plot next-character loss from a completed run."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    with (args.run_dir / "loss.csv").open(encoding="utf-8", newline="") as metrics:
        rows = list(csv.DictReader(metrics))
    steps = [int(row["step"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["validation_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    fig.set_facecolor("#f8faf9")
    ax.set_facecolor("#f8faf9")
    fig.suptitle("How the tiny transformer learns", x=0.07, ha="left",
                 fontsize=21, fontweight="bold", color="#122723")
    ax.set_title("Next-character prediction on fictional hospitality notes", loc="left",
                 fontsize=11, color="#475650", pad=13)
    ax.plot(steps, train, "o-", color="#006d53", linewidth=2.2, markersize=4,
            label="Training venues")
    ax.plot(steps, validation, "s--", color="#7850a3", linewidth=2.0, markersize=4,
            label="Validation venues")
    ax.set(xlabel="Optimizer updates", ylabel="Cross-entropy (nats per character)")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", color="#dbe4de")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(args.run_dir / "loss.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
