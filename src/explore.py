"""Plot one measurement with partial discharge and one without, raw and filtered.

Run this first: seeing the pulses on the filtered signal is what makes the features
in features.py make sense.

Usage:
    python src/explore.py --data-dir data
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import highpass
from load_data import load_metadata, load_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results/example_signals.png")
    args = parser.parse_args()

    meta = load_metadata(args.data_dir)
    faulty = meta[meta["target"] == 1]["id_measurement"].iloc[0]
    healthy = meta[meta["target"] == 0]["id_measurement"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, (label, measurement) in enumerate([("With PD", faulty), ("Without PD", healthy)]):
        rows = meta[meta["id_measurement"] == measurement]
        signals = load_signals(args.data_dir, rows["signal_id"])
        for _, row in rows.iterrows():
            raw = signals[str(row["signal_id"])].to_numpy()
            t = np.arange(len(raw)) / len(raw) * 20  # milliseconds across one cycle
            axes[0, col].plot(t, raw, linewidth=0.5, label=f"phase {row['phase']}")
            axes[1, col].plot(t, highpass(raw), linewidth=0.5)
        axes[0, col].set_title(f"{label}: measurement {measurement} (raw)")
        axes[1, col].set_title("High-pass filtered (50 Hz removed)")
        axes[1, col].set_xlabel("Time (ms)")
        axes[0, col].legend(loc="upper right")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
