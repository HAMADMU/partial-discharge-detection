"""Extract features for every signal and save them to results/features.csv.

This is the slow step (it filters every 800,000-sample signal), so it runs once and
caches the result. train.py then reads the CSV and runs in seconds.

Usage:
    python src/build_features.py --data-dir data
    python src/build_features.py --data-dir data --max-measurements 300   # quick trial
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from features import signal_features
from load_data import iter_signal_batches, load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results/features.csv")
    parser.add_argument("--max-measurements", type=int, default=None,
                        help="use a random subset of measurements for a quicker run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = load_metadata(args.data_dir)
    if args.max_measurements:
        # Sample whole measurements, so the three phases of a line stay together.
        keep = (meta["id_measurement"].drop_duplicates()
                .sample(n=args.max_measurements, random_state=args.seed))
        meta = meta[meta["id_measurement"].isin(keep)]

    rows = []
    start = time.time()
    for i, (sid, signal) in enumerate(iter_signal_batches(args.data_dir, meta["signal_id"])):
        rows.append({"signal_id": sid, **signal_features(signal)})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(meta)} signals processed ({time.time() - start:.0f} s)")

    features = meta.merge(pd.DataFrame(rows), on="signal_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out, index=False)
    print(f"Saved {len(features)} rows to {args.out}")


if __name__ == "__main__":
    main()
