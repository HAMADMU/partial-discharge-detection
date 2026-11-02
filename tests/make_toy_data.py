"""Create a small SYNTHETIC dataset in the same format as the real one.

This only checks that the pipeline runs end to end before you download 10 GB.
The numbers it produces mean nothing: never report results from toy data.

Usage:
    python tests/make_toy_data.py --out data_toy
    python src/build_features.py --data-dir data_toy --out results_toy/features.csv
    python src/train.py --features results_toy/features.csv --out-dir results_toy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_toy")
    parser.add_argument("--measurements", type=int, default=60)
    parser.add_argument("--samples", type=int, default=800_000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, args.samples, endpoint=False)
    signals, meta = {}, []
    for m in range(args.measurements):
        faulty = m % 8 == 0  # about 1 in 8 measurements has "discharge"
        for phase in range(3):
            sid = 3 * m + phase
            x = 20 * np.sin(t + phase * 2 * np.pi / 3) + rng.normal(0, 2, args.samples)
            if faulty:
                spikes = rng.choice(args.samples, size=40, replace=False)
                x[spikes] += rng.choice([-1, 1], 40) * rng.uniform(25, 60, 40)
            signals[str(sid)] = np.clip(x, -128, 127).astype(np.int8)
            meta.append({"signal_id": sid, "id_measurement": m, "phase": phase,
                         "target": int(faulty)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_csv(out / "metadata_train.csv", index=False)
    pd.DataFrame(signals).to_parquet(out / "train.parquet")
    print(f"Wrote {len(meta)} toy signals to {out}/")


if __name__ == "__main__":
    main()
