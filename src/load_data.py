"""Load the VSB Power Line Fault Detection data.

Each signal is 800,000 voltage samples covering one 50 Hz cycle (20 ms), so the
sampling rate is 40 MHz. The parquet file stores one signal per column, named by
its signal_id. The full file is about 10 GB, so signals are read in batches of
columns rather than all at once.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SAMPLES_PER_CYCLE = 800_000
CYCLE_SECONDS = 0.02
SAMPLING_RATE_HZ = SAMPLES_PER_CYCLE / CYCLE_SECONDS  # 40 MHz


def load_metadata(data_dir):
    """Return metadata_train.csv: signal_id, id_measurement, phase, target."""
    return pd.read_csv(Path(data_dir) / "metadata_train.csv")


def load_signals(data_dir, signal_ids):
    """Return a DataFrame with one column per requested signal_id."""
    columns = [str(s) for s in signal_ids]
    table = pq.read_table(Path(data_dir) / "train.parquet", columns=columns)
    return table.to_pandas()


def iter_signal_batches(data_dir, signal_ids, batch_size=300):
    """Yield (signal_id, numpy array) pairs, reading batch_size columns at a time."""
    signal_ids = list(signal_ids)
    for start in range(0, len(signal_ids), batch_size):
        batch = signal_ids[start:start + batch_size]
        frame = load_signals(data_dir, batch)
        for sid in batch:
            yield sid, frame[str(sid)].to_numpy()
