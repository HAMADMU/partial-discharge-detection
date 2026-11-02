# Partial Discharge Detection on Medium-Voltage Power Lines

Detecting partial discharge (PD) in real 3-phase signals from medium-voltage overhead lines, using the
public **VSB Power Line Fault Detection** dataset (Kaggle).

Partial discharge is small electrical breakdown inside damaged insulation. It shows up as short,
sharp pulses on top of the 50 Hz waveform, and it is an early warning sign of insulation failure,
which is why online PD monitoring is used on cables, switchgear, transformers and rotating machines.

Only a small share of signals contain PD, so this is a **rare-event detection** problem, the same kind
of problem as my final-year project on power-system instability.

## Approach

1. **Filter:** a high-pass filter (10 kHz cut-off) removes the 50 Hz waveform so the pulses stand out.
2. **Features:** for each signal, count and size the pulses relative to that signal's own noise level
   (number of peaks, peak heights, peak-to-noise ratio, positive vs negative pulses).
3. **Models:** a logistic regression baseline and an XGBoost classifier.
4. **Leakage-free evaluation:** 5-fold cross-validation grouped by measurement, so the three phases of
   one line never sit on both sides of the split.
5. **Metrics for rare events:** recall, precision, PR-AUC and Matthews correlation coefficient (MCC),
   not accuracy.

## Results

Status: in progress. The pipeline has been tested end to end on synthetic data only. Results on the real VSB dataset will be added once the full run is complete.

## How to run

### 1. Install the libraries

```
python -m pip install -r requirements.txt
```

### 2. Test the pipeline on toy data (2 minutes)

This uses a small **synthetic** dataset only to check that everything runs. Its numbers mean nothing.

```
python tests/make_toy_data.py --out data_toy
python src/build_features.py --data-dir data_toy --out results_toy/features.csv
python src/train.py --features results_toy/features.csv --out-dir results_toy
```

### 3. Get the real data

1. Sign in to Kaggle and open the competition **"VSB Power Line Fault Detection"**.
2. Accept the competition rules, then download `metadata_train.csv` and `train.parquet` from the Data tab.
3. Put both files in the `data/` folder. `train.parquet` is several GB, so check you have the disk space.

### 4. Run on the real data

```
python src/explore.py --data-dir data
python src/build_features.py --data-dir data --max-measurements 300
python src/train.py --features results/features.csv
```

Start with `--max-measurements 300` for a quick first run, then remove it to use all 8,712 signals
(about 2,904 measurements). Feature extraction is the slow step and is cached in `results/features.csv`,
so you can retrain as often as you like.

## Project structure

```
src/load_data.py       reads metadata and signals in batches (the full file is too big for memory)
src/features.py        high-pass filter and pulse features
src/build_features.py  extracts features for every signal and caches them
src/train.py           grouped cross-validation, baseline vs XGBoost, metrics and plots
src/explore.py         plots a measurement with and without PD
tests/make_toy_data.py synthetic data to test the pipeline
```

## Next steps

- Use where each pulse falls in the 50 Hz cycle: PD tends to cluster at particular phase angles,
  which is the idea behind phase-resolved PD (PRPD) patterns.
- Separate PD pulses from noise sources such as corona, which also produce pulses.
- Combine the three phases of each measurement into one prediction.
- Tune the decision threshold to the cost of a missed fault versus a false alarm.

## Use of AI-assisted tools

The initial project structure and code were drafted with help from an AI assistant (Claude). I’m working through each part, checking and adapting it as I run it on the real data, and I’ll note my own changes here as the project develops.
