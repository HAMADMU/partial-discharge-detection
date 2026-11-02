"""Train and compare a logistic regression baseline and XGBoost on the cached features.

Evaluation is leakage-free: the three phases of one measurement always fall in the
same fold (StratifiedGroupKFold grouped by id_measurement). Otherwise the model could
see one phase of a faulty line in training and an almost identical phase in testing.

Because only a small share of signals contain partial discharge, the report focuses
on recall, precision, PR-AUC and MCC rather than accuracy.

Usage:
    python src/train.py --features results/features.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ID_COLUMNS = ["signal_id", "id_measurement", "target"]


def make_models(pos_weight):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
            random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="results/features.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = pd.read_csv(args.features)
    feature_cols = [c for c in data.columns if c not in ID_COLUMNS]
    X = pd.get_dummies(data[feature_cols], columns=["phase"], dtype=float)
    y = data["target"].to_numpy()
    groups = data["id_measurement"].to_numpy()

    print(f"{len(y)} signals, {y.sum()} with partial discharge ({y.mean():.1%})")
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    out_of_fold = {name: np.zeros(len(y)) for name in make_models(pos_weight)}
    for train_idx, test_idx in cv.split(X, y, groups):
        for name, model in make_models(pos_weight).items():
            model.fit(X.iloc[train_idx], y[train_idx])
            out_of_fold[name][test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    results = []
    plt.figure(figsize=(6, 5))
    for name, scores in out_of_fold.items():
        predicted = (scores >= 0.5).astype(int)
        results.append({
            "model": name,
            "recall": recall_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "pr_auc": average_precision_score(y, scores),
            "mcc": matthews_corrcoef(y, predicted),
            "true_positives": int(((predicted == 1) & (y == 1)).sum()),
            "false_positives": int(((predicted == 1) & (y == 0)).sum()),
            "missed": int(((predicted == 0) & (y == 1)).sum()),
        })
        precision, recall, _ = precision_recall_curve(y, scores)
        plt.plot(recall, precision, label=f"{name} (PR-AUC {results[-1]['pr_auc']:.3f})")

    plt.axhline(y.mean(), linestyle="--", color="grey", label="no-skill baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Partial discharge detection, out-of-fold")
    plt.legend()
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pr_curve.png", dpi=150)
    table = pd.DataFrame(results)
    table.to_csv(out_dir / "metrics.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Which features the tree model leaned on most.
    final = make_models(pos_weight)["xgboost"].fit(X, y)
    importance = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", header=["importance"])
    print("\nTop features (XGBoost):")
    print(importance.head(5).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
