"""Combine independently trained GeoDynamics folds after protocol checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--expected-folds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = []
    configs = []
    for directory in sorted(args.root.glob(args.pattern)):
        metrics_path = directory / "fold_metrics.csv"
        config_path = directory / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        config.setdefault("feature_normalization", "none")
        config.setdefault("vectorization", "full")
        config.setdefault(
            "batchnorm_singleton_policy",
            "drop-last"
            if config.get("feature_normalization") == "batch"
            else "not-applicable",
        )
        with metrics_path.open(newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        if len(fold_rows) != 1:
            raise ValueError(f"expected one fold in {directory}")
        row = {key: int(value) if key == "fold" else float(value)
               for key, value in fold_rows[0].items()}
        rows.append(row)
        configs.append(config)

    folds = sorted(row["fold"] for row in rows)
    expected = list(range(1, args.expected_folds + 1))
    if folds != expected:
        raise ValueError(f"need exactly folds {expected}, found {folds}")
    comparable_keys = (
        "dataset", "input_dir", "label_path", "actor_path", "split_protocol",
        "architecture", "implementation_version", "adni_label_scheme",
        "dataset_warning",
        "florence_source", "max_time_points",
        "spd_normalization", "padding_policy",
        "window_size", "kernel_size", "tau", "stein_iterations",
        "learning_rate", "batch_size", "layers", "epochs",
        "readout", "feature_normalization", "vectorization",
        "batchnorm_singleton_policy",
        "evaluation_epoch", "seed", "torch_threads",
    )
    reference = configs[0]
    for config in configs[1:]:
        differences = {
            key: (reference.get(key), config.get(key))
            for key in comparable_keys if reference.get(key) != config.get(key)
        }
        if differences:
            raise ValueError(f"fold configurations differ: {differences}")

    metrics = ("accuracy", "precision", "recall", "f1")
    summary = {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)),
        }
        for metric in metrics
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["fold"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
