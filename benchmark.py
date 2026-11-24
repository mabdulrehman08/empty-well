"""Measure the effect of zeroing training yields and the value of flagging."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from fetch_data import DATA, verify

FEATURES = ("aryl_halide", "ligand", "base", "additive")
RATES = (0.05, 0.10, 0.20)
BUDGET = 0.10
METHODS = ("residual", "random", "lowest")
SPLITS = ("random", "held-out additive")


def load_data():
    verify(DATA.read_bytes())
    with DATA.open(encoding="utf-8", newline="") as source:
        rows = [
            row for row in csv.DictReader(source)
            if all(row[key] for key in (*FEATURES, "yield"))
        ]
    x = np.array([[row[key] for key in FEATURES] for row in rows])
    y = np.array([float(row["yield"]) for row in rows])
    return x, y


def split_indices(x, split, seed):
    if split == "random":
        return train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    held_out = np.unique(x[:, -1])[seed]
    test = x[:, -1] == held_out
    return np.flatnonzero(~test), np.flatnonzero(test)


def forest_predict(x_train, y_train, x_eval, seed):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train = encoder.fit_transform(x_train)
    model = RandomForestRegressor(n_estimators=64, min_samples_leaf=2, random_state=seed)
    model.fit(train, y_train)
    return model.predict(encoder.transform(x_eval))


def corrupt_yields(y_train, rate, rng):
    selected = np.zeros(len(y_train), dtype=bool)
    selected[rng.choice(len(y_train), round(rate * len(y_train)), replace=False)] = True
    observed = y_train.copy()
    observed[selected] = 0
    return observed, selected


def oof_predictions(x_train, observed, seed):
    predictions = np.empty(len(observed))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fit, val in folds.split(x_train):
        predictions[val] = forest_predict(x_train[fit], observed[fit], x_train[val], seed)
    return predictions


def flag_indices(observed, predicted, rng):
    count = round(BUDGET * len(observed))
    tie_break = rng.permutation(len(observed))
    return {
        "residual": np.lexsort((tie_break, observed - predicted))[:count],
        "random": rng.permutation(len(observed))[:count],
        "lowest": np.lexsort((tie_break, observed))[:count],
    }


def run_one(job):
    x, y, split, seed = job
    train, test = split_indices(x, split, seed)
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    clean_mae = mean_absolute_error(y_test, forest_predict(x_train, y_train, x_test, seed))
    rows = []
    for rate_index, rate in enumerate(RATES):
        corrupt_rng, flag_rng = [
            np.random.default_rng(stream)
            for stream in np.random.SeedSequence([seed, SPLITS.index(split), rate_index]).spawn(2)
        ]
        observed, selected = corrupt_yields(y_train, rate, corrupt_rng)
        dirty_mae = mean_absolute_error(y_test, forest_predict(x_train, observed, x_test, seed))
        predicted = oof_predictions(x_train, observed, seed)
        for method, flags in flag_indices(observed, predicted, flag_rng).items():
            hits = selected[flags].sum()
            keep = np.ones(len(train), dtype=bool)
            keep[flags] = False
            drop_mae = mean_absolute_error(
                y_test, forest_predict(x_train[keep], observed[keep], x_test, seed)
            )
            rows.append({
                "split": split, "seed": seed, "held_out": x_test[0, -1] if split == SPLITS[1] else "",
                "rate": rate, "method": method, "train_wells": len(train), "test_wells": len(test),
                "selected_wells": selected.sum(), "changed_wells": np.count_nonzero(observed != y_train),
                "flagged_wells": len(flags), "hits": hits,
                "precision": hits / len(flags), "recall": hits / selected.sum(),
                "clean_mae": clean_mae, "corrupt_mae": dirty_mae, "dropped_mae": drop_mae,
                "corruption_cost": dirty_mae - clean_mae, "dropping_gain": dirty_mae - drop_mae,
            })
    return rows


def mean_ci(values):
    values = np.asarray(values)
    mean = values.mean()
    half_width = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def summarize(rows):
    summary = []
    for split in SPLITS:
        for rate in RATES:
            block = [row for row in rows if row["split"] == split and row["rate"] == rate]
            by_method = {method: [row for row in block if row["method"] == method] for method in METHODS}
            for method, method_rows in by_method.items():
                metrics = ["precision", "recall", "dropped_mae", "dropping_gain"]
                if method == "residual":
                    metrics.extend(["clean_mae", "corrupt_mae", "corruption_cost"])
                values = {metric: [row[metric] for row in method_rows] for metric in metrics}
                if method != "residual":
                    values["residual_precision_advantage"] = [
                        residual["precision"] - baseline["precision"]
                        for residual, baseline in zip(by_method["residual"], method_rows, strict=True)
                    ]
                for metric, sample in values.items():
                    mean, lo, hi = mean_ci(sample)
                    summary.append({
                        "split": split, "rate": rate, "method": method, "metric": metric,
                        "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(sample),
                    })
    return summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="Parallel split runs (default: 2)")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    x, y = load_data()
    seeds = range(len(np.unique(x[:, -1])))
    jobs = [(x, y, split, seed) for split in SPLITS for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = [row for result in executor.map(run_one, jobs) for row in result]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "runs.csv", rows)
    write_csv(args.output / "summary.csv", summarize(rows))


if __name__ == "__main__":
    main()
