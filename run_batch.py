"""Run graph generation and model training for a sweep over one generation setting.

One generation setting (e.g. num_nodes) is the variable; every other setting
is fixed. For each value of the variable a graph is generated, then the model
is trained on it, one value after the other. At the end a table of
train/val/test accuracy is printed and saved as CSV.

Usage (from the repo root):
    venv/bin/python run_batch.py configs/example_nodes.json

Config (JSON):
    {
      "variable": {"name": "num_nodes", "values": [250, 500, 1000, 2000]},
      "fixed": {"num_edges": 2200, "feature_type": "random", ...},
      "model": "LINKX_multi_hide.py",
      "model_args": ["--patience", "50"]
    }

Generation settings (anything not in "fixed" uses the default shown):
    num_nodes     1000          num_features  16
    num_edges     2200          feature_type  "random" or "orthogonal"
    node_weights  [1, 1, 1]     noise_std     0.1 (orthogonal only)
    edge_weights  [1, 1, 1]     seed          0
    train_frac    0.5
    val_frac      0.25

Each run is saved under runs/<timestamp>/<variable>_<value>/: the graph in
data/graph.pt, the model's log (train.log) and its loss plot. The batch folder
also holds results.csv and accuracy_vs_<variable>.png, a plot of train, val
and test accuracy against the variable (numeric values at their actual
positions on the x axis, lists and text evenly spaced in config order).
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # save the plot to a file; never open a window mid-batch
import matplotlib.pyplot as plt
import torch

import generate_graph as gen
from model_utils import RESULTS_PREFIX

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "num_nodes": gen.DEFAULT_NUM_NODES,
    "num_edges": gen.DEFAULT_NUM_EDGES,
    "node_weights": [1, 1, 1],
    "edge_weights": [1, 1, 1],
    "train_frac": gen.DEFAULT_TRAIN_FRAC,
    "val_frac": gen.DEFAULT_VAL_FRAC,
    "num_features": 16,
    "feature_type": "random",
    "noise_std": 0.1,
    "seed": 0,
}
METRICS = ("train_acc", "val_acc", "test_acc")


def check_settings(s):
    """Return an error message if the generation settings are invalid, else None."""
    if not isinstance(s["num_nodes"], int) or s["num_nodes"] < 1:
        return "num_nodes must be an integer >= 1"
    if not isinstance(s["num_edges"], int) or s["num_edges"] < 0:
        return "num_edges must be an integer >= 0"
    for key in ("node_weights", "edge_weights"):
        w = s[key]
        if len(w) != 3 or any(x < 0 for x in w) or sum(w) == 0:
            return f"{key} must be 3 non-negative numbers, not all zero"
    if not 0 <= s["train_frac"] <= 1:
        return "train_frac must be between 0 and 1"
    if not 0 <= s["val_frac"] <= 1 - s["train_frac"] + 1e-9:
        return "val_frac must be between 0 and 1 - train_frac"
    if s["feature_type"] not in ("random", "orthogonal"):
        return 'feature_type must be "random" or "orthogonal"'
    if s["num_features"] < (len(gen.COLORS) if s["feature_type"] == "orthogonal" else 1):
        return "num_features too small (orthogonal features need at least 3)"
    if s["noise_std"] < 0:
        return "noise_std must be >= 0"
    return None


def generate(s, out_path):
    """Generate a graph with settings `s` and save it to `out_path`."""
    normalize = lambda w: [x / sum(w) for x in w]
    graph = gen.build_graph(s["num_nodes"], s["num_edges"], normalize(s["node_weights"]),
                            normalize(s["edge_weights"]), s["seed"])
    noise = s["noise_std"] if s["feature_type"] == "orthogonal" else 0.0
    data = gen.to_data(graph, s["train_frac"], s["val_frac"], s["num_features"],
                       s["feature_type"], noise)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)
    return gen.count_components(graph["num_nodes"], graph["edges"])


def run_model(model, run_dir, model_args):
    """Train `model` on run_dir/data/graph.pt; return (metrics or None, status)."""
    log_path = os.path.join(run_dir, "train.log")
    env = dict(os.environ, MPLBACKEND="Agg")  # save plots without opening windows
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, model), *model_args],
            cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, text=True)
    with open(log_path) as log:
        lines = [l for l in log if l.startswith(RESULTS_PREFIX)]
    if proc.returncode != 0 or not lines:
        return None, f"failed (exit {proc.returncode}), see {os.path.relpath(log_path, REPO_ROOT)}"
    return json.loads(lines[-1][len(RESULTS_PREFIX):]), "ok"


def plot_accuracies(rows, var, model, path):
    """Plot train/val/test accuracy against the variable and save it to `path`.

    Numeric variables are placed at their actual values on the x axis, so the
    spacing is proportional. Lists and text are evenly spaced in config order.
    Skipped or failed values appear as gaps.
    """
    numeric = all(isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool)
                  for r in rows)
    if numeric:
        rows = sorted(rows, key=lambda r: r["value"])
        positions = [r["value"] for r in rows]
    else:
        positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(rows)), 5))
    for key, label in (("train_acc", "Train"), ("val_acc", "Validation"), ("test_acc", "Test")):
        ys = [r.get(key, math.nan) for r in rows]
        ax.plot(positions, ys, marker="o", label=f"{label} accuracy")
    if not numeric:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(r["value"]) for r in rows],
                           rotation=30 if len(rows) > 6 else 0,
                           ha="right" if len(rows) > 6 else "center")
    ax.set_xlabel(var)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model}: accuracy vs {var}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(x):
    return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    var, values = config["variable"]["name"], config["variable"]["values"]
    if var not in DEFAULTS:
        sys.exit(f"Error: unknown variable {var!r}; choose one of {', '.join(DEFAULTS)}")
    unknown = set(config.get("fixed", {})) - set(DEFAULTS)
    if unknown:
        sys.exit(f"Error: unknown fixed setting(s): {', '.join(sorted(unknown))}")
    if var in config.get("fixed", {}):
        print(f"Note: {var} is the variable, so its fixed value is ignored.")
    model = config["model"]
    if not os.path.isfile(os.path.join(REPO_ROOT, model)):
        sys.exit(f"Error: model script not found: {model}")
    model_args = [str(a) for a in config.get("model_args", [])]

    batch_dir = os.path.join(REPO_ROOT, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(batch_dir)
    with open(os.path.join(batch_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    rows = []
    for value in values:
        settings = {**DEFAULTS, **config.get("fixed", {}), var: value}
        run_dir = os.path.join(batch_dir, f"{var}_{value}".replace(" ", ""))
        print(f"\n=== {var} = {value} ({model}) ===")
        error = check_settings(settings)
        if error is None:
            try:
                components = generate(settings, os.path.join(run_dir, "data", "graph.pt"))
                print(f"Generated graph ({components} connected component(s))")
            except ValueError as e:
                error = str(e)
        if error is not None:
            print(f"Skipped: {error}")
            rows.append({"value": value, "status": f"skipped: {error}"})
            continue
        print("Training ...", end=" ", flush=True)
        metrics, status = run_model(model, run_dir, model_args)
        print(status if metrics is None else
              "  ".join(f"{k}={metrics[k]:.4f}" for k in METRICS)
              + f"  violations={metrics.get('violations', '-')}")
        rows.append({"value": value, "status": status, **(metrics or {})})

    columns = ["value", *METRICS, "violations", "violation_rate", "best_epoch", "status"]
    csv_path = os.path.join(batch_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[var if c == "value" else c for c in columns])
        writer.writeheader()
        for r in rows:
            writer.writerow({(var if c == "value" else c): r.get(c, "") for c in columns})

    headers = [var, *METRICS, "violations", "violation_rate", "best_epoch", "status"]
    # Keep the table narrow: full skip/failure reasons are printed above and in the CSV.
    table = [[fmt(r.get(c)) for c in columns[:-1]] + [r["status"].split(":")[0]] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]
    print(f"\nModel: {model}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for t in table:
        print("  ".join(c.ljust(w) for c, w in zip(t, widths)))
    print(f"\nSaved results to {os.path.relpath(csv_path, REPO_ROOT)}")

    plot_path = os.path.join(batch_dir, f"accuracy_vs_{var}.png")
    plot_accuracies(rows, var, model, plot_path)
    print(f"Saved accuracy plot to {os.path.relpath(plot_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
