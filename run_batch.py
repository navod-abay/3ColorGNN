"""Run graph generation and model training for a sweep over one generation setting.

One generation setting (e.g. num_nodes) is the variable; every other setting
is fixed. For each value of the variable a graph is generated, then the model
is trained on it, one value after the other. At the end a table of the
results is printed and saved as CSV, and the results are plotted against the
variable. Flags choose what is printed and plotted; the CSV always holds every
metric.

Usage (from the repo root):
    venv/bin/python run_batch.py configs/example_nodes.json
    venv/bin/python run_batch.py configs/example_nodes.json --homophily
    venv/bin/python run_batch.py configs/example_nodes.json --accuracy --f1 --node-violations
    venv/bin/python run_batch.py configs/example_nodes.json --f1 --per-class --no-train --no-val

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
also holds results.csv and results_vs_<variable>.png, a plot against the
variable (numeric values at their actual positions on the x axis, lists and
text evenly spaced in config order). By default the plot has one panel, the
coloring violation ratio (violating edges / total edges). Flags add to it, and
to the printed table:
    --accuracy         a panel of train/val/test accuracy, on top
    --f1               a panel of train/val/test macro F1 (the unweighted mean
                       of the per-class F1 scores, so the minority classes
                       count as much as the majority one), below accuracy
    --node-violations  a line for nodes sharing a color with a neighbor /
                       total nodes, in the violation panel
--per-class adds a line per class to the accuracy panel (per-class accuracy:
the share of each class's nodes predicted correctly) and to the F1 panel
(per-class F1), and --no-train / --no-val leave those splits out of both.

--degree-hist passes --degree-hist on to the model, so each run folder also
gets a histogram of test node degrees with the misclassified test nodes on top
(only models listed in DEGREE_HIST_MODELS support it).

With --homophily, the CE and adjusted CE homophily of each generated graph
(see measure_homophily.py) are also added to the table and the CSV, and
plotted against the variable in homophily_vs_<variable>.png.
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
import measure_homophily as mh
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
F1_METRICS = ("train_f1", "val_f1", "test_f1")
PER_CLASS_METRICS = tuple(f"{split}_{kind}_class{c}" for kind in ("acc", "f1")
                          for split in ("train", "val", "test") for c in range(3))
# Fixed colors per split, so a split keeps its color when others are hidden.
SPLITS = (("train", "Train", "tab:blue"), ("val", "Validation", "tab:orange"),
          ("test", "Test", "tab:green"))
DEGREE_HIST_MODELS = ("GCN_multi_hide.py", "LINKX_multi_hide.py",
                      "LineGraphGNNs/train_multi_hide.py")  # model scripts that accept --degree-hist
CLASS_STYLES = (("--", "^"), (":", "s"), ("-.", "v"))  # line style and marker per class


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


def measure_homophily(graph_path):
    """Return the CE, chance level and adjusted CE homophily of a saved graph."""
    ce, chance, adjusted = mh.adjusted_ce_homophily(*mh.load_graph(graph_path))
    return {"ce_homophily": ce, "homophily_chance": chance, "adjusted_ce_homophily": adjusted}


def x_axis(rows):
    """Return (rows, x positions, numeric) for plotting against the variable.

    Numeric variables are placed at their actual values on the x axis, so the
    spacing is proportional. Lists and text are evenly spaced in config order.
    """
    numeric = all(isinstance(r["value"], (int, float)) and not isinstance(r["value"], bool)
                  for r in rows)
    if numeric:
        rows = sorted(rows, key=lambda r: r["value"])
        return rows, [r["value"] for r in rows], True
    return rows, list(range(len(rows))), False


def label_x_axis(ax, rows, positions, numeric, var):
    """Name the x axis; for non-numeric variables, label each position with its value."""
    if not numeric:
        ax.set_xticks(positions)
        ax.set_xticklabels([str(r["value"]) for r in rows],
                           rotation=30 if len(rows) > 6 else 0,
                           ha="right" if len(rows) > 6 else "center")
    ax.set_xlabel(var)


def plot_scores(ax, rows, positions, kind, splits, per_class):
    """Draw accuracy (`kind` "acc") or macro F1 (`kind` "f1") of each split in `splits`.

    With `per_class`, each class's accuracy (or F1) is added as a thinner line
    in its split's color, styled per class.
    """
    name = "macro F1" if kind == "f1" else "accuracy"
    for split, label, color in SPLITS:
        if split not in splits:
            continue
        ys = [r.get(f"{split}_{kind}", math.nan) for r in rows]
        ax.plot(positions, ys, marker="o", color=color, label=f"{label} {name}")
        if not per_class:
            continue
        for c, (linestyle, marker) in enumerate(CLASS_STYLES):
            ys = [r.get(f"{split}_{kind}_class{c}", math.nan) for r in rows]
            ax.plot(positions, ys, linestyle=linestyle, marker=marker, markersize=5,
                    linewidth=1.2, alpha=0.8, color=color,
                    label=f"{label} {'F1' if kind == 'f1' else 'accuracy'}, class {c} "
                          f"({gen.PLOT_COLOR_NAMES[c]})")
    ax.set_ylabel(name[0].upper() + name[1:])
    ax.set_ylim(0, 1)
    if per_class:  # too many entries to sit on top of the lines
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    else:
        ax.legend()


def plot_results(rows, var, model, path, accuracy=False, f1=False,
                 splits=("train", "val", "test"), per_class=False, node_violations=False):
    """Plot the violation ratio, and optionally accuracy and macro F1, against the variable.

    From top to bottom: accuracy if `accuracy`, macro F1 if `f1` (each for the
    splits in `splits`, with a line per class if `per_class`), and the
    violation ratio (violating edges / total edges), joined by violating nodes
    / total nodes if `node_violations`. The figure is saved to `path`.
    Skipped or failed values appear as gaps.
    """
    rows, positions, numeric = x_axis(rows)
    kinds = [kind for kind, wanted in (("acc", accuracy), ("f1", f1)) if wanted]
    fig, axes = plt.subplots(len(kinds) + 1, 1, sharex=True, squeeze=False,
                             figsize=(max(7, 0.9 * len(rows)), 4 * (len(kinds) + 1)))
    axes = axes[:, 0]
    for ax, kind in zip(axes, kinds):
        plot_scores(ax, rows, positions, kind, splits, per_class)
    ax_viol = axes[-1]
    ax_viol.plot(positions, [r.get("violation_ratio", math.nan) for r in rows],
                 marker="o", color="tab:red", label="Coloring violations / total edges")
    if node_violations:
        ax_viol.plot(positions, [r.get("node_violation_ratio", math.nan) for r in rows],
                     marker="o", color="tab:purple", label="Violating nodes / total nodes")
    label_x_axis(ax_viol, rows, positions, numeric, var)
    ax_viol.set_ylabel("Violation ratio")
    ax_viol.set_ylim(bottom=0)
    ax_viol.legend()
    shown = [name for name, wanted in (("accuracy", accuracy), ("macro F1", f1)) if wanted]
    shown.append("violation ratios" if node_violations else "violation ratio")
    title = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
    axes[0].set_title(f"{model}: {title} vs {var}")
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_homophily(rows, var, path):
    """Plot CE and adjusted CE homophily of the generated graphs against the variable.

    The y axis is fixed to [-1, 1] so plots from different batches compare
    directly; it only extends below -1 when adjusted CE does (a large majority
    class makes the chance level above 0.5). Skipped values appear as gaps.
    """
    rows, positions, numeric = x_axis(rows)
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(rows)), 4))
    for key, label in (("ce_homophily", "CE homophily"),
                       ("adjusted_ce_homophily", "Adjusted CE homophily")):
        ax.plot(positions, [r.get(key, math.nan) for r in rows], marker="o", label=label)
    ax.axhline(0, color="gray", linewidth=0.8)
    label_x_axis(ax, rows, positions, numeric, var)
    lowest = min((r["adjusted_ce_homophily"] for r in rows
                  if not math.isnan(r.get("adjusted_ce_homophily", math.nan))), default=-1)
    ax.set_ylim(min(-1, lowest - 0.1), 1)
    ax.set_ylabel("Homophily")
    ax.set_title(f"Graph homophily vs {var}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(x):
    return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a JSON config file")
    parser.add_argument("--homophily", action="store_true",
                        help="measure the CE and adjusted CE homophily of each generated "
                             "graph and plot it in homophily_vs_<variable>.png")
    parser.add_argument("--accuracy", action="store_true",
                        help="show accuracy in the table and plot it in a panel")
    parser.add_argument("--f1", action="store_true",
                        help="show macro F1 in the table and plot it in a panel")
    parser.add_argument("--node-violations", action="store_true",
                        help="show the share of nodes sharing a color with a neighbor in "
                             "the table and plot it in the violation panel")
    parser.add_argument("--degree-hist", action="store_true",
                        help="have the model save a histogram of test node degrees with the "
                             "misclassified test nodes on top, in each run folder")
    parser.add_argument("--per-class", action="store_true",
                        help="also plot each class's accuracy and F1 (with --accuracy / --f1)")
    parser.add_argument("--no-train", action="store_true",
                        help="leave the train split out of the accuracy and F1 panels")
    parser.add_argument("--no-val", action="store_true",
                        help="leave the validation split out of the accuracy and F1 panels")
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
    if args.degree_hist:
        if model not in DEGREE_HIST_MODELS:
            sys.exit(f"Error: --degree-hist is only supported for "
                     f"{', '.join(DEGREE_HIST_MODELS)}, not {model}")
        if "--degree-hist" not in model_args:
            model_args.append("--degree-hist")
    # Scores printed in the table and per run; the CSV always has every metric.
    shown = (METRICS if args.accuracy else ()) + (F1_METRICS if args.f1 else ())
    if (args.per_class or args.no_train or args.no_val) and not shown:
        print("Note: --per-class, --no-train and --no-val only affect the accuracy and F1 "
              "panels; add --accuracy or --f1 to plot them.")

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
        graph_path = os.path.join(run_dir, "data", "graph.pt")
        if error is None:
            try:
                components = generate(settings, graph_path)
                print(f"Generated graph ({components} connected component(s))")
            except ValueError as e:
                error = str(e)
        if error is not None:
            print(f"Skipped: {error}")
            rows.append({"value": value, "status": f"skipped: {error}"})
            continue
        homophily = measure_homophily(graph_path) if args.homophily else {}
        if homophily:
            print(f"Homophily: CE {homophily['ce_homophily']:.4f}, "
                  f"adjusted CE {homophily['adjusted_ce_homophily']:.4f}")
        print("Training ...", end=" ", flush=True)
        metrics, status = run_model(model, run_dir, model_args)
        print(status if metrics is None else
              "".join(f"{k}={fmt(metrics.get(k))}  " for k in shown)
              + f"violations={metrics.get('violations', '-')}"
              + f"  violation_ratio={fmt(metrics.get('violation_ratio'))}"
              + (f"  node_violation_ratio={fmt(metrics.get('node_violation_ratio'))}"
                 if args.node_violations else ""))
        rows.append({"value": value, "status": status, **homophily, **(metrics or {})})

    homophily_columns = (["ce_homophily", "homophily_chance", "adjusted_ce_homophily"]
                         if args.homophily else [])
    node_columns = ["violating_nodes", "node_violation_ratio"]
    csv_columns = ["value", *METRICS, *F1_METRICS, *PER_CLASS_METRICS, "violations",
                   "violation_rate", "violation_ratio", *node_columns, *homophily_columns,
                   "best_epoch", "status"]
    csv_path = os.path.join(batch_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[var if c == "value" else c for c in csv_columns])
        writer.writeheader()
        for r in rows:
            writer.writerow({(var if c == "value" else c): r.get(c, "") for c in csv_columns})

    columns = ["value", *shown, "violations", "violation_rate", "violation_ratio",
               *(node_columns if args.node_violations else []), *homophily_columns,
               "best_epoch", "status"]

    headers = [var if c == "value" else c for c in columns]
    # Keep the table narrow: full skip/failure reasons are printed above and in the CSV.
    table = [[fmt(r.get(c)) for c in columns[:-1]] + [r["status"].split(":")[0]] for r in rows]
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]
    print(f"\nModel: {model}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for t in table:
        print("  ".join(c.ljust(w) for c, w in zip(t, widths)))
    print(f"\nSaved results to {os.path.relpath(csv_path, REPO_ROOT)}")

    plot_path = os.path.join(batch_dir, f"results_vs_{var}.png")
    splits = [s for s, hidden in (("train", args.no_train), ("val", args.no_val),
                                  ("test", False)) if not hidden]
    plot_results(rows, var, model, plot_path, accuracy=args.accuracy, f1=args.f1,
                 splits=splits, per_class=args.per_class,
                 node_violations=args.node_violations)
    print(f"Saved results plot to {os.path.relpath(plot_path, REPO_ROOT)}")
    if args.homophily:
        homophily_path = os.path.join(batch_dir, f"homophily_vs_{var}.png")
        plot_homophily(rows, var, homophily_path)
        print(f"Saved homophily plot to {os.path.relpath(homophily_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
