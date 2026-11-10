"""Shared helpers for training and inspecting models.

Usage in a training script:

    import argparse
    from model_utils import (EarlyStopping, add_early_stopping_arg,
                             add_print_params_arg, print_params)

    parser = argparse.ArgumentParser()
    add_print_params_arg(parser)
    add_early_stopping_arg(parser)
    args = parser.parse_args()

    stopper = EarlyStopping(args.patience)
    for epoch in ...:
        ...  # train, compute val_loss
        if stopper.step(val_loss, model, epoch):
            break
    stopper.restore(model)
    if args.print_params:
        print_params(model)

Scripts in a subfolder (e.g. EnergyBasedGNNs/) need the repo root on the
import path, e.g. run them with PYTHONPATH=. from the repo root.
"""

import copy
import json
import math

import torch

PARAM_CONFIRM_LIMIT = 800  # above this many parameters, ask before printing them
PARAM_REFUSE_LIMIT = 1200  # above this many parameters, refuse to print them
GRID_COLUMNS = 8  # values per row block when printing a tensor
DEFAULT_PATIENCE = 50  # epochs without validation improvement before stopping


RESULTS_PREFIX = "RESULTS "


def report_results(**metrics):
    """Print final metrics as one JSON line that run_batch.py can parse."""
    print(RESULTS_PREFIX + json.dumps(metrics))


def split_metrics(pred, y, split, num_classes=3):
    """Macro F1 plus per-class F1 and accuracy of `pred` against `y`, for report_results.

    Keys are prefixed with `split` (e.g. "val"): `val_f1` is the macro F1,
    `val_f1_class<c>` the F1 of class c and `val_acc_class<c>` the share of
    class-c nodes predicted as c. Each class's F1 is 2*TP / (2*TP + FP + FN);
    the macro F1 is their unweighted mean, so a minority class counts as much
    as the majority class. A class absent from both `pred` and `y` gets NaN
    F1 and is left out of the mean; a class absent from `y` gets NaN accuracy.
    """
    nan = float("nan")
    metrics = {}
    f1s = []
    for c in range(num_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan
        metrics[f"{split}_f1_class{c}"] = f1
        metrics[f"{split}_acc_class{c}"] = tp / (tp + fn) if tp + fn else nan
        if not math.isnan(f1):
            f1s.append(f1)
    metrics[f"{split}_f1"] = sum(f1s) / len(f1s) if f1s else nan
    return metrics


def coloring_violations(pred, y, edge_index, known_mask, verbose=True):
    """Count edges whose two endpoints get the same color in the final coloring.

    The final coloring gives known nodes (`known_mask`, the train nodes) their
    true color `y` and every other node its predicted color `pred`. Edges
    between two known nodes can never violate, so the rate is taken over the
    edges that touch at least one predicted node; the ratio is taken over all
    edges. `edge_index` may list each undirected edge in both directions; each
    edge is counted once. A node violates the coloring when it shares its color
    with at least one neighbor (it is an endpoint of a violating edge); the node
    violation ratio is taken over all nodes.
    Prints a summary (unless `verbose` is False, e.g. when called every epoch)
    and returns the counts as a dict for report_results.
    """
    colors = torch.where(known_mask, y, pred)
    src, dst = edge_index
    once = src < dst
    src, dst = src[once], dst[once]
    violating = colors[src] == colors[dst]
    violations = int(violating.sum())
    checked = int((~(known_mask[src] & known_mask[dst])).sum())
    total = int(once.sum())
    rate = violations / checked if checked else float("nan")
    ratio = violations / total if total else float("nan")
    in_conflict = torch.zeros_like(known_mask)
    in_conflict[src[violating]] = True
    in_conflict[dst[violating]] = True
    violating_nodes = int(in_conflict.sum())
    num_nodes = colors.numel()
    node_ratio = violating_nodes / num_nodes if num_nodes else float("nan")
    if verbose:
        print(f"Coloring violations: {violations} of {checked} edges touching a predicted node "
              f"({rate:.2%}); {violations} of {total} edges in total ({ratio:.2%}); "
              f"{violating_nodes} of {num_nodes} nodes share a color with a neighbor "
              f"({node_ratio:.2%})")
    return {"violations": violations, "checked_edges": checked, "violation_rate": rate,
            "total_edges": total, "violation_ratio": ratio,
            "violating_nodes": violating_nodes, "node_violation_ratio": node_ratio}


def add_early_stopping_arg(parser, default=DEFAULT_PATIENCE):
    """Add the --patience flag (0 disables early stopping) to an argparse parser."""
    parser.add_argument("--patience", type=int, default=default,
                        help="stop after this many epochs without a lower validation "
                             f"loss; 0 trains all epochs (default: {default})")


class EarlyStopping:
    """Track the best validation loss, keep that model's weights, and say when to stop.

    `step` returns True once `patience` epochs in a row have not lowered the
    validation loss by more than `min_delta`. With patience 0 it never stops,
    but still keeps the best weights. A NaN validation loss (e.g. an empty
    validation set) never counts toward stopping.
    """

    def __init__(self, patience=DEFAULT_PATIENCE, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = None
        self.best_state = None
        self.epochs_without_improvement = 0
        self.stopped_epoch = None

    def step(self, val_loss, model, epoch):
        if math.isnan(val_loss):
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch = val_loss, epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        if self.patience > 0 and self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def restore(self, model):
        """Load the best weights into `model`, if any were recorded."""
        if self.stopped_epoch is not None:
            print(f"Early stopping at epoch {self.stopped_epoch}: no lower validation "
                  f"loss for {self.patience} epochs")
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"Restored model from epoch {self.best_epoch} "
                  f"(val loss {self.best_loss:.4f})")


def add_print_params_arg(parser):
    """Add the --print-params flag to an argparse parser."""
    parser.add_argument("--print-params", action="store_true",
                        help="print all model parameters after training")


def count_params(model):
    """Total number of individual parameter values in `model`."""
    return sum(p.numel() for p in model.parameters())


def _confirm(prompt):
    try:
        answer = input(prompt)
    except EOFError:  # no interactive input available
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def _print_table(rows, headers, align):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]

    def fmt(row):
        return "  ".join(str(c).rjust(w) if a == ">" else str(c).ljust(w)
                         for c, w, a in zip(row, widths, align))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def _print_grid(values, decimals):
    """Print a 1-D or 2-D tensor as a grid with row/column indices.

    Wide tensors are split into blocks of GRID_COLUMNS columns. Tensors with
    more than 2 dimensions are flattened to [dim0, rest].
    """
    if values.dim() == 0:
        print(f"  {values.item():+.{decimals}f}")
        return
    if values.dim() == 1:
        values = values.unsqueeze(0)
        row_labels = [""]
    else:
        values = values.reshape(values.size(0), -1)
        row_labels = [f"[{i}]" for i in range(values.size(0))]

    cell = decimals + 4  # sign, leading digit, point, padding
    label_w = max(len(label) for label in row_labels)
    for start in range(0, values.size(1), GRID_COLUMNS):
        cols = range(start, min(start + GRID_COLUMNS, values.size(1)))
        print("  " + " " * label_w + "".join(f"[{j}]".rjust(cell + 1) for j in cols))
        for label, row in zip(row_labels, values.tolist()):
            print("  " + label.rjust(label_w)
                  + "".join(f"{row[j]:+.{decimals}f}".rjust(cell + 1) for j in cols))


def print_params(model, decimals=3,
                 confirm_limit=PARAM_CONFIRM_LIMIT, refuse_limit=PARAM_REFUSE_LIMIT):
    """Print a summary table and the values of every parameter in `model`.

    Asks for confirmation when the model has more than `confirm_limit`
    parameters and refuses when it has more than `refuse_limit`.
    For weight matrices, rows are output units and columns are input units.
    """
    num_params = count_params(model)
    if num_params > refuse_limit:
        print(f"Not printing parameters: the model has {num_params} parameters "
              f"(limit is {refuse_limit}).")
        return
    if num_params > confirm_limit and not _confirm(
            f"The model has {num_params} parameters. Print them all? [y/N]: "):
        print("Not printing parameters.")
        return

    named = [(name, p.detach().cpu().float()) for name, p in model.named_parameters()]

    print(f"\nModel parameters: {num_params} total in {len(named)} tensors\n")
    rows = [
        (name, "x".join(map(str, p.shape)) or "scalar", p.numel(),
         f"{p.mean():+.{decimals}f}",
         f"{p.std():.{decimals}f}" if p.numel() > 1 else "-",
         f"{p.min():+.{decimals}f}", f"{p.max():+.{decimals}f}")
        for name, p in named
    ]
    _print_table(rows, ("Name", "Shape", "Count", "Mean", "Std", "Min", "Max"),
                 ("<", "<", ">", ">", ">", ">", ">"))

    for name, p in named:
        print(f"\n{name}  ({'x'.join(map(str, p.shape)) or 'scalar'})")
        _print_grid(p, decimals)
    print()
