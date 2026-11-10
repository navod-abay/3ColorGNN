"""GCN for semi-supervised node classification with random label masking.

The known colors are given to the GCN as extra input channels: each node's
input is [x, one-hot color] for clamped (known) nodes and [x, 0, 0, 0] for
unknown nodes. Each epoch the train nodes are split at random: half are
clamped as known, the other half are hidden (their color channels are zero)
and the cross-entropy on the hidden nodes is the loss. Val/test nodes are never
clamped or used in the loss. At evaluation all train nodes are clamped and the
rest are predicted.

With --degree-hist, a histogram of the test nodes' degrees is also saved, with
the misclassified test nodes' degrees drawn on top of it in the same figure.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import torch.nn.functional as F
from torch_geometric.nn import GCN

from model_utils import (EarlyStopping, add_early_stopping_arg,
                         add_print_params_arg, coloring_violations, split_metrics, print_params,
                         report_results)

torch.manual_seed(0)

NUM_CLASSES = 3
EPOCHS = 200
HIDDEN_FRACTION = 0.5  # share of train nodes hidden (predicted) each epoch
HIDE_SEED = 0  # seed for the per-epoch random hides, separate from the model's

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_print_params_arg(parser)
add_early_stopping_arg(parser)
parser.add_argument("--degree-hist", action="store_true",
                    help="also save a histogram of test node degrees, with the "
                         "misclassified test nodes drawn on top")
args = parser.parse_args()

data = torch.load("data/graph.pt", weights_only=False)
print(data)

print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"num_node_features: {data.num_node_features}")
print(f"Y: {data.y}")
print(f"Train mask: {data.train_mask}")
print(f"Val mask: {data.val_mask}")
print(f"Test mask: {data.test_mask}")

if int(data.train_mask.sum()) < 2:
    sys.exit("Error: need at least 2 labeled (train) nodes to train on.")


model = GCN(
    in_channels=data.num_node_features + NUM_CLASSES,
    hidden_channels=16,
    num_layers=2,
    out_channels=NUM_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
y_true_onehot = F.one_hot(data.y, NUM_CLASSES).float()
train_idx = data.train_mask.nonzero().view(-1)
hide_gen = torch.Generator().manual_seed(HIDE_SEED)


def forward(known_mask):
    """Run the GCN with the colors of `known_mask` nodes given as input."""
    label_input = y_true_onehot * known_mask.unsqueeze(-1)
    return model(torch.cat([data.x, label_input], dim=-1), data.edge_index)


def train():
    model.train()
    optimizer.zero_grad()

    # Randomly hide part of the train nodes; the rest are clamped as known.
    perm = train_idx[torch.randperm(train_idx.numel(), generator=hide_gen).to(device)]
    num_hidden = max(1, min(train_idx.numel() - 1, round(HIDDEN_FRACTION * train_idx.numel())))
    hidden_idx = perm[:num_hidden]
    known_mask = data.train_mask.clone()
    known_mask[hidden_idx] = False

    logits = forward(known_mask)
    loss = F.cross_entropy(logits[hidden_idx], data.y[hidden_idx])
    loss.backward()
    optimizer.step()

    train_acc = (logits[hidden_idx].argmax(dim=-1) == data.y[hidden_idx]).float().mean().item()
    train_scores = split_metrics(logits[hidden_idx].argmax(dim=-1), data.y[hidden_idx], "train")
    return loss.item(), train_acc, train_scores


def accuracy(pred, mask):
    """Accuracy over the nodes in `mask`, or NaN if the mask is empty."""
    total = int(mask.sum())
    if total == 0:
        return float("nan")
    return int((pred[mask] == data.y[mask]).sum()) / total


def masked_loss(logits, mask):
    """Cross-entropy over the nodes in `mask`, or NaN if the mask is empty."""
    if int(mask.sum()) == 0:
        return float("nan")
    return F.cross_entropy(logits[mask], data.y[mask]).item()


@torch.no_grad()
def evaluate():
    """Clamp all train nodes, predict the rest; return logits for every node."""
    model.eval()
    return forward(data.train_mask)


def test():
    logits = evaluate()
    pred = logits.argmax(dim=-1)

    val_acc = accuracy(pred, data.val_mask)
    test_acc = accuracy(pred, data.test_mask)

    val_loss = masked_loss(logits, data.val_mask)
    test_loss = masked_loss(logits, data.test_mask)

    # Same coloring as the final report: train nodes clamped, the rest predicted.
    counts = coloring_violations(pred, data.y, data.edge_index, data.train_mask, verbose=False)

    return (val_acc, test_acc, val_loss, test_loss,
            counts["violation_ratio"], counts["node_violation_ratio"])


losses = []
val_losses = []
train_accs = []
train_scores_per_epoch = []
val_accs = []
test_losses = []
test_accs = []
violation_ratios = []
node_violation_ratios = []
stopper = EarlyStopping(args.patience)
for epoch in range(1, EPOCHS + 1):
    loss, train_acc, train_scores = train()
    val_acc, test_acc, val_loss, test_loss, violation_ratio, node_violation_ratio = test()
    losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    train_scores_per_epoch.append(train_scores)
    val_accs.append(val_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    violation_ratios.append(violation_ratio)
    node_violation_ratios.append(node_violation_ratio)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Violation Ratio: {violation_ratio:.4f}, Node Violation Ratio: {node_violation_ratio:.4f}"
    )
    if stopper.step(val_loss, model, epoch):
        break

# Use the model from the epoch with the lowest validation loss.
stopper.restore(model)
best_epoch = stopper.best_epoch

val_acc, test_acc, val_loss, test_loss, _, _ = test()
print(f"Final: Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

predictions = evaluate().argmax(dim=-1)
unlabeled_predictions = predictions[~data.train_mask]
print("Predictions for unlabeled nodes:", unlabeled_predictions)

violations = coloring_violations(predictions, data.y, data.edge_index, data.train_mask)
report_results(
    train_acc=train_accs[best_epoch - 1] if best_epoch is not None else float("nan"),
    val_acc=val_acc, test_acc=test_acc,
    **(train_scores_per_epoch[best_epoch - 1] if best_epoch is not None else {}),
    **split_metrics(predictions[data.val_mask], data.y[data.val_mask], "val"),
    **split_metrics(predictions[data.test_mask], data.y[data.test_mask], "test"),
    best_epoch=best_epoch, **violations)

if args.print_params:
    print_params(model)

epochs = range(1, len(losses) + 1)
fig, (ax_loss, ax_acc, ax_viol) = plt.subplots(3, 1, sharex=True, figsize=(7, 11))
ax_loss.plot(epochs, losses, label="Training loss")
ax_loss.plot(epochs, val_losses, label="Validation loss")
ax_loss.plot(epochs, test_losses, label="Test loss")
for accs, label in ((train_accs, "Training accuracy "),
                    (val_accs, "Validation accuracy"), (test_accs, "Test accuracy")):
    if best_epoch is not None:
        label += f": {accs[best_epoch - 1]:.3f} at best epoch"
    (line,) = ax_acc.plot(epochs, accs, label=label)
    if best_epoch is not None:  # mark the value the restored model reports
        ax_acc.plot(best_epoch, accs[best_epoch - 1], "o", color=line.get_color())
ax_viol.plot(epochs, violation_ratios, color="tab:red",
             label="Coloring violations / total edges")
ax_viol.plot(epochs, node_violation_ratios, color="tab:purple",
             label="Violating nodes / total nodes")
for ax in (ax_loss, ax_acc, ax_viol):
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
ax_loss.set_ylabel("Loss")
ax_loss.set_title("GCN (multiple hides): loss, accuracy and violation ratio vs epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_viol.set_ylabel("Violation ratio")
ax_viol.set_ylim(bottom=0)
ax_viol.set_xlabel("Epoch")
plt.savefig("loss_GCN_multi_hide.png", dpi=150, bbox_inches="tight")
print("Saved loss/accuracy/violations plot to loss_GCN_multi_hide.png")

if args.degree_hist and int(data.test_mask.sum()) == 0:
    print("Not saving the degree histogram: the graph has no test nodes.")
elif args.degree_hist:
    # edge_index lists each undirected edge in both directions, so this counts neighbors.
    degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes).cpu()
    test_mask = data.test_mask.cpu()
    misclassified = test_mask & (predictions.cpu() != data.y.cpu())
    bins = [d - 0.5 for d in range(int(degree[test_mask].max()) + 2)]  # one bar per degree
    fig_deg, ax_deg = plt.subplots(figsize=(7, 4.5))
    ax_deg.hist(degree[test_mask].tolist(), bins=bins, color="tab:blue", alpha=0.45,
                label=f"All test nodes ({int(test_mask.sum())})")
    # Drawn second, so the misclassified nodes sit on top of all test nodes.
    ax_deg.hist(degree[misclassified].tolist(), bins=bins, color="tab:red",
                label=f"Misclassified test nodes ({int(misclassified.sum())})")
    ax_deg.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_deg.set_xlabel("Degree")
    ax_deg.set_ylabel("Number of test nodes")
    ax_deg.set_title("GCN (multiple hides): degree distribution of test nodes")
    ax_deg.legend()
    ax_deg.grid(True, alpha=0.3)
    fig_deg.savefig("degree_hist_GCN_multi_hide.png", dpi=150, bbox_inches="tight")
    print("Saved test node degree histogram to degree_hist_GCN_multi_hide.png")
plt.show()
