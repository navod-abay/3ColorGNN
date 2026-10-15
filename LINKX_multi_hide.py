"""LINKX for semi-supervised node classification with multiple random hides.

The known colors are given to LINKX as extra input channels: each node's
input is [x, one-hot color] for clamped (known) nodes and [x, 0, 0, 0] for
unknown nodes. Each epoch the train nodes are split at random: half are
clamped as known, the other half are hidden (their color channels are zero)
and the cross-entropy on the hidden nodes is the loss. Val/test nodes are never
clamped or used in the loss. At evaluation all train nodes are clamped and the
rest are predicted.
"""

import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import LINKX

from model_utils import (EarlyStopping, add_early_stopping_arg,
                         add_print_params_arg, print_params, report_results)

torch.manual_seed(0)

NUM_CLASSES = 3
EPOCHS = 200
HIDDEN_FRACTION = 0.5  # share of train nodes hidden (predicted) each epoch
HIDE_SEED = 0  # seed for the per-epoch random hides, separate from the model's

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_print_params_arg(parser)
add_early_stopping_arg(parser)
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


model = LINKX(
    num_nodes=data.num_nodes,
    in_channels=data.num_node_features + NUM_CLASSES,
    hidden_channels=12,
    num_layers=2,
    out_channels=NUM_CLASSES,
    num_node_layers=2,
    num_edge_layers=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
y_true_onehot = F.one_hot(data.y, NUM_CLASSES).float()
train_idx = data.train_mask.nonzero().view(-1)
hide_gen = torch.Generator().manual_seed(HIDE_SEED)


def forward(known_mask):
    """Run LINKX with the colors of `known_mask` nodes given as input."""
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
    return loss.item(), train_acc


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

    return val_acc, test_acc, val_loss, test_loss


losses = []
val_losses = []
train_accs = []
val_accs = []
test_losses = []
test_accs = []
stopper = EarlyStopping(args.patience)
for epoch in range(1, EPOCHS + 1):
    loss, train_acc = train()
    val_acc, test_acc, val_loss, test_loss = test()
    losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
    if stopper.step(val_loss, model, epoch):
        break

# Use the model from the epoch with the lowest validation loss.
stopper.restore(model)
best_epoch = stopper.best_epoch

val_acc, test_acc, val_loss, test_loss = test()
print(f"Final: Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
report_results(
    train_acc=train_accs[best_epoch - 1] if best_epoch is not None else float("nan"),
    val_acc=val_acc, test_acc=test_acc, best_epoch=best_epoch)

predictions = evaluate().argmax(dim=-1)
unlabeled_predictions = predictions[~data.train_mask]
print("Predictions for unlabeled nodes:", unlabeled_predictions)

if args.print_params:
    print_params(model)

epochs = range(1, len(losses) + 1)
fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
ax_loss.plot(epochs, losses, label="Training loss (hidden train nodes)")
ax_loss.plot(epochs, val_losses, label="Validation loss")
ax_loss.plot(epochs, test_losses, label="Test loss")
ax_acc.plot(epochs, train_accs, label="Training accuracy (hidden train nodes)")
ax_acc.plot(epochs, val_accs, label="Validation accuracy")
ax_acc.plot(epochs, test_accs, label="Test accuracy")
for ax in (ax_loss, ax_acc):
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
ax_loss.set_ylabel("Loss")
ax_loss.set_title("LINKX (multiple hides): loss and accuracy vs epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_acc.set_xlabel("Epoch")
plt.savefig("loss_LINKX_multi_hide.png", dpi=150, bbox_inches="tight")
print("Saved loss plot to loss_LINKX_multi_hide.png")
plt.show()
