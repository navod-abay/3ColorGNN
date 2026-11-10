"""Train EnergyBasedGNN for semi-supervised node classification.

The model scores a full (soft) coloring y with a scalar energy E(x, y). Colors
are predicted by minimizing E over the unknown nodes while the known nodes are
clamped to their true one-hot labels:

    z_0 = 0,   y_t = softmax(z_t) on unknown nodes, one-hot on known nodes
    z_{t+1} = z_t - step_size * dE/dz_t

Training unrolls this inference and backpropagates through it (end-to-end).
Each epoch the train nodes are split at random: half are clamped as known, the
other half are hidden and the cross-entropy of the inferred colors against
their true labels is the loss. Val/test nodes are never clamped or used in the
loss. At evaluation all train nodes are clamped and the rest are inferred.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from nodeOnlyModel import EnergyBasedGNN

# The shared helpers live in the repo root, one folder up.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_utils import (EarlyStopping, add_early_stopping_arg,  # noqa: E402
                         coloring_violations, split_metrics, report_results)

torch.manual_seed(0)

NUM_CLASSES = 3
EPOCHS = 200
INFERENCE_STEPS = 20
STEP_SIZE = 1.0
HIDDEN_FRACTION = 0.5  # share of train nodes hidden (predicted) each epoch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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


model = EnergyBasedGNN(
    num_node_features=data.num_node_features,
    hidden_channels=12,
    num_classes=NUM_CLASSES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
y_true_onehot = F.one_hot(data.y, NUM_CLASSES).float()
train_idx = data.train_mask.nonzero().view(-1)


def infer(known_mask, create_graph):
    """Minimize the energy over the unknown nodes; return their logits [N, C].

    With create_graph=True the steps stay differentiable w.r.t. the model
    parameters (for training); otherwise each step is detached (for evaluation).
    """
    z = torch.zeros(data.num_nodes, NUM_CLASSES, device=device, requires_grad=True)
    for _ in range(INFERENCE_STEPS):
        y = torch.where(known_mask.unsqueeze(-1), y_true_onehot, z.softmax(dim=-1))
        energy = model(data.x, data.edge_index, y)
        (grad,) = torch.autograd.grad(energy, z, create_graph=create_graph)
        z = z - STEP_SIZE * grad
        if not create_graph:
            z = z.detach().requires_grad_()
    return z


def train():
    model.train()
    optimizer.zero_grad()

    # Randomly hide part of the train nodes; the rest are clamped as known.
    perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
    num_hidden = max(1, min(train_idx.numel() - 1, round(HIDDEN_FRACTION * train_idx.numel())))
    hidden_idx = perm[:num_hidden]
    known_mask = data.train_mask.clone()
    known_mask[hidden_idx] = False

    logits = infer(known_mask, create_graph=True)
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


def evaluate():
    """Clamp all train nodes, infer the rest; return logits for every node."""
    model.eval()
    with torch.enable_grad():  # inference needs dE/dy even at evaluation time
        return infer(data.train_mask, create_graph=False).detach()


def test():
    logits = evaluate()
    pred = logits.argmax(dim=-1)

    val_acc = accuracy(pred, data.val_mask)
    test_acc = accuracy(pred, data.test_mask)

    val_loss = masked_loss(logits, data.val_mask)
    test_loss = masked_loss(logits, data.test_mask)

    # Same coloring as the final report: train nodes clamped, the rest predicted.
    violation_ratio = coloring_violations(pred, data.y, data.edge_index, data.train_mask,
                                          verbose=False)["violation_ratio"]

    return val_acc, test_acc, val_loss, test_loss, violation_ratio


losses = []
val_losses = []
train_accs = []
train_scores_per_epoch = []
val_accs = []
test_losses = []
test_accs = []
violation_ratios = []
stopper = EarlyStopping(args.patience)
for epoch in range(1, EPOCHS + 1):
    loss, train_acc, train_scores = train()
    val_acc, test_acc, val_loss, test_loss, violation_ratio = test()
    losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    train_scores_per_epoch.append(train_scores)
    val_accs.append(val_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    violation_ratios.append(violation_ratio)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Violation Ratio: {violation_ratio:.4f}"
    )
    if stopper.step(val_loss, model, epoch):
        break

# Use the model from the epoch with the lowest validation loss.
stopper.restore(model)
best_epoch = stopper.best_epoch

val_acc, test_acc, val_loss, test_loss, _ = test()
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

epochs = range(1, len(losses) + 1)
fig, (ax_loss, ax_acc, ax_viol) = plt.subplots(3, 1, sharex=True, figsize=(7, 11))
ax_loss.plot(epochs, losses, label="Training loss (hidden train nodes)")
ax_loss.plot(epochs, val_losses, label="Validation loss")
ax_loss.plot(epochs, test_losses, label="Test loss")
for accs, label in ((train_accs, "Training accuracy (hidden train nodes)"),
                    (val_accs, "Validation accuracy"), (test_accs, "Test accuracy")):
    if best_epoch is not None:
        label += f": {accs[best_epoch - 1]:.3f} at best epoch"
    (line,) = ax_acc.plot(epochs, accs, label=label)
    if best_epoch is not None:  # mark the value the restored model reports
        ax_acc.plot(best_epoch, accs[best_epoch - 1], "o", color=line.get_color())
ax_viol.plot(epochs, violation_ratios, color="tab:red",
             label="Coloring violations / total edges")
for ax in (ax_loss, ax_acc, ax_viol):
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax.legend()
    ax.grid(True, alpha=0.3)
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Loss, accuracy and violation ratio vs epoch (multiple hides)")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_viol.set_ylabel("Violation ratio")
ax_viol.set_ylim(bottom=0)
ax_viol.set_xlabel("Epoch")
plt.savefig("loss_ebm_multi_hide.png", dpi=150, bbox_inches="tight")
print("Saved loss/accuracy/violations plot to loss_ebm_multi_hide.png")
plt.show()
