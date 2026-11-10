import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import LINKX

from model_utils import (EarlyStopping, add_early_stopping_arg, coloring_violations,
                         split_metrics, report_results)

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Train LINKX for node classification.")
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

if int(data.train_mask.sum()) == 0:
    sys.exit("Error: the graph has no labeled (train) nodes to train on.")



model = LINKX(
    num_nodes=data.num_nodes,
    in_channels=data.num_node_features,
    hidden_channels=18,
    num_layers=1,
    out_channels=3,
    num_node_layers=1,
    num_edge_layers=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def accuracy(pred, mask):
    """Accuracy over the nodes in `mask`, or NaN if the mask is empty."""
    total = int(mask.sum())
    if total == 0:
        return float("nan")
    return int((pred[mask] == data.y[mask]).sum()) / total


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    train_acc = accuracy(pred, data.train_mask)
    val_acc = accuracy(pred, data.val_mask)
    test_acc = accuracy(pred, data.test_mask)

    if int(data.val_mask.sum()) == 0:
        val_loss = float("nan")
    else:
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()

    # Same coloring as the final report: train nodes clamped, the rest predicted.
    counts = coloring_violations(pred, data.y, data.edge_index, data.train_mask, verbose=False)

    return (train_acc, val_acc, test_acc, val_loss,
            counts["violation_ratio"], counts["node_violation_ratio"])


losses = []
val_losses = []
train_accs = []
val_accs = []
test_accs = []
violation_ratios = []
node_violation_ratios = []
stopper = EarlyStopping(args.patience)
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc, val_loss, violation_ratio, node_violation_ratio = test()
    losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)
    violation_ratios.append(violation_ratio)
    node_violation_ratios.append(node_violation_ratio)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Violation Ratio: {violation_ratio:.4f}, Node Violation Ratio: {node_violation_ratio:.4f}"
    )
    if stopper.step(val_loss, model, epoch):
        break

# Use the model from the epoch with the lowest validation loss.
stopper.restore(model)
train_acc, val_acc, test_acc, val_loss, _, _ = test()
print(f"Final: Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    predictions = out.argmax(dim=-1)

unlabeled_predictions = predictions[~data.train_mask]
print("Predictions for unlabeled nodes:", unlabeled_predictions)

violations = coloring_violations(predictions, data.y, data.edge_index, data.train_mask)
report_results(train_acc=train_acc, val_acc=val_acc, test_acc=test_acc,
               **split_metrics(predictions[data.train_mask], data.y[data.train_mask], "train"),
               **split_metrics(predictions[data.val_mask], data.y[data.val_mask], "val"),
               **split_metrics(predictions[data.test_mask], data.y[data.test_mask], "test"),
               best_epoch=stopper.best_epoch, **violations)

best_epoch = stopper.best_epoch
epochs = range(1, len(losses) + 1)
fig, (ax_loss, ax_acc, ax_viol) = plt.subplots(3, 1, sharex=True, figsize=(7, 11))
ax_loss.plot(epochs, losses, label="Training loss")
ax_loss.plot(epochs, val_losses, label="Validation loss")
for accs, label in ((train_accs, "Training accuracy"),
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
ax_loss.set_title("LINKX: loss, accuracy and violation ratio vs epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_ylim(0, 1)
ax_viol.set_ylabel("Violation ratio")
ax_viol.set_ylim(bottom=0)
ax_viol.set_xlabel("Epoch")
plt.savefig("loss_LINKX.png", dpi=150, bbox_inches="tight")
print("Saved loss/accuracy/violations plot to loss_LINKX.png")
plt.show()
