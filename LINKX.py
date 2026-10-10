import sys

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import LINKX

torch.manual_seed(0)

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
    hidden_channels=12,
    num_layers=12,
    out_channels=3,
    num_node_layers=3,
    num_edge_layers=3,
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

    return train_acc, val_acc, test_acc, val_loss


losses = []
val_losses = []
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc, val_loss = test()
    losses.append(loss)
    val_losses.append(val_loss)
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    predictions = out.argmax(dim=-1)

unlabeled_predictions = predictions[~data.train_mask]
print("Predictions for unlabeled nodes:", unlabeled_predictions)

epochs = range(1, len(losses) + 1)
plt.plot(epochs, losses, label="Training loss")
plt.plot(epochs, val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs epoch")
plt.grid(True, alpha=0.3)
plt.savefig("loss.png", dpi=150, bbox_inches="tight")
print("Saved loss plot to loss.png")
plt.show()
