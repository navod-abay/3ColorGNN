import argparse
import sys

import torch
from torch_geometric.nn import GCN

from model_utils import (EarlyStopping, add_early_stopping_arg,
                         add_print_params_arg, print_params, report_results)

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Train a GCN for node classification.")
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

if int(data.train_mask.sum()) == 0:
    sys.exit("Error: the graph has no labeled (train) nodes to train on.")

class EmbGCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, hidden, num_classes):
        super().__init__()
        self.emb = torch.nn.Embedding(num_nodes, emb_dim)   # row i = input vector of node i
        self.gnn = GCN(emb_dim, hidden, num_layers=2, out_channels=num_classes)

    def forward(self, edge_index):
        return self.gnn(self.emb.weight, edge_index)        # used in place of data.x

# model = EmbGCN(data.num_nodes, 16, 16, 3)


model = GCN(
    in_channels=data.num_node_features,
    hidden_channels=16,
    num_layers=2,
    out_channels=3,
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


stopper = EarlyStopping(args.patience)
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc, val_loss = test()
    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
    )
    if stopper.step(val_loss, model, epoch):
        break

# Use the model from the epoch with the lowest validation loss.
stopper.restore(model)
train_acc, val_acc, test_acc, val_loss = test()
print(f"Final: Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
report_results(train_acc=train_acc, val_acc=val_acc, test_acc=test_acc,
               best_epoch=stopper.best_epoch)

model.eval()
with torch.no_grad():
    out = model(data.x,  data.edge_index)
    predictions = out.argmax(dim=-1)

unlabeled_predictions = predictions[~data.train_mask]
print("Predictions for unlabeled nodes:", unlabeled_predictions)

if args.print_params:
    print_params(model)
