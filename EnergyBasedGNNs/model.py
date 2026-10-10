import torch
from torch_geometric.nn import GCNConv

class EnergyBasedGNN(torch.nn.Module):
    def __init__(self,num_node_features, hidden_channels, num_classes):
        super(EnergyBasedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.node_energy = torch.nn.Linear(hidden_channels + num_classes, 1)

        self.edge_energy = torch.nn.Sequential(
            torch.nn.Linear(2 *  num_classes, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, y_onehot):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)

        node_energy = self.node_energy(torch.cat([h, y_onehot], dim=-1)).sum()
        edge_energy = self.edge_energy(torch.cat([y_onehot[edge_index[0]], y_onehot[edge_index[1]]], dim=-1)).sum()

        total_energy = node_energy + edge_energy
        return total_energy


