"""Measure the adjusted CE homophily of a generated graph.

CE homophily is the share of co-member pairs that share a label. In a graph
(not a hypergraph) the clique expansion is the graph itself, so the pairs are
just the edges:

    CE = (# edges whose endpoints have the same color) / (# edges)

Adjusted CE corrects this for chance, using the class proportions p_k
(the share of nodes with color k):

    adjusted CE = (CE - sum_k p_k^2) / (1 - sum_k p_k^2)

0 means no more same-color pairs than chance, 1 means every pair shares a
color, and negative values mean neighbors differ more often than chance.
For a valid 3-coloring CE is always 0, so adjusted CE = -sum p^2 / (1 - sum p^2),
which is -0.5 for three equal classes.

Usage:
    python measure_homophily.py                  # reads data/graph.pt
    python measure_homophily.py path/to/graph.pt
"""

import argparse
from collections import Counter

import torch


def load_graph(path):
    """Return (colors, edges) from a .pt file saved by generate_graph.py."""
    data = torch.load(path, weights_only=False)
    colors = data.y.tolist()
    # edge_index stores each undirected edge in both directions; keep one.
    edges = [(u, v) for u, v in data.edge_index.t().tolist() if u < v]
    return colors, edges


def adjusted_ce_homophily(colors, edges):
    """Return (CE homophily, chance level sum p^2, adjusted CE homophily)."""
    n = len(colors)
    chance = sum((count / n) ** 2 for count in Counter(colors).values())
    if not edges:
        return float("nan"), chance, float("nan")
    ce = sum(colors[u] == colors[v] for u, v in edges) / len(edges)
    adjusted = float("nan") if chance == 1 else (ce - chance) / (1 - chance)
    return ce, chance, adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="data/graph.pt",
                        help="graph file from generate_graph.py (default: data/graph.pt)")
    args = parser.parse_args()

    colors, edges = load_graph(args.path)
    ce, chance, adjusted = adjusted_ce_homophily(colors, edges)
    counts = Counter(colors)

    print(f"Graph: {args.path} ({len(colors)} nodes, {len(edges)} edges)")
    print(f"Class sizes: " + ", ".join(f"color {k}: {counts[k]}" for k in sorted(counts)))
    print(f"CE homophily:          {ce:.4f}")
    print(f"Chance level (sum p^2): {chance:.4f}")
    print(f"Adjusted CE homophily: {adjusted:.4f}")


if __name__ == "__main__":
    main()
