#!/usr/bin/env python3
"""Interactively generate a random 3-colorable graph and save it as a .pt file.

The node set is split into three color classes (partitions). Edges are only
placed between different classes, so the class assignment is always a valid
3-coloring.

Inputs (prompted):
  1. Number of nodes N
  2. Number of edges E
  3. Node weights per color (w0, w1, w2): relative share of nodes in each class
  4. Edge weights per color pair (w01, w02, w12): relative share of edges
     between each pair of classes

With partition sizes l, m, n the maximum number of edges is
    l*m + l*n + m*n  ==  ((l+m+n)^2 - l^2 - m^2 - n^2) / 2
and each pair (i, j) can hold at most size_i * size_j edges.

Output is a torch_geometric.data.Data object saved with torch.save:
  edge_index      [2, 2E] long, each undirected edge in both directions
  x               [N, F] float node features, either
                    1. random: i.i.d. standard normal values, or
                    2. orthogonal: one random orthonormal vector per color plus
                       Gaussian noise per node (requires F >= 3)
  y               [N] long, color (0, 1, 2) of each node
  train_mask, val_mask, test_mask
                  [N] bool, disjoint random split for semi-supervised node
                  classification (labels of train nodes are known)
  num_nodes       N
  seed            random seed used
  feature_type    "random" or "orthogonal"; noise_std for "orthogonal"
  partition_sizes, node_weights, edge_weights, edges_per_pair
                  [1, 3] tensors (pairs ordered 0-1, 0-2, 1-2)
Load it with torch.load(path, weights_only=False).
"""

import os
import random
import sys

import torch
from torch_geometric.data import Data

COLORS = (0, 1, 2)
PAIRS = ((0, 1), (0, 2), (1, 2))

# Defaults used when the user presses Enter at a prompt.
DEFAULT_NUM_NODES = 1000
DEFAULT_NUM_EDGES = 2200
DEFAULT_NODE_WEIGHTS = "1 1 1"
DEFAULT_EDGE_WEIGHTS = "1 1 1"
DEFAULT_TRAIN_FRAC = 0.5
DEFAULT_VAL_FRAC = 0.25


def prompt_int(msg, minimum=0, default=None):
    while True:
        raw = input(msg).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if value < minimum:
            print(f"  Value must be >= {minimum}.")
            continue
        return value


def prompt_float(msg, default, maximum=1.0):
    while True:
        raw = input(msg).strip()
        if not raw:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("  Please enter a number.")
                continue
        if not 0 <= value <= maximum + 1e-9:  # tolerate float error in 1 - x
            print(f"  Value must be between 0 and {maximum:g}.")
            continue
        return value


def prompt_weights(msg, count, default=None):
    """Read `count` non-negative numbers (space/comma separated) and normalize them."""
    while True:
        raw = input(msg).strip() or (default or "")
        raw = raw.replace(",", " ").split()
        try:
            weights = [float(x) for x in raw]
        except ValueError:
            print("  Please enter numbers only.")
            continue
        if len(weights) != count:
            print(f"  Expected {count} values, got {len(weights)}.")
            continue
        if any(w < 0 for w in weights) or sum(weights) == 0:
            print("  Weights must be non-negative and not all zero.")
            continue
        total = sum(weights)
        return [w / total for w in weights]


def apportion(total, weights):
    """Split integer `total` proportionally to `weights` (largest-remainder method)."""
    raw = [total * w for w in weights]
    counts = [int(x) for x in raw]
    leftover = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:leftover]:
        counts[i] += 1
    return counts


def allocate_edges(num_edges, edge_weights, capacities):
    """Distribute edges over color pairs by weight, respecting each pair's capacity.

    Edges that do not fit in a pair are redistributed to the other pairs with a
    positive weight and spare room. Raises ValueError if that is impossible.
    """
    counts = [0] * len(PAIRS)
    remaining = num_edges
    active = [i for i, w in enumerate(edge_weights) if w > 0]
    while remaining > 0:
        open_pairs = [i for i in active if counts[i] < capacities[i]]
        if not open_pairs:
            raise ValueError(
                f"Cannot place {num_edges} edges: color pairs with non-zero edge "
                f"weight can hold at most {sum(capacities[i] for i in active)} edges."
            )
        total_w = sum(edge_weights[i] for i in open_pairs)
        share = apportion(remaining, [edge_weights[i] / total_w for i in open_pairs])
        for i, s in zip(open_pairs, share):
            add = min(s, capacities[i] - counts[i])
            counts[i] += add
            remaining -= add
    return counts


def build_graph(num_nodes, num_edges, node_weights, edge_weights, seed=None):
    rng = random.Random(seed)

    sizes = apportion(num_nodes, node_weights)
    capacities = [sizes[a] * sizes[b] for a, b in PAIRS]
    max_edges = sum(capacities)
    if num_edges > max_edges:
        raise ValueError(
            f"Too many edges: partition sizes {sizes} allow at most {max_edges} edges "
            f"(l*m + l*n + m*n), but {num_edges} were requested."
        )
    pair_counts = allocate_edges(num_edges, edge_weights, capacities)

    # Shuffle node ids so the color is not inferable from the id.
    ids = list(range(num_nodes))
    rng.shuffle(ids)
    members = []
    start = 0
    for size in sizes:
        members.append(ids[start:start + size])
        start += size

    colors = [0] * num_nodes
    for c in COLORS:
        for v in members[c]:
            colors[v] = c

    edges = []
    for (a, b), k in zip(PAIRS, pair_counts):
        A, B = members[a], members[b]
        # Sample k distinct (u, v) pairs from A x B without materializing it.
        for idx in rng.sample(range(len(A) * len(B)), k):
            u, v = A[idx // len(B)], B[idx % len(B)]
            edges.append(sorted((u, v)))
    rng.shuffle(edges)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "seed": seed,
        "node_weights": node_weights,
        "edge_weights": {f"{a}-{b}": w for (a, b), w in zip(PAIRS, edge_weights)},
        "partition_sizes": sizes,
        "edges_per_pair": {f"{a}-{b}": k for (a, b), k in zip(PAIRS, pair_counts)},
        "max_edges": max_edges,
        "colors": colors,
        "edges": edges,
        "num_classes": len(COLORS),
    }


def count_components(num_nodes, edges):
    """Number of connected components (isolated nodes count as one each)."""
    parent = list(range(num_nodes))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    components = num_nodes
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            components -= 1
    return components


def make_masks(num_nodes, train_frac, val_frac, seed):
    """Random disjoint train/val/test node masks; the test set gets the rest."""
    perm = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    num_train = round(train_frac * num_nodes)
    num_val = round(val_frac * num_nodes)
    masks = []
    for idx in (perm[:num_train], perm[num_train:num_train + num_val],
                perm[num_train + num_val:]):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        masks.append(mask)
    return masks


def make_features(colors, num_features, feature_type, noise_std, seed):
    """Node feature matrix [N, num_features] (see module docstring)."""
    gen = torch.Generator().manual_seed(seed + 1)
    num_nodes = len(colors)
    if feature_type == "random":
        return torch.randn(num_nodes, num_features, generator=gen)
    # Columns of Q are orthonormal: one random direction per color.
    q, _ = torch.linalg.qr(torch.randn(num_features, len(COLORS), generator=gen))
    base = q.t()[torch.tensor(colors)]
    return base + noise_std * torch.randn(num_nodes, num_features, generator=gen)


def to_data(graph, train_frac, val_frac, num_features, feature_type, noise_std):
    """Convert `graph` (from build_graph) to a PyG Data object with features and masks."""
    edges = torch.tensor(graph["edges"], dtype=torch.long).reshape(-1, 2).t()
    train_mask, val_mask, test_mask = make_masks(
        graph["num_nodes"], train_frac, val_frac, graph["seed"])
    data = Data(
        x=make_features(graph["colors"], num_features, feature_type, noise_std,
                        graph["seed"]),
        edge_index=torch.cat([edges, edges.flip(0)], dim=1),
        y=torch.tensor(graph["colors"], dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=graph["num_nodes"],
        seed=graph["seed"],
        partition_sizes=torch.tensor([graph["partition_sizes"]]),
        node_weights=torch.tensor([graph["node_weights"]]),
        edge_weights=torch.tensor([list(graph["edge_weights"].values())]),
        edges_per_pair=torch.tensor([list(graph["edges_per_pair"].values())]),
        feature_type=feature_type,
    )
    if feature_type == "orthogonal":
        data.noise_std = noise_std
    return data


def main():
    print("=== 3-colorable graph generator ===")
    num_nodes = prompt_int(
        f"Number of nodes [{DEFAULT_NUM_NODES}]: ", minimum=1, default=DEFAULT_NUM_NODES)
    num_edges = prompt_int(
        f"Number of edges [{DEFAULT_NUM_EDGES}]: ", minimum=0, default=DEFAULT_NUM_EDGES)
    node_weights = prompt_weights(
        f"Node weights for colors 0 1 2 [{DEFAULT_NODE_WEIGHTS}]: ", 3, DEFAULT_NODE_WEIGHTS)
    edge_weights = prompt_weights(
        f"Edge weights for color pairs 0-1 0-2 1-2 [{DEFAULT_EDGE_WEIGHTS}]: ", 3,
        DEFAULT_EDGE_WEIGHTS)
    train_frac = prompt_float(
        f"Fraction of labeled (train) nodes [{DEFAULT_TRAIN_FRAC}]: ", DEFAULT_TRAIN_FRAC)
    val_default = min(DEFAULT_VAL_FRAC, 1 - train_frac)
    val_frac = prompt_float(
        f"Fraction of validation nodes [{val_default:g}]: ", val_default, 1 - train_frac)
    num_features = prompt_int("Number of node features: ", minimum=1)
    print("Feature type:\n  1) Random values\n"
          "  2) Orthogonal vector per color + noise per node")
    while (choice := input("Choose 1 or 2: ").strip()) not in ("1", "2"):
        print("  Please enter 1 or 2.")
    feature_type = "random" if choice == "1" else "orthogonal"
    noise_std = 0.0
    if feature_type == "orthogonal":
        while num_features < len(COLORS):
            print(f"  Orthogonal features need at least {len(COLORS)} dimensions.")
            num_features = prompt_int("Number of node features: ", minimum=1)
        noise_std = prompt_float("Noise standard deviation [0.1]: ", 0.1, float("inf"))
    seed_raw = input("Random seed (blank for random): ").strip()
    seed = int(seed_raw) if seed_raw else random.randrange(2**32)
    out_path = input("Output file [data/graph.pt]: ").strip() or "data/graph.pt"

    try:
        graph = build_graph(num_nodes, num_edges, node_weights, edge_weights, seed)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = to_data(graph, train_frac, val_frac, num_features, feature_type, noise_std)
    torch.save(data, out_path)

    print(f"Partition sizes: {graph['partition_sizes']} (max edges {graph['max_edges']})")
    print(f"Edges per pair:  {graph['edges_per_pair']}")
    print(f"Split (train/val/test nodes): {int(data.train_mask.sum())}/"
          f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
    print(f"Saved graph to {out_path}")

    components = count_components(graph["num_nodes"], graph["edges"])
    if components > 1:
        print(f"Note: the generated graph is not connected ({components} components).")


if __name__ == "__main__":
    main()
