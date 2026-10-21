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

import argparse
import math
import os
import random
import sys

import torch
from torch_geometric.data import Data

COLORS = (0, 1, 2)
PAIRS = ((0, 1), (0, 2), (1, 2))
VISUALIZE_MAX_NODES = 20  # --visualize is ignored for bigger graphs
PLOT_COLORS = ("#E0524A", "#43A85B", "#3F7FD9")  # red, green, blue for classes 0, 1, 2
PLOT_COLOR_NAMES = ("Red", "Green", "Blue")
LAYOUT_TRIES = 50  # candidate layouts snapped to the grid for the drawing
LAYOUT_REFINE = 3  # how many of the best candidates are refined by local search

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


def _orient(a, b, c):
    """Sign of the turn a -> b -> c: 1 left, -1 right, 0 collinear."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (cross > 0) - (cross < 0)


def _on_segment(a, b, c):
    """For collinear a, b, c: is c within the segment a-b?"""
    return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _layout_problems(edges, pos):
    """Return (edges passing through another node, pairs of crossing edges)."""
    through = sum(
        1 for u, v in edges for w, p in pos.items()
        if w not in (u, v) and _orient(pos[u], pos[v], p) == 0 and _on_segment(pos[u], pos[v], p))
    crossings = 0
    for i, (a, b) in enumerate(edges):
        for c, d in edges[i + 1:]:
            pa, pb, pc, pd = pos[a], pos[b], pos[c], pos[d]
            if {a, b} & {c, d}:
                # Edges sharing an endpoint only clash if they overlap along one line.
                shared = ({a, b} & {c, d}).pop()
                x, y = (b if a == shared else a), (d if c == shared else c)
                s = pos[shared]
                if _orient(s, pos[x], pos[y]) == 0 and (
                        _on_segment(s, pos[x], pos[y]) or _on_segment(s, pos[y], pos[x])):
                    crossings += 1
                continue
            o1, o2 = _orient(pa, pb, pc), _orient(pa, pb, pd)
            o3, o4 = _orient(pc, pd, pa), _orient(pc, pd, pb)
            if (o1 != o2 and o3 != o4) or \
               (o1 == 0 and _on_segment(pa, pb, pc)) or (o2 == 0 and _on_segment(pa, pb, pd)) or \
               (o3 == 0 and _on_segment(pc, pd, pa)) or (o4 == 0 and _on_segment(pc, pd, pb)):
                crossings += 1
    return through, crossings


def grid_layout(num_nodes, edges, seed):
    """Place nodes on a square grid with as few edge crossings as possible.

    Like the 3Color puzzle game (archada.itch.io/3color), nodes sit on grid
    points and edges are straight lines. Many spring layouts (and a planar
    layout, when the graph is planar) are snapped to the grid. The best few are
    then improved by moving single nodes to free grid points or swapping two
    nodes, as long as that helps. A layout is better if fewer edges pass
    through a node, then if fewer edges cross, then if edges are shorter.
    Returns (positions, number of crossings).
    """
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    side = math.isqrt(num_nodes - 1) + 2 if num_nodes > 1 else 1
    points = [(x, y) for y in range(side) for x in range(side)]

    def score(pos):
        through, crossings = _layout_problems(edges, pos)
        return through, crossings, sum(math.dist(pos[u], pos[v]) for u, v in edges)

    def snap(raw):
        """Move each node to a grid point near its position in `raw`."""
        xs, ys = [raw[v][0] for v in g], [raw[v][1] for v in g]
        scale = lambda t, lo, hi: (t - lo) / (hi - lo) * (side - 1) if hi > lo else (side - 1) / 2
        target = {v: (scale(raw[v][0], min(xs), max(xs)), scale(raw[v][1], min(ys), max(ys)))
                  for v in g}
        # Greedy: closest (node, grid point) pairs first.
        pairs = sorted((_dist2(target[v], p), v, p) for v in g for p in points)
        pos, used = {}, set()
        for _, v, p in pairs:
            if v not in pos and p not in used:
                pos[v] = p
                used.add(p)
        return pos

    def improve(pos, current):
        """Hill-climb: move a node to another grid point (swapping if taken)."""
        at = {p: v for v, p in pos.items()}
        improved = True
        while improved and current[:2] != (0, 0):
            improved = False
            for v in list(pos):
                for p in points:
                    old, w = pos[v], at.get(p)
                    if p == old:
                        continue
                    pos[v] = p
                    if w is not None:
                        pos[w] = old
                    new = score(pos)
                    if new < current:
                        current, improved = new, True
                        at[p] = v
                        if w is not None:
                            at[old] = w
                        else:
                            del at[old]
                    else:
                        pos[v] = old
                        if w is not None:
                            pos[w] = p
        return pos, current

    raws = [nx.spring_layout(g, seed=seed + t) for t in range(LAYOUT_TRIES)]
    if nx.check_planarity(g)[0]:
        raws.append(nx.planar_layout(g))
    snapped = sorted(((score(pos), i, pos) for i, pos in enumerate(map(snap, raws))),
                     key=lambda c: c[:2])
    best_pos, best_score = None, None
    for start_score, _, pos in snapped[:LAYOUT_REFINE]:
        pos, s = improve(dict(pos), start_score)
        if best_score is None or s < best_score:
            best_pos, best_score = pos, s
    return best_pos, best_score[1]


def visualize(graph, path):
    """Draw the graph on a grid with each node in its color and save it to `path`."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    edges = [tuple(e) for e in graph["edges"]]
    pos, crossings = grid_layout(graph["num_nodes"], edges, graph["seed"])

    fig, ax = plt.subplots(figsize=(7, 7))
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#8C8C8C",
                linewidth=1.8, zorder=1)
    for v, (x, y) in pos.items():
        ax.scatter(x, y, s=900, color=PLOT_COLORS[graph["colors"][v]],
                   edgecolors="#2A2A2A", linewidths=1.2, zorder=2)
        ax.text(x, y, str(v), ha="center", va="center", color="#FFFFFF",
                fontweight="bold", fontsize=11, zorder=3)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=12,
                              markerfacecolor=PLOT_COLORS[c], markeredgecolor="#2A2A2A",
                              label=f"{PLOT_COLOR_NAMES[c]} (color {c})") for c in COLORS],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"3-colored graph: {graph['num_nodes']} nodes, {len(edges)} edges")
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {path}")
    if crossings:
        print(f"Note: the drawing has {crossings} edge crossing(s); "
              "this graph may not be drawable without crossings.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--visualize", action="store_true",
                        help="draw the graph with each node in its color "
                             f"(ignored above {VISUALIZE_MAX_NODES} nodes)")
    args = parser.parse_args()

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

    if args.visualize:
        if graph["num_nodes"] > VISUALIZE_MAX_NODES:
            print(f"Skipping --visualize: the graph has {graph['num_nodes']} nodes "
                  f"(limit is {VISUALIZE_MAX_NODES}).")
        else:
            visualize(graph, os.path.splitext(out_path)[0] + ".png")


if __name__ == "__main__":
    main()
