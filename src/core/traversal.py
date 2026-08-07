"""
Pipeline step §5.5: naive loop detection by greedy graph traversal (baseline).

Builds a k-NN graph from a precomputed distance matrix (§5.4) and greedily
traverses it into a single closed loop. Interpolation into a spline (§5.7) is a
separate step (see interpolation.interpolate_curves).
"""

import numpy as np

from core.graph import build_knn_graph


def detect_traversal(distance_matrix, k=2):
    """
    Naive baseline detection: k-NN graph -> greedy traversal -> single loop.

    Parameters
    ----------
    distance_matrix : (N, N) array
    k : int
        Number of neighbors per node in the k-NN graph.

    Returns
    -------
    result : dict with keys
        "curves" : [{"type": "loop", "component": 0, "order": [...]}],
        "graph"  : the k-NN graph dict (adjacency, edges, degrees).
    """
    graph = build_knn_graph(distance_matrix, k=k)
    order = traverse_graph(graph["adjacency"])
    curves = [{"type": "loop", "component": 0, "order": order}]
    return {"curves": curves, "graph": graph}


def traverse_graph(adjacency, start=0):
    """
    Greedy traversal:
    - always go to nearest unvisited neighbor

    Input:
    - adjacency : (N, N) weighted matrix (0 = no edge, >0 = weight)

    Output:
    - order : list of node indices
    """
    N = adjacency.shape[0]

    visited = set([start])
    order = [start]

    current = start

    for _ in range(N - 1):
        neighbors = np.where(adjacency[current] > 0)[0]

        candidates = [n for n in neighbors if n not in visited]

        if not candidates:
            break

        next_node = min(candidates, key=lambda j: adjacency[current, j])

        order.append(next_node)
        visited.add(next_node)

        current = next_node

    return order
