import numpy as np


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
