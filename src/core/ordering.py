"""
Pipeline step §5.5: order the MFA components along their intrinsic 1D structure
by graph traversal.

The core routine 'order_along_path' lays out a set of components along the
diameter of their minimum spanning tree.
The TDA detector (§5.6) reuses it to order each open component (components withoud loop).

'detect_traversal' orders all components into a single curve and decides
whether that curve closes into a loop.
It assumes one connected 1D structure and is the fast, simple counterpart to the
TDA detector, which also handles multiple components/loops.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra


# An ordered path is treated as closed when the gap between its two endpoints is
# no larger than this multiple of the typical consecutive step. 
_LOOP_CLOSURE_RATIO = 2.0


def _mst(distance_matrix: np.ndarray, node_indices: np.ndarray):
    """Symmetric minimum spanning tree over the given nodes."""
    sub_D = distance_matrix[np.ix_(node_indices, node_indices)]
    mst = minimum_spanning_tree(csr_matrix(sub_D))
    return mst.maximum(mst.T) # symmetrize (mst is upper-triangular)


def order_along_path(distance_matrix: np.ndarray, node_indices=None) -> list:
    """
    Order points along their 1D structure via the diameter of their minimum spanning tree.

    Parameters
    ----------
    distance_matrix : (N, N) geodesic distance matrix (§5.4).
    node_indices : indices into distance_matrix to order; None uses all nodes.

    Returns
    -------
    order : list of global vertex indices ordered along the path.
    """
    if node_indices is None:
        node_indices = np.arange(distance_matrix.shape[0])
    node_indices = np.asarray(node_indices)
    if node_indices.size < 2:
        return [int(i) for i in node_indices]

    mst = _mst(distance_matrix, node_indices)

    # start with an arbitrary node, and calculate the furthest node (far0).
    dist0, _ = dijkstra(mst, directed=False, indices=0, return_predecessors=True)
    far0 = int(np.argmax(dist0))

    # repeat this step. This time, we start with the node 'far0'.
    # The path from far0 to far1 is the longest possible path in the spanning tree.
    dist1, predecessors = dijkstra(mst, directed=False, indices=far0, return_predecessors=True)
    far1 = int(np.argmax(dist1))

    path_local = [far1]
    cur = far1
    while cur != far0:
        cur = predecessors[cur]
        path_local.append(cur)
    path_local.reverse()

    return [int(node_indices[i]) for i in path_local]


def mst_edges(distance_matrix: np.ndarray, node_indices=None) -> list:
    """Edges (i, j), i < j, of the minimum spanning tree, in global indices."""
    if node_indices is None:
        node_indices = np.arange(distance_matrix.shape[0])
    node_indices = np.asarray(node_indices)
    coo = _mst(distance_matrix, node_indices).tocoo()
    return [
        (int(node_indices[i]), int(node_indices[j]))
        for i, j in zip(coo.row, coo.col) if i < j
    ]


def closes_into_loop(
    distance_matrix: np.ndarray,
    order: list,
    ratio: float = _LOOP_CLOSURE_RATIO
) -> bool:
    """
    Decide whether an ordered open path actually closes into a loop.
    """
    if len(order) < 3:
        return False
    order = np.asarray(order)
    steps = distance_matrix[order[:-1], order[1:]]
    endpoint_gap = distance_matrix[order[0], order[-1]]
    return bool(endpoint_gap <= ratio * np.median(steps))


def detect_traversal(distance_matrix: np.ndarray, closed=None) -> dict:
    """
    Naive baseline detection: order all components into one curve along the MST
    diameter of the geodesic graph, then classify it as a loop or an open path.

    Parameters
    ----------
    distance_matrix : (N, N) geodesic distance matrix (§5.4).
    closed : bool or None
        Force the curve to be a loop (True) or an open path (False). None
        (default) decides automatically via closes_into_loop.

    Returns
    -------
    result : dict with keys
        "curves" : [{"type": "loop"|"path", "component": 0, "order": [...]}],
        "graph"  : {"edges": MST edges} (for visualization).
    """
    order = order_along_path(distance_matrix)
    is_loop = closes_into_loop(distance_matrix, order) if closed is None else bool(closed)
    curve_type = "loop" if is_loop else "path"

    curves = [{"type": curve_type, "component": 0, "order": order}]
    return {"curves": curves, "graph": {"edges": mst_edges(distance_matrix)}}

