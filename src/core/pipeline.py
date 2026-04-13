from core.atlas import extract_tangent_frame
from core.graph import compute_score_matrix, build_knn_graph
from core.traversal import traverse_graph
from core.interpolation import build_closed_spline


def build_graph(model, n_tangents=1, k=2):
    means = model.means
    covs = model.covariances

    tangents, variances, _ = extract_tangent_frame(covs, n_tangents)

    score = compute_score_matrix(means, tangents, variances)
    graph = build_knn_graph(score, k=k)

    return graph


def build_spline_from_graph(model, graph):
    traversal_order = traverse_graph(graph["adjacency"])
    points = model.means[traversal_order]
    spline = build_closed_spline(points)
    return spline


def build_spline(model, n_tangents=1, k=2):
    graph = build_graph(model, n_tangents=n_tangents, k=k)
    return build_spline_from_graph(graph)


