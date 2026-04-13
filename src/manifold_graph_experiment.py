import numpy as np
import matplotlib.pyplot as plt

from visualization.visualize import visualize_graph_on_mfa

from manifold_graph import (
    extract_tangent_directions,
    compute_score_matrix,
    build_knn_graph,
    compute_degrees,
    adjacency_to_edges,
)

from experiment import Experiment


def main():
    exp = Experiment(
        data_type='curve_in_3d',
        # data_type='circle',
        N=200,
        C=30,
        H=1,
        cov_type='mfa',
        shared=False,
        embed_dim=30,
        seed=2
    )
    exp.generate_data()
    exp.train()

    means = exp.model.means
    covariances = exp.model.covariances
    priors = exp.model.prior
    projection = exp.projection_matrix

    tangents = extract_tangent_directions(covariances)

    score_matrix = compute_score_matrix(means, tangents)

    adjacency = build_knn_graph(score_matrix, k=3)
    edges = adjacency_to_edges(adjacency)

    degrees = compute_degrees(adjacency)

    print("Degrees:", degrees)

    # fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    visualize_graph_on_mfa(
        ax,
        means @ projection,
        [projection.T @ cov @ projection for cov in covariances],
        priors,
        edges=edges
    )
    plt.show()


if __name__ == "__main__":
    main()
