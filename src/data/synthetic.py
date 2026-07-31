"""
Convenience loader for the synthetic toy manifolds used in the demos and
experiments. Turns a short name into a point cloud, optionally embedded into a
higher-dimensional ambient space with added noise.

Kept separate from the pipeline: data generation is a data source, not a
pipeline step. The pipeline (core/pipeline.py) only ever sees a numpy array X.
"""

from data.basic_manifolds import (
    line_in_2d,
    circle,
    swiss_roll,
    torus,
    curve_in_3d,
    embed_data_to_dimension,
)

_GENERATORS = {
    "line": line_in_2d,
    "circle": circle,
    "swiss_roll": swiss_roll,
    "torus": torus,
    "curve_in_3d": curve_in_3d,
}


def make_dataset(data_type, n, embed_dim=None, noise=0.005, seed=None):
    """
    Generate a synthetic manifold dataset.

    Parameters
    ----------
    data_type : str
        One of {"line", "circle", "swiss_roll", "torus", "curve_in_3d"}.
    n : int
        Number of points.
    embed_dim : int or None
        If given (and > 0), embed the data into this many ambient dimensions
        via a random orthonormal map plus isotropic noise.
    noise : float
        Standard deviation of the embedding noise (only used if embed_dim set).
    seed : int or None
        Seed for reproducibility.

    Returns
    -------
    X : (n, D) array
        The (possibly embedded) point cloud.
    projection_matrix : (embed_dim, D_intrinsic) array or None
        The embedding map W with X = data @ W.T, or None if not embedded.
        Useful for visualizing the model back in the intrinsic coordinates.
    """
    if data_type not in _GENERATORS:
        raise ValueError(
            f"unknown data_type '{data_type}', options: {sorted(_GENERATORS)}"
        )

    data = _GENERATORS[data_type](n=n, random_state=seed)

    if not embed_dim:
        return data, None

    return embed_data_to_dimension(data, embed_dim, noise=noise, random_state=seed)
