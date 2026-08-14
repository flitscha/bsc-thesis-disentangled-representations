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
    Generate a synthetic manifold dataset of n points.

    'data_type' is one of _GENERATORS. A positive 'embed_dim' embeds the points
    into that many ambient dimensions via a random orthonormal map plus isotropic
    noise of standard deviation 'noise'.

    Returns the point cloud and the embedding map W with X = data @ W.T (None if
    not embedded), which lets a model be visualized in intrinsic coordinates.
    """
    if data_type not in _GENERATORS:
        raise ValueError(
            f"unknown data_type '{data_type}', options: {sorted(_GENERATORS)}"
        )

    data = _GENERATORS[data_type](n=n, random_state=seed)

    if not embed_dim:
        return data, None

    return embed_data_to_dimension(data, embed_dim, noise=noise, random_state=seed)
