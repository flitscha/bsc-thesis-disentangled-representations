import numpy as np
from pathlib import Path


# Internal loader
def _load_raw_dsprites(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True, encoding="latin1")

    imgs = data["imgs"]                     # (N, 64, 64)
    latents_classes = data["latents_classes"]  # (N, 6)
    metadata = data["metadata"][()]

    return imgs, latents_classes, metadata


# Public API
def load_dsprites(
    path,
    shape=None,
    scale=None,
    orientation=None,
    posX=None,
    posY=None,
    flatten=True,
    as_float=True,
):
    """
    Loads a subset of the dSprites dataset

    Parameter
    ----------
    shape : int or None
        0, 1, 2 (None = not fixed)

    scale : int or None
        0..5

    orientation : int or None
        0..39

    posX : int or None
        0..31

    posY : int or None
        0..31

    flatten : bool
        if true -> returns an array of shape (N, 4096)

    as_float : bool
        if true -> float32 in [0,1]
    """

    imgs, latents_classes, metadata = _load_raw_dsprites(path)

    # latents_classes columns:
    # 0 = color
    # 1 = shape
    # 2 = scale
    # 3 = orientation
    # 4 = posX
    # 5 = posY

    mask = np.ones(len(imgs), dtype=bool)

    if shape is not None:
        mask &= (latents_classes[:, 1] == shape)

    if scale is not None:
        mask &= (latents_classes[:, 2] == scale)

    if orientation is not None:
        mask &= (latents_classes[:, 3] == orientation)

    if posX is not None:
        mask &= (latents_classes[:, 4] == posX)

    if posY is not None:
        mask &= (latents_classes[:, 5] == posY)

    imgs_subset = imgs[mask]

    if as_float:
        imgs_subset = imgs_subset.astype(np.float32)

    if flatten:
        imgs_subset = imgs_subset.reshape(len(imgs_subset), -1)

    return imgs_subset

