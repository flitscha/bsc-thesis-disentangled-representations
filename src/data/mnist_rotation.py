"""
Generates a dataset of MNIST digits rotated 0..360 degrees.

Noise is added in pixel space.
The returned X is then standardized: per-pixel mean removed and divided by a
single global scale.

Returns X, angles, pixel_mean and pixel_std.
mean and std are only needed to invert the standardization for display, i.e. to turn a
standardized vector back into a [0, 1] image:
    img = (v * pixel_std + pixel_mean).reshape(30, 30)
"""

import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def _load_mnist_images(
    digit: int = 3, n_images: int = 1, which: str = "train"
) -> list:
    """
    Load up to n_images examples of `digit` from MNIST.
    Returns list of (28, 28) float arrays in [0, 1].
    """
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T
        import tempfile
        import os
        cache = os.path.join(tempfile.gettempdir(), "mnist_cache")
        ds = dsets.MNIST(cache, train=(which == "train"), download=True, transform=T.ToTensor())
        imgs = []
        for img_tensor, label in ds:
            if label == digit:
                imgs.append(img_tensor.squeeze().numpy())
            if len(imgs) >= n_images:
                break
        if imgs:
            return imgs
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        src = x_train if which == "train" else x_test
        lbl = y_train if which == "train" else y_test
        idxs = np.where(lbl == digit)[0][:n_images]
        return [src[i].astype(float) / 255.0 for i in idxs]
    except Exception:
        pass

    print("WARNING: MNIST not found, using synthetic gaussian blob.")
    img = np.zeros((28, 28))
    cx, cy = 14, 10
    for x in range(28):
        for y in range(28):
            img[x, y] = np.exp(-((x - cx)**2 + (y - cy)**2) / 8.0)
    return [img] * n_images


def _rotate_frames(img: np.ndarray, angles: np.ndarray) -> list:
    """Rotate one (28, 28) image through `angles` (deg); return list of (900,) frames."""
    padded = np.pad(img, 3, mode="constant", constant_values=0.0)
    frames = []
    for angle in angles:
        rotated = scipy_rotate(padded, angle, reshape=False,
                               mode="constant", cval=0.0, order=1)
        rotated = rotated[2:-2, 2:-2]
        frames.append(rotated.flatten().astype(float))
    return frames


def make_rotation_dataset(
    digit: int = 3,
    n_angles: int = 360,
    n_images: int = 1,
    center: bool = True,
    add_noise: float = 0.0,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate MNIST digit images through 360 degrees.

    Returns
    -------
    X          : (n_images * n_angles, 900) standardized (+ noise) if center=True
    angles     : (n_images * n_angles,) in degrees
    pixel_mean : (900,) per-pixel mean of the clean images (0 if center=False)
    pixel_std  : (900,) global scale of the clean images, broadcast to every
                 pixel (all entries equal; 1 if center=False)
    """
    images = _load_mnist_images(digit, n_images)
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    rng = np.random.default_rng(random_state)
    frames = []
    angle_list = []

    for img in images:
        padded = np.pad(img, 3, mode="constant", constant_values=0.0)
        for angle in angles:
            rotated = scipy_rotate(padded, angle, reshape=False,
                                   mode="constant", cval=0.0, order=1)
            rotated = rotated[2:-2, 2:-2]
            frames.append(rotated.flatten().astype(float))
            angle_list.append(angle)

    X = np.stack(frames)
    angles_out = np.array(angle_list)

    # clean images in [0, 1] (removes rotation-interpolation over/undershoot)
    X = np.clip(X, 0.0, 1.0)

    # Standardization parameters from the clean images.
    pixel_mean = np.zeros(X.shape[1])
    pixel_std = np.ones(X.shape[1])
    if center:
        pixel_mean = X.mean(axis=0)
        scale = float((X - pixel_mean).std())
        scale = scale if scale > 1e-8 else 1.0
        pixel_std = np.full(X.shape[1], scale)

    # Add the noise before standardization, so it is spatially uniform on the actual image
    if add_noise > 0:
        X = X + rng.normal(0, add_noise, size=X.shape)

    # Standardize
    if center:
        X = (X - pixel_mean) / pixel_std

    return X, angles_out, pixel_mean, pixel_std


def make_multi_rotation_dataset(
    specs: list,
    samples_per: int = 120,
    center: bool = True,
    add_noise: float = 0.15,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray, np.ndarray]:
    """
    Several rotating digits stacked into one dataset, each with its own angle
    range. Different digits form separate connected components; a digit swept
    over the full circle (0..360) is a loop, a partial sweep is an open arc.

    Parameters
    ----------
    specs : list of dict
        One entry per component. Recognised keys:
            digit       : int, which MNIST digit
            start, end  : float, angle range in degrees (default 0, 360)
            samples     : int, frames for this component (default samples_per)
            image_index : int, which exemplar image of that digit (default 0)
        A spec is a loop iff start == 0 and end == 360.
    samples_per : int
        Default number of rotated frames per component (per-spec "samples"
        overrides it).
    center, add_noise, random_state : see make_rotation_dataset. Noise is added
        (in pixel space) before a single joint standardization over all frames.

    Returns
    -------
    X          : (n_specs * samples_per, 900) standardized (+ noise) frames
    angles     : (N,) per-frame angle in degrees (within its component's range)
    digit_id   : (N,) the digit value (0..9) each frame belongs to; the discrete
                 ground-truth factor / class label
    meta       : list of dict, one per spec, enriched with "is_loop", "digit",
                 "start", "end" and the sample index range "sl" (slice)
    pixel_mean : (900,) joint per-pixel mean (0 if center=False)
    pixel_std  : (900,) joint global scale, broadcast to every pixel
    """
    rng = np.random.default_rng(random_state)
    frames, angle_list, digit_list, meta = [], [], [], []

    for spec in specs:
        digit = int(spec["digit"])
        start = float(spec.get("start", 0.0))
        end = float(spec.get("end", 360.0))
        image_index = int(spec.get("image_index", 0))
        n = int(spec.get("samples", samples_per))
        is_loop = (start == 0.0 and end == 360.0)

        images = _load_mnist_images(digit, n_images=image_index + 1)
        img = images[min(image_index, len(images) - 1)]

        # a loop closes at 360==0, so drop the duplicate endpoint; an arc keeps it
        grid = np.linspace(start, end, n, endpoint=not is_loop)
        offset = len(frames)
        frames.extend(_rotate_frames(img, grid))
        angle_list.extend(grid.tolist())
        digit_list.extend([digit] * n)
        meta.append({
            "digit": digit, "start": start, "end": end, "is_loop": is_loop,
            "sl": slice(offset, offset + n),
        })

    X = np.stack(frames)
    angles_out = np.array(angle_list)
    digit_id = np.array(digit_list, dtype=int)

    # clean images in [0, 1] (removes rotation-interpolation over/undershoot)
    X = np.clip(X, 0.0, 1.0)

    # Joint standardization parameters over all components' clean frames.
    pixel_mean = np.zeros(X.shape[1])
    pixel_std = np.ones(X.shape[1])
    if center:
        pixel_mean = X.mean(axis=0)
        scale = float((X - pixel_mean).std())
        scale = scale if scale > 1e-8 else 1.0
        pixel_std = np.full(X.shape[1], scale)

    if add_noise > 0:
        X = X + rng.normal(0, add_noise, size=X.shape)

    if center:
        X = (X - pixel_mean) / pixel_std

    return X, angles_out, digit_id, meta, pixel_mean, pixel_std

