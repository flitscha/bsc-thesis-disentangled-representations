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

