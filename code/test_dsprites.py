from data_generation.dsprites import load_dsprites
import matplotlib.pyplot as plt
import numpy as np

PATH = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"


def show_random_images(X, n=64):
    """
    Zeigt n zufällige Bilder aus einem (N,4096)-Array.
    """
    assert X.ndim == 2 and X.shape[1] == 64*64

    indices = np.random.choice(len(X), size=n, replace=True)
    imgs = X[indices].reshape(-1, 64, 64)

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


X = load_dsprites(
    PATH,
    shape=0,
    scale=None,
    posX=16,
    posY=16,
    orientation=None,
)

print(X.shape)

show_random_images(X, n=64)

