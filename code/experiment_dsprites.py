# Trains an MFA model on (a subset of) dSprites, saves it to disk,
# and generates new images by sampling from the learned mixture.

import os
import numpy as np
import matplotlib.pyplot as plt
 
from data_generation.dsprites import load_dsprites
from vamm import Gaussian  # same import as in experiment.py

# paths
DATA_PATH  = "../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
MODEL_PATH = "../data/dsprites/mfa_model.npz"

# hyper parameters
C        = 1000
H        = 5
COV_TYPE = "mfa"
SHARED   = False
SEED     = 42
 
# dSprites subset – set values to None to include all
FILTER = dict(
    shape=2, # 0 = square, 1 = ellipse, 2 = heart
    scale=None,
    orientation=None,
    posX=None,
    posY=3,
)
 

def load_data():
    print("Loading dSprites ...")
    X = load_dsprites(DATA_PATH, flatten=True, as_float=True, **FILTER)
    print(f"  dataset shape: {X.shape}")
    return X
 
 
def train_model(X):
    print(f"Training MFA (C={C}, H={H}) ...")
    np.random.seed(SEED)
    _, D = X.shape
    model = Gaussian(C=C, D=D, covariance_type=COV_TYPE, shared=SHARED, H=H)
    obj, _ = model.fit(X, verbose=True)
    print(f"  final objective: {obj:.4f}")
    return model
 
 
def save_model(model):
    """
    Save learned MFA parameters to a .npz file.
    model._cpp holds the C++ object with:
      means    : (C, D)   - component means
      variance : (C, D)   - diagonal noise variances (Psi)
      A        : (C, D*H) - factor loading matrices (flattened)
      prior    : (C,)     - mixing weights
    """
    cpp = model._cpp
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        means    = cpp.means,
        variance = cpp.variance,
        A        = cpp.A,
        prior    = cpp.prior,
        C        = np.array(cpp.C),
        D        = np.array(cpp.D),
        H        = np.array(cpp.H),
    )
    print(f"Model saved -> {MODEL_PATH}")
 
 
def load_model():
    """
    Reconstruct a Gaussian model from saved parameters.
    Creates a fresh model, runs a dummy fit to initialise _cpp,
    then overwrites all parameters with the saved values.
    """
    print("Load model...")
    saved = np.load(MODEL_PATH)
    C_ = int(saved["C"])
    D_ = int(saved["D"])
    H_ = int(saved["H"])
 
    # Build fresh model and trigger C++ initialisation via a tiny dummy fit
    model = Gaussian(C=C_, D=D_, covariance_type=COV_TYPE, shared=SHARED, H=H_)
    dummy = np.random.randn(max(C_ * 10, 50), D_).astype(np.float32)
    model.fit(dummy, verbose=False)
 
    # Overwrite every parameter with saved values
    model._cpp.means[:]    = saved["means"]
    model._cpp.variance[:] = saved["variance"]
    model._cpp.A[:]        = saved["A"]
    model._cpp.prior[:]    = saved["prior"]
 
    print(f"Model loaded <- {MODEL_PATH}")
    return model
 
 
def sample_images(model, n=64):
    """
    Draw n samples from the MFA mixture.
 
    For each sample:
      1. pick component  k   ~ Categorical(prior)
      2. sample latent   z   ~ N(0, I_H)
      3. reconstruct     x   = mu_k + A_k @ z + eps
                         eps ~ N(0, diag(psi_k))
 
    Returns array of shape (n, 64, 64).
    """
    cpp = model._cpp
    C_  = cpp.C
    D_  = cpp.D
    H_  = cpp.H
 
    means    = cpp.means                        # (C, D)
    variance = cpp.variance                     # (C, D)  diagonal Psi
    A_flat   = cpp.A                            # (C, D*H)
    prior    = cpp.prior / cpp.prior.sum()      # (C,) normalise
 
    samples    = np.zeros((n, D_), dtype=np.float32)
    components = np.random.choice(C_, size=n, p=prior)
 
    for i, k in enumerate(components):
        mu_k  = means[k]                        # (D,)
        psi_k = variance[k]                     # (D,)
        A_k   = A_flat[k].reshape(D_, H_)       # (D, H)
        z     = np.random.randn(H_).astype(np.float32)
        eps   = (np.sqrt(psi_k) * np.random.randn(D_)).astype(np.float32)
        samples[i] = mu_k + A_k @ z + eps
 
    return samples.reshape(n, 64, 64)
 
 
def show_images(imgs, title="Generated dSprites samples"):
    n = len(imgs)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n:
            #ax.imshow(imgs[i], cmap="gray")
            ax.imshow(np.clip(imgs[i], 0, 1), cmap="gray")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
 
 
def main():
    # 1. load saved model, or train + save a new one
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        X = load_data()
        model = train_model(X)
        save_model(model)
 
    # 2. generate new images by sampling from the mixture
    print("Sampling from model ...")
    imgs = sample_images(model, n=64)
 
    # 3. display
    show_images(imgs, title=f"MFA samples  (C={C}, H={H})")
 
 
if __name__ == "__main__":
    main()



