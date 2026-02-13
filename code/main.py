import numpy as np
from itertools import product

from visualization.visualize import visualize_data, visualize_gmm_2d, visualize_gmm_3d
from data_generation.basic_manifolds import line_in_2d, circle, swiss_roll
from vamm import Gaussian


def gaussian_ellipse(mean, cov, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return width, height, angle


def run(cov_type, shared, data, num_components, manifold_dimension, seed):
    N, D = data.shape
    rng = np.random.default_rng(seed)

    model = Gaussian(
        C=num_components,
        D=D,
        H=manifold_dimension,  # H is only needed for "mfa"
        covariance_type=cov_type,
        shared=shared,
    )
    _shared = "with shared covariance matrices" if shared else ""
    print(f"Train {cov_type} {_shared} ...")
    obj, logs = model.fit(
        X=data,
        rng=rng,
    )
    print(f"Objective = {obj:<10.5f}\n")

    active = model.active
    means = model.means[:active]
    covs = model.covariances[:active]
    priors = model.prior[:active]

    # visualisation
    if D == 2:
        visualize_gmm_2d(data, means, covs, priors)
    if D == 3:
        visualize_gmm_3d(data, means, covs, priors)




def main():
    #cov_types = ["isotropic", "diagonal", "mfa", "full"]
    #shared_list = [True, False]
    cov_types = ["mfa", "full"]
    shared_list = [False]

    C = 15 # gaussian components
    H = 1 # internal manifold-dimension
    seed = 123
    rng = np.random.default_rng(seed)

    n = 100 # number of data points
    data = line_in_2d(n)

    print("Select dataset to visualize:")
    print("1: 1D curve in 2D")
    print("2: Circle")
    print("3: Swiss Roll")
    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        data = line_in_2d(n)
    elif choice == "2":
        data = circle(n)
    elif choice == "3":
        n = 1200
        C = 25
        H = 2 # this manifold is 2-dimensional. (TODO: why is the objective lower, when using H=1?)
        data = swiss_roll(n)
    else:
        print("Invalid choice")
    
    # train the gaussian model
    for cov_type, shared in product(cov_types, shared_list):
        obj = run(cov_type, shared, data, C, H, seed)

    #visualize_data(data)
    



if __name__ == "__main__":
    main()
