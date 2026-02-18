from data_generation.basic_manifolds import (
    line_in_2d,
    circle,
    swiss_roll,
    embed_data_to_dimension,
    torus,
    curve_in_3d
)
from vamm import Gaussian


class Experiment:
    def __init__(
        self,
        data_type,
        N,
        C,
        H,
        cov_type,
        shared=False,
        embed_dim=None,
        noise=0.005,
        seed=123,
    ):
        # type of toy-data ("line", "circle", "swiss_roll")
        self.data_type = data_type
        self.N = N  # number of data points
        self.C = C  # number of gaussian components
        self.H = H  # assumption that the data lives on H-dimensional manifold (needed for mfa)
        self.cov_type = cov_type  # ("isotropic", "diagonal", "mfa", "full")
        self.shared = shared  # use shared covariances?
        self.embed_dim = embed_dim
        if self.embed_dim == 0:
            self.embed_dim = None
        self.noise = 0.005
        self.seed = seed

        self.data = None
        self.projection_matrix = None
        self.model = None  # model for training
        self.obj = None  # objective (how good is the training-result?)


    def generate_data(self):
        data = None
        if self.data_type == "line":
            data = line_in_2d(n=self.N)
        elif self.data_type == "circle":
            data = circle(n=self.N)
        elif self.data_type == "swiss_roll":
            data = swiss_roll(n=self.N)
        elif self.data_type == "torus":
            data = torus(n=self.N)
        elif self.data_type == "curve_in_3d":
            data = curve_in_3d(n=self.N)
        else:
            print('warning: data type "' + self.data_type + '" does not exist')

        if self.embed_dim is None:
            self.data = data
        else:
            self.data, self.projection_matrix = embed_data_to_dimension(
                data, self.embed_dim, noise=self.noise
            )


    def train(self):
        _, D = self.data.shape

        self.model = Gaussian(C=self.C, D=D, covariance_type=self.cov_type, shared=self.shared, H=self.H)

        obj, _ = self.model.fit(self.data, verbose=False)
        self.obj = obj
