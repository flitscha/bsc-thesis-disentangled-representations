"""
Renders grayscale face images from the ICT-FaceKit morphable face model.

The model is linear in its shape modes,

    vertices = neutral + sum_i  w_i * (identity_i  - neutral)
                       + sum_j  e_j * (expression_j - neutral),

so a face is described completely by identity weights w (who it is), expression weights e
(smiling, jaw open) and a rigid head rotation. Each of those is a ground-truth generative factor
in physical units: degrees for the rotation, blendshape units in [0, 1] for the expressions.

The meshes live in 'data/ictfacekit/' and are not in the repository. Run 'python -m data.faces'
from src/ once to download them. Parsed .obj files are cached in a single .npz.

Images come from a small rasterizer written here: the quad mesh is sampled into a dense point
cloud, shaded with a Lambertian term and drawn back to front, so nearer points overwrite farther
ones.

'make_multi_face_dataset' returns X standardized. Undo that for display via
img = (v * pixel_std + pixel_mean).reshape(image_size, image_size).
"""

import os

import numpy as np

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "ictfacekit",
)

# The meshes are not kept in the repository, about 2.5 MB each. `download_model` fetches the ones
# this module needs from the ICT-FaceKit repository (MIT licence).
MODEL_URL = "https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/FaceXModel"

# Named factors mapped onto the blendshapes of the model. Left and right pairs are always driven
# together, so that a factor stays a symmetric, one-dimensional movement.
EXPRESSION_FACTORS = {
    "smile": ("mouthSmile_L", "mouthSmile_R"),
    "jaw_open": ("jawOpen",),
    "brow_raise": ("browInnerUp_L", "browInnerUp_R"),
    "blink": ("eyeBlink_L", "eyeBlink_R"),
    "pucker": ("mouthPucker",),
    "frown": ("mouthFrown_L", "mouthFrown_R"),
    "cheek_puff": ("cheekPuff_L", "cheekPuff_R"),
}

# rigid head rotations, in degrees
POSE_FACTORS = ("yaw", "pitch")

FACTOR_NAMES = tuple(EXPRESSION_FACTORS) + POSE_FACTORS

FACTOR_UNITS = {name: "deg" if name in POSE_FACTORS else "units"
                for name in FACTOR_NAMES}

# The full range of every factor, since a sweep always covers all of it. Yaw runs from profile to
# profile through the frontal view, pitch stops before the head leaves the frame, and a blendshape
# is defined on [0, 1].
FACTOR_RANGES = dict({name: (0.0, 1.0) for name in EXPRESSION_FACTORS},
                     yaw=(-90.0, 90.0), pitch=(-30.0, 30.0))

# The faces used in the experiment. The identity shape modes are principal modes, not people: at
# the natural weight 1.0 the higher modes barely differ from the mean face, with a pixel distance
# of 3.8 to 9.6 between them against 12.4 from mode 0 to the mean face. Deflecting a mode by 2.5
# gives faces that read as different persons and still look plausible. The six below were picked
# from modes 0 to 11 with both signs, by greedily maximising the smallest pairwise pixel distance.
# That leaves them 13.1 apart, and a seventh face would drop it to 12.2.
FACES = (
    {"name": "A", "mode": 0, "weight": 2.5},
    {"name": "B", "mode": 0, "weight": -2.5},
    {"name": "C", "mode": 3, "weight": 2.5},
    {"name": "D", "mode": 3, "weight": -2.5},
    {"name": "E", "mode": 2, "weight": -2.5},
    {"name": "F", "mode": 5, "weight": -2.5},
)

FACE_MODES = tuple(sorted({f["mode"] for f in FACES}))


def download_model(model_dir: str = None, identities=FACE_MODES, expressions=None):
    """
    Download the meshes this module needs into `model_dir`, skipping what is already there.

    That is the neutral mesh, the landmark indices, the requested identity shape modes and the
    blendshapes behind EXPRESSION_FACTORS.
    """
    import urllib.request

    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    if expressions is None:
        expressions = sorted({name for names in EXPRESSION_FACTORS.values()
                              for name in names})
    names = (["generic_neutral_mesh.obj", "vertex_indices.json"] +
             [f"identity{int(i):03d}.obj" for i in identities] +
             [f"{name}.obj" for name in expressions])

    for name in names:
        target = os.path.join(model_dir, name)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{MODEL_URL}/{name}", target)
    return model_dir


def _parse_obj(path: str, want_faces: bool = False):
    """Read vertices (and optionally quad indices) from a wavefront .obj."""
    verts, faces = [], []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append(line[2:].split()[:3])
            elif want_faces and line.startswith("f "):
                idx = [tok.split("/")[0] for tok in line[2:].split()]
                # the mesh is mostly quads with a few triangles, and a triangle is stored as a
                # quad with one repeated corner
                faces.append(idx if len(idx) == 4 else idx[:3] + idx[2:3])
    vertices = np.array(verts, dtype=np.float32)
    if not want_faces:
        return vertices, None
    quads = np.array(faces, dtype=np.int64) - 1 # .obj indices are 1-based
    return vertices, quads


def _landmark_indices(model_dir: str) -> np.ndarray:
    """The 68 facial landmark vertices, used to frame the camera on the face."""
    import json
    path = os.path.join(model_dir, "vertex_indices.json")
    with open(path) as handle:
        return np.array(json.load(handle)["idx_to_landmark_verts"], dtype=np.int64)


def _load_meshes(model_dir: str, identities: tuple, expressions: tuple) -> dict:
    """
    Load the neutral mesh together with the identity and expression shape modes.

    Shape modes are stored as offsets from the neutral mesh, which is how the ICT model is defined.
    Parsed meshes are cached in `model_cache.npz` inside `model_dir`, and the cache is rebuilt
    whenever a mesh is requested that it does not hold yet.
    """
    cache_path = os.path.join(model_dir, "model_cache.npz")
    have_ids, have_exs = [], []
    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        have_ids = list(cached["identity_names"])
        have_exs = list(cached["expression_names"])
        if (all(f"identity{i:03d}" in have_ids for i in identities) and
                all(name in have_exs for name in expressions)):
            return {
                "neutral": cached["neutral"],
                "quads": cached["quads"],
                "identity_modes": {n: m for n, m in
                                   zip(have_ids, cached["identity_modes"])},
                "expression_modes": {n: m for n, m in
                                     zip(have_exs, cached["expression_modes"])},
            }

    neutral_path = os.path.join(model_dir, "generic_neutral_mesh.obj")
    if not os.path.exists(neutral_path):
        raise FileNotFoundError(
            f"ICT-FaceKit meshes not found in {model_dir}. Run "
            "`python -m data.faces` (from src/) to download them."
        )
    neutral, quads = _parse_obj(neutral_path, want_faces=True)

    # take the union of cached and requested meshes, so that the cache only ever grows
    id_names = sorted({f"identity{i:03d}" for i in identities} | set(have_ids))
    ex_names = sorted(set(expressions) | set(have_exs))

    identity_modes, expression_modes = {}, {}
    for name in id_names:
        verts, _ = _parse_obj(os.path.join(model_dir, f"{name}.obj"))
        identity_modes[name] = verts - neutral
    for name in ex_names:
        verts, _ = _parse_obj(os.path.join(model_dir, f"{name}.obj"))
        expression_modes[name] = verts - neutral

    np.savez_compressed(
        cache_path,
        neutral=neutral, quads=quads,
        identity_names=np.array(list(identity_modes)),
        identity_modes=np.stack([identity_modes[n] for n in identity_modes]),
        expression_names=np.array(list(expression_modes)),
        expression_modes=np.stack([expression_modes[n] for n in expression_modes]),
    )
    return {"neutral": neutral, "quads": quads,
            "identity_modes": identity_modes,
            "expression_modes": expression_modes}


def _vertex_normals(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of a quad mesh, from the cross product of the diagonals."""
    v0, v1, v2, v3 = (vertices[quads[:, k]] for k in range(4))
    face_normals = np.cross(v2 - v0, v3 - v1)
    normals = np.zeros_like(vertices)
    for k in range(4):
        np.add.at(normals, quads[:, k], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-8)


def _surface_samples(quads: np.ndarray, subdiv: int):
    """
    Bilinear sample weights inside every quad.

    Returns an (n_quads * subdiv^2, 4) weight matrix and the matching corner indices, so that
    `weights @ corner_values` gives the sampled points. Sampling the surface instead of drawing
    the raw vertices is what keeps the rendered image free of holes.
    """
    steps = (np.arange(subdiv) + 0.5) / subdiv
    u, v = np.meshgrid(steps, steps, indexing="ij")
    u, v = u.ravel(), v.ravel()
    # bilinear weights of the four corners
    weights = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=1)
    return weights.astype(np.float32), quads


class FaceRenderer:
    """
    Renders the ICT face model into small grayscale images.

    The camera is orthographic and fixed. Its framing is derived once from the neutral mesh, so
    the only thing that moves in the image is the identity, the expression or the head pose.
    """

    def __init__(self, model_dir: str = None, identities=FACE_MODES,
                 expressions=None, image_size: int = 64, subdiv: int = None,
                 face_fill: float = 0.72, light_dir=(0.35, 0.45, 0.82),
                 ambient: float = 0.15, diffuse: float = 0.8):
        model_dir = model_dir or MODEL_DIR
        if expressions is None:
            expressions = sorted({name for names in EXPRESSION_FACTORS.values()
                                  for name in names})
        self.image_size = int(image_size)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.face_fill = float(face_fill)

        meshes = _load_meshes(model_dir, tuple(identities), tuple(expressions))
        self.neutral = meshes["neutral"]
        self.quads = meshes["quads"]
        self.identity_modes = meshes["identity_modes"]
        self.expression_modes = meshes["expression_modes"]

        light = np.asarray(light_dir, dtype=np.float32)
        self.light_dir = light / np.linalg.norm(light)

        # Fix the camera on the neutral head: the facial landmarks are centred and fill
        # `face_fill` of the image, so the face and not the neck and shoulders takes up most of
        # the pixels. The landmark centroid also serves as the pivot of the head rotation.
        landmarks = self.neutral[_landmark_indices(model_dir)]
        self._center = landmarks.mean(axis=0)
        extent = np.abs(landmarks - self._center)[:, :2].max()
        self._scale = self.image_size * self.face_fill / (2.0 * extent)

        self.subdiv = self._auto_subdiv() if subdiv is None else int(subdiv)
        self._weights, self._corners = _surface_samples(self.quads, self.subdiv)

    def _auto_subdiv(self) -> int:
        """
        How many samples per quad edge are needed to cover the image without holes.

        The renderer draws surface samples into pixels, so a quad spanning several pixels needs at
        least as many samples along each edge, otherwise it leaves gaps and the face looks
        speckled. Since the quads differ a lot in size, the count comes from a high quantile of the
        projected edge lengths of the neutral head. Rotating the head can only foreshorten a quad,
        never enlarge it.
        """
        projected = (self.neutral - self._center)[:, :2] * self._scale
        corners = projected[self.quads] # (Q, 4, 2)
        edges = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
        return int(np.clip(np.ceil(np.quantile(edges.max(axis=1), 0.99)), 1, 12))

    def vertices(self, identity: int = 0, identity_weight: float = 1.0,
                 expression: dict = None) -> np.ndarray:
        """The deformed vertices for one identity and a dict of blendshape weights."""
        verts = self.neutral.copy()
        name = f"identity{int(identity):03d}"
        if name in self.identity_modes:
            verts += float(identity_weight) * self.identity_modes[name]
        for shape_name, weight in (expression or {}).items():
            if weight == 0.0:
                continue
            verts += float(weight) * self.expression_modes[shape_name]
        return verts

    def render(self, identity: int = 0, identity_weight: float = 1.0,
               expression: dict = None, yaw: float = 0.0,
               pitch: float = 0.0) -> np.ndarray:
        """
        Render one face as an (image_size, image_size) array in [0, 1].

        yaw and pitch are in degrees, where yaw turns the head around the vertical axis.
        """
        verts = self.vertices(identity, identity_weight, expression)
        normals = _vertex_normals(verts, self.quads)

        centered = verts - self._center
        yaw_rad, pitch_rad = np.radians(yaw), np.radians(pitch)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        rot_yaw = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rot_pitch = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
        rotation = (rot_pitch @ rot_yaw).astype(np.float32)
        centered = centered @ rotation.T
        normals = normals @ rotation.T

        # sample the surface, giving (n_samples, 3) positions and normals
        points = np.einsum("sk,qkj->qsj", self._weights,
                           centered[self._corners]).reshape(-1, 3)
        point_normals = np.einsum("sk,qkj->qsj", self._weights,
                                  normals[self._corners]).reshape(-1, 3)

        shade = point_normals @ self.light_dir
        shade = self.ambient + self.diffuse * np.clip(shade, 0.0, None)

        half = self.image_size / 2.0
        col = points[:, 0] * self._scale + half
        row = half - points[:, 1] * self._scale
        col_i = np.floor(col).astype(np.int64)
        row_i = np.floor(row).astype(np.int64)
        inside = ((col_i >= 0) & (col_i < self.image_size) &
                  (row_i >= 0) & (row_i < self.image_size))

        depth = points[:, 2]
        order = np.argsort(depth[inside]) # back to front, so the nearest sample is written last
        flat = (row_i[inside] * self.image_size + col_i[inside])[order]

        image = np.zeros(self.image_size * self.image_size, dtype=np.float32)
        image[flat] = shade[inside][order]
        return image.reshape(self.image_size, self.image_size)

    def render_factor(self, identity: int, factor: str, value: float,
                      base: dict = None, identity_weight: float = 1.0) -> np.ndarray:
        """
        Render one identity with a single generative factor set to `value`.

        `base` can fix the remaining factors, for instance a constant yaw for a component that
        only varies its smile.
        """
        settings = dict(base or {})
        settings[factor] = value
        expression, yaw, pitch = {}, 0.0, 0.0
        for name, val in settings.items():
            if name == "yaw":
                yaw = float(val)
            elif name == "pitch":
                pitch = float(val)
            elif name in EXPRESSION_FACTORS:
                for shape_name in EXPRESSION_FACTORS[name]:
                    expression[shape_name] = float(val)
            else:
                raise ValueError(f"unknown factor {name!r}; "
                                 f"known: {sorted(FACTOR_NAMES)}")
        return self.render(identity=identity, identity_weight=identity_weight,
                           expression=expression, yaw=yaw, pitch=pitch)


def make_multi_face_dataset(
    specs: list,
    samples_per: int = 120,
    image_size: int = 64,
    center: bool = True,
    add_noise: float = 0.05,
    random_state: int = 0,
    renderer: FaceRenderer = None,
    progress=None,
) -> tuple:
    """
    Several faces, each sweeping one factor, stacked into one dataset.

    Every spec is a different face of FACES and therefore a separate component. A factor is swept
    over an interval and never closes, so each component is an open arc.

    Parameters
    ----------
    specs : one dict per component, with "face" (an index into FACES) and "factor" (one of
        FACTOR_NAMES), plus optionally "samples", "start"/"end" (by default the full range from
        FACTOR_RANGES) and "base" for the factors held fixed.
    samples_per : frames per component, overridden by a spec's own "samples".
    center, add_noise, random_state : as for the MNIST datasets. The noise goes on in pixel space,
        before one joint standardization over all frames.
    progress : callable(done, total), called at most once per percent. Rendering takes seconds to
        minutes depending on the image size.

    Returns
    -------
    X : (N, image_size**2) standardized frames; values : (N,) factor value in its component's own
    unit; component_id : (N,) spec index, used as the class label; meta : per spec "sl", "face",
    "name", "factor", "start", "end", "unit"; pixel_mean and pixel_std for undoing the scaling.
    """
    rng = np.random.default_rng(random_state)
    if renderer is None or renderer.image_size != int(image_size):
        needed = sorted({name for spec in specs
                         for name in EXPRESSION_FACTORS.get(spec.get("factor"), ())} |
                        {name for spec in specs
                         for key in (spec.get("base") or {})
                         for name in EXPRESSION_FACTORS.get(key, ())})
        renderer = FaceRenderer(image_size=image_size,
                                identities=tuple(sorted({FACES[int(s.get("face", 0))]["mode"]
                                                         for s in specs})),
                                expressions=tuple(needed) or None)

    total = sum(int(s.get("samples", samples_per)) for s in specs)
    last_percent = -1

    frames, value_list, comp_list, meta = [], [], [], []
    for index, spec in enumerate(specs):
        face = FACES[int(spec.get("face", 0))]
        factor = str(spec.get("factor", "yaw"))
        full = FACTOR_RANGES[factor]
        start = float(spec.get("start", full[0]))
        end = float(spec.get("end", full[1]))
        n = int(spec.get("samples", samples_per))
        base = spec.get("base") or {}

        grid = np.linspace(start, end, n)
        offset = len(frames)
        for value in grid:
            image = renderer.render_factor(face["mode"], factor, value, base=base,
                                           identity_weight=face["weight"])
            frames.append(image.ravel().astype(float))
            if progress is not None:
                percent = 100 * len(frames) // total
                if percent != last_percent:
                    last_percent = percent
                    progress(len(frames), total)
        value_list.extend(grid.tolist())
        comp_list.extend([index] * n)
        meta.append({
            "face": int(spec.get("face", 0)), "name": face["name"],
            "factor": factor, "start": start, "end": end,
            "unit": FACTOR_UNITS.get(factor, "units"), "base": dict(base),
            "is_loop": False, "sl": slice(offset, offset + n),
        })

    X = np.stack(frames)
    values = np.array(value_list)
    component_id = np.array(comp_list, dtype=int)

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

    return X, values, component_id, meta, pixel_mean, pixel_std


if __name__ == "__main__":
    print("meshes in", download_model())

