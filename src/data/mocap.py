"""
Motion capture poses from the CMU Graphics Lab Motion Capture Database.

The one experiment whose observations are not images: an observation is a pose,
the 3D positions of 31 skeleton joints, so a data vector is 93-dimensional and
can be drawn as a stick figure. A recording becomes a dataset in three steps:

1. 'Skeleton' reads the .asf (bone directions, lengths, local axes) and runs
   forward kinematics on every .amc frame (joint angles) to get world positions.
2. 'canonical_poses' removes global position and heading, leaving a body-centred
   frame (x = right, y = up, z = forward). The root height stays, since bobbing
   up and down is part of the movement.
3. Each motion is cut into repetitions along a physical signal, never the fitted
   model. Periodic motions give cycles and become loops; one-way movements such
   as sitting down give arcs.

The ground-truth factor is the progress within one repetition in percent, so
0 % and 100 % of a gait cycle are the same pose while 0 % of "sit down" is
standing and 100 % is seated.

Recordings live in 'data/cmu_mocap/' and are not in the repository; run
'python -m data.mocap' from src/ once to download them. Joint positions are
cached per subject. 'make_multi_motion_dataset' returns X standardized, invert
it for display via pose = (v * pose_std + pose_mean).reshape(31, 3).
"""

import os
import re

import numpy as np

MOCAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cmu_mocap",
)

# The database is free to use ("you may copy, modify, or redistribute"), see
# http://mocap.cs.cmu.edu/. Files are fetched per trial, a few hundred kB each.
BASE_URL = "http://mocap.cs.cmu.edu/subjects"

# ASF bone lengths are given in a unit of 1/0.45 inch; this maps them to metres.
LENGTH_SCALE = (1.0 / 0.45) * 2.54 / 100.0

# Only used to report durations. The database lists 120 Hz for every trial, but
# a walk cycle of this subject is 69 frames long, which is a normal 1.15 s
# stride at 60 Hz and an impossible 0.57 s one at 120 Hz. Nothing in the
# pipeline depends on it: the ground-truth factor is progress in percent, and
# the repetitions are cut with frame-rate free rules.
FPS = 60.0

# All motions come from subject 143, who performed walking, running, waving and
# sitting down in the same session: the skeleton is then identical across
# components, so the components differ by *movement* and not by body size.
SUBJECT = "143"

# The signal a stride is cut from: how far the left foot is in front of the
# right one. It is the cleanest gait signal there is - one smooth oscillation
# per stride, for walking as well as for running, with no half-cycle peaks from
# the swing of a single foot.
GAIT_SIGNAL = (("lfoot", "z", 1.0), ("rfoot", "z", -1.0))

# The motions of the experiment. `signal` is the physical scalar the repetitions
# are cut from, a weighted sum of joint coordinates in the body-centred frame.
# `cut` says how, which also fixes the expected topology:
#
#   "cycle"      peak to peak; the pose returns to its start, so a LOOP.
#   "swing"      extremum to extremum, i.e. half a period. A wave is periodic in
#                time, but the arm travels out and back along the SAME path, so
#                the component is an ARC. Progress is the signal itself (where
#                the hand is), not time, since swings differ in reach.
#   "transition" from the high plateau of the signal to the low one (or back,
#                with `rising`): a one-way movement, again an ARC.
#
# Four run trials are needed because a single run of this subject is only two
# strides long, while one walking trial is long enough on its own.
MOTIONS = (
    {"name": "walk", "trials": ("143_32", "143_29"), "kind": "loop",
     "cut": "cycle", "signal": GAIT_SIGNAL,
     "description": "walking, one gait cycle"},
    {"name": "run", "trials": ("143_42", "143_01", "143_02", "143_03"),
     "kind": "loop", "cut": "cycle", "signal": GAIT_SIGNAL,
     "description": "running, one gait cycle"},
    {"name": "walk_backwards", "trials": ("143_39",), "kind": "loop",
     "cut": "cycle", "signal": GAIT_SIGNAL,
     "description": "walking backwards, one gait cycle"},
    {"name": "wave", "trials": ("143_25",), "kind": "arc",
     "cut": "swing", "signal": (("rhand", "x", 1.0),), "tolerance": 0.4,
     "description": "waving, one swing of the arm"},
    {"name": "sit_down", "trials": ("143_18",), "kind": "arc",
     "cut": "transition", "signal": (("root", "y", 1.0),),
     "description": "sitting down, standing -> seated"},
    {"name": "stand_up", "trials": ("143_18",), "kind": "arc",
     "cut": "transition", "signal": (("root", "y", 1.0),), "rising": True,
     "description": "standing up, seated -> standing"},
)

MOTION_NAMES = tuple(m["name"] for m in MOTIONS)

# The factor is the progress within one repetition, for every motion.
FACTOR_UNIT = "%"

AXES = {"x": 0, "y": 1, "z": 2}


def motion_spec(name):
    for m in MOTIONS:
        if m["name"] == name:
            return m
    raise KeyError(f"unknown motion {name!r}; known: {MOTION_NAMES}")


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def download_motions(names=None, mocap_dir: str = None):
    """
    Fetch the skeleton and the trials of `names` (default: all MOTIONS) from the
    CMU database into `mocap_dir`, skipping files that are already there.
    """
    import urllib.request

    mocap_dir = mocap_dir or MOCAP_DIR
    os.makedirs(mocap_dir, exist_ok=True)
    motions = [motion_spec(n) for n in (names or MOTION_NAMES)]

    files = [f"{SUBJECT}.asf"]
    files += sorted({t for m in motions for t in m["trials"]}, key=str)
    for name in files:
        fname = name if name.endswith(".asf") else f"{name}.amc"
        target = os.path.join(mocap_dir, fname)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(f"{BASE_URL}/{SUBJECT}/{fname}", target)
    return mocap_dir


# --------------------------------------------------------------------------
# .asf / .amc  ->  joint positions
# --------------------------------------------------------------------------
def _euler_matrix(rx, ry, rz):
    """Rotation for the ASF axis order XYZ: rotate about x, then y, then z."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    return np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz],
        [-sy, cy * sx, cx * cy],
    ])


class Skeleton:
    """
    The bone hierarchy of one subject, read from its .asf, with forward
    kinematics for the joint angles of a .amc.

    A bone carries a rest direction, a length and a local axis frame C. Its
    global rotation is  M_bone = M_parent @ C @ R(angles) @ C^-1  and its
    endpoint is the parent's endpoint plus the rotated rest bone, which is the
    standard reading of the ASF/AMC format.
    """

    def __init__(self, asf_path: str):
        text = open(asf_path).read()
        sections = {}
        current = None
        for line in text.splitlines():
            if line.startswith(":"):
                current = line[1:].split()[0]
                sections[current] = [line[1:]]
            elif current is not None:
                sections[current].append(line)

        root_block = "\n".join(sections["root"])
        root_order = re.search(r"order\s+([A-Z ]+)", root_block).group(1).split()

        self.bones = {"root": {
            "direction": np.zeros(3), "length": 0.0, "axis": np.zeros(3),
            "dof": [d.lower() for d in root_order],
        }}
        for block in re.findall(r"begin(.*?)end", "\n".join(sections["bonedata"]), re.S):
            name = re.search(r"name\s+(\S+)", block).group(1)
            axis_m = re.search(r"axis\s+([-\d.eE\s]+?)\s*([A-Z]{3})", block)
            dof_m = re.search(r"dof\s+([rt xyz]+)", block)
            self.bones[name] = {
                "direction": np.array([
                    float(v) for v in
                    re.search(r"direction\s+([-\d.eE\s]+)", block).group(1).split()]),
                "length": float(re.search(r"length\s+([-\d.eE]+)", block).group(1)),
                "axis": np.array([float(v) for v in axis_m.group(1).split()]),
                "dof": dof_m.group(1).split() if dof_m else [],
            }

        self.parents = {"root": None}
        for line in sections["hierarchy"]:
            parts = line.split()
            if not parts or parts[0] in ("hierarchy", "begin", "end"):
                continue
            for child in parts[1:]:
                self.parents[child] = parts[0]

        # parents before children, so one pass of forward kinematics is enough
        self.joints = []
        pending = [n for n in self.bones]
        while pending:
            rest = []
            for name in pending:
                if self.parents[name] is None or self.parents[name] in self.joints:
                    self.joints.append(name)
                else:
                    rest.append(name)
            if len(rest) == len(pending):
                raise ValueError("broken bone hierarchy in the .asf")
            pending = rest

        for name, bone in self.bones.items():
            C = _euler_matrix(*np.deg2rad(bone["axis"]))
            bone["C"] = C
            bone["Cinv"] = np.linalg.inv(C)
            bone["offset"] = bone["length"] * LENGTH_SCALE * bone["direction"]

        self.index = {name: i for i, name in enumerate(self.joints)}
        # bones to draw: every joint except the root is connected to its parent
        self.edges = [(self.index[self.parents[n]], self.index[n])
                      for n in self.joints if self.parents[n] is not None]

    def positions(self, frames: list) -> np.ndarray:
        """Joint positions in metres for a list of .amc frames -> (T, J, 3)."""
        out = np.zeros((len(frames), len(self.joints), 3))
        for t, frame in enumerate(frames):
            coords, mats = {}, {}
            for name in self.joints:
                bone = self.bones[name]
                values = frame.get(name, ())
                angles = np.zeros(3)
                translation = np.zeros(3)
                for value, dof in zip(values, bone["dof"]):
                    if dof[0] == "r":
                        angles[AXES[dof[1]]] = value
                    else:
                        translation[AXES[dof[1]]] = value
                local = bone["C"] @ _euler_matrix(*np.deg2rad(angles)) @ bone["Cinv"]
                parent = self.parents[name]
                if parent is None:
                    mats[name] = local
                    coords[name] = translation * LENGTH_SCALE
                else:
                    mats[name] = mats[parent] @ local
                    coords[name] = coords[parent] + mats[name] @ bone["offset"]
                out[t, self.index[name]] = coords[name]
        return out


def _parse_amc(path: str) -> list:
    """Read an .amc into a list of {bone: [angles]} dicts, one per frame."""
    frames, current = [], None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(("#", ":")):
            continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            current = {}
            frames.append(current)
        elif current is not None:
            current[parts[0]] = [float(v) for v in parts[1:]]
    return frames


_SKELETON = None
_TRIAL_CACHE = {}


def _skeleton(mocap_dir: str = None) -> Skeleton:
    global _SKELETON
    if _SKELETON is None:
        mocap_dir = mocap_dir or MOCAP_DIR
        asf = os.path.join(mocap_dir, f"{SUBJECT}.asf")
        if not os.path.exists(asf):
            raise FileNotFoundError(
                f"{asf} not found - run `python -m data.mocap` (from src/) once "
                "to download the recordings.")
        _SKELETON = Skeleton(asf)
    return _SKELETON


def joint_names() -> list:
    return list(_skeleton().joints)


def bone_edges() -> list:
    """(parent, child) index pairs of the skeleton, for drawing a stick figure."""
    return list(_skeleton().edges)


def load_trial(trial: str, mocap_dir: str = None) -> np.ndarray:
    """
    World joint positions (T, J, 3) of one trial, in metres. Forward kinematics
    over a few thousand frames costs a second or two, so the result is cached in
    `data/cmu_mocap/positions_cache.npz` and in memory.
    """
    mocap_dir = mocap_dir or MOCAP_DIR
    if trial in _TRIAL_CACHE:
        return _TRIAL_CACHE[trial]

    cache_path = os.path.join(mocap_dir, "positions_cache.npz")
    cache = {}
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            cache = {k: data[k] for k in data.files}
    if trial not in cache:
        amc = os.path.join(mocap_dir, f"{trial}.amc")
        if not os.path.exists(amc):
            raise FileNotFoundError(
                f"{amc} not found - run `python -m data.mocap` (from src/) once "
                "to download the recordings.")
        cache[trial] = _skeleton(mocap_dir).positions(_parse_amc(amc))
        np.savez_compressed(cache_path, **cache)
    _TRIAL_CACHE[trial] = cache[trial]
    return cache[trial]


# --------------------------------------------------------------------------
# canonical poses
# --------------------------------------------------------------------------
def heading(P: np.ndarray) -> np.ndarray:
    """
    The direction the body faces, in radians, read off the hip axis and
    unwrapped, so that a turn shows up as a steady drift.
    """
    sk = _skeleton()
    right = P[:, sk.index["rhipjoint"]] - P[:, sk.index["lhipjoint"]]
    return np.unwrap(np.arctan2(right[:, 2], right[:, 0]))


def canonical_poses(P: np.ndarray) -> np.ndarray:
    """
    Body-centred poses: the root is moved to the origin and the body is turned
    to a common heading, so walking north and walking south give the same pose.
    The root height is added back, because rising and sinking is part of a
    movement while the position on the floor is not.

    The resulting frame is x = right, y = up, z = forward.
    """
    sk = _skeleton()
    root = sk.index["root"]

    out = P - P[:, root][:, None, :]
    angle = heading(P)
    cos, sin = np.cos(-angle), np.sin(-angle)
    R = np.zeros((len(P), 3, 3))
    R[:, 0, 0], R[:, 0, 2] = cos, sin
    R[:, 1, 1] = 1.0
    R[:, 2, 0], R[:, 2, 2] = -sin, cos
    out = np.einsum("tij,tkj->tki", R, out)
    out[:, :, 1] += P[:, root, 1][:, None]
    return out


def _signal(poses: np.ndarray, terms) -> np.ndarray:
    """
    The scalar a motion's repetitions are cut from: a weighted sum of joint
    coordinates in the body frame, e.g. the forward gap between the two feet.
    """
    index = _skeleton().index
    out = np.zeros(len(poses))
    for joint, axis, weight in terms:
        out += weight * poses[:, index[joint], AXES[axis]]
    return out


# --------------------------------------------------------------------------
# repetitions
# --------------------------------------------------------------------------
def _keep_typical(repetitions, tolerance):
    """
    Drop the repetitions whose duration is far from the median one, pooled over
    all trials of a motion: those are the ones interrupted by a turn, a stop or
    a change of pace, and they would not lie on the same curve as the rest.
    """
    if len(repetitions) < 3:
        return repetitions
    durations = np.array([len(poses) for poses, _ in repetitions], dtype=float)
    median = np.median(durations)
    return [r for r, d in zip(repetitions, durations)
            if abs(d - median) <= tolerance * median]


def _cycles(signal, prominence=0.2):
    """
    Cut a periodic signal into cycles, from one peak to the next.

    A peak must stand out by `prominence` of the signal range, which keeps the
    idle stretches of a recording out (the arm hanging down before the waving
    starts). Peaks are found twice: the first pass gives the typical cycle
    length, the second one uses 70 % of it as the minimum peak distance so that
    a wobble inside a cycle cannot split it. No frame rate enters anywhere.

    Returns a list of (start, end) frame indices, end exclusive.
    """
    from scipy.signal import find_peaks

    span = float(signal.max() - signal.min())
    if span <= 0:
        return []
    peaks, _ = find_peaks(signal, prominence=prominence * span)
    if len(peaks) < 2:
        return []
    gaps = np.diff(peaks)
    period = float(np.median(gaps)) if len(gaps) > 1 else float(gaps[0])
    peaks, _ = find_peaks(signal, prominence=prominence * span,
                          distance=max(1, int(0.7 * period)))
    return [(int(a), int(b)) for a, b in zip(peaks[:-1], peaks[1:])]


def _swings(signal, prominence=0.2):
    """
    Cut a periodic signal into half cycles, from one extremum to the next.

    Returns a list of (start, end, rising) with end exclusive; `rising` says
    whether the signal grows over the segment, which is what orients the two
    directions of the same swing onto the same progress.
    """
    from scipy.signal import find_peaks

    span = float(signal.max() - signal.min())
    if span <= 0:
        return []
    peaks, _ = find_peaks(signal, prominence=prominence * span)
    troughs, _ = find_peaks(-signal, prominence=prominence * span)
    extrema = np.sort(np.concatenate([peaks, troughs])).astype(int)
    from_trough = np.isin(extrema, troughs)
    return [(int(a), int(b) + 1, bool(up)) for a, b, up
            in zip(extrema[:-1], extrema[1:], from_trough[:-1])]


def _transitions(signal, rising=False, high=0.95, low=0.05, min_frames=5):
    """
    Cut a one-way movement out of a signal that steps between two levels: the
    stretches where it goes from the high plateau to the low one (sitting down)
    or, with `rising`, the other way round (standing up).

    Returns a list of (start, end) frame indices, end exclusive.
    """
    s = np.asarray(signal, dtype=float)
    span = s.max() - s.min()
    if span <= 0:
        return []
    s = (s - s.min()) / span
    if rising:
        s = 1.0 - s

    spans, start = [], None
    for i, value in enumerate(s):
        if value >= high:
            start = i  # keep sliding: the last frame still on the plateau
        elif value <= low and start is not None:
            if i - start >= min_frames:
                spans.append((start, i + 1))
            start = None
    return spans


def motion_repetitions(name: str, mocap_dir: str = None, max_turn: float = 30.0,
                       tolerance: float = 0.25) -> list:
    """
    The single repetitions of one motion, pooled over its trials.

    A repetition of a periodic motion is one cycle of its signal; of a one-way
    movement, one transition between the two levels. Cycles during which the
    body turns by more than `max_turn` degrees are dropped: while turning, the
    body leans into the curve and the pose leaves the curve the straight
    repetitions trace out. What survives is then cut down to the repetitions of
    typical duration (`tolerance`, relative to the median over all trials).

    Returns a list of (poses (n, J, 3), progress (n,)), progress in percent.
    """
    spec = motion_spec(name)
    cut = spec["cut"]
    prominence = spec.get("prominence", 0.2)
    reps = []

    for trial in spec["trials"]:
        P = load_trial(trial, mocap_dir)
        canonical = canonical_poses(P)
        signal = _signal(canonical, spec["signal"])

        if cut == "cycle":
            turn = np.rad2deg(heading(P))
            spans = [(a, b, True) for a, b in _cycles(signal, prominence)
                     if abs(turn[b - 1] - turn[a]) <= max_turn]
        elif cut == "swing":
            spans = _swings(signal, prominence)
        else:
            spans = [(a, b, True) for a, b in
                     _transitions(signal, rising=bool(spec.get("rising")))]

        for start, end, forward in spans:
            n = end - start
            if cut == "swing":
                # A swing is measured by *where* the body is, not by how long it
                # has been going: different swings cover different parts of the
                # same path, and the two directions retrace it, so the signal
                # itself (the hand position) is the factor. It is turned into
                # percent of its full range below, once all swings are in.
                progress = signal[start:end]
            else:
                # a cycle is periodic, so its last frame sits just short of 100 %
                progress = 100.0 * np.arange(n) / (n if cut == "cycle" else n - 1)
                if not forward:
                    progress = 100.0 - progress
            reps.append((canonical[start:end], progress))

    reps = _keep_typical(reps, spec.get("tolerance", tolerance))
    if not reps:
        raise ValueError(f"no repetitions found for motion {name!r}")
    if cut == "swing":
        values = np.concatenate([p for _, p in reps])
        lo, span = values.min(), np.ptp(values)
        reps = [(poses, 100.0 * (p - lo) / span) for poses, p in reps]
    if cut == "transition":
        # Two takes of a one-way movement start at the same pose and end at the
        # same pose, so their union closes into a loop - correct, but then the
        # component is no longer the arc the experiment is about. One take is
        # one traversal of the arc, exactly like one sweep of a rotating digit.
        reps = [max(reps, key=lambda rep: len(rep[0]))]
    return reps


def motion_frames(name: str, mocap_dir: str = None):
    """
    All usable frames of one motion, pooled over its repetitions.

    Returns (poses (N, J, 3), progress (N,)) where `progress` is the position
    within the repetition a frame belongs to, in percent. For a loop 100 % is
    the same pose as 0 %, so the endpoint is never included; for an arc 100 % is
    the far end, which is included.
    """
    reps = motion_repetitions(name, mocap_dir)
    return (np.concatenate([poses for poses, _ in reps]),
            np.concatenate([progress for _, progress in reps]))


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def make_multi_motion_dataset(
    specs: list,
    samples_per: int = 100,
    center: bool = True,
    add_noise: float = 0.0,
    random_state: int = 0,
    mocap_dir: str = None,
    progress=None,
) -> tuple:
    """
    Several motions of one person stacked into one dataset, one component each.

    Parameters
    ----------
    specs : one dict per component, keys "motion" (a MOTION_NAMES entry) and
        optionally "samples" (frames for this component, default samples_per).
    samples_per : frames per component, drawn evenly across all repetitions of
        the motion so every repetition contributes.
    center : standardize X by the mean pose and one global scale.
    add_noise : std of joint noise in metres, added before standardization.
    progress : callable(done, total), called while reading the recordings.

    Returns
    -------
    X : (N, 93) standardized poses, values : (N,) progress in percent,
    motion_id : (N,) index into `specs` (the discrete ground-truth label),
    meta : per spec "motion", "kind", "is_loop", "n_frames" and sample range "sl",
    pose_mean : (93,), pose_std : (93,) global scale broadcast to all coordinates.
    """
    rng = np.random.default_rng(random_state)
    blocks, values, motion_id, meta = [], [], [], []

    total = len(specs)
    for i, spec in enumerate(specs):
        name = str(spec["motion"])
        if progress is not None:
            progress(i, total)
        poses, prog = motion_frames(name, mocap_dir)

        n = int(spec.get("samples", samples_per))
        take = (np.linspace(0, len(poses) - 1, n).round().astype(int)
                if n < len(poses) else np.arange(len(poses)))
        offset = sum(len(b) for b in blocks)
        blocks.append(poses[take].reshape(len(take), -1))
        values.append(prog[take])
        motion_id.extend([i] * len(take))
        meta.append({
            "motion": name, "kind": motion_spec(name)["kind"],
            "is_loop": motion_spec(name)["kind"] == "loop",
            "description": motion_spec(name)["description"],
            "n_frames": int(len(poses)),
            "sl": slice(offset, offset + len(take)),
        })
    if progress is not None:
        progress(total, total)

    X = np.concatenate(blocks)
    values = np.concatenate(values)
    motion_id = np.array(motion_id, dtype=int)

    pose_mean = np.zeros(X.shape[1])
    pose_std = np.ones(X.shape[1])
    if center:
        pose_mean = X.mean(axis=0)
        scale = float((X - pose_mean).std())
        pose_std = np.full(X.shape[1], scale if scale > 1e-8 else 1.0)

    if add_noise > 0:
        X = X + rng.normal(0.0, add_noise, size=X.shape)

    if center:
        X = (X - pose_mean) / pose_std

    return X, values, motion_id, meta, pose_mean, pose_std


if __name__ == "__main__":
    download_motions()
    for spec in MOTIONS:
        reps = motion_repetitions(spec["name"])
        n = sum(len(poses) for poses, _ in reps)
        durations = [len(poses) for poses, _ in reps]
        print(f"{spec['name']:>9} ({spec['kind']:>4}): {n:5d} frames in "
              f"{len(reps)} repetition(s) of {durations} frames "
              f"({np.mean(durations) / FPS:.2f} s each)")
