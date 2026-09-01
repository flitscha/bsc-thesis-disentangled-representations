# Learning Data Densities and Data Coordinate Systems for Disentangled Representations

Bachelor thesis by **Felix Campidell**, supervised by **Till Kahlke** and **Jörg Lücke**.
Universität Innsbruck, Department of Computer Science.

## What this is about

High-dimensional data usually has far fewer degrees of freedom than it has dimensions.
A 900-pixel image of one handwritten digit turned through a full circle lives in a 900-dimensional space, but only a single number really varies: the angle.
The images trace out a one-dimensional curve (a *manifold*) inside that huge space, and here the curve closes into a loop, since turning by 360° returns to the start.

The quantities that generate the data this way are its **generative factors**.
A representation is **disentangled** if it recovers them as separate coordinates:
changing one coordinate changes exactly one factor and leaves the others alone.
That is what makes a representation useful.
You can read a factor off it, and you can control the data by moving along it.

The difficulty is that nothing in the raw data marks those directions.
The rotation angle is not a pixel and not a fixed direction in pixel space.
This thesis therefore treats disentanglement as a geometric question.
It shows that a representation is disentangled precisely when the way it splits up the tangent spaces of the data manifold agrees with the way the generative process does, and then builds an unsupervised method that estimates those tangent spaces and turns them into a coordinate system.

## How the method works

A mixture of factor analyzers (MFA) model is fitted to the data, so every mixture component becomes a local chart with its own tangent frame.
The charts are then linked by a tangent-aware distance that is cheap along the manifold and expensive across it. Persistent homology reads the connected components and the loops off that distance graph, and each detected structure is interpolated into a cubic spline.

That spline is a coordinate system for the data:
moving along it traverses one generative factor, learned without labels.

```python
from core.pipeline import ManifoldPipeline

pipe = ManifoldPipeline(n_components=100, pca_dim=50, detection="tda")
result = pipe.fit_detect(X)      # curves + splines
t, curve_id = pipe.transform(X)  # the learned coordinate per observation
```

Detection yields at least one curve per connected component, plus a loop wherever one was found. `result["curves"]` lists them, each with its `type` (`"loop"` or `"path"`), the H0 `component` it belongs to, the `order` of chart means along it and the `spline`.

`transform` is the encoder. It assigns every observation to the curve whose spline runs closest to it.


## Demos

`cd src && python main.py` opens one window with six tabs:

| Tab | Shows |
| --- | --- |
| Synthetic Data | demonstrates the MFA model |
| Loop Detection | the traversal baseline on a 1D manifold |
| Topology (TDA) | components, loops and the persistence diagram on toy manifolds |
| MNIST Rotation | rotating digits: a full sweep is a loop, a partial one an arc |
| Faces | rendered ICT-FaceKit faces, one generative factor each |
| Motion Capture | CMU poses (not images) |

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/80dcb648-de46-419a-bc9e-04532b1d6fce" />

## Experiments

The offline evaluations behind the thesis results run from the demo ("Run evaluation") or standalone, and write to `results/<experiment>/<tag>/` (`summary.json`, `arrays.npz`, figures).
The defaults reproduce the runs already in `results/`:

```bash
cd src
python -m experiments.mnist_rotation_eval   # §6.2  single rotating digit, TDA vs. baseline
python -m experiments.face_factor_eval      # §6.3  rendered faces
python -m experiments.multi_factor_eval     # §6.4  several digits at once
python -m experiments.mocap_eval            # §6.5  motion capture
```

## Installation

Linux, Python 3.9, plus a C++17 compiler and OpenMP for `vamm`.

```bash
conda create -n vamm python=3.9 && conda activate vamm
pip install -r requirements.txt
pip install ./external/vamm      # ships as a copy in external/, builds its C++ extension
```

## Data

MNIST is downloaded on first use. The other two datasets need a one-time download (neither is kept in the repository):

```bash
cd src
python -m data.faces    # ICT-FaceKit meshes -> data/ictfacekit/
python -m data.mocap    # CMU recordings     -> data/cmu_mocap/
```


## Code map

The repository follows the thesis one file per step. `§` refers to the section of the thesis a file implements.

### The pipeline - §5 Method

One module per step, all behind `ManifoldPipeline`.

| File | § | What it does |
| --- | --- | --- |
| `core/pipeline.py` | 5.1 | `ManifoldPipeline`: runs the six steps below on a plain numpy array, either as whole stages or one step at a time. |
| `core/preprocessing.py` | 5.2 | `PCARotation`: PCA down to a workable dimension, followed by a random orthogonal transformation that keeps the subspace but destroys the axis alignment PCA leaves behind. |
| `core/mfa.py` | 5.3 | Fits the MFA (via `vamm`) and reads it geometrically: a thin QR of each loading matrix turns it into an orthonormal tangent frame. |
| `core/graph.py` | 5.4 | Builds the distance matrix: a local metric that penalizes steps off the tangent spaces, then shortest paths in the k-nearest-neighbor graph of those edge lengths. |
| `core/ordering.py` | 5.5 | Orders all charts along the diameter of their minimum spanning tree and decides whether that path closes into a loop. |
| `core/tda.py` | 5.6 | Persistent homology (gudhi) over the distance matrix, giving connected components (H0) and loops (H1) plus a representative cycle for each loop. |
| `core/interpolation.py` | 5.7 | Fits a cubic Hermite spline through the chart means whose knot tangents are softly pulled towards the chart tangents. |

Two small studies justify the preprocessing step:

| File | § | What it does |
| --- | --- | --- |
| `experiments/pca_preprocessing.py` | 5.2 | Sweeps the PCA dimension and measures the held-out negative log-likelihood, with and without the random orthogonal transformation. |
| `experiments/tangent_alignment.py` | 5.2 | The same sweep, but measuring the angle between the estimated and the true manifold tangents. |
| `data/fourier_curve.py` | 5.2 | The synthetic ground truth both studies run on: a closed curve of known dimension, randomly embedded in 300 dimensions. |

### The evaluation - §6 Experiments

`experiments/eval/` is the shared protocol, one script per experiment on top of it.

| File | § | What it does |
| --- | --- | --- |
| `experiments/eval/align.py` | 6.1 | Fits out the freedoms an unsupervised model cannot recover - direction and offset for a loop, an affine map for an arc - so what is left is genuine error. |
| `experiments/eval/metrics.py` | 6.1 | M1 topology (H0 and H1 against the expected ones), M2 residual factor error, M4 adjusted Rand index of the detected components. |
| `experiments/eval/reconstruct.py` | 6.1 | M3: the reconstruction error split into the PCA floor, the MFA floor and the full round trip through the learned curve. |
| `experiments/eval/figures.py` | 6.1 | The plots: persistence diagram, component scatter, residual, and the image / pose strips. |
| `experiments/eval/runner.py` | 6.1 | Runs the metrics and figures for one fitted pipeline and writes a self-contained result directory. |
| `experiments/mnist_rotation_eval.py` | 6.2 | One rotating digit: the only single-structure setting, and therefore the one comparing TDA against the MST baseline, clean against noisy. |
| `experiments/face_factor_eval.py` | 6.3 | Three rendered faces, each sweeping a different factor: three separate components, all of them open arcs. |
| `experiments/multi_factor_eval.py` | 6.4 | Four rotating digits at once: several components with loops and an arc among them. |
| `experiments/mocap_eval.py` | 6.5 | Four motions of one person (non-image data): two periodic motions (loops) and two one-way ones (arcs). |

### Datasets

| File | § | What it does |
| --- | --- | --- |
| `data/mnist_rotation.py` | 6.2, 6.4 | Rotates MNIST digits through a full or partial circle; a full sweep is a loop, a partial one an arc. |
| `data/faces.py` | 6.3 | Renders grayscale faces from the ICT-FaceKit morphable model, sweeping one blendshape or head rotation per face. |
| `data/mocap.py` | 6.5 | Reads CMU motion capture recordings, runs forward kinematics on them and cuts each motion into repetitions along a physical signal. |
| `data/basic_manifolds.py` | - | Toy manifolds with known topology (circle, torus, two circles, linked circles, …), the ground truth of the demos. |
| `data/synthetic.py` | - | Turns a toy manifold name into a point cloud, optionally embedded in a higher-dimensional space with noise. |

### Demo application and plotting

| File | § | What it does |
| --- | --- | --- |
| `main.py`, `demo/main_app.py` | - | Entry point: one window with six tabs. |
| `demo/demo_synthetic.py` | - | Tab *Synthetic Data*: the fitted MFA components drawn over a toy manifold. |
| `demo/demo_manifold_1d.py` | 5.5 | Tab *Loop Detection*: the traversal baseline, stepping through MFA, graph, ordering and spline. |
| `demo/demo_tda.py` | 5.6 | Tab *Topology (TDA)*: components and loops on toy manifolds the baseline cannot handle, next to the persistence diagram. |
| `demo/demo_tab.py` | - | The machinery all three dataset tabs share: rendering thread, camera, settings blocks, evaluation button. |
| `demo/demo_mnist.py` | 6.2, 6.4 | Tab *MNIST Rotation* |
| `demo/demo_faces.py` | 6.3 | Tab *Faces* |
| `demo/demo_mocap.py` | 6.5 | Tab *Motion Capture* |
| `visualization/geometry.py` | - | Drawing of single Gaussian components (ellipse, ellipsoid, line, plane) |
| `visualization/gmm.py` | - | A fitted mixture over its data, in 2D or 3D. |
| `visualization/spline.py` | - | A spline path up to the current coordinate, over the data points. |
| `visualization/graph.py` | - | A traversal order over the chart means. |
| `visualization/mnist.py` | - | The image frames, the 3D PCA view and the persistence diagram of the demo tabs. |
| `visualization/faces.py` | - | The same for faces |
| `visualization/mocap.py` | - | The same for poses |

