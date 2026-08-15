# Learning Data Densities and Data Coordinate Systems for Disentangled Representations

Bachelor thesis by **Felix Campidell**, supervised by **Till Kahlke** and **Jörg Lücke**.
Universität Innsbruck, Department of Computer Science.

A mixture of factor analyzers model is fitted to the data, so every component becomes
a local chart with its own tangent frame. The charts are linked by a tangent-aware
distance, persistent homology reads the connected components and loops off that
distance graph, and each detected structure is interpolated into a spline. That
spline is a coordinate system for the data: moving along it traverses one
generative factor (a rotation angle, an expression, the progress of a motion),
learned without labels.

This repository holds the code and the evaluation runs.

## Pipeline

One module per step, all behind `ManifoldPipeline` in `src/core/pipeline.py`:

| Module | Step |
| --- | --- |
| `core/preprocessing.py` | PCA followed by a random orthogonal transformation |
| `core/mfa.py` | fit the MFA (via `vamm`), extract the tangent frames |
| `core/graph.py` | tangent-aware distance matrix between the charts |
| `core/tda.py` | persistent homology: components (H0) and loops (H1) |
| `core/ordering.py` | baseline ordering by Minimal Spanning Tree diameter |
| `core/interpolation.py` | cubic Hermite splines through the chart means |

```python
from core.pipeline import ManifoldPipeline

pipe = ManifoldPipeline(n_components=100, pca_dim=50, detection="tda")
result = pipe.fit_detect(X)      # curves + splines
t, curve_id = pipe.transform(X)  # the learned coordinate per observation
```

Detection yields at least one curve per connected component, plus a loop
wherever one was found. `result["curves"]` lists them, each with its `type`
(`"loop"` or `"path"`), the H0 `component` it belongs to, the `order` of chart
means along it and the `spline`.

`transform` is the encoder. It assigns every observation to the curve whose
spline runs closest to it.


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


## Experiments

The offline evaluations behind the thesis results run from the demo ("Run
evaluation") or standalone, and write to `results/<experiment>/<tag>/`
(`summary.json`, `arrays.npz`, figures). The defaults reproduce the runs already
in `results/`:

```bash
cd src
python -m experiments.mnist_rotation_eval   # single rotating digit, TDA vs. baseline
python -m experiments.multi_factor_eval     # several digits at once
python -m experiments.face_factor_eval      # rendered faces
python -m experiments.mocap_eval            # motion capture
```


## Installation

Linux, Python 3.9, plus a C++17 compiler and OpenMP for `vamm`.

```bash
conda create -n vamm python=3.9 && conda activate vamm
pip install numpy scipy matplotlib dearpygui gudhi torchvision
pip install ./external/vamm
```


## Data

MNIST is downloaded on first use. The other two datasets need a one-time download
(neither is kept in the repository):

```bash
cd src
python -m data.faces    # ICT-FaceKit meshes -> data/ictfacekit/
python -m data.mocap    # CMU recordings     -> data/cmu_mocap/
```
