# GeoDynamics

Implementation of **“GeoDynamics: A Geometric State-Space Neural Network for
Understanding Brain Dynamics on Riemannian Manifolds.”**


## Installation

```bash
python -m pip install -e .
```

## Data

Place them under `data/` or pass their paths on
the command line. See [data/README.md](data/README.md).

| Dataset | Evaluation |
|---|---|
| HCP-WM, ADNI, OASIS, PPMI, ABIDE | stratified 10-fold |
| Florence 3D Actions | leave-one-actor-out |
| UTKinect | leave-one-actor-out; requires `--actor-path` |
| HDM05 | 10 random stratified half splits |

## Training

All datasets use one entry point:

```bash
geodynamics-train --help
```

ADNI example:

```bash
geodynamics-train \
  --dataset ADNI \
  --input-dir data/ADNI/timeseries \
  --label-path data/ADNI/labels.csv \
  --adni-label-scheme five-class \
  --output-dir outputs/adni
```

Florence configuration:

```bash
geodynamics-train \
  --dataset F3D \
  --input-dir data/HAR/F3D.mat \
  --output-dir outputs/florence_actor1 \
  --florence-source paper \
  --readout final \
  --feature-normalization none \
  --vectorization full \
  --window-size 15 --kernel-size 5 --tau 1 \
  --stein-iterations 1 --learning-rate 5e-4 \
  --batch-size 16 --epochs 100 --fold-index 1 --device cuda:0
```

## Code

| File | Purpose |
|---|---|
| `geodynamics.py` | SPD operators, SPA, geometric state layers, classifier |
| `dataset.py` | HBC and HAR dataset loaders |
| `train_geodynamics.py` | training and dataset protocols |
| `summarize_folds.py` | fold validation and aggregation |


## License and citation

Code is released under Apache-2.0. Dataset licenses are separate. Citation
metadata is provided in [CITATION.cff](CITATION.cff).
