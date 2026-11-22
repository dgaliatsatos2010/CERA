# CERA — Holdout-Checked Projection Screening for Minority-Cluster Recovery

CERA is an **exploratory, fully unsupervised clustering method** that combines
conservative density/graph clustering with robust feature-unit canonicalization
and holdout-checked projection screening for selected minority-cluster settings.

**Research status:** experimental. There is **no calibrated false-split rate,
no formal family-wise error guarantee, and no demonstrated universal superiority**
over modern clustering approaches. Results vary substantially by cluster geometry.

## Quick installation

```bash
python -m pip install -e ".[test]"
```

Python 3.11 or newer is recommended. This repository currently includes the source code and documentation only; archived datasets and full benchmark outputs are not yet hosted here. A future PyPI release is **not yet claimed**.

## Quick start

```python
import numpy as np
from cera import CERA, predict_cera

rng = np.random.default_rng(42)
X = np.r_[rng.normal(size=(180, 8)), rng.normal(size=(20, 8)) + np.array([7, 0, 0, 0, 0, 0, 0, 0])]
labels = CERA().fit_predict(X)
print("Predicted groups:", len(set(labels) - {-1}))
print("Labels:", labels[:10])

# Direct, functional API
assert np.array_equal(labels, predict_cera(X))
```

Cluster label `-1` indicates unassigned/noise points when produced by an underlying
clustering stage. Output labels are **transductive**, not inductive predictions.

## Reproduce the archived CERA results

**The small source-only download does not include the 96 archived datasets or reference metrics.**
The archived full benchmark data and reference metrics have not yet been uploaded to GitHub.

```bash
pytest -q  # API unit tests; full-data replay requires separately archived data
```

The replay computes adjusted Rand index (ARI), adjusted mutual information (AMI),
cluster counts, and coverage. The original synthetic label arrays are used
**only to evaluate** the outputs, never in clustering or feature selection.
Dataset filenames and family IDs are not consumed by the predictor.

See [Methods](docs/METHODS.md) and [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md).
The manuscript's full third-party baseline comparisons and negative findings
remain separately reported in its supplementary materials.

## Reproducibility and versioning

The current Python package version is `0.1.0`, independently of
manuscript terminology. A GitHub release and Zenodo DOI are not yet published. Commit SHA and archived release DOI should be reported
in `Code and Data Availability` after the repository is published and archived.

## Citation and DOI

See [`CITATION.cff`](CITATION.cff). **A Zenodo DOI has not yet been minted.**
Do not add a placeholder or invented DOI to a manuscript.

## Licensing

CERA is released under the **MIT License**. See [LICENSE](LICENSE).
This license applies to the CERA software in this repository; externally sourced
datasets and third-party implementations retain their respective licenses.

## Authors

Dimitrios Galiatsatos · Hellenic Open University, Patras, Greece.
