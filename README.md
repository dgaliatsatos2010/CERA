# CERA — Holdout-Checked Projection Screening for Minority-Cluster Recovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24441562.svg)](https://doi.org/10.5281/zenodo.24441562)

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
manuscript terminology. The [GitHub v0.1.0 release](https://github.com/dgaliatsatos2010/CERA/releases/tag/v0.1.0)
was published on 2026-11-22 and archived via Zenodo.
The public archive DOI is [10.5281/zenodo.24441562](https://doi.org/10.5281/zenodo.24441562).
The full 96-dataset benchmark replay with the public package remains outstanding.

## Citation and DOI

Please cite the Zenodo archive: **[10.5281/zenodo.24441562](https://doi.org/10.5281/zenodo.24441562)**.
The machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).
The archive currently covers the source code and documentation, not the complete
96-dataset benchmark or all third-party comparison artifacts.

## Licensing

CERA is released under the **MIT License**. See [LICENSE](LICENSE).
This license applies to the CERA software in this repository; externally sourced
datasets and third-party implementations retain their respective licenses.

## Authors

Dimitrios Galiatsatos · Hellenic Open University, Patras, Greece.
