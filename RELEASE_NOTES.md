# CERA 0.1.0 — initial research software release

CERA is an experimental, unsupervised clustering research implementation
with holdout-checked projection screening for selected minority-cluster settings.

## Included
- Installable Python package with the `CERA` estimator and `predict_cera` API
- Documented algorithm components and basic usage example
- Automated API unit tests and GitHub Actions CI
- `CITATION.cff` metadata for software citation
- MIT License

## Installation

```bash
git clone https://github.com/dgaliatsatos2010/CERA.git
cd CERA
python -m pip install -e ".[test]"
python -m pytest -q
```

## Research and reproducibility notes

This software is **experimental**. There is no formal family-wise
false-split error guarantee and no established universal superiority
over other clustering methods. Performance depends on data geometry.

This first GitHub release contains **code and documentation only**.
The 96 archived synthetic benchmark datasets, complete result tables,
and third-party benchmark artifacts are **not bundled** in this release.
Selected local consistency tests were completed, but full 96-dataset
replay using this public package remains outstanding.

## Citation

The initial release was archived through Zenodo. Its DOI is
[10.5281/zenodo.24441562](https://doi.org/10.5281/zenodo.24441562).
See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.
