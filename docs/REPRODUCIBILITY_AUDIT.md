# Reproducibility audit of the public code package

Checked locally on 2026-11-16 with Python 3.13, NumPy 2.3.5,
SciPy 1.17.0, and scikit-learn 1.8.0.

- All **96/96** archived synthetic input files passed SHA-256 checks.
- Clean public package code vs archived implementation: **8/8 exact matching label vectors**
  on selected cases (six signal, two null families at n=240).
- Recorded metrics replay: **3/3** selected datasets matched ARI, AMI,
  number of groups and coverage at absolute tolerance 1e-10.
- Package installation succeeded via `pip install --no-deps --no-build-isolation -e .`.
- Automated API tests: **4/4 passed**.
- Source example completed successfully.
- The attempted **full 96-dataset prediction replay did not complete** within
  the available execution limit. This limitation must not be mistaken for a pass.
- GitHub Actions ran the API test matrix successfully on 2026-11-22
  (see https://github.com/dgaliatsatos2010/CERA/actions/runs/45024022666).
  This CI pass does not constitute a full 96-dataset benchmark replay.

These checks confirm packaging functionality on selected cases, not a claim of
statistical validity, algorithmic novelty, or broad empirical superiority.
