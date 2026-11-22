# CERA — method description

The implementation has a conservative density/graph-based baseline with a
holdout-checked rescue step for under-clustered partitions. Features are
robustly transformed using a 15th–85th-percentile scale. The exploratory
projection stage ranks up to eight features by absolute training-set kurtosis
and fits one-, two-, or three-component univariate Gaussian mixtures.

Data are split deterministically into **65% screening / 35% validation**.
Candidate acceptance requires prescribed improvements in BIC, separation,
component weights, density-valley shape, and support in the holdout sample.
The candidate is used only under the algorithm's internal underclustering
rule; otherwise the conservative base partition is retained.

The thresholds are **heuristic**, not a hypothesis-test calibration.
Sample splitting reduces selection/validation reuse, but does not imply
family-wise type-I error control. This algorithm should not be described as
universally novel or superior from the reported experiments alone.

The source modules are separated by role rather than development chronology:
`density_base.py`, `graph_mst.py`, `graph_coarsening.py`,
`projection_gate.py`, `scale_screen.py`, and `algorithm.py`.

Inputs must contain finite numeric features, with at least 33 observations.
Rescue-specific checks apply more restrictive size/dimensionality criteria.
