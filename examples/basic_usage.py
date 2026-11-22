"""Simple, finite sample clustering example without ground-truth labels."""
import numpy as np
from cera import CERA, predict_cera

rng = np.random.default_rng(42)
X = np.r_[rng.normal(size=(180, 8)), rng.normal(size=(20, 8)) + np.array([7, 0, 0, 0, 0, 0, 0, 0])]
model = CERA().fit(X)
print("Observed clusters:", model.n_clusters_)
print("First 12 labels:", model.labels_[:12])
print("Diagnostic branch:", model.diagnostics_.get('branch', 'unknown'))
assert np.array_equal(model.labels_, predict_cera(X))
