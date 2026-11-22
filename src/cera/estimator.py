"""Scikit-learn compatible, transductive interface to CERA."""
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from .algorithm import predict_cera
from .graph_mst import kcount


class CERA(ClusterMixin, BaseEstimator):
    """Cluster the rows of a numeric feature matrix.

    Unsupervised and transductive: fitted labels describe the supplied rows;
    the estimator does not implement out-of-sample label prediction.

    Valid input is finite with at least 33 rows (see full method limitations).
    """

    def fit(self, X, y=None):
        labels, diagnostics = predict_cera(X, return_info=True)
        self.labels_ = np.asarray(labels, dtype=int)
        self.n_clusters_ = int(kcount(self.labels_))
        self.diagnostics_ = diagnostics
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_.copy()
