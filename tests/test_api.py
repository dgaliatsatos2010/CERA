import numpy as np
import pytest
from cera import CERA, predict_cera


def test_direct_function_matches_estimator():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(180, 8))
    assert np.array_equal(CERA().fit_predict(X), predict_cera(X))


def test_deterministic():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(190, 6))
    assert np.array_equal(predict_cera(X), predict_cera(X))


def test_rejects_nonfinite():
    with pytest.raises(ValueError):
        predict_cera(np.full((190, 8), np.nan))


def test_estimator_attributes():
    X = np.random.default_rng(91).normal(size=(190, 8))
    model = CERA().fit(X)
    assert model.labels_.shape == (190,)
    assert 0 <= model.n_clusters_ <= 190
    assert model.n_features_in_ == 8
