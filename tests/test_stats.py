import pytest
import warnings

import numpy as np
from copia.data import to_copia_dataset
import copia.estimators
from copia.bootstrap import bootstrap_abundance_data


def test_bootstrap_seed():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")

    a = bootstrap_abundance_data(ds, copia.estimators.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=2021)
    
    b = bootstrap_abundance_data(ds, copia.estimators.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=2021)

    c = bootstrap_abundance_data(ds, copia.estimators.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=54656)
    
    d = bootstrap_abundance_data(ds, copia.estimators.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=54656)

    assert np.isclose(a['est'], b['est'], rtol=0.001)
    assert np.isclose(a['uci'], b['uci'], rtol=0.001)
    assert np.isclose(a['lci'], b['lci'], rtol=0.001)

    assert np.isclose(c['est'], d['est'], rtol=0.001)
    assert np.isclose(c['uci'], d['uci'], rtol=0.001)
    assert np.isclose(c['lci'], d['lci'], rtol=0.001)


COUNTS = np.array([88, 22, 16, 15, 13, 10, 8, 7, 5, 4, 3, 2, 2, 2] + [1] * 14)


def chao1_even(ds):
    # fails on the bootstrap samples with an odd number of singletons
    if ds.f1 % 2:
        raise ZeroDivisionError
    return copia.estimators.chao1(ds)


def data_only(ds):
    # fails on every bootstrap sample
    if not np.array_equal(ds.counts, COUNTS):
        raise ValueError
    return copia.estimators.chao1(ds)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_bootstrap_failures(n_jobs):
    ds = to_copia_dataset(COUNTS, input_type="counts")
    kwargs = dict(n_iter=50, n_jobs=n_jobs, seed=1, disable_pb=True)
    # failed bootstrap samples are reported, not silently dropped
    with pytest.warns(UserWarning, match="of 50 tasks failed"):
        bootstrap_abundance_data(ds, chao1_even, **kwargs)
    with pytest.raises(ValueError):
        bootstrap_abundance_data(ds, data_only, **kwargs)


def test_bootstrap_jobs():
    # the same result in this process (any function) and in worker processes
    ds = to_copia_dataset(COUNTS, input_type="counts")
    kwargs = dict(n_iter=50, seed=1, disable_pb=True)
    one = bootstrap_abundance_data(ds, lambda d: copia.estimators.chao1(d), n_jobs=1, **kwargs)
    two = bootstrap_abundance_data(ds, copia.estimators.chao1, n_jobs=2, **kwargs)
    assert (one == two).all()
