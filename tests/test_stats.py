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

