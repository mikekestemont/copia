import pytest
import warnings

import numpy as np
import copia.richness as diversity
import copia.stats
import copia.utils as u


def test_bootstrap_seed():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])

    a = copia.stats.bootstrap(x, diversity.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=2021)
    
    b = copia.stats.bootstrap(x, diversity.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=2021)

    c = copia.stats.bootstrap(x, diversity.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=54656)
    
    d = copia.stats.bootstrap(x, diversity.chao1,
              n_iter=1000, conf=0.95, n_jobs=4,
              seed=54656)

    assert sorted(a['bootstrap']) == sorted(b['bootstrap'])
    assert sorted(c['bootstrap']) == sorted(d['bootstrap'])
    assert sorted(a['bootstrap']) != sorted(d['bootstrap'])
