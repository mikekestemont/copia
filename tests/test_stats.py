import pytest
import warnings

import numpy as np
import copia.richness as diversity
import copia.stats


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

    assert np.isclose(sorted(a['bootstrap']), sorted(b['bootstrap']),
                      rtol=0.001).all()
    assert np.isclose(a['richness'], b['richness'], rtol=0.001)
    assert np.isclose(a['uci'], b['uci'], rtol=0.001)
    assert np.isclose(a['lci'], b['lci'], rtol=0.001)
    assert np.isclose(a['std'], b['std'], rtol=0.001)

    assert np.isclose(sorted(c['bootstrap']), sorted(d['bootstrap']),
                      rtol=0.001).all()
    assert np.isclose(c['richness'], d['richness'], rtol=0.001)
    assert np.isclose(c['uci'], d['uci'], rtol=0.001)
    assert np.isclose(c['lci'], d['lci'], rtol=0.001)
    assert np.isclose(c['std'], d['std'], rtol=0.001)
