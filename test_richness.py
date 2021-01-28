import pytest
import warnings

import numpy as np
import copia.richness as diversity


def test_chao1():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    assert np.isclose(diversity.chao1(x), 12.4, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    assert np.isclose(diversity.chao1(x), 12.4, rtol=0.001)    

    x = np.array([1, 1, 1, 1, 3, 5, 10, 25])
    assert np.isclose(diversity.chao1(x), 6)

    x = np.array([1, 1, 1, 1, -1, 0, 10, 25])
    assert np.isclose(diversity.chao1(x), 6)

    
def test_egghe_proot():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    assert np.isclose(diversity.egghe_proot(x, alpha=150), 16.38, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    assert np.isclose(diversity.egghe_proot(x, alpha=150), 16.38, rtol=0.001)


def test_nonnegative_counts():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, -1])
    with pytest.raises(ValueError):
        diversity.diversity(x)


def test_empty_counts():
    x = np.array([0, 0, 0])
    with pytest.raises(ValueError):
        diversity.diversity(x)


def test_egghe_proot_missing_p2():
    x = np.array([1, 1, 1, 1, 3, 5, 10, 25])
    with pytest.warns(UserWarning):
        diversity.egghe_proot(x)


def test_iChao4_f4():
    x = np.array([1, 1, 1, 6, 3, 3, 5, 5, 5, 5, 8])
    with pytest.warns(UserWarning):
        diversity.iChao1(x)
