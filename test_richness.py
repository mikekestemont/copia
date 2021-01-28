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

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, -1])
    assert np.isclose(diversity.egghe_proot(x, alpha=150), 16.38, rtol=0.001)        
    
