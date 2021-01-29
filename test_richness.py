import pytest
import warnings

import numpy as np
import copia.richness as diversity
import copia.utils as u


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


def test_minsample():
    # example from the appendix to the original paper:
    # ref: https://figshare.com/articles/dataset/Supplement_1_Excel-sheet_calculator_and_calculator_instructions_/3530930?backTo=/collections/Sufficient_sampling_for_asymptotic_minimum_species_richness_estimators/3300935
    # Characteristics of the tested assemblage:
    # f1 = 3, f2 = 2, n = 161, S_obs = 9, S_est = 11.250, x* = 2.221115367
    # m = nx* = 357.600 (for g=1)

    # mimic an assemblage with these properties:
    assemblage = ['A', 'B', 'C']
    assemblage += ['D'] * 2 + ['E'] * 2
    assemblage += ['F'] * 3 + ['G'] * 3 + ['H'] * 3
    assemblage += ['I'] * (161 - len(assemblage))
    x = u.to_abundance(assemblage)

    # check whether the assemblage is correctly constructed:
    counts = u.basic_stats(x)
    assert counts['f1'] == 3
    assert counts['f2'] == 2
    assert counts['S'] == 9
    assert counts['n'] == 161

    d = diversity.diversity(x, 'minsample', solver='grid', diagnostics=True)
    assert np.isclose(d['x*'], 2.221115367, rtol=0.01)
    assert np.isclose(d['richness'], 161 + 357.600, rtol=1)

    d = diversity.diversity(x, 'minsample', solver='fsolve', diagnostics=True)
    assert np.isclose(d['x*'], 2.221115367, rtol=0.01)
    assert np.isclose(d['richness'], 161 + 357.600, rtol=1)


def test_spider():
    # test chao1 on spider data (both girdled and logged):
    # https://cran.r-project.org/web/packages/iNEXT/vignettes/Introduction.html
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_girdled)
    assert np.isclose(diversity.chao1(x), 43.893, rtol=0.001)

    spider_logged = [88, 22, 16, 15, 13, 10, 8, 8, 7, 7, 7, 5,
                     4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_logged)
    assert np.isclose(diversity.chao1(x), 61.403, rtol=0.001)


    

    


