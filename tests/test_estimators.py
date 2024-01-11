import pytest
import warnings

import numpy as np
from copia.data import to_copia_dataset
import copia.estimators
import copia.diversity
import copia.stats
import copia.rarefaction_extrapolation
import copia.utils as u



def test_chao1():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 12.4, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 12.4, rtol=0.001)

    x = np.array([1, 1, 1, 1, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 13.8722, rtol=0.001)


def test_chao1_f2():
    """
    Test chao1 implementation against SpadeR shiny-app,
    spefically for the storm condition where f2 = 0, i.e.
    no doubletons are available in the assemblage. See
    https://chao.shinyapps.io/SpadeR/ <24 Oct 2021>

    """
    demo_default_data = np.array([752, 276, 194, 126, 121, 97,
        95, 83, 72, 44, 39, 0, 16, 15, 0, 13, 9, 9,
        9, 8, 7, 4, 0, 0, 2, 2, 1, 1, 1], dtype=np.int64)
    ds = to_copia_dataset(demo_default_data, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 27.249, rtol=0.001)

    # now remove doubletons:
    demo_no_f2 = np.array([752, 276, 194, 126, 121, 97,
        95, 83, 72, 44, 39, 0, 16, 15, 0, 13, 9, 9,
        9, 8, 7, 4, 0, 0, 1, 1, 1], dtype=np.int64)
    ds = to_copia_dataset(demo_no_f2, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 25.998, rtol=0.001)
    
def test_egghe_proot():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.egghe_proot(ds, alpha=150), 16.38, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.egghe_proot(ds, alpha=150), 16.38, rtol=0.001)

    # test kwargs:
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")
    assert copia.estimators.diversity(ds, method='egghe_proot', alpha=150) != \
           copia.estimators.diversity(ds, method='egghe_proot', alpha=50)

    # test against example from paper (pp. 260ff):
    assemblage = [f'{i}-1' for i in range(714)]
    for i in range(82):
        for j in range(2):
            assemblage.append(f'{i}-2')
    for i in range(3):
        for j in range(4):
            assemblage.append(f'{i}-3')
    for i in range(4):
        for j in range(3):
            assemblage.append(f'{i}-4')
    for i in range(1):
        for j in range(5):
            assemblage.append(f'{i}-5')
            
    x = to_copia_dataset(assemblage, input_type="observations")

    assert np.isclose(copia.estimators.egghe_proot(x, alpha=200), 3903, rtol=0.001)


def test_egghe_proot_missing_p2():
    x = np.array([1, 1, 1, 1, 3, 5, 10, 25])
    ds = to_copia_dataset(x, input_type="counts")
    with pytest.warns(UserWarning):
        copia.estimators.egghe_proot(ds)


def test_iChao1():
    x = np.array([1, 1, 1, 6, 3, 3, 5, 5, 5, 5, 8])
    ds = to_copia_dataset(x, input_type="counts")
    with pytest.warns(UserWarning):
        copia.estimators.iChao1(ds)
    
    # mimic "daytime beetle assemblage" from original paper
    # from Janzen (1973a, 1973b); p. 678:
    assemblage = [f'{i}-1' for i in range(59)]
    for i in range(9):
        for j in range(2):
            assemblage.append(f'{i}-2')
    for i in range(3):
        for j in range(3):
            assemblage.append(f'{i}-3')
    for i in range(2):
        for j in range(4):
            assemblage.append(f'{i}-4')
    for i in range(2):
        for j in range(5):
            assemblage.append(f'{i}-5')   
    for i in range(2):
        for j in range(6):
            assemblage.append(f'{i}-6')
    for i in range(1):
        for j in range(11):
            assemblage.append(f'{i}-11')
            
    ds = to_copia_dataset(assemblage)

    # check whether the assemblage is correctly constructed:

    assert ds.S_obs == 78
    assert ds.n == 127
    assert np.isclose(copia.estimators.iChao1(ds), 290.9, rtol=0.01)


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
    ds = to_copia_dataset(assemblage)

    # check whether the assemblage is correctly constructed:
    assert ds.f1 == 3
    assert ds.f2 == 2
    assert ds.S_obs == 9
    assert ds.n == 161

    d = copia.estimators.diversity(ds, method='minsample', solver='grid', diagnostics=True)
    assert np.isclose(d['x*'], 2.221115367, rtol=0.01)
    assert np.isclose(d['richness'], 161 + 357.600, rtol=1)

    d = copia.estimators.diversity(ds, method='minsample', solver='fsolve', diagnostics=True)
    assert np.isclose(d['x*'], 2.221115367, rtol=0.01)
    assert np.isclose(d['richness'], 161 + 357.600, rtol=1)


def test_spider():
    # test chao1 on spider data (both girdled and logged):
    # https://cran.r-project.org/web/packages/iNEXT/vignettes/Introduction.html
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_girdled)
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 43.893, rtol=0.001)

    spider_logged = [88, 22, 16, 15, 13, 10, 8, 8, 7, 7, 7, 5,
                     4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_logged)
    ds = to_copia_dataset(x, input_type="counts")
    assert np.isclose(copia.estimators.chao1(ds), 61.403, rtol=0.001)


def test_moths():
    # Moth data derived from Table 3 in Fisher et al. (1943); source:
    # https://github.com/piLaboratory/sads/blob/master/data/moths.rda
    # minsample estimate (`nx*`): first line in Table 1 in Chao et al. (2009)
    # for g = 1: nx* = 166509
    moths = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10,
    10, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 13, 14, 14, 15, 15, 15,
    15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 
    21, 22, 22, 22, 23, 23, 23, 24, 25, 25, 25, 26, 27, 28, 28, 28, 29,
    29, 32, 34, 34, 36, 36, 36, 37, 37, 43, 43, 44, 44, 45, 49, 49, 49,
    51, 51, 51, 51, 52, 53, 54, 54, 57, 58, 58, 60, 60, 60, 61, 64, 67,
    73, 76, 76, 78, 84, 89, 96, 99, 109, 112, 120, 122, 129, 135, 141,
    148, 149, 151, 154, 177, 181, 187, 190, 199, 211, 221, 226, 235, 239,
    244, 246, 282, 305, 306, 333, 464, 560, 572, 589, 604, 743, 823, 2349]
    x = np.array(moths)

    # grid solver
    ds = to_copia_dataset(x, input_type="counts")
    d = copia.estimators.min_add_sample(ds, solver="grid", diagnostics=True)
    assert np.isclose(d['n'] * d['x*'], 166509, rtol=1)


def test_species_accumulation():
    # verify whether species accumulation is stricly monotonic
    spider_girdled = np.array([46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    ds = to_copia_dataset(spider_girdled, input_type="counts")
    accumul = copia.rarefaction_extrapolation.species_accumulation(
        ds, max_steps=300)

    a = accumul['est']
    print(a)
    assert np.all(a.values[1:] >= a.values[:-1])

    a = accumul['lci']
    assert np.all(a.values[1:] >= a.values[:-1])

    a = accumul['uci']
    assert np.all(a.values[1:] >= a.values[:-1])


def test_hill():
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_girdled)
    ds = to_copia_dataset(x, input_type="counts")

    emp = copia.diversity.compute_hill_numbers(
        ds, n_iter=10, q_min=0, q_max=3, steps=100)

    q = np.linspace(0, 3, 100)
    # q = 0
    idx = np.where(q == 0)[0][0]
    q0 = emp[idx]
    assert np.isclose(q0, len(x), rtol=0.001)

    # q = 1
    idx = np.where(q == 1)[0][0]
    q1 = emp[idx]

    def shannon_entropy(counts, base=np.e):
        freqs = counts / counts.sum()
        freqs = freqs[freqs.nonzero()]
        return -(freqs * np.log(freqs)).sum() / np.log(base)

    print(shannon_entropy(ds.counts), np.log(q1))
    assert np.isclose(shannon_entropy(ds.counts), np.log(q1), rtol=0.001)

    # q = 2
    idx = np.where(q == 2)[0][0]
    q2 = emp[idx]

    def simpson_diversity(counts):
        return 1 - ((counts / counts.sum()) ** 2).sum()

    assert np.isclose(1 / (1 - simpson_diversity(ds.counts)), q2, rtol=0.1)


