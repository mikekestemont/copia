import pytest
import warnings

import numpy as np
import copia.richness as diversity
import copia.hill as hill
import copia.utils as u



def test_chao1():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    assert np.isclose(diversity.chao1(x), 12.4, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    assert np.isclose(diversity.chao1(x), 12.4, rtol=0.001)

    x = np.array([1, 1, 1, 1, 3, 5, 10, 25])
    assert np.isclose(diversity.chao1(x), 13.8722, rtol=0.001)


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
    assert np.isclose(diversity.chao1(demo_default_data),
                      27.249, rtol=0.001)

    # now remove doubletons:
    demo_no_f2 = np.array([752, 276, 194, 126, 121, 97,
        95, 83, 72, 44, 39, 0, 16, 15, 0, 13, 9, 9,
        9, 8, 7, 4, 0, 0, 1, 1, 1], dtype=np.int64)
    assert np.isclose(diversity.chao1(demo_no_f2),
                      25.998, rtol=0.001)

    
def test_egghe_proot():
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    assert np.isclose(diversity.egghe_proot(x, alpha=150), 16.38, rtol=0.001)

    x = np.array([1, 1, 1, 2, 3, 5, 10, 25, 0, 0])
    assert np.isclose(diversity.egghe_proot(x, alpha=150), 16.38, rtol=0.001)

    # test kwargs:
    x = np.array([1, 1, 1, 2, 3, 5, 10, 25])
    assert diversity.diversity(x, 'egghe_proot', alpha=150) != \
           diversity.diversity(x, 'egghe_proot', alpha=50)

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
            
    x = u.to_abundance(assemblage)

    assert np.isclose(diversity.egghe_proot(x, alpha=200), 3903, rtol=0.001)


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


def test_iChao1():
    x = np.array([1, 1, 1, 6, 3, 3, 5, 5, 5, 5, 8])
    with pytest.warns(UserWarning):
        diversity.iChao1(x)
    
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
            
    x = u.to_abundance(assemblage)

    # check whether the assemblage is correctly constructed:
    counts = u.basic_stats(x)
    assert counts['S'] == 78
    assert counts['n'] == 127
    assert np.isclose(diversity.iChao1(x), 290.9, rtol=0.01)


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
    d = diversity.min_add_sample(x, solver="grid", diagnostics=True)
    assert np.isclose(d['n'] * d['x*'], 166509, rtol=1)

    # fsolve should back off:
    with pytest.warns(UserWarning):
        d = diversity.min_add_sample(x, solver="fsolve",
                                     diagnostics=True)
        print(d)
        assert np.isclose(d['n'] * d['x*'], 166509, rtol=1)


def test_species_accumulation():
    # verify whether species accumulation is stricly monotonic
    spider_girdled = np.array([46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    accumul = diversity.species_accumulation(spider_girdled, max_steps=300)

    a = accumul['richness']
    assert np.all(a[1:] >= a[:-1])

    a = accumul['lci']
    assert np.all(a[1:] >= a[:-1])

    a = accumul['uci']
    assert np.all(a[1:] >= a[:-1])


def test_hill():
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    x = np.array(spider_girdled)

    emp, _ = hill.hill_numbers(x, n_iter=10, q_min=0, q_max=3, step=0.1)

    q = np.arange(0, 3, 0.1)
    # q = 0
    idx = np.where(q == 0)[0][0]
    q0 = emp['richness'][idx]
    assert np.isclose(q0, len(x), rtol=0.001)

    # q = 1
    idx = np.where(q == 1)[0][0]
    q1 = emp['richness'][idx]

    def shannon_entropy(counts, base=np.e):
        freqs = counts / counts.sum()
        freqs = freqs[freqs.nonzero()]
        return -(freqs * np.log(freqs)).sum() / np.log(base)

    assert np.isclose(shannon_entropy(x), np.log(q1), rtol=0.001)

    # q = 2
    idx = np.where(q == 2)[0][0]
    q2 = emp['richness'][idx]

    def simpson_diversity(counts):
        return 1 - ((counts / counts.sum()) ** 2).sum()

    assert np.isclose(1 / (1 - simpson_diversity(x)), q2, rtol=0.1)


def test_shared_richness():
    # Example with folktale collections taken from: 
    # https://web.archive.org/web/20220526135551/https://www.karsdorp.io/posts/20220316142536-two_assemblage_good_turing_estimation/
    # Drachten data collection:
    c1 = np.array([65, 56, 42, 38, 33, 28, 27, 22, 21, 19, 18, 15, 14, 14, 13, 12, 12, 11, 11, 11, 10, 10, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    # Harkema data collection:
    c2 = np.array([123, 84, 39, 56, 45, 26, 16, 46, 36, 18, 12, 26, 15, 15, 13, 19, 8, 19, 6, 18, 8, 17, 20, 4, 23, 27, 31, 14, 0, 18, 30, 14, 22, 3, 0, 10, 9, 2, 12, 7, 5, 2, 15, 17, 19, 0, 12, 2, 9, 24, 0, 3, 3, 11, 2, 1, 0, 2, 1, 16, 2, 2, 2, 0, 8, 3, 0, 3, 10, 4, 0, 6, 2, 1, 0, 8, 0, 0, 0, 1, 1, 1, 3, 5, 11, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 3, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 6, 5, 1, 1, 5, 1, 1, 2, 1, 0, 2, 2, 12, 0, 6, 12, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 7, 6, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    # Both collections are aligned and should have the same length:
    assert len(c1) == len(c2)

    est = diversity.shared_richness(c1, c2)
    print(est)
    assert np.isclose(est['richness'], 244, rtol=0.1)
    assert np.isclose(est['observed shared'], 136, rtol=0.1)
    assert np.isclose(est['f0+'], 26, rtol=0.1)
    assert np.isclose(est['f+0'], 48, rtol=0.1)
    assert np.isclose(est['f00'], 34, rtol=0.1)
