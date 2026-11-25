import numpy as np
import pandas as pd
import pytest

import copia.estimators
from copia.data import ListData, to_copia_dataset
from copia.lists import care1, dependence, lincoln_petersen, list_coverage


def published(*freqs):
    """Numbers of items per pattern, in the order 0...01, 0...10, 0...11, ..., 1...11."""
    t = int(np.log2(len(freqs) + 1))
    return {format(i, f"0{t}b"): n for i, n in enumerate(freqs, start=1)}


# The data in Chao, A. et al., 'The applications of capture-recapture models to
# epidemiological data', Statistics in Medicine (2001), Tables II and V:

# Hepatitis A in a college in Taiwan, 1995: serum tests (P), hospital records (Q) and
# epidemiologists (E); Chao, D. et al., Epidemiology Bulletin (1997); data set HAV of
# the R package CARE1. A later screening put the number of cases at about 545.
HAV = published(63, 55, 18, 69, 17, 21, 28)

# Spina bifida: birth certificates (B), death certificates (D), rehabilitation files (M);
# Hook, Albright & Cross (1980).
SPINA_BIFIDA = published(60, 49, 4, 247, 112, 142, 12)

# Diabetes in a community in Italy, four lists; Bruno, G. et al., 'National diabetes
# programs', Diabetes Care 17 (1994), 548; data set Diabetes of CARE1.
DIABETES = published(10, 182, 8, 74, 7, 20, 14, 709, 12, 650, 46, 104, 18, 157, 58)

# Infants with a congenital anomaly in Massachusetts, five lists; Wittes, J.T. et al.,
# 'Capture-recapture methods for assessing the completeness of case ascertainment',
# Journal of Chronic Diseases 27 (1974), 25; data set Congenital of CARE1.
CONGENITAL = published(83, 97, 30, 4, 3, 2, 0, 37, 34, 37, 23, 1, 0, 3, 0, 27, 36, 22,
                       5, 4, 5, 1, 3, 19, 18, 25, 8, 1, 2, 5, 2)

# Microgastrinae wasps in Costa Rica: reared, core and peripheral Malaise traps;
# Colwell, R.K. et al., 'Constructing a lower-bound estimate of the global number of
# insect species on a hyperdiverse empirical foundation', PNAS 123 (2026),
# e2524283123, Fig. 2.
WASPS = published(246, 132, 147, 666, 114, 40, 69)


def lists(patterns):
    return to_copia_dataset(patterns, data_type="lists", input_type="counts")


def test_list_data():
    # the same data in every format
    ds = lists(HAV)
    assert isinstance(ds, ListData)
    assert (ds.S_obs, ds.T, ds.n) == (271, 3, 135 + 122 + 126)

    table = pd.DataFrame([[int(p) for p in pattern] + [n] for pattern, n in HAV.items()],
                         columns=["P", "Q", "E", "cases"])
    rows = table.loc[table.index.repeat(table["cases"]), ["P", "Q", "E"]].reset_index(drop=True)
    rows.loc[len(rows)] = [0, 0, 0]             # a case that is on no list is left out
    pairs = [(i, name) for i, row in rows.iterrows() for name in "PQE" if row[name]]

    for data, input_type, names in (
            ({tuple(map(int, p)): n for p, n in HAV.items()}, "counts", (1, 2, 3)),
            (pd.Series(HAV), "counts", (1, 2, 3)),
            (table, "counts", ("P", "Q", "E")),
            (rows, "observation_matrix", ("P", "Q", "E")),
            (rows.values, "observation_matrix", (1, 2, 3)),
            (pairs, "observation_list", ("E", "P", "Q")),
            (pd.DataFrame(pairs, columns=["case", "source"]), "observation_list", ("E", "P", "Q"))):
        other = to_copia_dataset(data, data_type="lists", input_type=input_type,
                                 index_column="case", location_column="source",
                                 count_column="cases")
        assert other.lists == names
        assert other.S_obs == 271
        assert np.isclose(care1(other, variant="low"), care1(ds, variant="low"))

    with pytest.raises(ValueError):
        lists({"1": 5})
    with pytest.raises(ValueError):
        lists({"10": 5, "01": 3, "11": -1})


def test_lists_as_incidence_data():
    # Chao2, with the lists as sampling units
    ds = lists(HAV)
    assert np.isclose(copia.estimators.chao1(ds), 271 + 2 / 3 * 187**2 / (2 * 56))


def test_lincoln_petersen():
    """
    Reference values: estN.pair in the R package CARE1 1.1.0.
    """
    ds = lists(HAV)
    for pair, petersen, reference in (((1, 2), 336.1224, (333.5600, 28.6997, 288.6791, 403.4078)),
                                      ((1, 3), 378.0000, (374.4783, 35.5818, 318.6127, 460.7587)),
                                      ((2, 3), 334.1739, (331.3617, 29.9287, 284.6868, 404.3836))):
        estimate = lincoln_petersen(ds, lists=pair, CI=True)
        assert np.allclose([estimate[k] for k in ("est", "se", "lci", "uci")], reference, atol=1e-4)
        assert np.isclose(lincoln_petersen(ds, lists=pair, chapman=False), petersen, atol=1e-4)

    # two of the three lists: no need to choose
    two = lists({"11": 49, "10": 86, "01": 73})
    assert np.isclose(lincoln_petersen(two), 333.5600, atol=1e-4)
    estimate = copia.estimators.diversity(two, method="lincoln_petersen", CI=True, conf=0.9)
    assert 288.6791 < estimate["lci"] < estimate["est"] < estimate["uci"] < 403.4078

    with pytest.raises(ValueError):
        lincoln_petersen(ds)


@pytest.mark.filterwarnings("ignore:The sample coverage")
@pytest.mark.parametrize("data, coverage, independent, high, low", [
    (HAV, 0.513, 407.0037897, 970.8047636, 507.7719955),
    (DIABETES, 0.803, 2271.688912, 2609.02984, 2457.968685),
    (CONGENITAL, 0.774, 629.5419453, 659.4523515, 649.3473626),
])
def test_care1(data, coverage, independent, high, low):
    """
    Reference values: estN in the R package CARE1 1.1.0 (Indep, HSC and LSC).
    """
    ds = lists(data)
    assert np.isclose(list_coverage(ds), coverage, atol=5e-4)
    assert np.isclose(care1(ds, variant="independent"), independent)
    assert np.isclose(care1(ds, variant="high"), high)
    assert np.isclose(care1(ds, variant="low"), low)


@pytest.mark.filterwarnings("ignore:The sample coverage")
@pytest.mark.parametrize("data, coverage, estimates, dependences", [
    (HAV, 0.513, (407, 971, 508),
     ((0.21, 0.08, 0.22), (1.89, 1.57, 1.91), (0.51, 0.34, 0.52))),
    (SPINA_BIFIDA, 0.654, (775, 752, 767),
     ((0.12, 0.00, -0.68), (0.09, -0.03, -0.69), (0.11, -0.01, -0.68))),
    (DIABETES, 0.803, (2272, 2609, 2458),
     ((-0.03, 0.04, 0.00, 0.10, 1.82), (0.11, 0.19, 0.15, 0.27, 2.24),
      (0.04, 0.12, 0.09, 0.19, 2.05))),
    (CONGENITAL, 0.774, (630, 659, 649),
     ((0.28, 1.20, -0.07, 0.08, 0.14, 0.15, 0.01, 0.06, 0.04, -0.33),
      (0.34, 1.30, -0.03, 0.13, 0.19, 0.20, 0.06, 0.11, 0.09, -0.29),
      (0.32, 1.27, -0.04, 0.11, 0.17, 0.18, 0.04, 0.10, 0.07, -0.30))),
])
def test_care1_2001(data, coverage, estimates, dependences):
    """
    The output of CARE-1 printed in Chao et al. (2001): sample coverage, N-hat-0, N-hat and
    N-hat-1, and the dependences they imply, in the order of the pairs. (The last pair of
    the diabetes data is cut off there.)
    """
    ds = lists(data)
    assert round(list_coverage(ds), 3) == coverage
    for variant, estimate, gammas in zip(("independent", "high", "low"), estimates, dependences):
        assert round(care1(ds, variant=variant)) == estimate
        assert np.allclose(dependence(ds, variant=variant)["est"][:len(gammas)], gammas, atol=0.0051)


def test_pairs_2001():
    """
    Petersen and Chapman estimates for the spina bifida data, with standard errors and
    intervals, in Chao et al. (2001).
    """
    ds = lists(SPINA_BIFIDA)
    for pair, petersen, *chapman in (((1, 2), 690, 689, 23, 651, 743),
                                     ((1, 3), 778, 776, 35, 718, 857),
                                     ((2, 3), 2432, 2311, 498, 1554, 3556)):
        estimate = lincoln_petersen(ds, lists=pair, CI=True)
        assert [round(estimate[k]) for k in ("est", "se", "lci", "uci")] == chapman
        assert round(lincoln_petersen(ds, lists=pair, chapman=False)) == petersen


@pytest.mark.filterwarnings("ignore:The sample coverage")
def test_care1_wasps():
    """
    N-hat-1 = 2,394 (2,221-2,604) and N-hat = 3,405 for the wasps, as published.
    """
    ds = lists(WASPS)
    assert ds.S_obs == 1414
    assert round(care1(ds)) == 3405
    estimate = care1(ds, variant="low", CI=True, n_iter=5000, seed=1)
    assert round(estimate["est"]) == 2394
    assert np.isclose(estimate["lci"], 2221, rtol=0.01)
    assert np.isclose(estimate["uci"], 2604, rtol=0.01)


def test_care1_interval():
    """
    Reference values: estN with se=TRUE in CARE1 1.1.0 (from a bootstrap with 500
    samples, hence the tolerance).
    """
    ds = lists(DIABETES)
    for variant, se, lci, uci in (("independent", 23.9, 2230, 2324), ("high", 82.1, 2471, 2795),
                                  ("low", 49.8, 2372, 2568)):
        estimate = care1(ds, variant=variant, CI=True, n_iter=5000, seed=1)
        assert np.isclose(estimate["se"], se, rtol=0.08)
        assert np.isclose(estimate["lci"], lci, rtol=0.01)
        assert np.isclose(estimate["uci"], uci, rtol=0.01)
        # through the wrapper, with the same seed
        assert copia.estimators.diversity(ds, method="care1", CI=True, variant=variant,
                                          n_iter=5000, seed=1) == estimate
    assert copia.estimators.diversity(ds, method="care1") == care1(ds)


def test_dependence():
    """
    Reference values: estN.para in the R package CARE1 1.1.0.
    """
    ds = lists(HAV)
    assert np.allclose(dependence(ds)["est"], [1.88825, 1.56827, 1.90509], atol=1e-5)
    assert np.allclose(dependence(ds, variant="low")["est"], [0.51068, 0.34331, 0.51948], atol=1e-5)

    estimate = dependence(lists(DIABETES), variant="low", CI=True, n_iter=500, seed=1)
    reference = {(1, 2): 0.04481, (1, 3): 0.12478, (2, 3): 0.19300,
                 (1, 4): 0.08544, (2, 4): 2.04904, (3, 4): 0.57727}
    assert np.allclose(estimate["est"][list(reference)], list(reference.values()), atol=1e-5)
    assert (estimate["lci"] < estimate["est"]).all() and (estimate["est"] < estimate["uci"]).all()

    # how far an estimate is above the Lincoln-Petersen estimates of the pairs
    for (i, j), gamma in dependence(ds, variant="low")["est"].items():
        petersen = lincoln_petersen(ds, lists=(i, j), chapman=False)
        assert np.isclose(care1(ds, variant="low") / petersen - 1, gamma)


def test_care1_data():
    with pytest.warns(UserWarning, match="below 55%"):
        care1(lists(HAV))
    with pytest.raises(ValueError):
        care1(lists({"11": 49, "10": 86, "01": 73}))            # two lists
    with pytest.raises(ValueError):
        care1(lists({"100": 5, "010": 8, "001": 3}))            # no overlap
    with pytest.raises(ValueError):
        care1(lists(DIABETES), variant="unknown")
    with pytest.raises(TypeError):
        care1(to_copia_dataset(np.array([1, 1, 2, 5]), input_type="counts"))
