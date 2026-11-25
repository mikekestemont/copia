import numpy as np
import pytest

from copia.data import to_copia_dataset
from copia.diagnostics import one_inflation

# Heroin users in Chiang Mai, 2013-2018, treated up to six times; Lerdsuwansri, R. et al.,
# 'Identifying one-inflation using ratio regression in capture-recapture problems', Journal
# of Statistical Computation and Simulation (2025), Table 2.
HEROIN = (537, 152, 80, 34, 15, 8)

# Contacts per index case of Covid-19 in Thailand, 2020 (counts 1 to 20); Böhning, D.,
# Lerdsuwansri, R. & Sangnawakij, P., 'Modeling COVID-19 contact-tracing using the ratio
# regression capture-recapture approach', Biometrics (2023), Table 1.
COVID = (44, 22, 24, 16, 15, 10, 11, 10, 9, 9, 9, 10, 6, 7, 8, 16, 3, 2, 3, 8)

# Homeless people in Utrecht by the number of nights in the shelter, of 14 nights in 2013,
# and perpetrators of domestic violence in the Netherlands, 2009; Böhning, D., 'Ratio plot
# and ratio regression with applications to social and medical sciences', Statistical
# Science (2016), Tables 1 and 2.
HOMELESS = (36, 11, 6, 11, 5, 7, 6, 11, 3, 8, 7, 12, 22, 77)
VIOLENCE = (15169, 1957, 393, 99, 28, 8, 6, 1, 1)

# Submissions per farm in Britain: all submissions (counts 1 to 20; larger counts play no
# role here), and carcasses only; Böhning, D. et al., 'A generalization of Chao's
# estimator for covariate information', Biometrics (2013); data sets farmsubmission and
# carcassubmission of the R package singleRcapture.
FARMS = (6340, 2520, 1149, 709, 380, 249, 173, 135, 94, 80, 52, 26, 30, 23, 8, 8, 14, 5, 7, 5)
CARCASSES = (1532, 231, 51, 27, 6, 5, 2, 1, 3)

# Police records of immigrants without a permit in four Dutch cities, 1995; van der
# Heijden, P.G.M. et al., 'Point and interval estimation of the population size using the
# truncated Poisson regression model', Statistical Modelling (2003).
IMMIGRANTS = (1645, 183, 37, 13, 1, 1)

# Cottontail rabbits captured over 18 nights; Edwards, W.R. & Eberhardt, L.L., Journal of
# Wildlife Management (1967).
RABBITS = (43, 16, 8, 6, 0, 2, 1)

# Yellow-bellied prinia captured on 17 occasions in Hong Kong, 1993; Hwang & Huang (2003),
# as given in Liu, Y. et al., 'Semiparametric empirical likelihood inference for abundance
# from one-inflated capture-recapture data', arXiv:2507.10388, Table 6.
PRINIA = (133, 24, 5, 1, 1)

# Female methamphetamine users in Bangkok, 2001; Böhning, D. et al. (2004), as given in
# Liu et al., Table 6.
DRUG_USERS = (261, 10, 2, 1)


def dataset(frequencies, T=None):
    counts = np.repeat(range(1, len(frequencies) + 1), frequencies)
    return to_copia_dataset(counts, data_type="incidence" if T else "abundance",
                            input_type="counts", n_sampling_units=T)


def test_one_inflation_published():
    """
    The heroin users in Lerdsuwansri et al. (2025), Table 3: the regression without (M0) and
    with (M1) one-inflation, of which M1 is chosen.
    """
    test = one_inflation(dataset(HEROIN))
    assert round(test["excess"], 2) == -0.70
    assert round(test["aic0"], 2) == 5.01 and round(test["bic0"], 2) == 3.84
    assert round(test["aic1"], 2) == -4.37 and round(test["bic1"], 2) == -5.93
    # intercept -0.38 and slope -0.26 for the ratios without the factor x + 1
    assert np.isclose(np.log(test["r1_expected"] / 2), -0.38 - 0.26 * np.log(2), atol=0.01)
    assert test["p"] < 0.05

    # 'there is little evidence for one-inflation' in the Covid data (Böhning et al. 2023)
    test = one_inflation(dataset(COVID), max_count=20)
    assert test["aic0"] < test["aic1"] and test["bic0"] < test["bic1"]
    assert test["p"] > 0.05


@pytest.mark.parametrize("frequencies, r1, r1_expected, f1_expected, z, p", [
    (HEROIN, 0.56611, 1.13677, 267.42459, -2.27752, 0.01138),
    (COVID, 1.00000, 1.68841, 26.06007, -1.27166, 0.10175),
    (VIOLENCE, 0.25803, 0.29077, 13460.87048, -0.85192, 0.19713),
    (FARMS, 0.79495, 0.81571, 6178.65405, -0.46394, 0.32135),
    (CARCASSES, 0.30157, 0.26507, 1742.93038, 0.42123, 0.66321),
    (IMMIGRANTS, 0.22249, 0.25249, 1449.58242, -0.23961, 0.40532),
    (RABBITS, 0.74419, 0.95601, 33.47234, -0.27454, 0.39184),
    (PRINIA, 0.36090, 0.16962, 282.99003, 0.55927, 0.71201),
])
def test_one_inflation(frequencies, r1, r1_expected, f1_expected, z, p):
    """
    Reference values: the same weighted regression with lm() in R,
    log(r) ~ log(x + 1) + I(x == 1), with the standard error from cov.unscaled. Tests
    under a Poisson or binomial distribution do find one-inflation in IMMIGRANTS
    (likelihood ratio 37.8, 1 df, in the vignette of singleRcapture) and PRINIA (p < 0.05
    for both tests in Liu et al.), where the present test attributes the singletons to
    heterogeneity.
    """
    test = one_inflation(dataset(frequencies))
    assert test["f1"] == frequencies[0]
    for name, reference in (("r1", r1), ("r1_expected", r1_expected),
                            ("f1_expected", f1_expected), ("z", z), ("p", p)):
        assert np.isclose(test[name], reference, atol=1e-5)


def test_one_inflation_excess():
    # singletons added to the data of the farms
    for extra, level in ((1000, 0.01), (3000, 1e-9)):
        test = one_inflation(dataset((FARMS[0] + extra,) + FARMS[1:]))
        assert test["p"] < level
        # the number expected does not depend on the singletons
        assert np.isclose(test["f1_expected"], 6178.65405)


def test_one_inflation_incidence():
    # the ratios of the binomial distribution, which are those of the Poisson for a large T
    test = one_inflation(dataset(RABBITS, T=18))
    assert np.isclose(test["r1"], 18 / 17 * 2 * 16 / 43)
    large = one_inflation(dataset(RABBITS, T=10**8))
    assert np.isclose(large["z"], one_inflation(dataset(RABBITS))["z"])

    # with lm() in R, as in test_one_inflation
    test = one_inflation(dataset(HOMELESS, T=14), max_count=14)
    for name, reference in (("r1", 0.65812), ("r1_expected", 0.23520), ("z", 1.88126),
                            ("aic0", 40.71076), ("aic1", 41.70274)):
        assert np.isclose(test[name], reference, atol=1e-5)

    # captures with the same probability on every occasion: no trend, and no excess
    rng = np.random.RandomState(1)
    counts = rng.binomial(10, 0.15, size=20000)
    ds = to_copia_dataset(counts, data_type="incidence", input_type="counts",
                          n_sampling_units=10)
    test = one_inflation(ds)
    assert np.isclose(test["r1"], 10 * 0.15 / 0.85, rtol=0.05)
    assert test["p"] > 0.05


def test_one_inflation_level():
    # the test rejects about 5% of samples without one-inflation at the 5% level
    rng = np.random.RandomState(1)
    for draw in (lambda: rng.poisson(2, 2000), lambda: rng.geometric(0.4, 2000) - 1):
        p = [one_inflation(to_copia_dataset(draw(), input_type="counts"))["p"]
             for _ in range(400)]
        assert 0.02 < np.mean(np.array(p) < 0.05) < 0.08


def test_one_inflation_sparse():
    # too few ratios for a trend
    test = one_inflation(dataset(DRUG_USERS))
    assert np.isclose(test["r1"], 2 * 10 / 261)
    assert np.isnan(test["z"]) and np.isnan(test["p"])
    assert np.isnan(one_inflation(dataset((0, 5, 4, 3, 2, 1)))["r1"])
