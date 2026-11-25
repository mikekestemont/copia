"""
Estimating the size of a population from several incomplete lists of its members
(capture-recapture, or multiple systems estimation)
"""
import warnings

import numpy as np
import pandas as pd

from copia.data import ListData, accepts
from copia.stats import lognormal_ci
from copia.utils import check_random_state


@accepts(ListData)
def lincoln_petersen(ds: ListData, lists=None, chapman=True, CI=False, conf=0.95):
    r"""
    Lincoln-Petersen estimate of the size of a population from two lists

    Parameters
    ----------
    ds : ListData
        The items on two or more lists.
    lists : tuple (default = None)
        The names of the two lists to use. Required if there are more than two.
    chapman : bool (default = True)
        Whether to apply the correction of Chapman (1951), which reduces the bias
        in small samples.
    CI : bool (default = False)
        Whether to return the standard error and the confidence interval.
    conf : float (default = 0.95)
        Confidence level for the confidence interval.

    Returns
    -------
    richness : float
        With :math:`n_1` and :math:`n_2` items on the lists and :math:`m` on both:

        .. math::
            \hat{N} = \left\{\begin{aligned}
            \frac{(n_1 + 1)(n_2 + 1)}{m + 1} - 1 \qquad \text{(Chapman)};\\
            \frac{n_1 n_2}{m} \qquad \text{(Lincoln-Petersen)}
            \end{aligned}\right.

    If CI evaluates to True, a dict with the keys "est", "se", "lci" and "uci". The
    standard error and the log-normal interval (Chao 1987) are those of Chapman's
    estimate.

    Note
    ----
    The estimate rests on the independence of the lists: it is too low if items on
    one list are more likely to be on the other (see `dependence`).

    References
    ----------
    - D.G. Chapman, 'Some properties of the hypergeometric distribution with
      applications to zoological censuses', University of California Publications
      in Statistics (1951), 131-160.
    - G.A.F. Seber, The Estimation of Animal Abundance (1982), 59-60.
    - A. Chao, 'Estimating the population size for capture-recapture data with
      unequal catchability', Biometrics (1987), 783-791.
    """
    if lists is None:
        if ds.T > 2:
            raise ValueError(f"choose two of the lists {ds.lists}")
        lists = ds.lists
    i, j = (ds.lists.index(name) for name in lists)
    on_i, on_j = ds.patterns[:, i] == 1, ds.patterns[:, j] == 1
    n1, n2, m = (int(ds.freqs[on].sum()) for on in (on_i, on_j, on_i & on_j))

    est_chapman = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    if chapman:
        est = est_chapman
    else:
        est = n1 * n2 / m if m > 0 else np.inf
    if not CI:
        return est

    se = np.sqrt((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2)))
    lci, uci = lognormal_ci(n1 + n2 - m, est_chapman, se, conf)
    return {"est": est, "se": se, "lci": lci, "uci": uci}


def _care1(freqs, patterns, variant):
    """CARE-1 estimates for one or more (rows) sets of numbers of items per pattern."""
    t = patterns.shape[1]
    n_lists = patterns.sum(axis=1, keepdims=True)           # lists of each pattern
    i, j = np.triu_indices(t, 1)                            # the pairs of lists
    in_pair = patterns[:, i] + patterns[:, j]

    n = freqs @ patterns                                    # items on each list
    single = freqs @ (patterns * (n_lists == 1))            # items on one list only
    both = freqs @ (patterns[:, i] * patterns[:, j])        # items on both lists of a pair
    A = freqs @ (in_pair * (in_pair == n_lists))            # items on no list but these two

    M = freqs.sum(axis=-1)
    D = M - single.mean(axis=-1)
    C = 1 - (single / n).mean(axis=-1)                      # sample coverage
    overlap = both / (n[..., i] * n[..., j])

    if variant == "independent":
        return D / C

    if variant == "low":
        N = D / C
        for _ in range(2):      # preliminary and final estimate of the dependences
            N = D / C + (A * (N[..., None] * overlap - 1)).sum(axis=-1) / (t * C)
        return N

    numerator = t * D - A.sum(axis=-1)                      # 0 without any overlap
    N = numerator / (t * C - (A * overlap).sum(axis=-1))
    return np.where(numerator > 0, np.maximum(N, M), M)


def _bootstrap(ds, variant, n_iter, seed):
    """Numbers of items per pattern, in samples from the estimated population."""
    N = max(round(float(_care1(ds.freqs, ds.patterns, variant))), ds.S_obs)
    p = np.append(N - ds.S_obs, ds.freqs) / N
    return check_random_state(seed).multinomial(N, p, size=n_iter)[:, 1:]


@accepts(ListData)
def list_coverage(ds: ListData):
    r"""
    Sample coverage of the lists: 1 minus the average proportion of the items of a
    list that are on no other list (Chao & Tsay 1998)
    """
    single = ds.freqs @ (ds.patterns * (ds.patterns.sum(axis=1, keepdims=True) == 1))
    return 1 - (single / (ds.freqs @ ds.patterns)).mean()


@accepts(ListData)
def care1(ds: ListData, variant="high", CI=False, conf=0.95, n_iter=1000, seed=None):
    r"""
    Estimates of the size of a population from three or more lists, based on the
    sample coverage of the lists (Chao & Tsay 1998; the program CARE-1)

    Parameters
    ----------
    ds : ListData
        The items on three or more lists.
    variant : str (default = 'high')
        One of the estimators:
            - 'high' (:math:`\hat{N}`): for a sufficiently high sample coverage.
            - 'low' (:math:`\hat{N}_1`): one-step estimator for a low sample
              coverage, and a lower (upper) bound if the lists are positively
              (negatively) dependent. It is recommended if the coverage is below
              55%, or if the standard error of :math:`\hat{N}` is more than one
              third of the estimate.
            - 'independent' (:math:`\hat{N}_0`): for independent lists.
    CI : bool (default = False)
        Whether to return the standard error and the confidence interval.
    conf : float (default = 0.95)
        Confidence level for the confidence interval.
    n_iter : int (default = 1000)
        Number of bootstrap samples for the standard error.
    seed : int (default = None)
        A seed to initialize the random number generator.

    Returns
    -------
    richness : float
        With :math:`M` items in all, :math:`n_j` items on list :math:`j`, of which
        :math:`s_j` on that list only, :math:`n_{jk}` items on both lists of a pair,
        and :math:`A_{jk}` the items on :math:`j`, on :math:`k`, or on both (counted
        twice) that are on no other list:

        .. math::
            D = M - \frac{1}{t} \sum_j s_j, \qquad
            \hat{C} = 1 - \frac{1}{t} \sum_j \frac{s_j}{n_j}, \qquad
            \hat{N}_0 = D / \hat{C}

        .. math::
            \hat{N} = \hat{N}_0 + \frac{1}{t\hat{C}} \sum_{j<k} A_{jk}
            \hat{\gamma}_{jk}, \qquad \hat{\gamma}_{jk} =
            \hat{N} \frac{n_{jk}}{n_j n_k} - 1

        solved for :math:`\hat{N}`. The one-step estimator :math:`\hat{N}_1` takes
        the dependences :math:`\hat{\gamma}_{jk}` from a preliminary estimate, which
        has those of :math:`\hat{N}_0`.

    If CI evaluates to True, a dict with the keys "est", "se", "lci" and "uci". The
    standard error is that of the estimates on samples from the estimated
    population, the interval is log-normal (Chao 1987).

    Note
    ----
    The formulas are equations 7 to 13 (three lists) and 14 to 17 (any number) in
    Chao et al. (2001). The bootstrap is that of the R package CARE1 1.1.0. With three
    lists, its standard errors of :math:`\hat{N}` and :math:`\hat{N}_1` are not those
    printed in Chao et al. (2001), e.g. 48 and not 40 for :math:`\hat{N}_1` in their
    first example.

    References
    ----------
    - A. Chao & P.K. Tsay, 'A sample coverage approach to multiple-system estimation
      with application to census undercount', Journal of the American Statistical
      Association (1998), 283-293.
    - A. Chao et al., 'The applications of capture-recapture models to
      epidemiological data', Statistics in Medicine (2001), 3123-3157.
    """
    _check(ds, variant)
    if variant == "high" and list_coverage(ds) < 0.55:
        warnings.warn("The sample coverage of the lists is below 55%: consider "
                      "variant='low'", UserWarning)

    est = float(_care1(ds.freqs, ds.patterns, variant))
    if not CI:
        return est

    samples = _bootstrap(ds, variant, n_iter, seed)
    se = _care1(samples, ds.patterns, variant).std(ddof=1)
    lci, uci = lognormal_ci(ds.S_obs, est, se, conf)
    return {"est": est, "se": se, "lci": lci, "uci": uci}


@accepts(ListData)
def dependence(ds: ListData, variant="high", CI=False, conf=0.95, n_iter=1000, seed=None):
    r"""
    Dependence between the lists, as implied by a CARE-1 estimate

    Parameters
    ----------
    ds : ListData
        The items on three or more lists.
    variant : str (default = 'high')
        The CARE-1 estimator used for the size of the population (see `care1`).
    CI : bool (default = False)
        Whether to return confidence intervals (percentiles of the bootstrap).
    conf : float (default = 0.95)
        Confidence level for the confidence intervals.
    n_iter : int (default = 1000)
        Number of bootstrap samples.
    seed : int (default = None)
        A seed to initialize the random number generator.

    Returns
    -------
    dependence : pd.DataFrame
        For each pair of lists the coefficient of covariation (Chao & Tsay 1998),
        :math:`\hat{\gamma}_{jk} = \hat{N} n_{jk} / (n_j n_k) - 1` ("est"; "lci" and
        "uci" if CI evaluates to True): how far the estimate :math:`\hat{N}` is above
        the Lincoln-Petersen estimate of the pair. It is 0 for independent lists, and
        positive if items on one list are more likely to be on the other.
    """
    _check(ds, variant)
    i, j = np.triu_indices(ds.T, 1)

    def gamma(freqs):
        n, both = freqs @ ds.patterns, freqs @ (ds.patterns[:, i] * ds.patterns[:, j])
        N = np.asarray(_care1(freqs, ds.patterns, variant))
        return N[..., None] * both / (n[..., i] * n[..., j]) - 1

    pairs = pd.MultiIndex.from_tuples([(ds.lists[a], ds.lists[b]) for a, b in zip(i, j)])
    out = pd.DataFrame({"est": gamma(ds.freqs)}, index=pairs)
    if CI:
        samples = gamma(_bootstrap(ds, variant, n_iter, seed))
        out["lci"], out["uci"] = np.quantile(
            samples, [(1 - conf) / 2, (1 + conf) / 2], axis=0)
    return out


def _check(ds, variant):
    if variant not in ("high", "low", "independent"):
        raise ValueError(f"Unknown variant of CARE-1: `{variant}`")
    if ds.T < 3:
        raise ValueError("CARE-1 needs at least three lists")
    if ds.f1 == ds.S_obs:
        raise ValueError("no item is on more than one list")


__all__ = ['lincoln_petersen', 'care1', 'dependence', 'list_coverage']
