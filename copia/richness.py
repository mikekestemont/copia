# -*- coding: utf-8 -*-
"""
Bias-correcting richness estimators for abundance data
"""
import warnings

import numpy as np
import scipy.stats
from scipy.optimize import fsolve

from .stats import bootstrap, dbinom


def empirical_richness(x, species=True):
    """
    Empirical species richness of an assemblage

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.

    Returns
    -------
    richness : float
        The empirically observed number of distinct species
    """

    if species:
        return np.count_nonzero(x > 0)
    else:
        return x.sum()


def chao1(x):
    """
    Chao1 estimate of bias-corrected species richness

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.

    Returns
    -------
    richness : float
        Estimate ($\hat{f_0}$) of the bias-corrected species richness:

        $\hat{f_0} = \left\{\begin{aligned}
        \frac{(n - 1)}{n} \frac{f_1^2}{(2f_2)} \qquad if f_2 > 0;\\
        \frac{(n - 1)}{n} \frac{f_1(f_1 - 1)}{2} \qquad if f_2 = 0
        \end{aligned}$

        With:
            - $f_1$ = the number of species sighted exactly once in
            the sample (singletons),
            - $f_2$ = the number of species that were sighted twice
            (doubletons)
            - $n$ = the observed, total sample size.
            - $\hat{f_0}$ = the estimated lower bound for the number
            of species that do exist in the assemblage, but which were
            sighted zero times, i.e. the number of undetected species.       

    References
    ----------
    - A. Chao, 'Non-parametric estimation of the classes in a population',
    Scandinavian Journal of Statistics (1984), 265-270.
    - A. Chao, et al., 'Quantifying sample completeness and comparing
    diversities among assemblages', Ecological research (2020), 292-314.
    """

    x = x[x > 0]
    n = x.sum()
    t = x.shape[0]
    f1 = np.count_nonzero(x == 1)
    f2 = np.count_nonzero(x == 2)

    if f2 > 0:
        return t + (n - 1) / n * (f1 ** 2 / 2 / f2)
    else:
        return f1 * (f1 - 1) / 2


def iChao1(x):
    """
    "Improved" iChao1 estimate of bias-corrected species richness

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.

    Returns
    -------
    richness : float
        The "improved" estimate iChao1, that extends Chao1 to also
        consider f3 ("tripletons") and f4 ("quadrupletons") in assemblages.

    Note
    ----
        We follow the original paper's recommendation to add 1
        to f4, if there are no quadrupletons in the assemblage,
        so that iChao1 is always obtainable.  A user warning will b
        raised in this case.

    References
    -------
    - C.-H. Chiu et al., 'An Improved Nonparametric Lower Bound of
    Species Richness via a Modified Good–Turing Frequency Formula',
    Biometrics (2014), 671–682.
    """

    ch1 = chao1(x)
    f1 = np.count_nonzero(x == 1)
    f2 = np.count_nonzero(x == 2)
    f3 = np.count_nonzero(x == 3)
    f4 = np.count_nonzero(x == 4)

    if f4 == 0:
        warnings.warn("Add-one smoothing for f4 = 0", UserWarning)
        f4 += 1

    iCh1 = ch1 + (f3 / (4 * f4)) * np.max((f1 - ((f2 * f3) / (2 * f4)), 0))
    return iCh1


def egghe_proot(x, alpha=150):
    """
    Egghe & Proot estimate of bias-corrected species richness

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    alpha : int (default = 150)
        An estimate of the average print run

    Returns
    -------
    richness : float
        Estimate ($\hat{f_0}$) of the bias-corrected species richness:

        $\hat{f_0} = \left( \frac{1}{1 + \frac{2f_2}{(a-1)f_1}} \right)^a$

        With:
            - $f_1$ = the number of species sighted exactly once in
            the sample (singletons),
            - $f_2$ = the number of species that were sighted twice
            (doubletons)
            - $\hat{f_0}$ = the estimated number of species that once
            existed in the assemblage, but which were sighted zero times,
            i.e. the number of undetected species.

    Note
    ----
        If no doubletons are available in the samples, we apply add-one-
        smoothing to P2. A user warning will be raised in this case.

    References
    ----------
    - L. Egghe and G. Proot, 'The estimation of the number of lost
    multi-copy documents: A new type of informetrics theory', Journal
    of Informetrics (2007), 257-268.
    - Q.L. Burrell, 'Some comments on "The estimation of lost multi-copy
    documents: A new type of informetrics theory" by Egghe and Proot',
    Journal of Informetrics (2008), 101–105.
    """

    ft = np.bincount(x)[1:]
    S = ft.sum()

    P1 = np.count_nonzero(x == 1)
    P2 = np.count_nonzero(x == 2)

    if P2 == 0:
        warnings.warn("Add-one smoothing for P2 = 0", UserWarning)
        P2 += 1

    P0 = (1 / (1 + (2 / (alpha - 1)) * (P2 / P1))) ** alpha

    S_lost = S * (P0 / (1 - P0))
    S_lost = S + S_lost

    if not np.isinf(S_lost):
        return S_lost
    else:
        return np.nan


def ace(x, k=10):
    """
    ACE estimate of bias-corrected species richness (Chao & Lee 1992)

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    k : int (default = 10)
        The abudance threshold for considering a species
        "rare". Species with counts <= k will be considered
        "rare".

    Note
    ----
        - Regarding k, we follow the recommendation from the
        "EstimateS" package and assume that the upper limit
        for considering a species "rare" is 10 observations.
        - Our implementation mirrors that in the "fossil" R
        package (https://cran.r-project.org/web/packages/fossil).

    Returns
    -------
    richness : float
        Estimate $\hat{S}$ of the bias-corrected species richness.

    References
    ----------
    - A. Chao & S.-M. Lee, 'Estimating the number of classes via
    sample coverage'. Journal of the American Statistical Association
    87 (1992), 210-217.
    - R.K. Colwell & J.E. Elsensohn, 'EstimateS turns 20: statistical
    estimation of species richness and shared species from samples,
    with non-parametric extrapolation', Ecography 37 (2014), 609–613.
    - M.J. Vavrek, 'fossil: palaeoecological and palaeogeographical
    analysis tools', Palaeontologia Electronica 14 (2011), 1T.
    """

    nr = sum(x[x <= k])
    sa = np.count_nonzero(x > k)
    sr = np.count_nonzero(x <= k)
    f1 = np.count_nonzero(x == 1)
    ca = 1 - (f1 / nr)
    sumf = np.sum([i * (x == i).sum() for i in range(1, k + 1)])
    g2a = np.max((sr / ca) * (sumf / (nr * (nr - 1))) - np.array((1.0, 0.0)))
    S = sa + sr / ca + (f1 / ca) * g2a
    return S


def jackknife(x, k=5, return_order=False, return_ci=False, conf=0.95):
    """
    Jackknife estimate of bias-corrected species richness

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    k : int (default = 5)
        Maximum number of orders to consider (0 < k >= 5).
    return_order : bool (default = False)
        Whether to return the selected order for the Jackknife
    return_ci : bool (default = False)
        Whether to return the confidence interval for the Jackknife
    conf : float (default = 0.95)
        Confidence level for the confidence interval (e.g. 0.95).

    Returns
    -------
    richness : float
        Jackknife estimate of the bias-corrected species richness. The Jackknife
        is a general-purpose, resampling method for statistical bias estimation.

    By default, only the richness will be returned. If return_order and/or return_ci
    evaluate to True, a dict will be returned with the appropriate, additional keys:
      - "richness" (always included)
      - "order"
      - "lci" (lower confidence interval)
      - "uci" (upper confidence interval)

    e.g.
        {'richness': 177.0,
        'order': 3,
        'lci': 127.80529442066658,
        'uci': 226.1947055793334}

    Notes
    -------
    This is a literal translation of the reference implementation in the
    [SPECIES package](https://github.com/jipingw/SPECIES/blob/master/R/jackknife.R).

    References
    -------
    - K.P. Burnham and W.S. Overton, 'Estimation of the size of a closed population
    when capture probabilities vary among animals' Biometrika (1978), 625–633.
    - J.-P. Wang, 'SPECIES: An R Package for Species Richness Estimation',
    Journal of Statistical Software (2011), 1-15.
    """

    k0, k = k, min(len(np.unique(x)) - 1, 10)
    n = np.bincount(x)[1:]
    n = np.array((np.arange(1, n.shape[0] + 1), n)).T
    total = n[:, 1].sum()
    gene = np.zeros((k + 1, 5))
    gene[0, 0] = total

    for i in range(1, k + 1):
        gene[i, 0] = total
        gene[i, 3] = total
        for j in range(1, i + 1):
            gene[i, 0] = (
                gene[i, 0] + (-1) ** (j + 1) * 2 ** i * dbinom(j, i, 0.5) * n[j - 1, 1]
            )
            gene[i, 3] = gene[i, 3] + (-1) ** (j + 1) * 2 ** i * dbinom(j, i, 0.5) * n[
                j - 1, 1
            ] * np.prod(np.arange(1, j + 1))
        gene[i, 1] = -gene[i, 0]
        for j in range(1, i + 1):
            gene[i, 1] = (
                gene[i, 1]
                + ((-1) ** (j + 1) * 2 ** i * dbinom(j, i, 0.5) + 1) ** 2 * n[j - 1, 1]
            )
        gene[i, 1] = np.sqrt(gene[i, 1] + n[i:, 1].sum())

    if k > 1:
        for i in range(2, k + 1):
            gene[i - 1, 2] = -((gene[i, 0] - gene[i - 1, 0]) ** 2) / (total - 1)
            for j in range(1, i):
                gene[i - 1, 2] = gene[i - 1, 2] + (
                    (-1) ** (j + 1) * 2 ** (i) * dbinom(j, i, 0.5)
                    - (-1) ** (j + 1) * 2 ** (i - 1) * dbinom(j, i - 1, 0.5)
                ) ** 2 * n[j - 1, 1] * total / (total - 1)
            gene[i - 1, 2] = np.sqrt(gene[i - 1, 2] + n[i - 1, 1] * total / (total - 1))
            gene[i - 1, 4] = (gene[i, 0] - gene[i - 1, 0]) / gene[i - 1, 2]

    coe = scipy.stats.norm().ppf(1 - (1 - conf) / 2)
    x = gene[1 : k + 1, 4] < coe

    if x.sum() == 0:
        jackest = gene[k, 0]
        sej = gene[k, 1]
        order = 1
    else:
        indicator = np.arange(1, k + 1)
        jackest = gene[indicator[x][0], 0]
        sej = gene[indicator[x][0], 1]
        order = np.arange(1, k + 2)[indicator[x][0]] - 1

    if k0 <= order:
        jackest = gene[k0, 0]
        sej = gene[k0, 1]
        order = k0

    if return_order or return_ci:
        d = {"richness": jackest}
        if return_order:
            d["order"] = order
        if return_ci:
            d["lci"] = jackest - coe * sej
            d["uci"] = jackest + coe * sej
        return d
    else:
        return jackest


def min_add_sample(x, solver="grid", search_space=(0, 100, 1e6), tolerance=1e-1):
    """
    Minimum additional sampling estimate (of population size)

    Parameters
    ----------
    x : array-like, with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    solver : str (default = 'grid')
        Solver to find x* = the intersection between h() and v():
            - 'grid': hardcode grid search (less precise, but recommended)
            - 'fsolve': numpy optimization (more precise, less stable in practice)
    search_space : 3-way tuple (default = (0, 100, 1e5))
        Search space to be used in the grid search:
            (start, end, number of samples)
    tolerance : float (default = 1e-2)
        Allowed divergence (from zero) in finding the intersection
        between h() and v().

    Returns
    -------
    estimate : float
        Lower-bound estimate of the minimum additional samples
        (observations) that would have to be taken to observe
        each of the hypothesized species (i.e. $\hat{f_0}$) at
        least once. (In some cases, this number can approximate
        the estimated number of individuals in the original
        population.)

    References
    -------
    - A. Chao et al., 'Sufficient sampling for asymptotic minimum
    species richness estimators', Ecology (2009), 1125-1133.
    - M. Kestemont & F. Karsdorp, 'Estimating the Loss of Medieval
    Literature with an Unseen Species Model from Ecodiversity',
    Computational Humanities Research (2020), 44-55.
    """

    n = x.sum()
    x = x[x > 0]
    f1 = np.count_nonzero(x == 1)
    f2 = np.count_nonzero(x == 2)

    h = lambda x: 2 * f1 * (1 + x)
    v = lambda x: np.exp(x * (2 * f2 / f1))

    if solver == "grid":
        search_space = np.linspace(*[int(i) for i in search_space])
        hs = np.array(h(search_space))
        vs = np.array(v(search_space))
        diffs = np.abs(hs - vs)
        x_ast = search_space[diffs.argmin()]

    elif solver == "fsolve":

        def intersection(func1, func2, x0):
            return fsolve(lambda x: func1(x) - func2(x), x0)[0]

        x_ast = intersection(h, v, n)

    else:
        raise ValueError(f'Unsupported "solver" argument: {solver}')

    diff_intersect = abs(h(x_ast) - v(x_ast))
    if not diff_intersect < tolerance:
        warnings.warn(f"Tolerance criterion not met: {diff_intersect} > {tolerance}")

    return n * x_ast


estimators = {
    "empirical": empirical_richness,
    "chao1": chao1,
    "ichao1": iChao1,
    "egghe_proot": egghe_proot,
    "jackknife": jackknife,
    "minsample": min_add_sample,
    "ace": ace,
}


def diversity(
    x, method=None, CI=False, conf=0.95, n_iter=1000, n_jobs=1, seed=None, **kwargs):
    """
    Wrapper for various bias-corrected richness functions

    Parameters
    ----------
    x : array-like, with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    method : str (default = None)
        One estimator of:
            - 'chao1'
            - 'egghe_proot'
            - 'jackknife'
            - 'minsample'
            - 'empirical' (same as None)
    **kwargs : additional parameters passed to selected method

    Returns
    -------
    Consult the documentation of selected method.
    """

    x = np.array(x, dtype=np.int64)

    if (x < 0).any():
        msg = "Elements of `x` should be strictly non-negative"
        raise ValueError(msg)

    if x.sum() <= 0:
        msg = "`x` appears to be empty"
        raise ValueError(msg)

    if method is not None and method.lower() not in estimators:
        raise ValueError(f"Unknown estimation method `{method}`.")

    if method is None:
        method = "empirical"

    if CI:
        estimate = bootstrap(
            x, fn=estimators[method.lower()], n_iter=n_iter, n_jobs=n_jobs, seed=seed
        )
    else:
        estimate = estimators[method.lower()](x, **kwargs)

    return estimate
