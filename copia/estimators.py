"""
Bias-correcting richness estimators for abundance data
"""
import warnings
from functools import partial

import numpy as np
import scipy.stats

from scipy.optimize import fsolve
from copia.bootstrap import bootstrap_abundance_data, bootstrap_incidence_data
from copia.data import AbundanceData, IncidenceData

import copia.stats as stats


def chao1(ds: AbundanceData):
    r"""
    Chao1 estimate of bias-corrected species richness.
    Formulas taken from Chao & Jost (2012), p. 2538.

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.

    Returns
    -------
    richness : float
        Estimate of the bias-corrected species richness (:math:`S + \hat{f_0}`) with:

        .. math::
            \hat{f_0} = \left\{\begin{aligned}
            \frac{(n - 1)}{n} \frac{f_1^2}{(2f_2)} \qquad \text{if } f_2 > 0;\\
            \frac{(n - 1)}{n} \frac{f_1(f_1 - 1)}{2} \qquad \text{if } f_2 = 0
            \end{aligned}\right.

        With:
            - :math:`f_1` = the number of species sighted exactly once in
              the sample (singletons),
            - :math:`f_2` = the number of species that were sighted twice
              (doubletons)
            - :math:`n` = the observed, total sample size.
            - :math:`S` = the observed number of distinct species.
            - :math:`\hat{f_0}` = the estimated lower bound for the number
              of species that do exist in the assemblage, but which were
              sighted zero times, i.e. the number of undetected species.       

    References
    ----------
    - A. Chao, 'Non-parametric estimation of the classes in a population',
      Scandinavian Journal of Statistics (1984), 265-270.
    - A. Chao & Jost, 'Coverage-based rarefaction and extrapolation:
      standardizing samples by completeness rather than size', Ecology (2012),
      2533–2547.
    """

    f1, f2, n, S_obs = ds.f1, ds.f2, ds.n, ds.S_obs
    if f2 > 0:
        S_est = S_obs + (n - 1) / n * (f1**2 / (2 * f2))
    else:
        S_est = S_obs + (n - 1) / n * f1 * (f1 - 1) / 2 * (f2 + 1)
    return S_est


def iChao1(ds: AbundanceData):
    r"""
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
        so that iChao1 is always obtainable. A user warning will be
        raised in this case.

    References
    -------
    - C.-H. Chiu et al., 'An Improved Nonparametric Lower Bound of
      Species Richness via a Modified Good–Turing Frequency Formula',
      Biometrics (2014), 671–682.
    """

    ch1 = chao1(ds)
    f1, f2, counts = ds.f1, ds.f2, ds.counts
    f3 = np.count_nonzero(counts == 3)
    f4 = np.count_nonzero(counts == 4)

    if f4 == 0:
        warnings.warn("Add-one smoothing for f4 = 0", UserWarning)
        f4 += 1

    iCh1 = ch1 + (f3 / (4 * f4)) * np.max((f1 - ((f2 * f3) / (2 * f4)), 0))
    return iCh1


def egghe_proot(ds: AbundanceData, alpha=150):
    r"""
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
        Estimate of the bias-corrected species richness (:math:`S + \hat{f_0}`) with:

        .. math::
           \hat{f_0} = \left( \frac{1}{1 + \frac{2f_2}{(a-1)f_1}} \right)^a

        With:
            - :math:`f_1` = the number of species sighted exactly once in
              the sample (singletons),
            - :math:`f_2` = the number of species that were sighted twice
              (doubletons)
            - :math:`S` = the observed number of distinct species.
            - :math:`\hat{f_0}` = the estimated number of species that once
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

    ft = np.bincount(ds.counts)[1:]
    S = ft.sum()

    f1, f2 = ds.f1, ds.f2
    if f2 == 0:
        warnings.warn("Add-one smoothing for f2 = 0", UserWarning)
        f2 += 1

    f0 = (1 / (1 + (2 / (alpha - 1)) * (f2 / f1))) ** alpha

    S_lost = S * (f0 / (1 - f0))
    S_lost = S + S_lost

    return S_lost if not np.isinf(S_lost) else np.nan


def ace(ds: AbundanceData, k=10):
    r"""
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
        Estimate :math:`\hat{S}` of the bias-corrected species richness.

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

    x, f1 = ds.counts, ds.f1
    nr = sum(x[x <= k])
    sa = np.count_nonzero(x > k)
    sr = np.count_nonzero(x <= k)
    ca = 1 - (f1 / nr)
    sumf = np.sum([i * (x == i).sum() for i in range(1, k + 1)])
    g2a = np.max((sr / ca) * (sumf / (nr * (nr - 1))) - np.array((1.0, 0.0)))
    S = sa + sr / ca + (f1 / ca) * g2a
    return S


def jackknife(ds: AbundanceData, k=5, CI=False, conf=0.95):
    r"""
    Jackknife estimate of bias-corrected species richness

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    k : int (default = 5)
        Maximum number of orders to consider (0 < k >= 5).
    CI : bool (default = False)
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
        {'est': 177.0,
         'obs': 110,
         'lci': 127.80529442066658,
         'uci': 226.1947055793334}

    Note
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

    x = ds.counts
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
                gene[i, 0] + (-1) ** (j + 1) * 2 ** i * stats.dbinom(j, i, 0.5) * n[j - 1, 1]
            )
            gene[i, 3] = gene[i, 3] + (-1) ** (j + 1) * 2 ** i * stats.dbinom(j, i, 0.5) * n[
                j - 1, 1
            ] * np.prod(np.arange(1, j + 1))
        gene[i, 1] = -gene[i, 0]
        for j in range(1, i + 1):
            gene[i, 1] = (
                gene[i, 1]
                + ((-1) ** (j + 1) * 2 ** i * stats.dbinom(j, i, 0.5) + 1) ** 2 * n[j - 1, 1]
            )
        gene[i, 1] = np.sqrt(gene[i, 1] + n[i:, 1].sum())

    if k > 1:
        for i in range(2, k + 1):
            gene[i - 1, 2] = -((gene[i, 0] - gene[i - 1, 0]) ** 2) / (total - 1)
            for j in range(1, i):
                gene[i - 1, 2] = gene[i - 1, 2] + (
                    (-1) ** (j + 1) * 2 ** (i) * stats.dbinom(j, i, 0.5)
                    - (-1) ** (j + 1) * 2 ** (i - 1) * stats.dbinom(j, i - 1, 0.5)
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

    d = jackest
    if CI:
        d = {"est": d}
        d["lci"] = jackest - coe * sej
        d["uci"] = jackest + coe * sej
    return d


def chao_wor(ds: AbundanceData, q, CI=0.95):
    # Convert the int64 to longs to ensure the numbers
    # don't get too big in the variance calculation
    n, f1, f2, S_obs = ds.n, ds.f1, ds.f2, ds.S_obs
    w = n / (n - 1)
    r = q / (1 - q)
    f0 = (f1 ** 2) / (2 * w * f2 + r * f1)

    # compute sd
    t1 = ((2 * w * f2 * (f0**2) + (f1**2) * f0) ** 2) / (f1 ** 5)
    t2 = (4 * (w**2) * f2) * ((f0 / f1)**4)
    var = f0 + t1 + t2

    z = abs(scipy.stats.norm.ppf((1 - CI) / 2))
    K = np.exp(z * np.sqrt(np.log(1 + var / f0 ** 2)))
    lci, uci = S_obs + f0 / K, S_obs + f0 * K
    return {
        "est": S_obs + f0,
        "lci": lci,
        "uci": uci,
    }


def shared_richness(ds1: AbundanceData, ds2: AbundanceData, CI=False):
    r"""
    Estimate (shared) unseen species in two assemblages

    Parameters
    ----------
    s1 : 1D Numpy array representing the observed counts for
        each individual species in the *first* assemblage.
        (Should have the same length as `s2`.)
    s2 : 1D Numpy array representing the observed counts for
        each individual species in the *second* assemblage.
        (Should have the same length as `s1`.)
    CI : bool (default = False)
        Whether to return the confidence interval for the estimates
    conf : float (default = 0.95)
        Confidence level for the confidence interval (e.g. 0.95).

    Returns
    -------
    results : dict
        Results dictionary, with the following fields:
        - "richness": the estimated total number of species
                      across both assemblages                      
        - "observed shared": the observed number of shared 
                      species across both assemblages
        - "f0+": the number of unseen species unobserved,
                      missing in `s1`, but present in `s2`
        - "f+0": the number of unseen species unobserved,
                      missing in `s2`, but present in `s1`
        - "f00": the number of species unobserved and
                      and missing from both `s1` and s2`

    References
    -------
    - Chao, Anne, et al. 2017. 'Deciphering the Enigma of Undetected
      Species, Phylogenetic, and Functional Diversity Based on Good-Turing
      Theory.' Ecology (2017), 2914-2929.
    - Code taken from: Karsdorp, F, 'Estimating Unseen Shared Cultural Diversity' (2022).
      https://web.archive.org/web/20220526135551/https://www.karsdorp.io/\
      posts/20220316142536-two_assemblage_good_turing_estimation/
    """

    assert len(ds1.counts) == len(ds2.counts)
    if not CI:
        n1, n2 = ds1.n, ds2.n
        s1_counts, s2_counts = ds1.counts, ds2.counts

        # Compute f_{0, +}
        f1p = ((s1_counts == 1) & (s2_counts >= 1)).sum()
        f2p = ((s1_counts == 2) & (s2_counts >= 1)).sum()
        f0p = ((n1 - 1) / n1) * ((f1p ** 2) / (2 * f2p))

        # Compute f_{+, 0}
        fp1 = ((s1_counts >= 1) & (s2_counts == 1)).sum()
        fp2 = ((s1_counts >= 1) & (s2_counts == 2)).sum()
        fp0 = ((n2 - 1) / n2) * ((fp1 ** 2) / (2 * fp2))

        # Compute f_{0, 0}
        f11 = ((s1_counts == 1) & (s2_counts == 1)).sum()
        f22 = ((s1_counts == 2) & (s2_counts == 2)).sum()
        f00 = ((n1 - 1) / n1) * ((n2 - 1) / n2) * ((f11 ** 2) / (4 * f22))

        obs_shared = ((s1_counts > 0) & (s2_counts > 0)).sum()
        S = obs_shared + f0p + fp0 + f00

        return {
            "est": round(S),
            "obs": obs_shared,
        }
    else:
        raise NotImplementedError('No CI available yet for this estimator.')


def min_add_sample(ds: AbundanceData, solver="grid", search_space=(0, 100, 1e6),
                   tolerance=1e-1, diagnostics=False):
    r"""
    Observed population size added to the minimum additional sampling estimate
    (~ original population size)

    Parameters
    ----------
    x : array-like, with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    solver : str (default = 'grid')
        Solver to find x* = the intersection between h() and v():
            - 'grid': hardcode grid search (slower, but recommended)
            - 'fsolve': scipy optimization (faster, but less stable in practice)
    search_space : 3-way tuple (default = (0, 100, 1e5))
        Search space to be used in the grid search:
            (start, end, number of samples)
    tolerance : float (default = 1e-2)
        Allowed divergence (from zero) in finding the intersection
        between h() and v()
    diagnostics : bool (default = False)
            If True, a diagnostics dict is returned with the keys
            "richness", "x*", "n".

    Returns
    -------
    estimate : float
        :math:`n + m (= nx*)`
        Observed :math:`n + m`, i.e. lower-bound estimate of the minimum
        additional samples (observations) that would have to be taken
        to observe each of the hypothesized species (i.e. :math:`\hat{f_0}`) at
        least once. (In some cases, this number can approximate
        the estimated number of individuals in the original
        population.) We only implement the case :math:`g = 1`.

    Note
    -------
    If the "fsolve" solver fails, the function will automatically back
    off to the "grid". A user warning will be raised in this case.

    References
    ----------
    - A. Chao et al., 'Sufficient sampling for asymptotic minimum
      species richness estimators', Ecology (2009), 1125-1133.
    - M. Kestemont & F. Karsdorp, 'Estimating the Loss of Medieval
      Literature with an Unseen Species Model from Ecodiversity',
      Computational Humanities Research (2020), 44-55.
    """

    if solver not in ('grid', 'fsolve'):
        raise ValueError(f'Unsupported "solver" argument: {solver}')

    n, f1, f2 = ds.n, ds.f1, ds.f2

    h = lambda x: 2 * f1 * (1 + x)
    v = lambda x: np.exp(x * (2 * f2 / f1))

    if solver == "fsolve":
        def intersection(func1, func2, x0):
            return fsolve(lambda x: func1(x) - func2(x), x0)[0]
        x_ast = intersection(h, v, n)

        # check result
        diff_intersect = abs(h(x_ast) - v(x_ast))
        if diff_intersect > tolerance:
            print('Diff_intersect:', diff_intersect)
            msg = f"Tolerance criterion not met via fsolve: {diff_intersect} > {tolerance}"
            msg += "-> backing off to grid-solver."
            warnings.warn(msg)
            solver = "grid" # set for back-off

    elif solver == "grid":
        start, end, num = search_space
        search = np.linspace(start, end, num=int(num))
        hs = np.array(h(search))
        vs = np.array(v(search))
        diffs = np.abs(hs - vs)
        x_ast = search[diffs.argmin()]

    else:
        raise ValueError("Solver must be either 'grid or fsolve'.")

    # check result
    diff_intersect = abs(h(x_ast) - v(x_ast))
    if not diff_intersect < tolerance:
        warnings.warn(f"Tolerance criterion not met: {diff_intersect} > {tolerance}")
        
    if x_ast <= 0:
        warnings.warn(f"Optimization failure likely: {x_ast} <= 0")
    
    m = n * x_ast

    if diagnostics:
        return {'richness': n + m, 'x*': x_ast, 'n': n}
    else:
        return n + m


ESTIMATORS = {
    "chao1": chao1,
    "ichao1": iChao1,
    "egghe_proot": egghe_proot,
    "jackknife": jackknife,
    "minsample": min_add_sample,
    "ace": ace,
    "shared_richness": shared_richness,
}


def diversity(
        ds: AbundanceData, ds2: AbundanceData=None, method=None, CI=False,
        conf=0.95, n_iter=1000, n_jobs=1, seed=None, disable_pb=False, **kwargs):
    r"""
    Wrapper for various bias-corrected richness functions

    Parameters
    ----------
    x : array-like, with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    x2: array-like, with shape (number of species) (default = None)
        An array representing the abundances (observed
        counts) for each individual species. Only used for shared
        species estimation.
    method : str (default = None)
        One estimator of:
            - 'chao1'
            - 'egghe_proot'
            - 'jackknife'
            - 'minsample'
            - 'shared_richness'
            - 'empirical' (same as None)
    **kwargs : additional parameters passed to selected method

    Note
    ----
    If `CI` is True, a bootstrap procedure will be called on the
    specified method to compute the confidence intervals around
    the central estimate etc. For the Jackknife procedure, the
    CI is calculated analytically and no bootstrap values will
    be included in the returned dict. For chao1_shared no confidence
    intervals have been implemented yet.

    Returns
    -------
    Consult the documentation of selected method.
    """

    if method is not None and method.lower() not in ESTIMATORS:
        raise ValueError(f"Unknown estimation method `{method}`.")

    if method is None:
        method = "empirical"

    method = method.lower()

    if method == "shared_richness":
        estimate = ESTIMATORS[method](ds, ds2, CI=CI)
    elif CI and method != 'jackknife':
        if isinstance(ds, AbundanceData):
            bootstrap_fn = bootstrap_abundance_data
        elif isinstance(ds, IncidenceData):
            bootstrap_fn = bootstrap_incidence_data
        else:
            raise ValueError(
                "ds must be an instance of AbundanceData or IncidenceData")
        estimate = bootstrap_fn(
            ds, fn=partial(ESTIMATORS[method], **kwargs),
            n_iter=n_iter, n_jobs=n_jobs, seed=seed, disable_pb=disable_pb
        )
    elif CI and method == 'jackknife':
        estimate = ESTIMATORS[method](ds, CI=CI,
                                      conf=conf, **kwargs)
    else:
        estimate = ESTIMATORS[method](ds, **kwargs)

    return estimate

__all__ = ['chao1', 'iChao1', 'egghe_proot',
           'ace', 'jackknife', 'min_add_sample',
           'diversity', 'shared_richness']
