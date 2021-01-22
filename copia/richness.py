# -*- coding: utf-8 -*-
"""
Bias-correcting richness estimators for abundance data
"""
import warnings

import numpy as np
import scipy.stats as stats


def empirical_richness(x):
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
    return x[x > 0].shape[0]


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
    -------
    - A. Chao, 'Non-parametric estimation of the classes in a population',
    Scandinavian Journal of Statistics (1984), 265-270.
    - A. Chao, et al., 'Quantifying sample completeness and comparing
    diversities among assemblages', Ecological research (2020), 292-314.
    """

    x = x[x > 0]
    n = x.sum()
    t = x.shape[0]
    f1 = (x == 1).sum()
    f2 = (x == 2).sum()

    if f2 > 0:
        return t + (n - 1) / n * (f1 ** 2 / 2 / f2) 
    else:
        return f1 * (f1 - 1) / 2


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

    References
    -------
    - L. Egghe and G. Proot, 'The estimation of the number of lost
    multi-copy documents: A new type of informetrics theory', Journal
    of Informetrics (2007), 257-268.
    - Q.L. Burrell, 'Some comments on "The estimation of lost multi-copy
    documents: A new type of informetrics theory" by Egghe and Proot',
    Journal of Informetrics (2008), 101–105.
    """

    ft = np.bincount(x)[1:]
    S = ft.sum()

    P1 = (x == 1).sum()
    P2 = (x == 2).sum()
    P0 = (1 / (1 + (2 / (alpha - 1)) * (P2 / P1))) ** alpha

    S_lost = S * (P0 / (1 - P0))
    S_lost = S + S_lost

    if not np.isinf(S_lost):
        return S_lost
    else:
        return np.nan


def jackknife(x, k=5, return_order=False, return_ci=False,
              conf=0.95):
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

    def dbinom(x, size, prob):
        d = stats.binom(size, prob).pmf(x)
        return 1 if np.isnan(d) else d

    x = np.array(x, dtype=np.int64)

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
                gene[i, 0] +
                (-1)**(j + 1) * 2**i * dbinom(j, i, 0.5) * n[j - 1, 1])
            gene[i, 3] = gene[i, 3] + (-1)**(j + 1) * 2**i * dbinom(
                j, i, 0.5) * n[j - 1, 1] * np.prod(np.arange(1, j + 1))
        gene[i, 1] = -gene[i, 0]
        for j in range(1, i + 1):
            gene[i, 1] = (gene[i, 1] + (
                (-1)**(j + 1) * 2**i * dbinom(j, i, 0.5) + 1)**2 * n[j - 1, 1])
        gene[i, 1] = np.sqrt(gene[i, 1] + n[i:, 1].sum())
    
    if k > 1:
        for i in range(2, k + 1):
            gene[i - 1, 2] = -(gene[i, 0] - gene[i - 1, 0])**2 / (total - 1)
            for j in range(1, i):
                gene[i - 1, 2] = gene[i - 1, 2] + (
                    (-1)**(j + 1) * 2**(i) * dbinom(j, i, 0.5) -
                    (-1)**(j + 1) * 2**(i - 1) * dbinom(j, i - 1, 0.5)
                )**2 * n[j - 1, 1] * total / (total - 1)
            gene[i - 1, 2] = np.sqrt(gene[i - 1, 2] + n[i - 1, 1] * total /
                                     (total - 1))
            gene[i - 1, 4] = (gene[i, 0] - gene[i - 1, 0]) / gene[i - 1, 2]
    
    coe = stats.norm().ppf(1 - (1 - conf) / 2)
    x = gene[1:k + 1, 4] < coe

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
    
    if (return_order or return_ci):
        d = {'richness': jackest}
        if return_order:
            d['order']  = order
        if return_ci:
            d['lci'] = jackest - coe * sej
            d['uci'] = jackest + coe * sej
        return d
    else:
        return jackest


def min_add_sample(x, solver='grid', search_space=(0, 100, 1e6),
                  tolerance=1e-2):
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
            (start, end, stepsize)
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
    t = x.shape[0]
    f1, f2 = (x == 1).sum(), (x == 2).sum()
    
    h = lambda x: 2 * f1 * (1 + x)
    v = lambda x: np.exp(x * (2 * f2 / f1))
    
    if solver == 'grid':
        search_space = np.linspace(*search_space)
        hs = np.array(h(search_space))
        vs = np.array(v(search_space))
        diffs = np.abs(hs - vs)
        x_ast = search_space[diffs.argmin()]

    elif solver == 'fsolve':
        from scipy.optimize import fsolve
        def intersection(func1, func2, x0):
            return fsolve(lambda x: func1(x) - func2(x), x0)[0]
        x_ast = intersection(h, v, n)
    
    else:
        raise ValueError(f'Unsupported "solver" argument: {solver}')
    
    diff_intersect = abs(h(x_ast) - v(x_ast))
    if not diff_intersect < tolerance:
        warnings.warn(f'Tolerance criterion not met: {diff_intersect} > {tolerance}')

    return n * x_ast


estimators = {'empirical': empirical_richness,
              'chao1': chao1,
              'egghe_proot': egghe_proot,
              'jackknife': jackknife,
              'minsample': min_add_sample}


def richness(x, method=None, **kwargs):
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
    if method is None:
        method = 'empirical'

    estimate = estimators[method](x, **kwargs)
    return estimate
