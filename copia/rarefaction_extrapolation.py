import numpy as np

from functools import partial
from scipy.special import gammaln

from copia.bootstrap import bootstrap_abundance_data
from copia.bootstrap import bootstrap_incidence_data
from copia.coverage import estimate_coverage
from copia.data import AbundanceData, CopiaData, IncidenceData


def rarefaction_extrapolation(
        ds: CopiaData, max_steps, step_size=1):
    r"""
    Species accumulation curve (calculation)

    Parameters
    ----------
    ds : CopiaData
        An instance of CopiaData (either AbundanceData or IncidenceData).
    max_steps : int
        Maximum number of steps in the accumulation. Should
        be a positive integer, but can be smaller (rarefaction)
        or larger (extrapolation) than the empirical population
        size (:math:`n`).
    step_size : int
        Specify the increment to max_steps. Defaults to 1.

    Returns
    -------
    accumulation : np.ndarray
        Species accumulation curve as a 1D numpy array of shape
        (max_steps, ). Contains the estimated richness of the
        assemblage, for each step in the range [0, max_steps, step_size].

    References
    ----------
    - N.J. Gotelli and R.K. Colwell, 'Estimating Species Richness',
      Biological Diversity: Frontiers in Measurement and Assessment,
      OUP (2011), 39-54.
    - A. Chao, et al. 'Rarefaction and extrapolation with Hill numbers:
      a framework for sampling and estimation in species diversity studies',
      Ecological Monographs (2014), 84, 45–67.
    """
    if isinstance(ds, AbundanceData):
        n, f1, f2, counts = ds.n, ds.f1, ds.f2, ds.counts
    elif isinstance(ds, IncidenceData):
        n, f1, f2, counts = ds.T, ds.f1, ds.f2, ds.counts
    else:
        raise ValueError("ds must be an instance of AbundanceData or IncidenceData")
    
    if f2 == 0:
        f0 = (n - 1) / n * f1 * (f1 - 1) / 2
    else:
        f0 = (n - 1) / n * f1**2 / 2 / f2
    A = 1
    if f1 > 0:
        A = n * f0 / (n * f0 + f1)
    
    def _sub(m):
        if m <= n:
            xx = counts[(n - counts) >= m]
            return np.sum(1 - np.exp(
                gammaln(n - xx + 1) +
                gammaln(n - m + 1) - 
                gammaln(n - xx - m + 1) -
                gammaln(n + 1))) + np.count_nonzero(n - counts < m)
        elif f1 == 0:
            return ds.S_obs
        else:
            return ds.S_obs + f0 * (1 - A**(m - n))
    return np.array([_sub(mi) for mi in range(1, max_steps + step_size, step_size)])


def species_accumulation(ds, max_steps=None, step_size=1, compute_coverage=False,
                         n_iter=100, n_jobs=1):

    if isinstance(ds, AbundanceData):
        bootstrap_fn = bootstrap_abundance_data
        n_sampling_units = ds.n
    elif isinstance(ds, IncidenceData):
        bootstrap_fn = bootstrap_incidence_data
        n_sampling_units = ds.T
    else:
        raise ValueError("ds must be an instance of AbundanceData or IncidenceData")
        
    if max_steps is None:
        max_steps = n_sampling_units * 2

    accumulation = bootstrap_fn(
        ds, fn=partial(
            rarefaction_extrapolation,
            max_steps=max_steps, step_size=step_size),
        n_iter=n_iter, n_jobs=n_jobs)

    if compute_coverage:
        coverage = estimate_coverage(ds, max_steps=max_steps, step_size=step_size)
        accumulation['coverage'] = coverage    

    steps = np.arange(1, max_steps + step_size, step_size)
    if isinstance(ds, AbundanceData):
        interpolated = np.arange(1, max_steps + step_size, step_size) < ds.n
    else:
        interpolated = np.arange(1, max_steps + step_size, step_size) < ds.T
    accumulation['interpolated'] = interpolated
    accumulation['steps'] = steps
    accumulation = accumulation.set_index('steps')
    return accumulation
