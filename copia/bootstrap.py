import numpy as np
import pandas as pd
from copia.data import to_copia_dataset
import copia.utils

def bt_prob_abundance(ds):
    x, n, f1, f2 = ds.counts, ds.n, ds.f1, ds.f2
    C = 1 - f1 / n * (((n - 1) * f1 / ((n - 1) * f1 + 2 * f2)) if f2 > 0 else
                      ((n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)) if f1 > 0 else
                      0)
    W = (1 - C) / np.sum(x / n * (1 - x / n) ** n)
    p = x / n * (1 - W * (1 - x / n) ** n)
    f0 = np.ceil(((n - 1) / n * f1 ** 2 / (2 * f2)) if f2 > 0 else
                 ((n - 1) / n * f1 * (f1 - 1) / 2))
    p0 = (1 - C) / f0
    p = np.hstack((p, np.array([p0 for _ in np.arange(f0)])))
    return p


def bootstrap_abundance_data(ds, fn,
                             n_iter=1000,
                             conf=0.95,
                             n_jobs=1,
                             disable_pb=False,
                             seed=None):
    """Bootstrap method to construct confidence intervals of a specified 
    richness index.

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    fn : Callable representing the target richness index.
    n_iter : int (default = 1000)
        Number of bootstrap samples.
    conf : float (default = 0.95)
        Compute the confidence interval at the specified level.
    n_jobs : int (default = 1)
        Number of cores to use for computation.
    seed : int (default = None)
        A seed to initialize the random number generator. 

    Returns
    -------
    estimates : dict
        A dictionary providing the empirical richness index keyed with 
        `richness`, the bootstrapped estimates `bootstrap`, the lower and 
        upper endpoint of the specified confidence interval (`lci` and `uci`), 
        and the standard deviation of the richness index. 
    """
    rnd = copia.utils.check_random_state(seed)
    pro = fn(ds) 
    p, n = bt_prob_abundance(ds), ds.n
    data_bt = rnd.multinomial(n, p, n_iter)
    
    pool = copia.utils.Parallel(n_jobs, n_iter, disable_pb=disable_pb)
    for row in data_bt:
        row = to_copia_dataset(row, data_type="abundance", input_type="counts")
        pool.apply_async(fn, args=(row,))
    pool.join()

    bt_pro = np.array(pool.result())
    pro_mean = bt_pro.mean(0)
    
    lci_pro = -np.quantile(bt_pro, (1 - conf) / 2, axis=0) + pro_mean
    uci_pro = np.quantile(bt_pro, 1 - (1 - conf) / 2, axis=0) - pro_mean

    bt_pro = pro_mean - bt_pro

    lci_pro, uci_pro = pro - lci_pro, pro + uci_pro
    bt_pro = pro - bt_pro

    out = pd.DataFrame(
        np.vstack([pro, lci_pro, uci_pro]).T, columns=["est", "lci", "uci"])

    if out.shape[0] == 1:
        out = out.iloc[0]

    return out


def bt_prob_incidence(ds):
    incidence_freqs = ds.counts
    T, Q1, Q2, U = ds.T, ds.f1, ds.f2, ds.n
    if Q2 == 0:
        Q0_hat = (T - 1) / T * Q1 * (Q1 - 1) / 2
    else:
        Q0_hat = (T - 1) / T * Q1**2 / 2 / Q2
    A = 1
    if Q1 > 0:
        A = T * Q0_hat / (T * Q0_hat + Q1)
    C = 1 - Q1 / U * A
    Q0 = max(np.ceil(Q0_hat), 1)
    tau = 0
    if Q0_hat != 0:
        tau = (U / T * (1 - C) /
               sum(incidence_freqs / T *
                   (1 - incidence_freqs / T) ** T))
    p = (incidence_freqs / T *
         (1 - tau * (1 - incidence_freqs / T) ** T))
    p0 = U / T * (1 - C) / Q0
    return np.hstack((p, np.array([p0 for _ in np.arange(Q0)])))


def bootstrap_incidence_data(ds,
                             fn,
                             n_iter=1000,
                             conf=0.95,
                             n_jobs=1,
                             disable_pb=False,
                             seed=None):
    rnd = copia.utils.check_random_state(seed)
    pro = fn(ds) 
    p = bt_prob_incidence(ds)
    data_bt = rnd.binomial(ds.T, p, size=(n_iter, p.shape[0]))
    
    pool = copia.utils.Parallel(n_jobs, n_iter, disable_pb=disable_pb)
    for row in data_bt:
        row = to_copia_dataset(
            row, data_type="incidence", input_type="counts",
            n_sampling_units=ds.T)
        pool.apply_async(fn, args=(row,))
    pool.join()

    bt_pro = np.array(pool.result())
    pro_mean = bt_pro.mean(0)
    
    lci_pro = -np.quantile(bt_pro, (1 - conf) / 2, axis=0) + pro_mean
    uci_pro = np.quantile(bt_pro, 1 - (1 - conf) / 2, axis=0) - pro_mean

    bt_pro = pro_mean - bt_pro

    lci_pro, uci_pro = pro - lci_pro, pro + uci_pro
    bt_pro = pro - bt_pro

    out = pd.DataFrame(
        np.vstack([pro, lci_pro, uci_pro]).T, columns=["est", "lci", "uci"])

    if out.shape[0] == 1:
        out = out.iloc[0]

    return out

