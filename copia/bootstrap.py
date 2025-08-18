import numpy as np
import pandas as pd
from scipy import stats

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


def bt_prob_shared(s1, s2):
    """Calculate bootstrap probabilities for shared species estimation."""
    # Remove species with zero total abundance
    mask = (s1 + s2) > 0
    s1, s2 = s1[mask], s2[mask]
    
    # Initial computations
    n = np.sum(s1) + np.sum(s2)  # Joint total
    n1, n2 = np.sum(s1), np.sum(s2)
    
    # Compute f0j (total unseen)
    joint = s1 + s2
    f1j = np.sum(joint == 1)
    f2j = np.sum(joint == 2)
    f0j = ((n - 1) / n * f1j**2 / (2 * f2j)) if f2j > 0 else ((n - 1) / n * f1j * (f1j - 1) / 2)
    f0j_star = int(np.ceil(f0j))
    
    # Extend arrays with zeros
    s1_ext = np.concatenate([s1, np.zeros(f0j_star)])
    s2_ext = np.concatenate([s2, np.zeros(f0j_star)])
    
    # Calculate f01 and f02
    f11 = np.sum(s1 == 1)
    f21 = np.sum(s1 == 2)
    f12 = np.sum(s2 == 1)
    f22 = np.sum(s2 == 2)
    
    f01 = ((n1 - 1) / n1 * f11**2 / (2 * f21)) if f21 > 0 else ((n1 - 1) / n1 * f11 * (f11 - 1) / 2)
    f02 = ((n2 - 1) / n2 * f12**2 / (2 * f22)) if f22 > 0 else ((n2 - 1) / n2 * f12 * (f12 - 1) / 2)
    
    f01_star = int(np.ceil(f01))
    f02_star = int(np.ceil(f02))
    
    # Adjust f0*_star based on available zeros
    zero1 = np.sum(s1_ext == 0)
    zero2 = np.sum(s2_ext == 0)
    f01_star = min(zero1, f01_star)
    f02_star = min(zero2, f02_star)
    
    # Calculate C1 and C2
    if f21 == 0:
        C1 = 1 - f11/n1 * (n1-1) * (f11-1) / ((n1-1) * (f11-1) + 2)
    else:
        C1 = 1 - f11/n1 * (n1-1) * f11 / ((n1-1) * f11 + 2 * f21)
        
    if f22 == 0:
        C2 = 1 - f12/n2 * (n2-1) * (f12-1) / ((n2-1) * (f12-1) + 2)
    else:
        C2 = 1 - f12/n2 * (n2-1) * f12 / ((n2-1) * f12 + 2 * f22)
    
    # Calculate probabilities
    p1_term = s1/n1 * (1 - s1/n1)**n1
    p2_term = s2/n2 * (1 - s2/n2)**n2
    
    lambda1 = (1 - C1) / np.sum(p1_term)
    lambda2 = (1 - C2) / np.sum(p2_term)
    
    # Probabilities for seen species
    pseen1 = s1/n1 * (1 - lambda1 * (1 - s1/n1)**n1)
    pseen2 = s2/n2 * (1 - lambda2 * (1 - s2/n2)**n2)
    
    # Probabilities for unseen species
    punseen1 = (1 - C1) / f01_star if f01_star > 0 else 0
    punseen2 = (1 - C2) / f02_star if f02_star > 0 else 0
    
    # Create extended probability arrays
    prob1 = np.zeros_like(s1_ext, dtype=float)
    prob2 = np.zeros_like(s2_ext, dtype=float)
    
    # Assign seen probabilities
    prob1[:len(s1)] = pseen1
    prob2[:len(s2)] = pseen2
    
    # Randomly assign unseen probabilities
    zero_indices1 = np.where(s1_ext == 0)[0]
    zero_indices2 = np.where(s2_ext == 0)[0]
    
    if f01_star == 1 and len(zero_indices1) >= 1:
        idx1 = np.random.choice(zero_indices1, 1)
        prob1[idx1] = punseen1
    elif f01_star > 1:
        idx1 = np.random.choice(zero_indices1, f01_star, replace=False)
        prob1[idx1] = punseen1
        
    if f02_star == 1 and len(zero_indices2) >= 1:
        idx2 = np.random.choice(zero_indices2, 1)
        prob2[idx2] = punseen2
    elif f02_star > 1:
        idx2 = np.random.choice(zero_indices2, f02_star, replace=False)
        prob2[idx2] = punseen2
    
    # Final scaling
    prob1 = prob1 * (n1 / n)
    prob2 = prob2 * (n2 / n)
    
    return prob1, prob2, n1, n2


def percentile_ci(estimates, conf=0.95):
    """
    Calculate percentile-based confidence intervals from bootstrap estimates.
    
    Parameters:
    estimates: array of bootstrap estimates (n_iter x n_components)
    conf: confidence level (default 0.95)
    
    Returns:
    lci, uci: lower and upper confidence intervals
    """
    alpha = 1 - conf
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100
    
    lci = np.percentile(estimates, lower_p, axis=0)
    uci = np.percentile(estimates, upper_p, axis=0)
    
    return lci, uci


def bootstrap_shared_species(s1, s2, fn, n_iter=1000, conf=0.95, **kwargs):
    """Bootstrap procedure for shared species estimation."""
    rnd = np.random.RandomState(kwargs.get('seed', None))
    z_score = stats.norm.ppf((1 + conf) / 2)
    
    # Get bootstrap probabilities and generate samples
    pseen1, pseen2, n1, n2 = bt_prob_shared(s1, s2)
    
    # Get original estimates for reference
    orig_est = fn(s1, s2)
    shared_obs = orig_est[1]
    
    estimates = []
    for _ in range(n_iter):
        boot_s1 = rnd.multinomial(int(n1), pseen1/np.sum(pseen1))
        boot_s2 = rnd.multinomial(int(n2), pseen2/np.sum(pseen2))
        result = fn(boot_s1, boot_s2)
        estimates.append(result)
    
    estimates = np.array(estimates)
    est_mean = estimates.mean(0)
    est_sd = estimates.std(0, ddof=1)
    
    # Initialize CI arrays
    lci = np.zeros_like(orig_est)
    uci = np.zeros_like(orig_est)
    
    # For completeness, the code block below illustrates how to
    # obtain confidence intervals using the analytical method:
    
    # # For total shared (index 0):
    # var_ratio = est_sd[1]**2 / (est_mean[1] - shared_obs)**2 
    # R = np.exp(z_score * np.sqrt(np.log(1 + var_ratio)))
    # lci[0] = shared_obs + (orig_est[0] - shared_obs) / R
    # uci[0] = shared_obs + (orig_est[0] - shared_obs) * R
    
    # # For other components: symmetric intervals
    # lci[1:] = orig_est[1:] - z_score * est_sd[1:]
    # uci[1:] = orig_est[1:] + z_score * est_sd[1:]

    lci, uci = percentile_ci(estimates, conf)
    
    return est_mean, lci, uci, est_sd