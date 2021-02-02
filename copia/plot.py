# -*- coding: utf-8 -*-
"""
Miscellaneous visualization routines
"""
from functools import partial
from collections import Counter

import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy.stats import logser
import pandas as pd


from .stats import quantile, rarefaction_extrapolation, bootstrap
from .richness import *


def abundance_counts(x, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.array(sorted(x, reverse=True))
    ax.bar(range(len(x)), x, alpha=.7, align='center',
           color=next(ax._get_lines.prop_cycler)['color'])
    ax.tick_params(axis='x', which='both', bottom=False,
                   top=False, labelbottom=False)
    ax.set(xlabel='Species', ylabel='Number of sightings',
           title='Distribution of sightings over species')
    
    textstr = '\n'.join((
        f'Species: {np.count_nonzero(x)}',
        f'Observations: {x.sum()}',
        f'$f_1$: {np.count_nonzero(x == 1)}',
        f'$f_2$: {np.count_nonzero(x == 2)}',
        ))
    ax.annotate(textstr, xy=(0.75, 0.75), xycoords='axes fraction',
                 va='center', backgroundcolor='white')

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, _ = curve_fit(func, range(len(x)), x,
          bounds=([-np.inf, 0.0001, -np.inf], [np.inf, 10, np.inf]))
    ax2 = ax.twinx()
    ax2.grid(None)
    ax2.plot(range(len(x)), func(range(len(x)), *popt), 'r--', 
             label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    ax2.set(ylabel="Exponential fit", ylim=(1, max(x)))
    return ax


def abundance_histogram(x, ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    x = np.array(sorted(x, reverse=True))

    textstr = (f'Species: {np.count_nonzero(x)}\n'
               f'Observations: {x.sum()}\n'
               f'$f_1$: {np.count_nonzero(x == 1)}\n'
               f'$f_2$: {np.count_nonzero(x == 2)}')
    
    counter = Counter(x)
    max_count = max(counter.keys())
    pos = [k for k in range(1, max_count + 1)]
    x = np.array([counter[k] for k in pos])

    ax.bar(pos, x, alpha=.7, align='center',
           color=next(ax._get_lines.prop_cycler)['color'])

    ax.set(xlabel='Species', title='Sightings histogram')
    
    ax.annotate(textstr, xy=(0.7, 0.7), xycoords='axes fraction',
                va='center', backgroundcolor='white')
    
    # https://github.com/jkitzes/macroeco/blob/master/macroeco/models/_distributions.pys
    mu = np.mean(x)
    eq = lambda p, mu: -p/np.log(1-p)/(1-p) - mu
    p = optim.brentq(eq, 1e-16, 1-1e-16, args=(mu), disp=True)
    estims = logser.pmf(pos, p)

    ax2 = ax.twinx()
    ax2.plot(pos, estims, 'r--')
    ax2.grid(None)
    ax2.set(ylabel="Fisher's log series (pmf)", ylim =(0, 1))
    return ax


def richness_density(d, empirical=None, normalize=False, title=None, ax=None, figsize=(15, 15)):
    if normalize and not empirical:
        msg = """If normalize is  set to True, `empirical`
                 richness must be provided."""
        raise ValueError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        d['bootstrap'] = empirical / d['bootstrap']

    sb.histplot(d['bootstrap'], kde=True, ax=ax)

    q_11, q_50, q_89 = quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
    q_m, q_p = q_50 - q_11, q_89 - q_50

    ax.axvline(q_50, color='red')
    ax.axvline(q_11, ls='--', color='red')
    ax.axvline(q_89, ls='--', color='red')

    if not normalize and empirical:
        ax.axvline(empirical, ls='--', color='green', linewidth=2)

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(".2f").format
    textstr = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    textstr = textstr.format(fmt(q_50), fmt(q_m), fmt(q_p))
    textstr = 'Estimate: ' + textstr

    ax.annotate(textstr, xy=(0.5, 0.7), xycoords='axes fraction',
                va='center_baseline', backgroundcolor='white', 
                fontsize=12)
    
    if normalize:
        ax.set_xlim([0, 1])


    ax.set(
        xlabel="Richness" if not survival else "Survival ratio",
        ylabel="Kernel density",
        title = "Estimate: bootstrap values (KDE and CI)" if not title else title
    )

    return ax


def survival_kde(assemblages, ax=None, figsize=(16, 8),
                  xlim=(0, 1), ylabel='Survival ratio (KDE)',
                  xlabel=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, (label, assemblage) in enumerate(assemblages.items()):
        sb.kdeplot(assemblage['bootstrap'], label=label,
                   ax=ax, color=f"C{i}", shade=True)
        ax.axvline(assemblage['survival'], linewidth=2, color=f"C{i}")
    
    ax.legend()

    return ax


def survival_errorbar(df, ax=None, figsize=(12, 4)):
    estimates = []

    estimates = df.sort_values('survival')
    errors = np.array(list(zip(estimates['lci'], estimates['uci']))).T
    errors[0] = estimates['survival'] - errors[0]
    errors[1] -= estimates['survival']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(np.arange(len(estimates)),
            estimates['survival'],
            yerr=errors,
            fmt='.',
            c='green',
            label='diversity',
            ms=12)
    
    ax.set_ylabel('Estimated survival ratio')
    ax.set_xticks(np.arange(len(estimates)))
    ax.set_xticklabels(estimates['label'], fontsize=12)
    return ax


def species_accumulation_curve(x, max_steps=None, incl_minsample=False,
                               ax=None, figsize=(16, 12), n_iter=100):
    x = np.array(x, dtype=np.int)

    # min sample estimate to estimate max steps:
    minsample_est = diversity(x, method='minsample', 
                              solver='fsolve', CI=True)
    q_11, q_50, q_89 = quantile(minsample_est['bootstrap'],
                               [0.11, 0.5, 0.89], weights=None)
    
    if max_steps is None:
        if incl_minsample:
            max_steps = int(max(minsample_est['bootstrap']))
        else:
            max_steps = int(q_89)
    
    steps = np.arange(1, max_steps)
    interpolated = np.arange(1, max_steps) < x.sum()

    estim = bootstrap(x, fn=partial(rarefaction_extrapolation,
                                    max_steps=max_steps),
                      n_iter=n_iter)
    
    lci_pro = estim['lci']
    uci_pro = estim['uci']
    Dq = estim['richness']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # species accumulation
    ax.plot(x.sum(), Dq[x.sum() - 1], 'o', markersize=8)
    ax.plot(steps[interpolated], Dq[interpolated], color='C0')
    ax.plot(steps[~interpolated], Dq[~interpolated], '--', color='C0')
    ax.fill_between(steps, lci_pro, uci_pro, alpha=0.3)

    # min sample:
    if incl_minsample:
        ax2 = ax.twinx()
        sb.kdeplot(minsample_est['bootstrap'], ax=ax2,
                color='green', fill=True)

        ax.axvline(q_50, color='green')
        ax.axvline(q_11, ls='--', color='green')
        ax.axvline(q_89, ls='--', color='green')

        ax2.grid(None)

    # cosmetics etc.
    ax.set(xlabel='sightings', ylabel='species', title='Species Accumulation Curve')
    return ax


def hill_plot(emp, est, q_min=0, q_max=3, step=0.1,
              figsize=None, ax=None, add_densities=True,
              title=None):

    c_emp, c_est = 'C0', 'C1'
    q = np.arange(q_min, q_max + step, step)

    lci_emp, lci_est = emp['lci'], est['lci']
    uci_emp, uci_est = emp['uci'], est['uci']
    bt_emp, bt_est = emp['bootstrap'], est['bootstrap']
    emp, est =  emp['richness'], est['richness']

    y_min = min(min(lci_emp), min(lci_est)) - 2
    y_max = max(max(uci_emp), max(uci_est)) + 2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(q, emp, color=c_emp, label='empirical')
    ax.plot(q, est, color=c_est, label='estimation')

    ax.fill_between(q, lci_emp, uci_emp, color=c_emp, alpha=0.3)
    ax.fill_between(q, lci_est, uci_est, color=c_est, alpha=0.3)

    ax.set(xlabel='Order $q$', ylabel='Hill numbers', ylim=(y_min, y_max),
           title=title)

    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc='upper center',
        ncol=3,
        mode='expand',
        borderaxespad=0.0,
        facecolor='white',
        framealpha=1,
    )

    if add_densities:
        left, bottom, width, height = [0.58, 0.5, 0.3, 0.35]
        ax2 = fig.add_axes([left, bottom, width, height])
        labels = 'Richness', 'Shannon', 'Simpson'

        for k, (i, label) in enumerate(zip(np.where(np.isin(q, (0, 1, 3)))[0], labels)):
            sb.kdeplot(bt_est[:, i], label=label, c=f"C{k}", ax=ax2)
        
        l1, l2, l3 = ax2.lines

        # Get the xy data from the lines so that we can shade
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        x3 = l3.get_xydata()[:, 0]
        y3 = l3.get_xydata()[:, 1]

        ax2.fill_between(x1, y1, color="C0", alpha=0.3)
        ax2.fill_between(x2, y2, color="C1", alpha=0.3)
        ax2.fill_between(x3, y3, color="C2", alpha=0.3)

        ax2.set_xlabel('Hill numbers')
        ax2.set_ylabel('Density')

    return ax


def evenness_plot(assemblages, incl_CI=False, q_min=0, q_max=3, step=0.1, ax=None, figsize=(14, 14)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    q = np.arange(q_min, q_max + step, step)

    for i, (label, d) in enumerate(assemblages.items()):
        lci, uci, richness = d['lci'], d['uci'], d['richness']
        
        richness = (richness - 1) / (richness[0] - 1)
        ax.plot(q, richness, label=label, c=f"C{i}", linewidth=2)

        if incl_CI:
            # experimental...
            lci = (lci - 1) / (max(max(lci), lci[0]) - 1)
            uci = (uci - 1) / (max(max(uci), uci[0]) - 1)

            lci = np.maximum(richness, lci)
            uci = np.minimum(richness, uci)
            
            ax.plot(q, lci, c=f"C{i}", linewidth=.8)
            ax.plot(q, uci, c=f"C{i}", linewidth=.8)
            ax.fill_between(q, lci, uci, color=f"C{i}", alpha=0.3)
    
    ax.set_xlabel('Diversity order ($q$)', fontsize=16)
    ax.set_ylabel(r'Evenness: $({}^qD - 1) / (\hat{S} - 1)$', fontsize=16)
    ax.set_title('Evenness profile', fontsize=20)
    ax.legend(fontsize=14)

    return ax
