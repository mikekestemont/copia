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

import copia.stats as stats
import copia.richness as richness
import copia.utils as utils


def abundance_counts(x, ax=None, figsize=None, trendline=False):
    r"""
    Plot per-species abundance in an assemblage, as a ranked bar plot

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)
    trendline : bool (default = False)
        If True, an trendline (exponential) is fitted
        and added to the barplot as a reading aid.

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
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

    if trendline:
        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        popt, _ = curve_fit(
            func, range(len(x)), x,
            bounds=([-np.inf, 0.0001, -np.inf], [np.inf, 10, np.inf]))
        ax2 = ax.twinx()
        ax2.grid(None)
        ax2.plot(range(len(x)), func(range(len(x)), *popt), 'r--',
                 label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        ax2.set(ylabel="Exponential fit", ylim=(1, max(x)))

    return ax


def abundance_histogram(x, ax=None, figsize=None, trendline=False):
    r"""
    Plot an assemblage's frequency histogram as a bar plot

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)
    trendline : bool (default = False)
        If True, a trendline (Fisher log-series) is
        fitted and added to the barplot as a reading
        aid.

    Note
    ----
    The code for fitting the trendline is based on the
    [macroeco package](https://github.com/jkitzes/macroeco/\
    blob/master/macroeco/models/_distributions.py).

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
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

    if trendline:
        mu = np.mean(x)
        eq = lambda p, mu: -p/np.log(1-p)/(1-p) - mu
        p = optim.brentq(eq, 1e-16, 1-1e-16, args=(mu), disp=True)
        estims = logser.pmf(pos, p)

        ax2 = ax.twinx()
        ax2.plot(pos, estims, 'r--')
        ax2.grid(None)
        ax2.set(ylabel="Fisher's log series (pmf)", ylim=(0, 1))

    return ax


def density(d, empirical=None, title=None, ax=None, xlim=None,
            xlabel=None, ylabel=None, figsize=None):
    r"""
    Plot histogram and kernel density of a bootstrapped
    richness estimate, with vertical lines for the quartiles
    (q11, q50, q89) and an info box.

    Parameters
    ----------
    d : dict
        A dict resulting from a (bootstrapped) richness
        estimate (see copia.richness)
    empirical : int (default = None)
        The empirical richness for the samples for which
        the true richness has been annotated. Will be ad-
        added as a vertical line to the plot if a value
        is specified.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    xlim : 2-way tuple (default = None)
        The xlim for the main axis of the plot
    xlabel : str (default = None)
        The label for the primary x-axis in the plot
    ylabel : str (default = None)
        The label for the primary y-axis in the plot
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sb.histplot(d['bootstrap'], kde=True, ax=ax)

    q_11, q_50, q_89 = stats.quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
    q_m, q_p = q_50 - q_11, q_89 - q_50

    ax.axvline(q_50, color='red')
    ax.axvline(q_11, ls='--', color='red')
    ax.axvline(q_89, ls='--', color='red')

    if empirical:
        ax.axvline(empirical, ls='--', color='green', linewidth=2)

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(".2f").format
    textstr = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    textstr = textstr.format(fmt(q_50), fmt(q_m), fmt(q_p))
    textstr = 'Estimate: ' + textstr

    ax.annotate(textstr, xy=(0.5, 0.7), xycoords='axes fraction',
                va='center_baseline', backgroundcolor='white')

    ax.set(
        xlim=xlim,
        xlabel=xlabel,
        ylabel=ylabel,
        title="Estimate: bootstrap values (KDE and quartiles)" if not title else title
    )

    return ax


def multi_kde(assemblages, ax=None, figsize=None,
              xlim=(0, 1), ylabel=None, xlabel=None):
    r"""
    Takes a dict of survival estimates for labeled assem-
    blages and plots the bootstrap value as kernel density
    estimates. Point estimates are added as vertical lines.

    Parameters
    ----------
    d : dict
        An assemblage dict, with labels (keys) and survival
        estimates for each assemblage (values) that come
        from `copia.utils.survival_ratio()`.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    xlim : 2-way tuple (default = (0, 1))
        The xlim for the main axis of the plot
    xlabel : str (default = None)
        The label for the primary x-axis in the plot
    ylabel : str (default = None)
        The label for the primary y-axis in the plot
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for i, (label, assemblage) in enumerate(assemblages.items()):
        sb.kdeplot(assemblage['bootstrap'], label=label,
                   ax=ax, color=f"C{i}", shade=True)
        ax.axvline(assemblage['survival'], linewidth=2, color=f"C{i}")

    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax


def survival_errorbar(survival, ax=None, figsize=None, xlabel=None,
                      ylabel='label'):
    r"""
    Takes a dict of survival estimates for labeled assem-
    blages and plots the bootstrap value as error bars,
    with whiskers for lower and upper CI.

    Parameters
    ----------
    survival : dict
        An assemblage dict, with labels (keys) and survival
        estimates for each assemblage (values) that come
        from `copia.utils.survival_ratio()`.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    xlim : 2-way tuple (default = (0, 1))
        The xlim for the main axis of the plot
    xlabel : str (default = None)
        The label for the primary x-axis in the plot
    ylabel : str (default = None)
        The label for the primary y-axis in the plot
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    estimates = []
    for l in survival:
        estimates.append([l] + [survival[l][k] for k in ['survival', 'lci', 'uci']])
    estimates = pd.DataFrame(estimates, columns=[ylabel, 'survival', 'lci', 'uci'])
    estimates = estimates.sort_values('survival')
    errors = np.array(list(zip(estimates['lci'], estimates['uci']))).T
    errors[0] = estimates['survival'] - errors[0]
    errors[1] -= estimates['survival']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        np.arange(len(estimates)),
        estimates['survival'],
        yerr=errors,
        fmt='.',
        c='green',
        label='diversity',
        ms=12)

    ax.set_ylabel(ylabel)
    ax.set_ylabel(xlabel)
    ax.set_xticks(np.arange(len(estimates)))
    ax.set_xticklabels(estimates[ylabel])

    return ax


def accumulation_curve(x, accumulation, minsample=None,
                       ax=None, figsize=None, **kwargs):
    r"""
    Plots the species accumulation curve for an assemblage,
    with the option of adding a kernel density plot for a
    corresponding minimum additional sampling estimate.

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    accumulation : dict
        The species accumulation curve, obtained from
        `copia.richness.species_accumulation()`.
    minsample : dict (default = None)
        The result of a call to copia.richness.min_add\
        _sample(). If specified, a KDE for the bootstrap-
        ped values will be included.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Note
    ----
    Any `**kwargs` will be passed to ax.set() to con-
    trol figure aesthetics.

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    lci = accumulation['lci']
    uci = accumulation['uci']
    Dq = accumulation['richness']
    steps = accumulation['steps']
    interpolated = accumulation['interpolated']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # mark empirical situation:
    ax.plot(x.sum(), Dq[x.sum() - 1], 'o', markersize=8)
    ax.plot(steps[interpolated], Dq[interpolated], color='C0')
    ax.plot(steps[~interpolated], Dq[~interpolated], '--', color='C0')
    ax.fill_between(steps, lci, uci, alpha=0.3)

    if minsample:
        ax2 = ax.twinx()
        sb.kdeplot(minsample['bootstrap'], ax=ax2,
                   color='green', fill=True)

        ax.axvline(minsample['richness'], color='green')

        ax2.grid(None)
        ax2.set(xlabel='Min. add. sample')

    # cosmetics etc.
    ax.set(**kwargs)
    return ax


def minsample_diagnostic_plot(x, diagnostics, max_x_ast=100, ax=None,
                              figsize=None, **kwargs):
    r"""
    A diagnostic plot showing the detected intersection
    between v() and h() for checking whether the optimization
    in `copia.richness.species_accumulation()` has converged.

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    diagnostics : dict
        The result of a call to copia.richness.min_add\
        _sample(..., solver='grid', CI=False, diagnostics=True)
    max_x_ast : int (default = 100)
        Controls the spacing of the search space to
        be visualized
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Note
    ----
    Any `**kwargs` will be passed to ax.set() to con-
    trol figure aesthetics.

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    x_ast = diagnostics['x*']
    sp = np.linspace(x_ast - 1, x_ast + 1, max_x_ast)

    basics = utils.basic_stats(x)
    f1 = basics['f1']
    f2 = basics['f2']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sp, 2 * f1 * (1 + sp), label='$h(x)$')
    ax.plot(sp, np.exp(sp * (2 * f2 / f1)), label='$v(x)$')
    ax.axvline(x_ast, linestyle='--', c='grey')
    ax.set_xlabel('$x$')
    ax.set_ylabel('h(x) and v(x)')
    ax.legend()

    ax.set(**kwargs)
    return ax


def hill_plot(emp, est, q_min=0, q_max=3, step=0.1,
              figsize=None, ax=None, add_densities=True,
              title=None, **kwargs):
    r"""
    Plots the Hill number profiles (with CI) for an
    assemblage (both empirical and estimated), with
    the option of adding kernel densitity estimates
    for the main orders q in a separate subplot.

    Parameters
    ----------
    emp : dict
        The empirical Hill number profile, i.e. the
        first dict returned by `copia.hill.hill_numbers()`.
    est : dict
        The estimated Hill number profile, i.e. the
        second dict returned by `copia.hill.hill_numbers()`.
    q_min : float (default = 0)
        Minimum order to consider
    q_max : float (default = 3)
        Maximum order to consider
    add_densities: bool (default = True)
        If True, kernel densitity estimates for the main
        orders q will be added in a separate subplot
    step : float (default = 0.1)
        Step size in between consecutive orders
    title : str (default = None)
        The main title for the figure
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Note
    ----
    Any `**kwargs` will be passed to ax.set() to con-
    trol figure aesthetics.

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    c_emp, c_est = 'C0', 'C1'
    q = np.arange(q_min, q_max + step, step)

    lci_emp, lci_est = emp['lci'], est['lci']
    uci_emp, uci_est = emp['uci'], est['uci']
    bt_emp, bt_est = emp['bootstrap'], est['bootstrap']
    emp, est = emp['richness'], est['richness']

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

    ax.set(**kwargs)
    return ax


def evenness_plot(evennesses, q_min=0, q_max=3, step=0.1, ax=None,
                  figsize=None, **kwargs):
    r"""
    Plots evenness curves for a dictionary of assemblages

    Parameters
    ----------
    evennesses : dict
        An assemblage dict, with labels (keys) and evenness
        profiles for each assemblage (values) that come
        from `copia.utils.evenness()`.
    q_min : float (default = 0)
        Minimum order to consider
    q_max : float (default = 3)
        Maximum order to consider
    step : float (default = 0.1)
        Step size in between consecutive orders
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    figsize : 2-way tuple (default = None)
        The size of the new plt.Figure to be plotted
        (Ignored if an axis is passed.)

    Note
    ----
    Any `**kwargs` will be passed to ax.set() to con-
    trol figure aesthetics.

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    q = np.arange(q_min, q_max + step, step)

    for i, (label, evenness) in enumerate(evennesses.items()):
        ax.plot(q, evenness, label=label, c=f"C{i}", linewidth=2)

    ax.set_xlabel('Diversity order ($q$)')
    ax.set_ylabel(r'Evenness: $({}^qD - 1) / (\hat{S} - 1)$')
    ax.set_title('Evenness profile')
    ax.legend(loc='best')

    ax.set(**kwargs)
    return ax


__all__ = ['abundance_counts', 'abundance_histogram', 'density',
           'multi_kde', 'survival_errorbar', 'accumulation_curve',
           'minsample_diagnostic_plot', 'hill_plot', 'evenness_plot']
