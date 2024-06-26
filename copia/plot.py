"""
Miscellaneous visualization routines
"""
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from copia.data import AbundanceData


def abundance_barplot(
        ds: AbundanceData, ax=None, figsize=None,
        xlabel="Species", ylabel="Number of sightings",
        title='Distribution of sightings over species',
):
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

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    counts = np.array(sorted(ds.counts, reverse=True))
    ax.bar(range(ds.S_obs), counts, alpha=.7, align='center',
           color=next(ax._get_lines.prop_cycler)['color'])
    ax.tick_params(axis='x', which='both', bottom=False,
                   top=False, labelbottom=False)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    textstr = '\n'.join((
        f'Categories: {ds.S_obs}',
        f'Observations: {ds.n}',
        f'$f_1$: {ds.f1}',
        f'$f_2$: {ds.f2}',
        ))
    ax.annotate(textstr, xy=(0.75, 0.75), xycoords='axes fraction',
                va='center', backgroundcolor='white')

    return ax


def abundance_histogram(
        ds: AbundanceData, ax=None, figsize=None,
        xlabel='Species', title='Sightings histogram'
):
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

    Returns
    -------
    ax : plt.Axes
        The resulting plot's (primary) axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    counts = np.array(sorted(ds.counts, reverse=True))

    textstr = (f'Categories: {ds.S_obs}\n'
               f'Observations: {ds.n}\n'
               f'$f_1$: {ds.f1}\n'
               f'$f_2$: {ds.f2}')

    counter = Counter(counts)  # TODO: stick with numpy 
    max_count = max(counter.keys())
    pos = [k for k in range(1, max_count + 1)]
    x = np.array([counter[k] for k in pos])

    ax.bar(pos, x, alpha=.7, align='center',
           color=next(ax._get_lines.prop_cycler)['color'])

    ax.set(xlabel=xlabel, title=title)

    ax.annotate(textstr, xy=(0.7, 0.7), xycoords='axes fraction',
                va='center', backgroundcolor='white')

    return ax


def accumulation_curve(ds, acc_df, ax=None, figsize=None, c0='C0', **kwargs):
    r"""
    Plots the species accumulation curve for an assemblage,
    with the option of adding a kernel density plot for a
    corresponding minimum additional sampling estimate.

    Parameters
    ----------
    ds : CopiaData
        An instance of AbundanceData or IncidenceData.
    acc_df : pd.DataFrame
        The species accumulation curve, obtained from
        `copia.rarefaction_extrapolation.species_accumulation()`.
    ax : plt.Axes (default = None)
        The ax to plot on or None if a new plt.Figure
        is required.
    c0 : str (default = "C0")
        Color string for the species accumulation curve
    c1 : str (default = "C1")
        Color string for the minsample curve
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
    lci = acc_df['lci']
    uci = acc_df['uci']
    Dq = acc_df['est']
    steps = acc_df.index
    interpolated = acc_df['interpolated'].values

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n = ds.n if isinstance(ds, AbundanceData) else ds.T
    # mark empirical situation:
    ax.plot(n, ds.S_obs, 'o', markersize=8, color=c0)
    ax.plot(steps[interpolated], Dq[interpolated], color=c0)
    ax.plot(steps[~interpolated], Dq[~interpolated], '--', color=c0)
    ax.fill_between(steps, lci, uci, alpha=0.3, color=c0)

    # cosmetics etc.
    ax.set(**kwargs)
    return ax


def minsample_diagnostic_plot(ds: AbundanceData, diagnostics, max_x_ast=100, ax=None,
                              figsize=None, **kwargs):
    r"""
    A diagnostic plot showing the detected intersection
    between v() and h() for checking whether the optimization
    in `copia.stats.species_accumulation()` has converged.

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

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(sp, 2 * ds.f1 * (1 + sp), label='$h(x)$')
    ax.plot(sp, np.exp(sp * (2 * ds.f2 / ds.f1)), label='$v(x)$')
    ax.axvline(x_ast, linestyle='--', c='grey')
    ax.set_xlabel('$x$')
    ax.set_ylabel('h(x) and v(x)')
    ax.legend()

    ax.set(**kwargs)
    return ax


def hill_plot(emp, est, q_min=0, q_max=3, step=0.1,
              figsize=None, ax=None,
              title=None, **kwargs):
    r"""
    Plots the Hill number profiles (with CI) for an
    assemblage (both empirical and estimated), with
    the option of adding kernel densitity estimates
    for the main orders q in a separate subplot.

    Parameters
    ----------
    emp : pd.DataFrame
        The empirical Hill number profile, returned by
        `copia.diversity.hill_numbers()`.
    est : pd.DataFrane
        The estimated Hill number profile, returned by
        `copia.diversity.hill_numbers()`.
    q_min : float (default = 0)
        Minimum order to consider
    q_max : float (default = 3)
        Maximum order to consider
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
    emp, est = emp['est'], est['est']

    y_min = min(min(lci_emp), min(lci_est)) - 2
    y_max = max(max(uci_emp), max(uci_est)) + 2

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(q, emp, color=c_emp, label='empirical')
    ax.plot(q, est, color=c_est, label='estimation')

    ax.fill_between(q, lci_emp, uci_emp, color=c_emp, alpha=0.3)
    ax.fill_between(q, lci_est, uci_est, color=c_est, alpha=0.3)

    ax.set(xlabel='Diversity order $q$', ylabel='Hill numbers', ylim=(y_min, y_max),
           title=title)

    ax.legend(
        bbox_to_anchor=(0.5, 1.05),
        loc='upper center',
        ncol=2,
        # mode='expand',
        borderaxespad=0.0,
        facecolor='white',
        framealpha=1,
    )

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
        from `copia.diversity.evenness()`.
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
        _, ax = plt.subplots(figsize=figsize)

    q = np.arange(q_min, q_max + step, step)

    for i, (label, evenness) in enumerate(evennesses.items()):
        ax.plot(q, evenness, label=label, c=f"C{i}")

    ax.set_xlabel('Diversity order $q$')
    ax.set_ylabel(r'Evenness: $({}^qD - 1) / (\hat{S} - 1)$')
    ax.set_title('Evenness profile')
    ax.legend(loc='best')

    ax.set(**kwargs)
    return ax


__all__ = ['abundance_barplot', 'abundance_histogram', 'accumulation_curve',
           'minsample_diagnostic_plot', 'hill_plot', 'evenness_plot']
