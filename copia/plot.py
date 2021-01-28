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

def abundance_counts(x):
    plt.clf()
    plt.Figure(figsize=(14, 5))
    ax = plt.gca()
    x = np.array(sorted(x, reverse=True))
    plt.bar(range(len(x)), x, alpha=.7, align='center',
            color=next(ax._get_lines.prop_cycler)['color'])
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    plt.xlabel('Species')
    plt.ylabel('Number of sightings')
    plt.title('Distribution of sightings over species')
    
    textstr = '\n'.join((
        f'Species: {len(x[x > 0])}',
        f'Observations: {x.sum()}',
        f'$f_1$: {(x == 1).sum()}',
        f'$f_2$: {(x == 2).sum()}',
        ))
    plt.annotate(textstr, xy=(0.75, 0.75), xycoords='axes fraction',
                 va='center', backgroundcolor='white')

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, _ = curve_fit(func, range(len(x)), x,
          bounds=([-np.inf, 0.0001, -np.inf], [np.inf, 10, np.inf]))
    ax2 = plt.gca().twinx()
    ax2.grid(None)
    ax2.plot(range(len(x)), func(range(len(x)), *popt), 'r--', 
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    ax2.set_ylabel("Exponential fit")
    ax2.set_ylim((1, max(x)))


def abundance_histogram(x):
    plt.clf()
    x = np.array(sorted(x, reverse=True))
    plt.Figure(figsize=(14, 5))
    ax = plt.gca()

    textstr = '\n'.join((
        f'Species: {len(x[x > 0])}',
        f'Observations: {x.sum()}',
        f'$f_1$: {(x == 1).sum()}',
        f'$f_2$: {(x == 2).sum()}',
        ))
    
    counter = Counter(x)
    max_count = max(counter.keys())
    pos = [k for k in range(1, max_count + 1)]
    x = np.array([counter[k] for k in pos])

    plt.bar(pos, x, alpha=.7, align='center',
            color=next(ax._get_lines.prop_cycler)['color'])

    plt.xlabel('Species')
    plt.title('Sightings histogram')
    
    
    plt.annotate(textstr, xy=(0.7, 0.7), xycoords='axes fraction',
                 va='center', backgroundcolor='white')
    
    # https://github.com/jkitzes/macroeco/blob/master/macroeco/models/_distributions.pys
    mu = np.mean(x)
    eq = lambda p, mu: -p/np.log(1-p)/(1-p) - mu
    p = optim.brentq(eq, 1e-16, 1-1e-16, args=(mu), disp=True)
    estims = logser.pmf(pos, p)

    ax2 = plt.gca().twinx()
    ax2.plot(pos, estims, 'r--')
    ax2.grid(None)
    ax2.set_ylabel("Fisher's log series (pmf)")
    ax2.set_ylim((0, 1))


def richness_density(d, empirical=None, normalize=False, title=None):
    if normalize and not empirical:
        msg = """If normalize is  set to True, `empirical`
                 richness must be provided."""
        raise ValueError(msg)
    plt.Figure(figsize=(15, 15))

    if normalize:
        d['bootstrap'] = empirical / d['bootstrap']

    sb.displot(d['bootstrap'], kde=True)

    q_11, q_50, q_89 = quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
    q_m, q_p = q_50 - q_11, q_89 - q_50

    plt.axvline(q_50, color='red')
    plt.axvline(q_11, ls='--', color='red')
    plt.axvline(q_89, ls='--', color='red')

    if not normalize and empirical:
        plt.axvline(empirical, ls='--', color='green', linewidth=2)

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(".2f").format
    textstr = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    textstr = textstr.format(fmt(q_50), fmt(q_m), fmt(q_p))
    textstr = 'Estimate: ' + textstr

    plt.annotate(textstr, xy=(0.5, 0.7), xycoords='axes fraction',
                 va='center_baseline', backgroundcolor='white', 
                 fontsize=12)
    
    if normalize:
        plt.xlim([0, 1])
    
    if not survival:
        plt.xlabel(r"Richness")
    else:
        plt.xlabel(r"Survival ratio")
    plt.ylabel(r"Kernel density")
    if not title:
        plt.title(r"Estimate: bootstrap values (KDE and CI)")
    else:
        plt.title(title)


def survival(assemblages, method='chao1'):
    survival_estimates = []
    plt.Figure(figsize=(16, 8))

    for label, assemblage in assemblages.items():
        d = bootstrap(assemblage, fn=estimators[method])
        
        if method == 'minsample':
            # normalize to proportions:
            empirical = empirical_richness(assemblage, species=False)
            surv_norm = 1 / (d['richness'] / empirical)
            bt_norm = 1 / (d['bootstrap'] / empirical)
            lci = 1 / (d['lci'] / empirical)
            uci = 1 / (d['uci'] / empirical)
        
            # plot and append:
            survival_estimates.append((label, surv_norm, lci, uci))
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            sb.kdeplot(bt_norm, label=label, ax=plt.gca(),
                       color=color, shade=True)
            plt.axvline(surv_norm, linewidth=2, color=color)
        
        else:
            # normalize to proportions:
            empirical = empirical_richness(assemblage, species=True)
            surv_norm = empirical / d['richness']
            bt_norm = empirical / d['bootstrap']
            lci = empirical / d['lci']
            uci = empirical / d['uci']

            # plot and append:
            survival_estimates.append((label, surv_norm, lci, uci))
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            sb.kdeplot(bt_norm, label=label, ax=plt.gca(),
                       color=color, shade=True)
            plt.axvline(surv_norm, linewidth=2, color=color)
            #q_11, q_50, q_89 = quantile(bt_norm, [0.11, 0.5, 0.89], weights=None)
            #plt.axvline(q_11, ls='--', color=color)
            #plt.axvline(q_89, ls='--', color=color)
        
    if method == 'minsample':
        plt.xlim([0, .5])
        plt.ylabel('Kernel density estimate (sightings)')
        plt.xlabel('Proportion of attested sightings')
    else:
        plt.xlim([0, 1])
        plt.ylabel('Kernel density estimate (species survival)')
        plt.xlabel('Percentage of surviving species')
    
    plt.legend()

    return pd.DataFrame(survival_estimates,
                        columns=['label', 'survival', 'lCI', 'uCI'])

def survival_error(df):
    estimates = df.sort_values('survival')
    errors = np.array(list(zip(estimates['lCI'], estimates['uCI']))).T
    errors[0] = estimates['survival'] - errors[0]
    errors[1] -=  estimates['survival']

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(np.arange(len(estimates)),
            estimates['survival'],
            yerr=errors,
            fmt='.',
            c='green',
            label='diversity',
            ms=12)
    
    plt.ylabel('Estimated survival ratio')
    ax.set_xticks(np.arange(len(estimates)))
    ax.set_xticklabels(estimates['label'], fontsize=12)


def species_accumulation_curve(x, max_steps=None, incl_minsample=False):
    x = np.array(x, dtype=np.int)
    plt.Figure(figsize=(16, 12))

    # min sample estimate to estimate max steps:
    minsample_est = diversity(x, method='minsample', 
                              solver='fsolve', CI=True)
    q_11, q_50, q_89 = quantile(minsample_est['bootstrap'],
                               [0.11, 0.5, 0.89], weights=None)
    
    if not max_steps:
        if incl_minsample:
            max_steps = int(max(minsample_est['bootstrap']))
        else:
            max_steps = int(q_89)
    
    steps = np.arange(1, max_steps)
    interpolated = np.arange(1, max_steps) < x.sum()

    estim = bootstrap(x, fn=partial(rarefaction_extrapolation,
                                    max_steps=max_steps),
                      n_iter=100)
    
    lci_pro = estim['lci']
    uci_pro = estim['uci']
    Dq = estim['richness']

    # species accumulation
    plt.plot(x.sum(), Dq[x.sum() - 1], 'o', markersize=8)
    plt.plot(steps[interpolated], Dq[interpolated], color='C0')
    plt.plot(steps[~interpolated], Dq[~interpolated], '--', color='C0')
    plt.fill_between(steps, lci_pro, uci_pro, alpha=0.3)
    plt.gca().set_ylim((0, plt.gca().get_ylim()[1]))

    # min sample:
    if incl_minsample:
        ax2 = plt.gca().twinx()
        sb.kdeplot(minsample_est['bootstrap'], ax=ax2,
                color='green', fill=True)

        plt.axvline(q_50, color='green')
        plt.axvline(q_11, ls='--', color='green')
        plt.axvline(q_89, ls='--', color='green')

        plt.gca().set_ylim((0, plt.gca().get_ylim()[1]))
        ax2.grid(None)

    # cosmetics etc.
    plt.xlabel('sightings')
    plt.ylabel('species')
    plt.title('Species Accumulation Curve')    


def hill_plot(emp, est, q_min=0, q_max=3, step=0.1,
              figsize=None, ax=None, add_densities=True,
              title=None):

    plt.style.use('bmh')
    COLORS = BLUE, RED, PURPLE = '#348ABD', '#A60628', '#7A68A6'
    
    c_emp, c_est = RED, BLUE
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

    ax.set_xlabel('Order $q$')
    ax.set_ylabel('Hill numbers')
    ax.set_ylim(y_min, y_max)

    if title:
        plt.title(title)

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
        ax2 = plt.gcf().add_axes([left, bottom, width, height])
        labels = 'Richness', 'Shannon', 'Simpson'

        for k, (i, label) in enumerate(zip(np.where(np.isin(q, (0, 1, 3)))[0], labels)):
            sb.kdeplot(
                bt_est[:, i], label=label, c = COLORS[k], ax=ax2
            )
        
        l1,  l2, l3 = ax2.lines

        # Get the xy data from the lines so that we can shade
        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        x3 = l3.get_xydata()[:, 0]
        y3 = l3.get_xydata()[:, 1]

        ax2.fill_between(x1, y1, color=BLUE, alpha=0.3)
        ax2.fill_between(x2, y2, color=RED, alpha=0.3)
        ax2.fill_between(x3, y3, color=PURPLE, alpha=0.3)

        ax2.set_xlabel('Hill numbers')
        ax2.set_ylabel('Density')

