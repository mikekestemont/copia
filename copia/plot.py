from collections import Counter

import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import scipy.optimize as optim
from scipy.optimize import curve_fit
from scipy.stats import logser


from .stats import quantile

def abundance_counts(x):
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 5))
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
    fig, ax = plt.subplots(figsize=(14, 5))

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


def richness_density(d):
    plt.Figure(figsize=(15, 15))
    sb.displot(d['bootstrap'], kde=True)

    q_11, q_50, q_89 = quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
    q_m, q_p = q_50 - q_11, q_89 - q_50
    plt.axvline(q_50, color='red')

    plt.axvline(q_11, ls="--", color="red")
    plt.axvline(q_89, ls="--", color="red")        

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(".3f").format
    textstr = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    textstr = textstr.format(fmt(q_50), fmt(q_m), fmt(q_p))
    textstr = 'Estimate: ' + textstr

    plt.annotate(textstr, xy=(0.5, 0.7), xycoords='axes fraction',
                 va='center_baseline', backgroundcolor='white', 
                 fontsize=12)
    
    plt.xlabel(r"Richness")
    plt.ylabel(r"Kernel density")
    plt.title(r"Estimate: bootstrap values (KDE and CI)")


def species_accumulation_curve(x):
    
    

def autoplot(d):
    if isinstance(d, dict):
        richness_density(d)
    else:
        abundance_histogram(d)
        

def species_accumulation_curve(x, incl_minsample=True):

    pass