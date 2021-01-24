import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.optimize import curve_fit

from .stats import quantile

def abundance_counts(x):
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.array(sorted(x, reverse=True))
    plt.bar(range(len(x)), x,
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
    c = next(ax._get_lines.prop_cycler)['color']
    plt.plot(range(len(x)), func(range(len(x)), *popt), 'b--', c=c,
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))


def abundance_histogram(x):
    plt.clf()
    x = np.array(sorted(x, reverse=True))
    sb.displot(x, kde=True)

    plt.xlabel('Species')
    plt.title('Sightings histogram')
    
    textstr = '\n'.join((
        f'Species: {len(x[x > 0])}',
        f'Observations: {x.sum()}',
        f'$f_1$: {(x == 1).sum()}',
        f'$f_2$: {(x == 2).sum()}',
        ))
    plt.annotate(textstr, xy=(0.7, 0.7), xycoords='axes fraction',
                 va='center', backgroundcolor='white')


def richness_density(d):
    sb.displot(d['bootstrap'], kde=True)

    q_11, q_50, q_89 = quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
    q_m, q_p = q_50 - q_11, q_89 - q_50
    plt.axvline(q_50, color='red')

    plt.axvline(q_11, ls="--", color="red")
    plt.axvline(q_89, ls="--", color="red")        

    # Format the quantile display.
    fmt = "{{0:{0}}}".format(".3f").format
    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
    plt.title(title, {'fontname':'Arial'})
    plt.xlabel(r"Richness")
    plt.ylabel(r"Kernel density")

def autoplot(d):
    if isinstance(d, dict):
        richness_density(d)
    else:
        abundance_histogram(d)
        

def species_accumulation_curve(x, incl_minsample=True):

    pass