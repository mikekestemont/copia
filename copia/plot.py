import seaborn as sb
import matplotlib.pyplot as plt

from .stats import quantile

def autoplot(d):
    fig, ax = plt.subplots(figsize=(15, 10))

    if isinstance(d, dict):
        sb.distplot(d['bootstrap'], ax=ax)

        ax.set_xlabel(r"Richness")
        ax.set_ylabel(r"Kernel density")

        q_11, q_50, q_89 = quantile(d['bootstrap'], [0.11, 0.5, 0.89], weights=None)
        q_m, q_p = q_50 - q_11, q_89 - q_50
        plt.axvline(q_50, color='red')

        ax.axvline(q_11, ls="--", color="red")
        ax.axvline(q_89, ls="--", color="red")        

        # Format the quantile display.
        fmt = "{{0:{0}}}".format(".3f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
        plt.title(title, {'fontname':'Arial'})

def species_accumulation_curve():
    pass

def CI():
    pass