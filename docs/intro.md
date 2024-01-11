(intro)= 

# Copia - Diversity Estimation in Cultural Data

Copia is a statistical software package for estimating diversity and
richness on the basis of abundance data. The package contains several
bias-correcting richness estimators, such as the Chao1 and the Jacknife
estimator.

Because this package targets so-called "abundance" data from ecology, it has
been named using the corresponding Latin term copia. The artwork for the logo
has been kindly contributed by Lauren Fonteyn. The depicted goat is a
mythological reference to the cornucopia or "horn of plenty", the legendary horn
of the goat Amaltheia, who fed the infant Zeus with her milk.

Copia is licensed under [BSD3](https://opensource.org/licenses/BSD-3-Clause).
The source is on [GitHub](https://github.com/mikekestemont/copia).

## Currently implemented diversity estimation methods

- Empirical Richness: empirical species richness of an assemblage;
- Chao1: Chao1 estimate of bias-corrected species richness;
- "Improved" iChao1 estimate of bias-corrected species richness;
- Egghe & Proot: Egghe & Proot estimate of bias-corrected species richness;
- ACE: ACE estimate of bias-corrected species richness (Chao & Lee 1992);
- Jacknife: Jackknife estimate of bias-corrected species richness;
- Minsample: Observed population size added to the minimum additional sampling
  estimate;
  
## Installation

Copia requires Python 3.11 or greater. To install Copia, execute the following
command in your terminal:

``` bash
python3 -m pip install copia
```

    
### Development
If you wish to help developing Copia, download the source files from GitHub and
install the package using the following command:

``` bash
pip install -e .
```

## Citation

Copia is developed by Folgert Karsdorp and Mike Kestemont. If you wish to cite Copia,
please cite one of the following papers: 

``` bibtex
@article{doi:10.1126/science.abl7655,
  author = {Mike Kestemont  and Folgert Karsdorp
            and Elisabeth de Bruijn and Matthew Driscoll
            and Katarzyna A. Kapitan and Pádraig Ó Macháin
            and Daniel Sawyer  and Remco Sleiderink and Anne Chao},
  title = {Forgotten books: The application of unseen species models to
           the survival of culture},
  journal = {Science},
  volume = {375},
  number = {6582},
  pages = {765-769},
  year = {2022},
  doi = {10.1126/science.abl7655},
  URL = {https://www.science.org/doi/abs/10.1126/science.abl7655}}
  
@inproceedings{karsdorpDarkNumbersModeling2023,
  title = {Dark Numbers: Modeling the Historical Vulnerability to Arrest in Brussels (1879-1880) Using Demographic Predictors},
  booktitle = {Proceedings of the DHBenelux Conference 2023},
  author = {Karsdorp, Folgert and Kestemont, Mike and De Koster, Margo},
  year = {2023},
  address = {Brussels}
}

@inproceedings{karsdorpWhatShallWe2022,
  title = {What Shall We Do With the Unseen Sailor? Estimating the Size of the Dutch East India Company Using an Unseen Species Model},
  booktitle = {Proceedings of the Computational Humanities Research Conference, 2022},
  author = {Karsdorp, Folgert and Wevers, Melvin and {van Lottum}, Jelle},
  year = {2022},
  pages = {189--197},
  address = {Antwerp},
}

@inproceedings{karsdorpIntroducingFunctionalDiversity2022,
  title = {Introducing Functional Diversity: A Novel Approach to Lexical Diversity in (Historical) Corpora},
  booktitle = {Proceedings of the Computational Humanities Research Conference, 2022},
  author = {Karsdorp, Folgert and Manjavacas, Enrique and Fonteyn, Lauren},
  year = {2022},
  pages = {114--126},
  publisher = {CEUR-WS},
  address = {Antwerp},
  langid = {english},
  file = {/Users/folgert/Zotero/storage/9H2Z4PFR/Karsdorp et al. - Introducing Functional Diversity A Novel Approach.pdf}
}


```

