.. Copia documentation master file, created by
   sphinx-quickstart on Wed Feb  3 09:47:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Copia - Bias correction for richness in abundance data
======================================================

.. image:: _static/img/copia.png
  :width: 200
  :align: right
  :alt: Copia Logo by Lauren Fonteyn

.. include:: description.rst
             
.. include:: etymology.rst

See :doc:`the quickstart <quickstart>` to get started.

Copia is licensed under BSD3_. The source is on GitHub_.

.. _BSD3: https://opensource.org/licenses/BSD-3-Clause
.. _GitHub: https://github.com/mikekestemont/copia

Currently implemented diversity estimation methods
--------------------------------------------------

- Empirical Richness: empirical species richness of an assemblage;
- Chao1: Chao1 estimate of bias-corrected species richness;
- "Improved" iChao1 estimate of bias-corrected species richness;
- Egghe & Proot: Egghe & Proot estimate of bias-corrected species richness;
- ACE: ACE estimate of bias-corrected species richness (Chao & Lee 1992);
- Jacknife: Jackknife estimate of bias-corrected species richness;
- Minsample: Observed population size added to the minimum additional sampling estimate; 


.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    quickstart

.. toctree::
    :maxdepth: 2
    :caption: Documentation

    api
    


Citation
--------

Copia is developed by Mike Kestemont and Folgert Karsdorp. If you wish to cite Copia,
please cite the following paper: 

.. code-block:: console

    @article{kestemont,
      author = {Mike Kestemont}
    }
