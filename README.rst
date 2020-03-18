gvar
------
This package facilitates the creation and manipulation of arbitrarily
complicated (correlated) multi-dimensional Gaussian random variables.
The random variables are represented by a new data type (``gvar.GVar``)
that can be used in arithmetic expressions and pure Python functions. Such
expressions/functions create new Gaussian random variables
while automatically tracking statistical correlations between the new
and old variables. This data type is useful for simple error propagation,
but also is heavily used by the Bayesian least-squares fitting module
``lsqfit.py`` to define priors and specify fit results, while accounting
for correlations between all variables. Documentation can is in the
``doc/`` subdirectory: see ``doc/html/index.html``
or look online at <https://gvar.readthedocs.io>.

These packages use ``numpy`` for efficient array arithmetic, and ``cython``
to compile efficient code. ``gvar`` uses automatic differentiation to
track covariances through arbitrary arithmetic.

Information on how to install the components is in the ``INSTALLATION`` file.

To test the libraries try ``make tests``. Some
examples are give in the ``examples/`` subdirectory.

``gvar`` version numbers have the form ``major.minor.patch`` where:
incompatible changes are signaled by incrementing the ``major`` version
number, the ``minor`` number signals new features, and the ``patch`` number
signals bug fixes.

| Created by G. Peter Lepage (Cornell University) 2008
| Copyright (c) 2008-2020 G. Peter Lepage

.. image:: https://zenodo.org/badge/37556070.svg
   :target: https://zenodo.org/badge/latestdoi/37556070

