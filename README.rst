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
lsqfit.py — to define priors and specify fit results, while accounting
for correlations between all variables. Documentation can is in the 
``doc/`` subdirectory (see ``doc/html/index.html`` for the html version or 
``doc/lsqfit.pdf`` for a pdf version).

These packages use numpy for efficient array arithmetic, and cython 
to compile efficient code. ``gvar`` uses automatic differentiation to 
track covariances through arbitrary arithmetic.

Information on how to install the components is in the ``INSTALLATION`` file. 

To test the libraries try ``make tests``. (Some tests involve random
numbers and so may occasionally — less than 1 in 100 runs — fail due to
rare multi-sigma fluctuations; rerun the tests if they do fail.) Some
examples are give in the ``examples/`` subdirectory.

Versioning: Version numbers for ``gvar`` are now (5.0 and later) based upon
*semantic  versioning* (http://semver.org). Incompatible changes will be
signaled by incrementing the major version number, where version numbers have
the form major.minor.patch. The minor number signals new features, and the
patch number bug fixes.

| Created by G. Peter Lepage (Cornell University) 2008
| Copyright (c) 2008-2015 G. Peter Lepage
