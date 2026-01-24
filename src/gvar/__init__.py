r""" Correlated gaussian random variables.

Objects of type :class:`gvar.GVar` represent gaussian random variables,
which are specified by a mean and standard deviation. They are created
using :func:`gvar.gvar`: for example, ::

    >>> x = gvar.gvar(0, 0.3)          # 0 +- 0.3
    >>> y = gvar.gvar(2, 0.4)          # 2 +- 0.4
    >>> z = x + y                      # 2 +- 0.5
    >>> print(z)
    2.00(50)
    >>> print(z.mean)
    2.0
    >>> print(z.sdev)
    0.5

This module contains tools for creating and manipulating gaussian random
variables including:

    - ``mean(g)`` --- extract means.

    - ``sdev(g)`` --- extract standard deviations.

    - ``var(g)`` --- extract variances.

    - ``correlate(g, corr)`` --- add correlations to GVars in dict/array g.

    - ``chi2(g1, g2)`` --- ``chi**2`` of ``g1-g2``.

    - ``equivalent(g1, g2)`` --- ``g1`` and ``g2`` the same?

    - ``evalcov(g)`` --- compute covariance matrix.

    - ``cov(g1, g2)`` --- covariance of :class:`gvar.GVar` ``g1`` with ``g2``.

    - ``evalcov_blocks(g)`` --- compute diagonal blocks of covariance matrix.

    - ``evalcorr(g)`` --- compute correlation matrix.

    - ``corr(g1, g2)`` --- correlation between :class:`gvar.GVar` ``g1`` and ``g2``.

    - ``tabulate(g)`` --- create a table of GVar values in dict/array g.

    - ``fmt_values(g)`` --- create table of values.

    - ``fmt_errorbudget(g)`` --- create error-budget table.

    - ``fmt_chi2(f)`` --- format chi**2 information in f.

    - ``sample(g, nbatch)`` --- random sample from |GVar|\s.

    - ``gvar_from_sample(gs)`` --- reconstruct Gaussian distribution from sample.
    
    - ``raniter(g, n, nbatch)`` --- iterator for random numbers.

    - ``bootstrap_iter(g, n)`` --- bootstrap iterator.

    - ``ranseed(seed)`` --- seed random number generator.

    - ``random(size)`` --- generate array of random numbers on [0.0,1.0).

    - ``regulate(g, eps|svdcut)`` --- regulate correlation matrix.

    - ``svd(g, svdcut)`` --- SVD regulation of correlation matrix.

    - ``PDF(g)`` --- (class) probability density function.

    - ``PDFStatistics`` --- (class) statistical analysis of moments of a random variable.

    - ``BufferDict`` --- (class) ordered dictionary with data buffer.

    - ``disassemble(g)`` --- disassemble |GVar|\s in ``g``.

    - ``reassemble(data, cov)`` --- reassemble into |GVar|\s.

    - ``load(inputfile)`` --- read |GVar|\s from a file.

    - ``loads(inputstr)`` --- read |GVar|\s from a string.

    - ``dump(g, outputfile)`` --- store |GVar|\s in file.

    - ``dumps(g)`` --- store |GVar|s in a string.

    - ``dataset.bin_data(data)`` --- bin random sample data.

    - ``dataset.avg_data(data)`` --- estimate means from data.

    - ``dataset.bootstrap_iter(data,N)`` --- bootstrap data.

    - ``dataset.Dataset`` --- class for collecting data.

There are also sub-modules that implement some standard numerical analysis
tools for use with |GVar|\s (or ``float``\s):

    - ``cspline`` --- cubic splines for 1-d data.

    - ``ode`` --- integration of systems of ordinary differential
        equations; one dimensional integrals.

    - ``linalg`` --- basic linear algebra.

    - ``pade`` --- Pade approximants of functions.

    - ``powerseries`` --- power series representations
        of functions.

    - ``root`` --- root-finding for one-dimensional functions.
"""

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-24  G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

__version__='13.1.9'

import collections
import sys 
import numpy

from ._gvarcore import sin, cos, tan, exp, log, sqrt, fabs, sinh, cosh, tanh
from ._gvarcore import arcsin, arccos, arctan, arctan2, arcsinh, arccosh, arctanh, square
from ._gvarcore import GVar, GVarFactory, gvar_function, abs, wsum_der, msum_gvar, wsum_gvar
gvar = GVarFactory()            # order matters for this statement (don't move down)

from ._svec_smat import svec, smask, smat
from ._bufferdict import BufferDict, asbufferdict, _BDict_UDistribution, BUFFERDICTDATA
from ._bufferdict import has_dictkey, dictkey, get_dictkeys
from ._bufferdict import trim_redundant_keys    # legacy
from ._bufferdict import add_parameter_parentheses, nonredundant_keys   # legacy
# from ._utilities import *
from ._utilities import rebuild, mean, fmt, sdev, deriv, is_primary
from ._utilities import dependencies, missing_dependencies, uncorrelated, evalcorr, corr, cov
from ._utilities import correlate, evalcov_blocks_dense, evalcov_blocks, var, evalcov
from ._utilities import distribute_gvars, remove_gvars, collect_gvars, filter
from ._utilities import dumps, loads, dump, load, gdumps, gdump, gloads, gload #, olddump
from ._utilities import disassemble, reassemble, fmt_values
from ._utilities import fmt_errorbudget, bootstrap_iter, sample, gvar_from_sample, raniter 
from ._utilities import valder, gammaQ, gammaP, regulate, svd, erf, SVD, GVarRef

from gvar import dataset
from gvar import ode
from gvar import cspline
from gvar import linalg
from gvar import pade
from gvar import powerseries
from gvar import root

_GVAR_LIST = []
_CONFIG = dict(evalcov=15, evalcov_blocks=6000, var=50)


def _v0_RNG(): 
    " RNG for numpy versions less than 1.17. " 
    def integers(low, high, size=None, dtype=numpy.int64):
        return numpy.random.randint(low=low, high=high, size=size, dtype=dtype)
    numpy.random.integers = integers
    return numpy.random

if tuple([int(si) for si in numpy.__version__.split('.')]) >= (1, 17, 0):
    RNG = numpy.random.default_rng()
else:
    RNG = _v0_RNG()

# ``gvar.RNG`` is set equal to ``numpy.random.default_rng()`` by default
# (or ``numpy.random`` for ``numpy`` versions earlier than 1.17). Could 
# replace this with some other generator provided it has methods ``random``,
# ``normal``, ``uniform``, and ``integers`` analogous to those provided 
# by the ``numpy`` generators. Setting gvar.RNG = gvar._v0_RNG() restores
# the random number generator used before :mod:`gvar` version 12.1 (but 
# see gvar.old_ranseed()).

def ranseed(seed=None, size=3, version=None):
    r""" Seed random number generators with tuple ``seed``.

    Argument ``seed`` is an integer or
    a :class:`tuple` of integers that is used to seed
    the random number generators used by by :mod:`gvar`. Reusing
    the same ``seed`` results in the same set of random numbers.

    ``ranseed`` generates its own seed when called without an argument
    or with ``seed=None``. This seed is stored in ``ranseed.seed`` and
    is also returned by the function. The seed can be used to regenerate
    the same set of random numbers at a later time. Keyword argument 
    ``size`` specifies the size or shape of the new seed.

    :mod:`gvar`'s random number generator is ``gvar.RNG``, which 
    is (currently) an instance of the ``numpy.random.Generator`` class.
    (Random numbers are generated by calling routines like 
    ``gvar.RNG.random(size)``; see the documentation 
    for ``numpy.random.Generator``.) Calling ``gvar.ranseed`` assigns 
    ``gvar.RNG`` to a new generator of this type, corresponding
    to the seed.

    Note: The random number generator changed in version 12.1 
    of :mod:`gvar` in order to track changes in version 1.17 of 
    :mod:`numpy`, which provides the generator. To restore the 
    old generator set keyword ``version=0``. This option should
    only be used for legacy code as it is likely to 
    disappear eventually. Note that ``gvar.RNG=numpy.random`` in 
    this case. This is also the choice if the ``numpy`` version 
    is earlier than |~| 1.17.

    Args:
        seed (int, tuple, or None): Seed for generator. Generates a
            random tuple for the seed if ``None``.
        size (int): Size of the tuple of integers used 
            to seed the random number generator when ``seed=None``.
            Default is ``size=3``. Ignored if ``seed`` is not
            ``None``.
        version (int or None): Version of random number generator:
            ``version=0`` specifies the generator used in :mod:`gvar` 
            versions earlier than 12.1; ``version=None`` (default) 
            specifies the current random number generator.
    Returns:
        The seed used to reseed the generator.
    """
    global RNG
    if RNG == numpy.random and version is None:
        # version=0 implies numpy.random
        version = 0
    elif version in [1, None]:
        version = None
    else:
        raise ValueError('unknown version = {}'.format(version))
    if version == 0:
        # old generator
        RNG = _v0_RNG()
    if seed is None:
        seed = RNG.integers(1, min(2**30, sys.maxsize), size=size)
    try:
        seed = tuple(seed)
    except TypeError:
        pass
    if version is None:
        RNG = numpy.random.default_rng(seed)
    else:
        # old generator
        numpy.random.seed(seed)
    ranseed.seed = seed
    return seed

def switch_gvar(cov=None):
    r""" Switch :func:`gvar.gvar` to new :class:`gvar.GVarFactory`.

    |GVar|\s created from different factory functions (see 
    :func:`gvar.gvar_factory`), with different 
    covariance matrices, should never be mixed in arithmetic or other 
    expressions. Such mixing is unsupported and will result in 
    unpredictable behavior. Arithmetic that mixes |GVar|\s in 
    this way will generate an error message: "incompatible GVars".

    Args:
        cov: Covariance matrix for new :func:`gvar.gvar`. A new 
            covariance matrix is created if ``cov=None`` (default). 
            If ``cov`` is a |GVar|, the covariance matrix of 
            the |GVar| is used. 

    Returns:
        New :func:`gvar.gvar`.
    """
    global gvar
    _GVAR_LIST.append(gvar)
    if isinstance(cov, GVar):
        cov = cov.cov
    gvar = GVarFactory(cov)
    return gvar

def restore_gvar():
    r""" Restore previous :func:`gvar.gvar`.

    |GVar|\s created from different factory functions (see 
    :func:`gvar.gvar_factory`), with different 
    covariance matrices, should never be mixed in arithmetic or other 
    expressions. Such mixing is unsupported and will result in 
    unpredictable behavior. Arithmetic that mixes |GVar|\s in 
    this way will generate an error message: "incompatible GVars".

    :returns: Previous :func:`gvar.gvar`.
    """
    global gvar
    try:
        gvar = _GVAR_LIST.pop()
    except IndexError:
        raise RuntimeError("no previous gvar")
    return gvar

def gvar_factory(cov=None):
    r""" Return new function for creating |GVar|\s (to replace
    :func:`gvar.gvar`).

    If ``cov`` is specified, it is used as the covariance matrix
    for new |GVar|\s created by the function returned by
    ``gvar_factory(cov)``. Otherwise a new covariance matrix is created
    internally.

    |GVar|\s created from different factory functions, with different 
    covariance matrices, should never be mixed in arithmetic or other 
    expressions. Such mixing is unsupported and will result in 
    unpredictable behavior. Arithmetic that mixes |GVar|\s in 
    this way will generate an error message: "incompatible GVars".
    """
    return GVarFactory(cov)

def qqplot(g1, g2=None, plot=None, eps=None, svdcut=None, dof=None, nocorr=False):
    r""" QQ plot ``g1-g2``.

    The QQ plot compares the distribution of the means of Gaussian 
    distribution ``g1-g2`` to that of random samples from the 
    distribution. The resulting plot will approximate a straight 
    line along the diagonal of the plot (dashed black line) if 
    the means have a Gaussian distribution about zero with
    the correct standard deviations.

    Usually ``g1`` and ``g2`` are dictionaries with the same keys,
    where ``g1[k]`` and ``g2[k]`` are |GVar|\s or arrays of
    |GVar|\s having the same shape. Alternatively ``g1`` and ``g2``
    can be |GVar|\s, or arrays of |GVar|\s having the same shape.

    One of ``g1`` or ``g2`` can contain numbers instead of |GVar|\s.

    One or the other of ``g1`` or ``g2`` can be missing keys, or missing
    elements from arrays. Only the parts of ``g1`` and ``g2`` that
    overlap are used. Also setting ``g2=None`` is equivalent 
    to replacing its elements by zeros.

    In a typical application, the plot displayed by ::

        gvar.qqplot(gsample, g).show()

    tests whether ``gsample`` is likely a random sample  
    from a multi-dimensional Gaussian distribtuion ``g``. 
    It is consistent with being a random sample if the 
    QQ plot is a straight line through the origin
    with unit slope. The is most useful when there are 
    many variables (``g.size >> 1``).

    Args:
        g1: |GVar| or array of |GVar|\s, or a dictionary whose values are
            |GVar|\s or arrays of |GVar|\s. Specifies a multi-dimensional
            Gaussian distribution. Alternatively the elements can be 
            numbers instead of |GVar|\s, in which case ``g1`` specifies 
            a sample from a distribution.
        g2: |GVar| or array of |GVar|\s, or a dictionary whose values are
            |GVar|\s or arrays of |GVar|\s. Specifies a multi-dimensional
            Gaussian distribution. Alternatively the elements can be 
            numbers instead of |GVar|\s, in which case ``g2`` specifies 
            a sample from a distribution. Setting ``g2=None`` 
            (default) is equivalent to setting its elements all to zero.
        plot: a :mod:`matplotlib` plotter. If ``None`` (default), 
            uses ``matplotlib.pyplot``.
        eps (float or None): ``eps`` used by :func:`gvar.regulate` when 
            inverting the covariance matrix. Ignored if ``svdcut`` is 
            set (and not ``None``). 
        svdcut (float): SVD cut used when inverting the covariance
            matrix of ``g1-g2``. See documentation for :func:`gvar.svd` 
            for more information. Default is ``svdcut=1e-12``.
        dof (int or None): Number of independent degrees of freedom in
            ``g1-g2``. This is set equal to the number of elements in
            ``g1-g2`` if ``dof=None`` is set. This parameter affects
            the ``Q`` value assigned to the ``chi**2``.

    Returns:
        Plotter ``plot``.

    This method requires the :mod:`scipy` and :mod:`matplotlib` modules.
    """
    try:
        import scipy 
        from scipy import stats 
    except ImportError:
        warnings.warn('scipy module not installed; needed for qqplot_residuals()')
        return
    if plot is None:
        import matplotlib.pyplot as plot
    chi2g1g2 = chi2(g1, g2, eps=eps, svdcut=svdcut, nocorr=nocorr, dof=dof)
    (x, y), (s,y0,r) = stats.probplot(chi2g1g2.residuals, plot=plot, fit=True)
    minx = min(x)
    maxx = max(x)
    plot.plot([minx, maxx], [minx, maxx], 'k:')
    text = (
        '{}\n' r'residual = {:.2f} + {:.2f} $\times$ theory' '\nr = {:.2f}').format(
        fmt_chi2(chi2g1g2), y0, s, r
        )
    plot.title('Q-Q Plot')
    plot.ylabel('Ordered residuals')
    ylim = plot.ylim()
    plot.text(minx, ylim[0] + (ylim[1] - ylim[0]) * 0.84,text, color='r')
    return plot 

def chi2(g1, g2=None, eps=None, svdcut=None, dof=None, nocorr=False):
    r""" Compute chi**2 of ``g1-g2``.

    chi**2 equals ``dg.invcov.dg, where ``dg = g1 - g2`` and 
    ``invcov`` is the inverse of ``dg``'s covariance matrix. 
    It is a measure  of how well multi-dimensional Gaussian 
    distributions ``g1`` and ``g2`` (dictionaries or arrays)  
    agree with each other ---  that is, do their means agree 
    within errors for corresponding elements. The probability 
    is high if ``chi2(g1,g2)/dof`` is of order 1 or smaller,
    where ``dof`` is the number  of degrees of freedom
    being compared.

    Usually ``g1`` and ``g2`` are dictionaries with the same keys,
    where ``g1[k]`` and ``g2[k]`` are |GVar|\s or arrays of
    |GVar|\s having the same shape. Alternatively ``g1`` and ``g2``
    can be |GVar|\s, or arrays of |GVar|\s having the same shape.

    One of ``g1`` or ``g2`` can contain numbers instead of |GVar|\s,
    in which case ``chi**2`` is a measure of the likelihood that
    the numbers came from the distribution specified by the other
    argument.

    One or the other of ``g1`` or ``g2`` can be missing keys, or missing
    elements from arrays. Only the parts of ``g1`` and ``g2`` that
    overlap are used. Also setting ``g2=None`` is equivalent 
    to replacing its elements by zeros.

    A typical application tests whether distribution ``g1`` and ``g2`` 
    are consistent with each other (within errors): the code ::

        >>> chi2 = gvar.chi2(g1, g2)
        >>> print(gvar.fmt_chi2(chi2))
        chi2/dof = 1.1 [100]    Q = 0.26

    shows that the distributions are reasonably consistent. The 
    number of degrees of freedom (here 100) in this example equals the 
    number of variables from ``g1`` and ``g2`` that are compared;
    this can be changed using the ``dof`` argument.

    Args:
        g1: |GVar| or array of |GVar|\s, or a dictionary whose values are
            |GVar|\s or arrays of |GVar|\s. Specifies a multi-dimensional
            Gaussian distribution. Alternatively the elements can be 
            numbers instead of |GVar|\s, in which case ``g1`` specifies 
            a sample from a distribution.
        g2: |GVar| or array of |GVar|\s, or a dictionary whose values are
            |GVar|\s or arrays of |GVar|\s. Specifies a multi-dimensional
            Gaussian distribution. Alternatively the elements can be 
            numbers instead of |GVar|\s, in which case ``g2`` specifies 
            a sample from a distribution. Setting ``g2=None`` 
            (default) is equivalent to setting its elements all to zero.
        eps (float): If positive, singularities in the correlation matrix 
            for ``g1-g2`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. Ignored if ``svdcut`` is specified (and 
            not ``None``).
        svdcut (float): If nonzero, singularities in the correlation 
            matrix for ``g1-g2`` are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. Default is ``svdcut=1e-12``.
        dof (int or None): Number of independent degrees of freedom in
            ``g1-g2``. This is set equal to the number of elements in
            ``g1-g2`` if ``dof=None`` is set. This parameter affects
            the ``Q`` value assigned to the ``chi**2``.
    
    Returns:
        The return value is the ``chi**2``. Extra attributes attached 
        to this number give additional information:

        - **dof** --- Number of degrees of freedom (that is, the number 
            of variables compared if not specified).

        - **Q** --- The probability that the ``chi**2`` could have 
            been larger, by chance, even if ``g1`` and ``g2`` agree. Values 
            smaller than 0.1 or so suggest that they do not agree. 
            Also called the *p-value*.

        - **residuals** --- Decomposition of the ``chi**2`` in terms of the 
            independent modes of the correlation matrix: ``chi**2 = sum(residuals**2)``.
    """
    # customized class for answer
    class ans(float):
        def __new__(cls, chi2, dof, res):
            return float.__new__(cls, chi2)
        def __init__(self, chi2, dof, res):
            self.dof = dof
            self.chi2 = chi2
            self.residuals = res
        def _get_Q(self):
            return gammaQ(self.dof / 2., self.chi2 / 2.)  
        Q = property(_get_Q)

    # leaving nocorr (turn off correlations) undocumented because I
    #   suspect I will remove it
    if g2 is None:
        diff = (
            BufferDict(g1).buf if hasattr(g1, 'keys') else 
            numpy.asarray(g1).flatten()
            )
    elif hasattr(g1, 'keys') and hasattr(g2, 'keys'):
        # g1 and g2 are dictionaries
        g1 = BufferDict(g1)
        g2 = BufferDict(g2)
        diff = BufferDict()
        keys = set(g1.keys())
        keys = keys.intersection(g2.keys())
        for k in keys:
            g1k = g1[k]
            g2k = g2[k]
            shape = tuple(
                [min(s1,s2) for s1, s2 in zip(numpy.shape(g1k), numpy.shape(g2k))]
                )
            diff[k] = numpy.zeros(shape, object)
            if len(shape) == 0:
                diff[k] = g1k - g2k
            else:
                for i in numpy.ndindex(shape):
                    diff[k][i] = g1k[i] - g2k[i]
        diff = diff.buf
    elif not hasattr(g1, 'keys') and not hasattr(g2, 'keys'):
        # g1 and g2 are arrays or scalars
        g1 = numpy.asarray(g1)
        g2 = numpy.asarray(g2)
        shape = tuple(
            [min(s1,s2) for s1, s2 in zip(numpy.shape(g1), numpy.shape(g2))]
            )
        diff = numpy.zeros(shape, object)
        if len(shape) == 0:
            diff = numpy.array(g1 - g2)
        else:
            for i in numpy.ndindex(shape):
                diff[i] = g1[i] - g2[i]
        diff = diff.flatten()
    else:
        # g1 and g2 are something else
        raise ValueError(
            'cannot compute chi**2 for types ' + str(type(g1)) + ' ' +
            str(type(g2))
            )
    if diff.size == 0:
        return ans(0.0, 0)
    if nocorr:
        # ignore correlations
        res = mean(diff) / sdev(diff)
        chi2 = numpy.sum(res ** 2)
        if dof is None:
            dof = len(diff)
    else:
        diffmod, i_wgts = regulate(diff, eps=eps, svdcut=svdcut, wgts=-1)
        diffmean = mean(diffmod)
        res = numpy.zeros(diffmod.shape, float)
        i, wgts = i_wgts[0]
        res[i] = diffmean[i] * wgts
        ilist = i.tolist()
        for i, wgts in i_wgts[1:]:
            res[i[:wgts.shape[0]]] = wgts.dot(diffmean[i])
            ilist.extend(i[:wgts.shape[0]])
        res = res[ilist]
        chi2 = numpy.sum(res ** 2)
        if dof is None:
            dof = len(res)
    return ans(chi2, dof, res)

def equivalent(g1, g2, rtol=1e-10, atol=1e-10):
    r""" Determine whether ``g1`` and ``g2`` contain equivalent |GVar|\s.

    Compares sums and differences of |GVar|\s stored in ``g1``
    and ``g2`` to see if they agree within tolerances. Operationally,
    agreement means that::

        abs(diff) < abs(summ) / 2 * rtol + atol

    where ``diff`` and ``summ`` are the difference and sum of the
    mean values (``g.mean``) or derivatives (``g.der``) associated with
    each pair of |GVar|\s.

    |GVar|\s that are equivalent are effectively interchangeable with respect
    to both their means and also their covariances with any other |GVar|
    (including ones not in ``g1`` and ``g2``).

    ``g1`` and ``g2`` can be individual |GVar|\s or arrays of |GVar|\s
    or dictionaries whose values are |GVar|\s and/or arrays of |GVar|\s.
    Comparisons are made only for shared keys when they are dictionaries.
    Array dimensions must match between ``g1`` and ``g2``, but the shapes
    can be different; comparisons are made for the parts of the arrays that
    overlap in shape.

    :param g1: A |GVar| or an array of |GVar|\s or a dictionary of
        |GVar|\s and/or arrays of |GVar|\s.
    :param g2: A |GVar| or an array of |GVar|\s or a dictionary of
        |GVar|\s and/or arrays of |GVar|\s.
    :param rtol: Relative tolerance with which mean values and derivatives
        must agree with each other. Default is ``1e-10``.
    :param atol: Absolute tolerance within which mean values and derivatives
        must agree with each other. Default is ``1e-10``.
    """
    atol = abs(atol)
    rtol = abs(rtol)
    if hasattr(g1, 'keys') and hasattr(g2, 'keys'):
        # g1 and g2 are dictionaries
        g1 = BufferDict(g1)
        g2 = BufferDict(g2)
        diff = BufferDict()
        summ = BufferDict()
        keys = set(g1.keys())
        keys = keys.intersection(g2.keys())
        for k in keys:
            g1k = g1[k]
            g2k = g2[k]
            shape = tuple(
                [min(s1,s2) for s1, s2 in zip(numpy.shape(g1k), numpy.shape(g2k))]
                )
            diff[k] = numpy.zeros(shape, object)
            summ[k] = numpy.zeros(shape, object)
            if len(shape) == 0:
                diff[k] = g1k - g2k
                summ[k] = g1k + g2k
            else:
                for i in numpy.ndindex(shape):
                    diff[k][i] = g1k[i] - g2k[i]
                    summ[k][i] = g1k[i] + g2k[i]
        diff = diff.buf
        summ = summ.buf
    elif not hasattr(g1, 'keys') and not hasattr(g2, 'keys'):
        # g1 and g2 are arrays or scalars
        g1 = numpy.asarray(g1)
        g2 = numpy.asarray(g2)
        shape = tuple(
            [min(s1,s2) for s1, s2 in zip(numpy.shape(g1), numpy.shape(g2))]
            )
        diff = numpy.zeros(shape, object)
        summ = numpy.zeros(shape, object)
        if len(shape) == 0:
            diff = numpy.array(g1 - g2)
            summ = numpy.array(g1 + g2)
        else:
            for i in numpy.ndindex(shape):
                diff[i] = g1[i] - g2[i]
                summ[i] = g1[i] + g2[i]
        diff = diff.flatten()
        summ = summ.flatten()
    else:
        # g1 and g2 are something else
        raise ValueError(
            'cannot compare types ' + str(type(g1)) + ' ' +
            str(type(g2))
            )
    if diff.size == 0:
        return True

    avgg = summ / 2.
    # check means
    dmean = mean(diff)
    amean = mean(avgg)
    if not numpy.all(numpy.abs(dmean) < (numpy.abs(amean) * rtol + atol)):
        return False

    # check derivatives
    for ai, di in zip(avgg, diff):
        # focus on large derivatives to avoid comparing noise to noise
        ai_der = numpy.abs(ai.der)
        di_der = numpy.abs(di.der)
        idx = (ai_der > rtol * max(ai_der))
        if not numpy.all(di_der[idx] < ai_der[idx] * rtol + atol):
            return False
    return True


def fmt_chi2(f):
    r""" Return string containing ``chi**2/dof``, ``dof`` and ``Q`` from ``f``.

    Assumes ``f`` has attributes ``chi2``, ``dof`` and ``Q``. The
    logarithm of the Bayes factor will also be printed if ``f`` has
    attribute ``logGBF``.
    """
    from scipy.special import gammaincc as gammaQ
    Q = gammaQ(f.dof/2., f.chi2/2.)
    fmt = 'chi2/dof [dof] = {:.2g} [{}]    Q = {:.2g}    {} = {:.5g}'
    if hasattr(f, 'logGBF') and f.logGBF is not None:
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt.format(chi2_dof, f.dof, Q, 'logGBF', f.logGBF)
    elif hasattr(f, 'logBF') and f.logBF is not None:
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt.format(chi2_dof, f.dof, Q, 'logBF', f.logBF)
    else:
        fmt = 'chi2/dof [dof] = {:.2g} [{}]    Q = {:.2g}'
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt.format(chi2_dof, f.dof, Q)

def tabulate(g, ncol=1, headers=True, offset='', ndecimal=None, keys=None):
    r""" Tabulate contents of an array or dictionary of |GVar|\s.

    Given an array ``g`` of |GVar|\s or a dictionary whose values are
    |GVar|\s or arrays of |GVar|\s, ``gvar.tabulate(g)`` returns a
    string containing a table of the values of ``g``'s entries.
    For example, the code ::

        import collections
        import gvar as gv

        g = collections.OrderedDict()
        g['scalar'] = gv.gvar('10.3(1)')
        g['vector'] = gv.gvar(['0.52(3)', '0.09(10)', '1.2(1)'])
        g['tensor'] = gv.gvar([
            ['0.01(50)', '0.001(20)', '0.033(15)'],
            ['0.001(20)', '2.00(5)', '0.12(52)'],
            ['0.007(45)', '0.237(4)', '10.23(75)'],
            ])
        print(gv.tabulate(g, ncol=2))

    prints the following table::

           key/index          value     key/index          value
        ---------------------------  ---------------------------
              scalar     10.30 (10)           1,0     0.001 (20)
            vector 0     0.520 (30)           1,1     2.000 (50)
                   1      0.09 (10)           1,2      0.12 (52)
                   2      1.20 (10)           2,0     0.007 (45)
          tensor 0,0      0.01 (50)           2,1    0.2370 (40)
                 0,1     0.001 (20)           2,2     10.23 (75)
                 0,2     0.033 (15)

    Args:
        g: Array of |GVar|\s (any shape) or dictionary whose values are
            |GVar|\s or arrays of |GVar|\s (any shape).
        ncol: The table is split over ``ncol`` columns of key/index values
            plus |GVar| values. Default value is 1.
        headers: Prints standard header on table if ``True``; omits the
            header if ``False``. If ``headers`` is a 2-tuple, then
            ``headers[0]`` is used in the header over the indices/keys
            and ``headers[1]`` over the |GVar| values. (Default is ``True``.)
        offset (str): String inserted at the beginning of each line in
            the table. Default is ``''``.
        ndecimal: Number of digits displayed after the decimal point.
            Default is ``ndecimal=None`` which adjusts table entries to
            show 2 digits of error.
        keys: When ``g`` is a dictionary and ``keys`` is a list of 
            keys, entries will be tabulated only for keys in the ``keys``
            list; ignored if ``keys=None`` (default).
    """
    entries = []
    if hasattr(g, 'keys'):
        if headers is True:
            headers = ('key/index', 'value')
        g = BufferDict(g)
        if keys is None:
            keys = g.keys()
        for k in keys:
            if g[k].shape == ():
                entries.append((
                    str(k), fmt(g[k], sep=' ', ndecimal=ndecimal)
                    ))
            else:
                prefix = str(k) + ' '
                gk = g[k]
                for idx in numpy.ndindex(gk.shape):
                    str_idx = str(idx)[1:-1]
                    str_idx = ''.join(str_idx.split(' '))
                    if str_idx[-1] == ',':
                        str_idx = str_idx[:-1]
                    entries.append((
                        prefix + str_idx,
                        fmt(gk[idx], sep=' ', ndecimal=ndecimal),
                        ))
                    if prefix != '':
                        prefix = ''
    else:
        if headers is True:
            headers = ('index', 'value')
        g = numpy.asarray(g)
        if g.shape == ():
            return fmt(g)
        for idx in numpy.ndindex(g.shape):
            str_idx = str(idx)[1:-1]
            str_idx = ''.join(str_idx.split(' '))
            if str_idx[-1] == ',':
                str_idx = str_idx[:-1]
            entries.append((
                str_idx, fmt(g[idx], sep=' ', ndecimal=ndecimal)
                ))
    w0 = max(len(ei[0]) for ei in entries)
    w1 = max(len(ei[1]) for ei in entries)
    linefmt = '  {e0:>{w0}}    {e1:>{w1}}'
    table = ncol * [[]]
    # nl = length of long columns; ns = lenght of short columns
    nl = len(entries) // ncol
    if nl * ncol < len(entries):
        nl += 1
    ns = len(entries) - (ncol - 1) * nl
    ne = (ncol - 1) * [nl] + [ns]
    iter_entries = iter(entries)
    if headers is not False and len(headers) != 2:
        raise ValueError('headers must be True, False or a 2-tuple')
    for col in range(ncol):
        if headers is not False:
            e0, e1 = headers
            w0 = max(len(e0), w0)
            w1 = max(len(e1), w1)
            table[col] = [linefmt.format(e0=e0, w0=w0, e1=e1, w1=w1)]
            table[col].append(len(table[col][0]) * '-')
        else:
            table[col] = []
        for ii in range(ne[col]):
            e0, e1 = next(iter_entries)
            table[col].append(linefmt.format(e0=e0, w0=w0, e1=e1, w1=w1))
    mtable = []
    if headers is not False:
        ns += 2
        nl += 2
    for i in range(ns):
        mtable.append('  '.join([tabcol[i] for tabcol in table]))
    for i in range(ns, nl):
        mtable.append('  '.join([tabcol[i] for tabcol in table[:-1]]))
    return offset + ('\n' + offset).join(mtable)

# default extensions
BufferDict.add_distribution('log', numpy.exp)
BufferDict.add_distribution('sqrt', numpy.square)
BufferDict.add_distribution('erfinv', erf)


class PDF(object):
    r""" Probability density function (PDF) for ``g``.

    Given an array or dictionary ``g`` of |GVar|\s, ``pdf=PDF(g)`` is
    the probability density function for the (correlated)
    multi-dimensional Gaussian distribution defined by ``g``. That is
    ``pdf(p)`` is the probability density for random sample ``p``
    drawn from ``g``. The logarithm of the PDF is obtained using
    ``pdf.logpdf(p)``.

    Args:
        g: |GVar| or array of |GVar|\s, or dictionary of |GVar|\s
            or arrays of |GVar|\s.

        svdcut (float): If nonzero, singularities in the correlation
            matrix are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. Default is ``svdcut=1e-12``.

        eps (non-negative float): If positive, singularities in the correlation matrix 
            for ``g`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. Ignored if ``svdcut`` is specified (and 
            not ``None``).

        noise (bool): If ``True`` adds noise to the corrections caused by 
            ``svdcut/eps`` (see documentation for :func:`gvar.regulate`).
            Default value is ``False``.

        mode (str or None): The PDF is evaluated simultaneously 
            for batches of points in the parameter space 
            when ``mode`` equals either ``rbatch`` or ``lbatch``.
            For example, in the code ::

                import gvar as gv 
                g = gv.gvar([1, 2], [[1., .1], [.1, 2]])
                pdf = gvar.PDF(g, mode='rbatch')
                p = gv.sample(g, nbatch=1000, mode='rbatch')
                pdf_p = pdf(p)

            ``p[d, i]`` represents a collection of 1000 points in 
            the parameter space, labeled by batch index ``i=0..999``.
            Index ``d=0..3`` labels directions in parameter space.
            ``pdf_p=pdf(p)`` returns the PDF value ``pdf_p[i]`` for  
            each point. This is generally much more efficient than 
            calculating the PDF values one at a time (for example,
            in PDF integrals using :mod:`vegas`).

            In batch mode an extra batch index is added to each 
            parameter. The index is the rightmost index when 
            ``mode='rbatch'`` and the leftmost index when 
            ``mode='lbatch'`` (e.g., ``p[d,i]`` and ``p[i,d]``,
            respectively, in the example above).
            
            Default is ``mode=None`` which turns batch mode off.

        decorrelate (bool): If ``True`` correlations between parameters 
            are ignored. Ignored otherwise. Default value is ``False``.

    Objects of type :class:`PDF` have the following attributes:

    Attributes:    
        size (int): Dimension of the parameter space.

        nchiv (int): Dimension of the ``self.chiv(p)`` vector. This 
            can be smaller than ``self.size`` if ``svdcut`` is 
            negative.
        
        shape (tuple or None): Shape of ``g`` array or ``None`` 
            if ``g`` is a dictionary.

        mean (array or dictionary): Mean values of ``g``.

        meanflat (array): Flattened array containing mean values of ``g``.

        cov (array): Modified covariance matrix of the elements in ``g.flat[:]``.

        invcov (array): Inverse of modified covariance matrix.

        i_wgts: :func:`gvar.regulate` decomposition of the modified covariance matrix.

        i_invwgts: :func:`gvar.regulate` decomposition of the inverse modified covariance matrix.

        logdet: Determinant of modified covariance matrix.

        dp_dchiv: Jacobian :math:`\mathrm{det}(\partial p_i/\partial \chi_j)`.

        distribution: Array or dictionary of |GVar|\s describing the distribution 
            corresponding to the PDF. This may differ from the initial distribution
            if ``svdcut`` or ``eps`` is nonzero.
        
        correction: ``self.distribution - self.correction`` is the original  
            distribution before applying ``svdcut/eps`` corrections.

        nmod: Number of eigenmodes modified by the SVD cut.

        nblocks (dict): ``nblocks[s]`` equals the number of block-diagonal
            sub-matrices of the ``y``--``prior`` covariance matrix that are
            size ``s``-by-``s``. This is sometimes useful for debugging.
    """
    def __init__(self, g, svdcut=1e-12, eps=None, noise=False, mode=None, decorrelate=False):
        if decorrelate:
            g = gvar(mean(g), sdev(g))
        if hasattr(g, 'keys'):
            # g is a dict
            g = asbufferdict(g)
            gflat = g.buf
        else:
            # g is an array
            g = numpy.asarray(g)
            gflat = g.reshape(-1)
        if decorrelate:
            i_wgts = [(numpy.arange(gflat.size), sdev(gflat))]
            i_invwgts = [(numpy.arange(gflat.size), 1. / sdev(gflat))]
            logdet = numpy.sum(numpy.log(var(gflat)))
            self.nblocks = {1:gflat.size}
            self.svdcut = None
            self.eps = None
        elif eps is None or svdcut is not None:
            gflat, i_wgts, i_invwgts = svd(gflat, svdcut=svdcut, noise=noise, wgts=True)
            logdet = gflat.logdet
            self.nblocks = gflat.nblocks
            self.svdcut = gflat.svdcut
            self.eps = None
        else:
            gflat, i_wgts, i_invwgts = regulate(gflat, eps=eps, noise=noise, wgts=True)
            logdet = gflat.logdet
            self.nblocks = gflat.nblocks
            self.svdcut = None
            self.eps = gflat.eps
        self.mode = mode
        if mode is not None:
            self.fcntype = mode
        self.i_wgts = i_wgts        # -> cov
        self.i_invwgts = i_invwgts  # -> 1/cov
        self.logdet = logdet
        self.dp_dchiv = numpy.exp(logdet  / 2)
        self.size = g.size 
        self.shape = g.shape 
        self._cov = None 
        self._invcov = None
        if hasattr(g, 'keys'):
            self._keys = g.keys()
        self.mean = mean(g)
        self.meanflat = mean(gflat)
        if decorrelate:
            self.nmod = 0 
            self.correction = 0
            self.distribution = g 
        else:
            self.nmod = gflat.nmod
            self.correction = gflat.correction
            self.distribution = self._unflatten(gflat, mode=None)
        if self.nmod > 0 and svdcut is not None and svdcut < 0:
            self.nchiv = self.size - self.nmod
        else:
            self.nchiv = self.size 
        # devide exp(-chi**2/2) by norm
        self.lognorm = 0.5 *  (g.size * numpy.log(2 * numpy.pi) + logdet)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        gvstate = dumps(dict(
            correction=state['correction'], 
            distribution=state['distribution'],
            ))
        del state['correction']
        del state['distribution']
        state['gvstate'] = gvstate
        return state
    
    def __setstate__(self, state):
        gvstate = loads(state['gvstate'])
        del state['gvstate']
        self.__dict__.update(state)
        self.__dict__.update(gvstate)

    @staticmethod
    def _make_mat(size, i_wgts):
        mat = numpy.zeros((size, size), dtype=float)
        # 1x1 sub-matrices
        i, wgts = i_wgts[0]
        mat[i, i] = wgts ** 2
        # nxn sub-matrices (n>1)
        for i, wgts in i_wgts[1:]:
            mat[i[:, None], i] = wgts.T.dot(wgts)
        return mat

    def _getcov(self):
        if self._cov is None:
            self._cov = PDF._make_mat(self.size, self.i_wgts)
        return self._cov
    def _getinvcov(self):
        if self._invcov is None:
            self._invcov = PDF._make_mat(self.size, self.i_invwgts)
        return self._invcov     
    cov = property(_getcov, doc="Covariance matrix.")
    invcov = property(_getinvcov, doc="Inverse covariance matrix.")
    
    def _unflatten(self, pflat, mode='default'):
        r""" pflat -> p """  
        if mode == 'default':
            mode = self.mode
        if self.shape is None:
            if mode is None:
                return BufferDict(self.mean, buf=pflat)  
            elif mode == 'rbatch':
                return BufferDict(self.mean, rbatch_buf=pflat)
            else:       
                return BufferDict(self.mean, lbatch_buf=pflat)
        else:
            pflat = numpy.asarray(pflat)
            if mode is None:
                return pflat.reshape(self.shape) if self.shape != () else pflat.flat[0]
            elif mode == 'rbatch':
                return pflat.reshape(self.shape + (-1,))
            elif mode == 'lbatch':
                return pflat.reshape((-1,) + self.shape)

    def _flatten(self, p, mode='default'):
        r""" p -> pflat """ 
        if mode == 'default':
            mode = self.mode
        if self.shape is None:
            p = BufferDict(p, keys=self._keys)
            if mode is None:
                return p.buf 
            else:
                return p.rbatch_buf if mode == 'rbatch' else p.lbatch_buf
        else:
            p = numpy.asarray(p)
            if mode is None:
                return p.flat[:]
            else:
                return (
                    p.flat[:].reshape(-1, p.shape[-1]) if mode=='rbatch' else
                    p.flat[:].reshape(p.shape[0], -1)
                    )

    def chiv(self, p=None, pflat=None, mode='default'): 
        r""" Returns :math:`\chi_i(p)` where :math:`\chi^2(p) = \sum_i \chi_i^2(p)`.
        
        The PDF is proportional to :math:`\exp(-\chi^2(p)/2)` where

        .. math::

            \chi^2(p) = \Delta p \cdot\mathrm{cov}^{-1}_p\cdot\Delta p

        :math:`\Delta p_i\equiv \overline p_i - p_i`, and :math:`\overline p_i`
        and :math:`\mathrm{cov}_p` are the means and covariance matrix of the 
        distribution. By replacing the parameters :math:`p_i` with a new 
        set of uncorrelated parameters (linear combinations of the originals),
        :math:`\chi^2(p)` is reexpressed as a sum of terms 
        :math:`(\chi_i(p))^2`. By default the individual terms
        correspond to individual eigenmodes of the correlation matrix.
        If ``eps`` is specified (instead of ``svdcut``), a Cholesky 
        decomposition is used instead (see :func:`gvar.regulate`).

        Args:
            p: An array or dictionary represention a point in the 
                distribution's parameter space (or multiple points 
                if in batch mode). Ignored if ``pflat`` is specified.
            pflat (array): The coordinates of a point in parameter 
                space flattened into a 1-d array. In batch mode 
                an extra batch index is added to the array, either
                on the right (``mode='rbatch'``) or on the left 
                (``mode='lbatch'``), to represent multiple
                points.
            mode: Batch mode. Defaults to ``self.mode`` if 
                not specified (``mode='default'``).

        Returns:
            An array containing :math:`\chi_i(p)`. In batch 
            mode the array has an extra batch index (on the right or 
            left depending upon ``mode``). 
        """ 
        if mode == 'default':
            mode = self.mode 
        if pflat is None:
            pflat = self._flatten(p, mode)
        else:
            pflat = numpy.asarray(pflat)
        # dpflat in rbatch mode
        if mode != 'rbatch':
            dpflat = (pflat - self.meanflat).T
        else:
            dpflat = pflat - self.meanflat[:, None]
        if len(dpflat.shape) > 1:
            chiv_shape = (self.nchiv, dpflat.shape[-1])
        else:
            chiv_shape = self.nchiv
        chiv = numpy.zeros(chiv_shape, dtype=pflat.dtype)
        i, iwgts = self.i_invwgts[0]
        i1 = i2 = 0
        if len(i) > 0:
            i1 = i2
            i2 += len(i)
            if mode is not None:
                chiv[i1:i2] = iwgts[:, None] * dpflat[i]
            else:
                chiv[i1:i2] = iwgts * dpflat[i]
        for i, iwgts in self.i_invwgts[1:]:
            i1 = i2 
            i2 += len(i)
            chiv[i1:i2] = iwgts.dot(dpflat[i])
        return chiv.T if mode == 'lbatch' else chiv
                    
    def pflat(self, chiv, mode='default'):
        r""" Inverse of :meth:`PDF.chiv`.
        
        Args:
            chiv: An array of values for :math:`\chi_i(p)` 
                evaluate a point ``p`` in paramter space. In batch 
                mode the array has an extra batch index, 
                either on the right (``mode='rbatch'``) or on the l
                left (``mode='lbatch'``), to identify multiple
                points in parameter space.
            mode: Batch mode used. Defaults to ``self.mode`` if 
                not specified (``mode='default'``).

        Returns:
            An array containing the flattened version of ``p`` 
            corresponding to ``chiv``. In batch mode the array 
            has an extra batch index (on the right or 
            left depending upon ``mode``). 
        """
        if mode == 'default':
            mode = self.mode
        # put in rbatch mode (or None)
        if mode == 'lbatch':
            chiv = chiv.T
        if mode is not None:
            pshape = (self.size, chiv.shape[-1])
        else:
            pshape = self.size
        dpflat = numpy.zeros(pshape, dtype=chiv.dtype)
        i, wgts = self.i_wgts[0]
        i1 = i2 = 0
        if len(i) > 0:
            # print('***', i, wgts)
            i1 = i2 
            i2 += len(i)
            if mode is not None:
                dpflat[i] = wgts[:, None] * chiv[i1:i2]
            else:
                dpflat[i] = wgts * chiv[i1:i2]
        for i, wgts in self.i_wgts[1:]:
            # print('***', i, wgts)
            i1 = i2
            i2 += len(i)
            dpflat[i] = wgts.T.dot(chiv[i1:i2])
        if mode != 'rbatch':
            return dpflat.T + self.meanflat
        else:
            return dpflat + self.meanflat[:, None]

    def chi2(self, p, mode='default'):
        r""" ``sum(self.chiv(p)**2)``. """
        if mode == 'default':
            mode = self.mode
        return numpy.sum(
            self.chiv(p, mode=mode) ** 2, 
            axis=0 if mode != 'lbatch' else -1
            )

    def logpdf(self, p, mode='default'):
        r""" Logarithm of the PDF evaluated at ``p``.

        The logarithm of the PDF is 
        ``-sum(self.chiv(p)**2)/2 - self.lognorm``.
        """
        if mode == 'default':
            mode = self.mode
        return - self.lognorm - numpy.sum(
            self.chiv(p, mode=mode) ** 2, 
            axis=0 if mode != 'lbatch' else -1
            ) / 2. 
    
    def __call__(self, p, mode='default'):
        r""" The PDF evaluated at ``p``. 
        
        The PDF is ``exp(-sum(self.chiv(p)**2)/2 - self.lognorm)``.
        """
        if mode == 'default':
            mode = self.mode
        return numpy.exp(self.logpdf(p, mode=mode))
    
    def sample(self, nbatch=None, uniform=None, mode='default'):
        r""" Generate random sample(s) from the PDF.

        Args:
            nbatch (int or None): If not ``None``, the iterator will return
                ``nbatch`` samples drawn from the PDF. 
                The results are packaged in arrays or dictionaries
                whose elements have an extra index labeling the different 
                samples in the batch. The batch index is 
                the rightmost index if ``mode='rbatch'``; it is 
                the leftmost index if ``mode`` is ``'lbatch'``. Ignored
                if ``nbatch`` is ``None`` (default).
            mode (bool): Batch mode. Allowed 
                modes are ``'rbatch'``or ``'lbatch'``,
                corresponding to batch indices that are on the 
                right or the left, respectively. 
                Defaults to ``self.mode`` if 
                not specified (``mode='default'``).
            uniform (float or None): Replace Gaussian distribution specified 
                by ``g`` with a uniform distribution covering the interval 
                ``[-uniform, uniform]`` times the standard deviation centered 
                on the mean along each principal axis the distribution.
                Ignored if ``None`` (default).
        """
        if mode == 'default':
            mode = self.mode
        if nbatch is None:
            chiv = RNG.normal(size=self.nchiv) if uniform is None else RNG.uniform(-uniform, uniform, size=self.nchiv)
            return self._unflatten(self.pflat(chiv, mode=None), mode=None)
        else:
            if mode is None:
                mode = 'rbatch'
            shape = (self.nchiv, nbatch) if mode == 'rbatch' else (nbatch, self.nchiv)
            if uniform is None:
                chiv = RNG.normal(size=shape)
            else:
                chiv = RNG.uniform(-uniform, uniform, size=shape)
            return self._unflatten(self.pflat(chiv, mode=mode), mode=mode)



class oldPDF(object):
    r""" Probability density function (PDF) for ``g``.

    Given an array or dictionary ``g`` of |GVar|\s, ``pdf=PDF(g)`` is
    the probability density function for the (usually correlated)
    multi-dimensional Gaussian distribution defined by ``g``. That is
    ``pdf(p)`` is the probability density for random sample ``p``
    drawn from ``g``. The logarithm of the PDF is obtained using
    ``pdf.logpdf(p)``.

    Args:
        g: |GVar| or array of |GVar|\s, or dictionary of |GVar|\s
            or arrays of |GVar|\s.

        svdcut (non-negative float or None): If not ``None``, replace
            covariance matrix of ``g`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``1e-12``.
    """
    def __init__(self, g, svdcut=1e-12):
        if hasattr(g, 'keys'):
            # g is a dict
            g = BufferDict(g)
            gflat = g.buf
        else:
            # g is an array
            g = numpy.asarray(g)
            gflat = g.reshape(-1)
        s = SVD(
            evalcov(gflat),
            svdcut=abs(svdcut) if svdcut is not None else None
            )
        self.vec_sig = numpy.array(s.vec)
        self.vec_isig = numpy.array(s.vec)
        self.sig = s.val ** 0.5
        self.dpdx = numpy.prod(self.sig)
        for i, sigi in enumerate(self.sig):
            self.vec_sig[i] *= sigi
            self.vec_isig[i] /= sigi
        self.meanflat = mean(gflat)
        self.size = g.size
        self.shape = g.shape
        if hasattr(g, 'keys'):
            self._keys = g.keys()
        self.mean = mean(g)
        self.sample = sample(g)
        self.log_gnorm = numpy.sum(0.5 * numpy.log(2 * numpy.pi * s.val))

    def x2dpflat(self, x, mode=None):
        r""" Map vector ``x`` in x-space into the displacement from ``g.mean``.

        x-space is a vector space of dimension ``p.size``. Its axes are
        in the directions specified by the eigenvectors of ``p``'s covariance
        matrix, and distance along an axis is in units of the standard
        deviation in that direction.

        Batches of ``x`` vectors can be processed together bu specifying the 
        batch mode: ``mode='lbatch'`` means the batch index ``i`` is on the left, 
        ``x[i,d]``; ``mode='rbatch'`` means the batch index is on the right, 
        ``x[d,i]``.
        """
        if mode != 'rbatch':
            return x.dot(self.vec_sig)
        else:
            return x.T.dot(self.vec_sig).T

    def dpflat2x(self, dpflat, mode=None):
        r""" Map the displacement from ``mean(g)`` to vector ``x`` in x-space.

        x-space is a vector space of dimension ``p.size``. Its axes are
        in the directions specified by the eigenvectors of ``p``'s covariance
        matrix, and distance along an axis is in units of the standard
        deviation in that direction.

        Batches of ``dpflat`` vectors can be processed together bu specifying the 
        batch mode: ``mode='lbatch'`` means the batch index ``i`` is on the left, 
        ``dpflat[i,d]``; ``mode='rbatch'`` means the batch index is on the right, 
        ``dpflat[d,i]``.
        """
        if mode != 'lbatch':
            return self.vec_isig.dot(dpflat)
        else:
            return self.vec_isig.dot(dpflat.T).T

    def p2x(self, p, mode=None):
        r""" Map parameters ``p`` to vector in x-space.

        x-space is a vector space of dimension ``p.size``. Its axes are
        in the directions specified by the eigenvectors of ``p``'s covariance
        matrix, and distance along an axis is in units of the standard
        deviation in that direction.
        """
        if hasattr(p, 'keys'):
            p = BufferDict(p, keys=self._keys)
            if mode == 'rbatch':
                dpflat = p.rbatch_buf - self.meanflat[:, None]
            elif mode == 'lbatch':
                dpflat = p.lbatch_buf - self.meanflat[None, :]
            else:
                dpflat = p.buf - self.meanflat
        else:
            p = numpy.array(p)
            if mode == 'lbatch':
                dpflat = p.flat[:].reshape(p.shape[0],-1) - self.meanflat[None, :]
            elif mode == 'rbatch':
                dpflat = p.flat[:].reshape(-1, p.shape[-1]) - self.meanflat[:, None]
            else:
                dpflat = p.flat[:] - self.meanflat
        return self.dpflat2x(dpflat, mode=mode)

    def logpdf(self, p, mode=None):
        r""" Logarithm of the probability density function evaluated at ``p``. """
        x2 = self.p2x(p, mode=mode) ** 2
        return -0.5 * numpy.sum(x2, axis=1 if mode=='lbatch' else 0) - self.log_gnorm

    def __call__(self, p, mode=None):
        r""" Probability density function evaluated at ``p``."""
        return numpy.exp(self.logpdf(p, mode))

class PDFHistogram(object):
    r""" Utility class for creating PDF histograms. (Deprecated.)

    This class is designed to facilitate studies of probability
    density functions associated with |GVar|\s. The following code,
    for example, makes a histogram of probabilities for the Gaussian
    distribution corresponding to |GVar| 1.0(5)::

        g = gv.gvar('1.0(5)')
        data = [g() for i in range(10000)]

        hist = gv.PDFHistogram(g)
        count = hist.count(data)
        a = hist.analyze(count)
        print('probabilities:', a.prob)
        print('statistics:\n', a.stats)

    Here ``hist`` defines a histogram with 8 bins, centered on
    ``g.mean``, and each with width equal to ``g.sdev``. The data in
    array ``data`` is a random sampling from ``g``'s distribution.
    The number of data elements in each bin is determined by
    ``hist.count(data)`` and turned into probabilities by
    ``hist.analyze(count)``. The probabilities (``a.prob``) and
    a statistical analysis of the probability distribution based
    on the histogram (``a.stats``) are then printed out::

        probabilities: [ 0.0017  0.0213  0.1358  0.3401  0.3418  0.1351  0.0222  0.0018]
        statistics:
              mean = 1.001   sdev = 0.52334   skew = 0.0069999   ex_kurt = 0.034105
           median = 1.00141666398   plus = 0.499891542549   minus = 0.501710986504

    A plot of the histogram can be created and displayed using,
    for example::

        plt = hist.make_plot(count)
        plt.xlabel('g')
        plt.ylabel('probabilities')
        plt.show()

    :class:`vegas.PDFIntegrator` can be used to create histograms for more
    complicated, multi-dimensional distributions: the expectation value of
    ``hist.count(f(p))`` over values of ``p`` drawn from a multi-dimensional
    distribution gives the the probability distribution for function ``f(p)``.

    Args:
        g (|GVar| or ``None``): The mean and standard deviation of ``g`` are
            used to design the histogram bins, which are centered
            on ``g.mean``. Ignored if ``None`` (in which case ``bins`` must
            be specified).
        nbin (int): The number of histogram bins. Set equal to
            ``PDFHistogram.default_nbin`` (=8 initially) if ``None``.
        binwidth (float): The width of each bin is ``binwidth * g.sdev``.
            Set equal to ``PDFHistogram.default_binwidth`` (=1. initially)
            if ``None``.
        bins (array or ``None``): Ignored if ``None`` (default). Otherwise specifies the
            histogram's bin edges, overriding the default bin design specified
            by ``g``. ``len(bins)`` is one larger than the number of bins. If
            specified it overrides the default bin design indicated by ``g``.
            One of ``g`` or ``bins`` must be specified.

    The main attributes are:

    Attributes:
        g: |GVar| used to design the histogram.
        bins: Bin edges for the histogram (see above).
        midpoints: Bin midpoints.
        widths: Bin widths.
    """

    Histogram = collections.namedtuple('Histogram', 'bins, prob, stats, norm ')
    default_nbin = 8
    default_binwidth = 1.

    def __init__(self, g=None, nbin=None, binwidth=None, bins=None):
        import warnings
        warnings. warn(
            'PDFHistogram is deprecated. Use numpy.histogram.', 
            DeprecationWarning, stacklevel=2
            )
        if nbin is None or nbin < 0:
            nbin = self.default_nbin
        if binwidth is None:
            binwidth = self.default_binwidth
        if g is not None:
            g = gvar(g)
            limit = binwidth * nbin / 2.
            if bins is None:
                self.bins = numpy.linspace(
                    g.mean - binwidth * nbin * g.sdev / 2.,
                    g.mean + binwidth * nbin * g.sdev / 2.,
                    nbin + 1,
                    )
            else:
                self.bins = numpy.array(bins)
        elif bins is not None:
            self.bins = numpy.array(bins)
        else:
            raise ValueError('must specify either g or bins')
        self.bins.sort()
        self.midpoints = (self.bins[1:] + self.bins[:-1]) / 2.
        self.widths = self.bins[1:] - self.bins[:-1]
        self.g = g
        self.nbin = len(self.bins) - 1

    def count(self, data):
        r""" Compute histogram of data.

        Counts the number of elements from array ``data`` in each bin of the
        histogram. Results are returned in an array, call it ``h``, of
        length ``nbin+2`` where ``h[0]`` is the number of data elements
        that fall below the range of the histogram, ``h[-1]``
        (i.e., ``h[nbin+1]``) is the number that fall above the range,
        and ``h[i]`` is the number in the ``i``-th bin for ``i=1...nbin``.

        Argument ``data`` can also be a float, in which case the result is the
        same as from ``histogram([data])``. Note that the expectation value of
        ``count(f(p))`` over parameter values ``p`` drawn from a random
        distribution gives the probabilities for values of ``f(p)`` to fall
        in each histogram bin. Dividing  by the bin widths gives the average
        probability density for random variable ``f(p)`` in each bin.

        Bin intervals are closed on the left and open on the right,
        except for the last interval which is closed on both ends.
        """
        if isinstance(data, float) or isinstance(data, int):
            hist = numpy.zeros(self.nbin + 2, float)
            if data > self.bins[-1]:
                hist[-1] = 1.
            elif data < self.bins[0]:
                hist[0] = 1.
            elif data == self.bins[-1]:
                if self.nbin > 1:
                    hist[-2] = 1.
            else:
                hist[numpy.searchsorted(self.bins, data, side='right')] = 1.
            return hist
        if numpy.ndim(data) != 1:
            data = numpy.reshape(data, -1)
        else:
            data = numpy.asarray(data)
        middle = numpy.histogram(data, self.bins)[0]
        below = numpy.sum(data < self.bins[0])
        above = numpy.sum(data > self.bins[-1])
        return numpy.array([below] + middle.tolist() + [above], float)

    def analyze(self, count):
        r""" Analyze count data from :meth:`PDFHistogram.count`.

        Turns an array of counts (see :meth:`PDFHistogram.count`) into a
        histogram of probabilities, and  estimates the mean, standard
        deviation, and other statistical characteristics of the corresponding
        probability distribution.

        Args:
            count (array): Array of length ``nbin+2`` containing histogram
                data where ``count[0]`` is the count for values that are
                below the range of the histogram, ``count[-1]`` is the count
                for values above the range, and ``count[i]`` is the count
                for the ``i``-th bin where ``i=1...nbin``.

        Returns a named tuple containing the following information (in order):

            *bins*: Array of bin edges for histogram (length ``nbin+1``)

            *prob*: Array of probabilities for each bin.

            *stats*: Statistical data about histogram. See :class:`PDFStatistics`.

            *norm*: Convert counts into probabilities by dividing by ``norm``.
        """
        if numpy.ndim(count) != 1:
            raise ValueError('count must have dimension 1')
        if len(count) == len(self.midpoints) + 2:
            norm = numpy.sum(count)
            data = numpy.asarray(count[1:-1]) / norm
        elif len(count) != len(self.midpoints):
            raise ValueError(
                'wrong data length: %s != %s'
                    % (len(count), len(self.midpoints))
                )
        else:
            data = count
            norm = 1.
        mid = self.midpoints
        stats = PDFStatistics(histogram=(self.bins, count))
        return PDFHistogram.Histogram(self.bins, data, stats, norm)

    @staticmethod
    def gaussian_pdf(x, g):
        r""" Gaussian probability density function at ``x`` for |GVar| ``g``. """
        return (
            numpy.exp(-(x - g.mean) ** 2 / 2. /g.var) /
            numpy.sqrt(g.var * 2 * numpy.pi)
            )

    def make_plot(
        self, count, plot=None, show=False, plottype='probability',
        bar=dict(alpha=0.15, color='b', linewidth=1.0, edgecolor='b'),
        errorbar=dict(fmt='b.'),
        gaussian=dict(ls='--', c='r')
        ):
        r""" Convert histogram counts in array ``count`` into a plot.

        Args:
            count (array): Array of histogram counts (see
                :meth:`PDFHistogram.count`).
            plot (plotter): :mod:`matplotlib` plotting window. If ``None``
                uses the default window. Default is ``None``.
            show (boolean): Displayes plot if ``True``; otherwise returns
                the plot. Default is ``False``.
            plottype (str): The probabilities in each bin are plotted if
                ``plottype='probability'`` (default). The average probability
                density is plot if ``plottype='density'``. The
                cumulative probability is plotted if ``plottype=cumulative``.
            bar (dictionary): Additional plotting arguments for the bar graph
                showing the histogram. This part of the plot is omitted
                if ``bar=None``.
            errorbar (dictionary): Additional plotting arguments for the
                errorbar graph, showing error bars on the histogram. This
                part of the plot is omitted if ``errorbar=None``.
            gaussian (dictionary): Additional plotting arguments for the
                plot of the Gaussian probability for the |GVar| (``g``)
                specified in the initialization. This part of the plot
                is omitted if ``gaussian=None`` or if no ``g`` was
                specified.
        """
        if numpy.ndim(count) != 1:
            raise ValueError('count must have dimension 1')
        if plot is None:
            import matplotlib.pyplot as plot
        if len(count) == len(self.midpoints) + 2:
            norm = numpy.sum(count)
            data = numpy.asarray(count[1:-1]) / norm
        elif len(count) != len(self.midpoints):
            raise ValueError(
                'wrong data length: %s != %s'
                    % (len(count), len(self.midpoints))
                )
        else:
            data = numpy.asarray(count)
        if plottype == 'cumulative':
            data = numpy.cumsum(data)
            data = numpy.array([0.] + data.tolist())
            data_sdev = sdev(data)
            if not numpy.all(data_sdev == 0.0):
                data_mean = mean(data)
                plot.errorbar(self.bins, data_mean, data_sdev, **errorbar)
            if bar is not None:
                plot.fill_between(self.bins, 0, data_mean, **bar)
                # mean, +- 1 sigma lines
                plot.plot([self.bins[0], self.bins[-1]], [0.5, 0.5], 'k:')
                plot.plot([self.bins[0], self.bins[-1]], [0.158655254, 0.158655254], 'k:')
                plot.plot([self.bins[0], self.bins[-1]], [0.841344746, 0.841344746], 'k:')
        else:
            if plottype == 'density':
                data = data / self.widths
            if errorbar is not None:
                data_sdev = sdev(data)
                if not numpy.all(data_sdev == 0.0):
                    data_mean = mean(data)
                    plot.errorbar(self.midpoints, data_mean, data_sdev, **errorbar)
            if bar is not None:
                plot.bar(self.bins[:-1], mean(data), width=self.widths, align='edge', **bar)
        if gaussian is not None and self.g is not None:
            # spline goes through the errorbar points for gaussian stats
            if plottype == 'cumulative':
                x = numpy.array(self.bins.tolist() + self.midpoints.tolist())
                x.sort()
                dx = (x - self.g.mean) / self.g.sdev
                y = (erf(dx / 2**0.5) + 1) / 2.
                yspline = cspline.CSpline(x, y, alg='cspline')
                plot.ylabel('cumulative probability')
                plot.ylim(0, 1.0)
            elif plottype in ['density', 'probability']:
                x = self.bins
                dx = (x - self.g.mean) / self.g.sdev
                y = (erf(dx / 2**0.5) + 1) / 2.
                x = self.midpoints
                y = (y[1:] - y[:-1])
                if plottype == 'density':
                    y /= self.widths
                    plot.ylabel('probability density')
                else:
                    plot.ylabel('probability')
                yspline = cspline.CSpline(x, y, alg='cspline')
            else:
                raise ValueError('unknown plottype: ' + str(plottype))
            if len(x) < 100:
                ny = int(100. / len(x) + 0.5) * len(x)
            else:
                ny = len(x)
            xplot = numpy.linspace(x[0], x[-1], ny)
            plot.plot(xplot, yspline(xplot), **gaussian)
        if show:
            plot.show()
        return plot


class _TwoSided:
    " Two-sided Gaussian "
    def __init__(self, loc, plus, minus, type='splitnormal'):
        self.loc = loc 
        self.plus = plus 
        self.minus = minus 
        self.type = type   # splitnormal or median
    def __str__(self):
        if self.loc is None:
            return None
        loc = str(self.loc)
        def fmt(x):
            if isinstance(x, float) or isinstance(x, int):
                return '{:.5g}'.format(x)
            else:
                return str(x)
        return '{} +/- {}/{}'.format(loc, fmt(self.plus), fmt(self.minus))
    def __repr__(self):
        return '_TwoSided({}, plus={}, minus={}, type={})'.format(self.loc, self.plus, self.minus, self.type)
    def __call__(self, x):
        if self.loc is None or self.plus is None or self.minus is None:
            return None
        loc = mean(self.loc)
        plus = mean(self.plus) 
        minus = mean(self.minus)
        x = numpy.asarray(x)
        xflat = x.flat[:]
        ans = 0 * xflat
        if self.type == 'splitnormal':
            wp = wm = numpy.sqrt(2 / numpy.pi) / (plus + minus)
        else:
            wp = numpy.sqrt(0.5 / numpy.pi) / plus
            wm = numpy.sqrt(0.5 / numpy.pi) / minus
        idx = xflat >= loc
        ans[idx] = wp * numpy.exp(-(xflat[idx] - loc)**2 / (2 * plus**2)) 
        idx = xflat < loc
        ans[idx] = wm * numpy.exp(-(xflat[idx] - loc)**2 / (2 * minus**2)) 
        return ans.reshape(x.shape)

class PDFStatistics(object):
    r""" Compute statistical information about a distribution.

    Given moments ``mom[i]`` of a random variable, :class:`PDFStatistics`
    computes the mean, standard deviation, skewness, and excess kurtosis. 
    
    With histogram data, it estimates the median and the intervals 
    on either side of the median (each) containing 34% of the probability.
    Also it does a maximum likelihood fit of a split-normal distribution
    to the histogram. Finally the moments are estimated from the histogram
    if they are not otherwise specified.

    Typical usage::

        >>> import gvar as gv 
        >>> import numpy as np
        >>> p = gv.gvar(['1(1)', '0.05(5)'])
        >>> N = 10_000
        >>> psample = gv.sample(p, nbatch=N)
        >>> p01sample = psample[0] * psample[1]
        >>>
        >>> moments = np.mean([p01sample, p01sample**2, p01sample**3, p01sample**4], axis=-1)
        >>> counts, bins = np.histogram(p01sample, bins=50)
        >>>
        >>> stats = gv.PDFStatistics(moments=mom, histogram=(bins, counts, N))
        >>> print(stats)
        mean = 0.05190738402779899   sdev = 0.088379   skew = 1.2499   ex_kurt = 3.5594
        split-normal: 0.0069(38) +/- 0.1144(49)/0.0517(37)
              median: 0.0327(28) +/- 0.0960(56)/0.0486(26)
        >>> stats.plot_histogram().show()
    
    The distribution in this example is 
    skewed, with the tail on the positive side of the peak roughly 
    twice as wide as that on the negative side. The last line displays 
    a plot of the histogram, overlayed with plots of the Gaussians or 
    two-sided Gaussians corresponding to the mean and standard deviation,
    or the median or split-normal distributions. 

    Args:
        moments (array of floats or ``GVar``\s): ``moments[i]`` is the 
            (i+1)-th moment. Optional unless ``histogram=None``.  

        histogram (tuple): (Optional unless ``moments=None``) Tuple ``(bins,prob)``
            specifying histogram bins and the probabilities contained in each bin.
            
            If ``len(prob) == len(bins) + 1``, ``prob[0]`` is the probability 
            below ``bins[0]`` and ``prob[-1]`` is the probability above ``bins[-1]``,
            while ``prob[i]`` is the probability between ``bins[i-1]`` and ``bins[i]``;
            the sum of probabilities is normalized to equal one. 
            
            If ``len(prob) == len(bins) - 1``, ``prob[i]`` is the probability  
            betwen ``bins[i]`` and ``bins[i+1]``. 

            Alternatively if ``histogram=(bins, counts, N)``, 
            array ``counts`` is the number of random samples in each bin, 
            where ``N`` is the total number of samples. The probability 
            associated with each bin is then ``prob = counts / N``. 
            An uncertainty is assigned to each probability (assuming 
            a binomial distribution). 

    The attributes are as follows:

    Attributes:
        mean: mean value
        sdev: standard deviation
        skew: skewness coefficient
        ex_kurt: excess kurtosis
        median: ``self.median.loc`` is the location of the median estimated 
            from the histogram, while the intervals ::
            
                (self.median.loc, self.median.loc + self.median.plus)
                (self.median.loc - self.median.minus, self.median.loc)
            
            each contain 34% of the probability.
        splitnormal: ``self.splitnormal.loc`` is the location of the peak
            of the (continuous) split-normal distribution fit to the histogram. The 
            standard deviations above and below that point are 
            ``self.splitnormal.plus`` and ``self.splitnormal.minus``, respectively.
        gvar: ``gvar.gvar(mean, sdev)``
        moments: array containing the values of (up to) the first four moments.
        bins: array of bin edges used for the histogram
        prob: array of probabilities ``prob[i]`` associated with the 
            intervals ``(bins[i-1], bins[i])``. ``prob[0]`` is the 
            probability from below ``bins[0]``; ``prob[-1]`` is the 
            probability from above ``bins[-1]``. The ``prob[i]`` can
            be numbers or |GVar|\s.
    """
    def __init__(self, moments=None, histogram=None, prefix='   '):
        self.prefix = prefix
        if histogram is not None:
            if len(histogram) == 3:
                bins, counts, N = histogram 
                prob = gvar(counts, (counts * (1 - counts / N)) ** 0.5) / N
            else:
                bins, prob = histogram
            self.bins = numpy.array(bins)
            self.prob = numpy.fabs(prob)
            if len(self.prob) == len(self.bins) - 1:
                # add out-of-bounds probabilities
                self.prob = numpy.array([0.] + self.prob.tolist() + [0.])
            elif len(self.prob) != len(self.bins) + 1:
                raise ValueError('length mismatch: len(bins)!=len(prob)-1 in histogram')
            self.prob /= numpy.sum(self.prob)
            self.splitnormal = self._fit_splitnormal(self.bins, self.prob)
            self.median = self._fit_median(self.bins, self.prob)
            # self.bins = bins 
            # self.prob = prob / numpy.sum(prob)
            if moments is None:
                mid = (bins[1:] + bins[:-1]) / 2.
                moments = numpy.sum(
                    [mid * self.prob[1:-1], mid**2 * self.prob[1:-1],
                    mid**3 * self.prob[1:-1], mid**4 * self.prob[1:-1],
                    ],
                    axis=1
                    )
        else:
            # self.splitnormal = None 
            # self.median = None 
            self.bins = None 
            self.prob = None
        if moments is not None:
            x = self.moments = numpy.array(moments)
            self.mean = x[0]
            if len(x) > 1:
                self.sdev = numpy.fabs(x[1] - x[0] ** 2) ** 0.5
            if len(x) > 2:
                self.skew = (
                    x[2] - 3. * self.mean * self.sdev ** 2 - self.mean ** 3
                    ) / self.sdev ** 3
            if len(x) > 3:
                self.ex_kurt = (
                    x[3] - 4. * x[2] * self.mean + 6 * x[1] * self.mean ** 2
                    - 3 * self.mean ** 4
                    ) / self.sdev ** 4 - 3.
            if not hasattr(self, 'gvar'):
                self.gvar = self.mean + gvar(0, mean(self.sdev))
        else:
            raise ValueError('need moments and/or histogram')
        
    def _fit_median(self, bins, prob):
        r""" Fit median model to histogram """
        # prob = prob / sum(prob)
        cumprob = numpy.cumsum(prob)[:-1]
        probspline = cspline.CSpline(bins, cumprob, alg='cspline')
        x0 = []
        for p0 in [0.317310507863 / 2., 0.5, 1 - 0.317310507863 / 2.]:
            if cumprob[0] < p0 and cumprob[-1] > p0:
                def f(x):
                    return probspline(x) - p0
                x0.append(root.refine(f, (bins[0], bins[-1])))
            else:
                x0.append(None)
        minus, median, plus = x0 
        if median is None or minus is None or plus is None:
            return None
        return _TwoSided(median, plus=plus - median, minus=median - minus, type='median')

    def _fit_splitnormal(self, bins, prob):
        r""" Fit split-normal model to histogram. """
        x = (bins[1:] + bins[:-1]) / 2
        wgt = prob[1:-1] / sum(prob[1:-1])
        def sum_plus(mu, n=2):
            idx = x > mu
            ans = sum(wgt[idx] * (x[idx] - mu) ** n)
            return ans 
        def sum_minus(mu, n=2):
            idx = x < mu
            ans = sum(wgt[idx] * (x[idx] - mu) ** n)
            return ans
        def Lhat(mu):
            return sum_plus(mu) ** (1/3) + sum_minus(mu) ** (1/3)
        def dLhat_dmu(mu):
            return (2/3) * (
                sum_plus(mu, n=2) ** (-2/3) * sum_plus(mu, n=1)  +
                sum_minus(mu, n=2) ** (-2/3) * sum_minus(mu, n=1)
                )
        # find search interval for mu
        lhatmu = [mean(Lhat(mu)) for mu in x]
        i = numpy.argmin(lhatmu)
        if i < 1 or i > len(x) - 2:
            # print('****', i, [dLhat_dmu(xx) for xx in x])
            return None
        if wgt[i-1] < 1e-8 * wgt[i] or 1e-8 * wgt[i] > wgt[i+1]:
            return None
        # search for root
        try:
            mu = root.refine(dLhat_dmu, interval=(x[i-1], x[i+1]))
        except:
            return None
        sigp = (Lhat(mu) * sum_plus(mu) ** (2/3)) ** 0.5
        sigm = (Lhat(mu) * sum_minus(mu) ** (2/3)) ** 0.5
        if mu is None or sigp is None or sigm is None:
            return None
        return _TwoSided(mu, plus=sigp, minus=sigm, type='splitnormal')

    # def __repr__(self):
    #     return str(self)
    
    def __str__(self):
        ans = self.prefix
        ans += 'mean = {}'.format(self.mean)
        for attr in ['sdev', 'skew', 'ex_kurt']:
            if hasattr(self, attr):
                x = getattr(self, attr)
                if isinstance(x, GVar):
                    ans += '   {} = {}'.format(attr, x)
                else:
                    ans += '   {} = {:.5}'.format(attr, x)
        if hasattr(self, 'splitnormal'):
            ans += '\n' + self.prefix
            ans += 'split-normal: {}'.format(self.splitnormal) 
        if hasattr(self, 'median'):
            ans += '\n' + self.prefix
            ans += '      median: {}'.format(self.median)
        return ans

    def plot_histogram(self, plot=None, show=False, fits=['mean', 'split-normal', 'median'], errorbars=True):
        r""" Plot histogram of probability density.

        Plots the histogram, overlayed with plots of the Gaussian or 
        two-sided Gaussians corresponding to the mean and standard deviation,
        or the median or split-normal distributions.

        Args:
            plot: Plotter. Set to ``matplotlib.pyplot`` if ``plot=None`` (default).

            show (bool): Plot is displayed if ``show=True``, and not otherwise (default).

            fits: Array indicating histogram fits to be plotted on the histogram.
                Default is ``fits=['mean', 'split-normal', 'median']`` which draws 
                all of  the fits normally examined by :class:`gvar.PDFStatistics`:
                a fit to a Gaussian based on the mean and standard deviation of 
                the distribution; a fit to a continuous two-sided Gaussian (split-normal); 
                and a fit to a (discontinuous) two-sided Gaussian centered on the median. 
                Set ``fits=[]`` to omit the fits from the plot.

            errorbars (bool): Plot errorbars on histogram if ``True`` (default) when 
                the input probabilities are |GVar|\s; ignored otherwise.
            
        Returns:
            The plotter.
        """
        if plot is None:
            import matplotlib.pyplot as plot 
        if self.prob is None or self.bins is None:
            return plot
        density = mean(self.prob[1:-1]) / (self.bins[1:] - self.bins[:-1])
        if errorbars:
            errors = sdev(self.prob[1:-1]) / (self.bins[1:] - self.bins[:-1])
            plot.errorbar(
                x=(self.bins[1:] + self.bins[:-1])/2., y=density, yerr=errors,
                alpha=0.5, lw=0.5, elinewidth=0.5, mew=0.5, ms=0.5,
                fmt='k.'
                )
        plot.bar(
            self.bins[:-1], density, width=self.bins[1:]-self.bins[:-1], align='edge',
            color='k', ec='k', alpha=0.1, lw=0.5, label='data'
            )
        x = numpy.linspace(*plot.xlim(), num=4000)
        g = _TwoSided(self.gvar.mean, self.gvar.sdev, self.gvar.sdev)
        sn = g(x) 
        if fits is None:
            fits = []
        if 'mean' in fits:
            plot.plot(x, sn, 'b:', label='mean')
        if 'split-normal' in fits and self.splitnormal is not None:
            sn = self.splitnormal(x)
            if sn is not None:
                plot.plot(x, sn, 'g', lw=1, label='split-normal')
        if 'median' in fits and self.median is not None:
            sn = self.median(x)
            if sn is not None:
                plot.plot(x, sn, 'r--', lw=1, label='median')
        plot.legend()
        plot.xlabel(r'$f(p)$')
        plot.ylabel(r'probability density')
        if show:
            plot.show()
        return plot
    

def make_fake_data(g, fac=1.0):
    r""" Make fake data based on ``g``.

    This function replaces the |GVar|\s in ``g`` by  new |GVar|\s with similar
    means and a similar covariance matrix, but multiplied by ``fac**2`` (so
    standard deviations are ``fac`` times smaller or larger). The changes are random.
    The function was designed to create fake data for testing fitting
    routines, where ``g`` is set equal to ``fitfcn(x, prior)`` and ``fac<1``
    (e.g.,  set ``fac=0.1`` to get fit parameters whose standard deviations
    are  10x smaller than those of the corresponding priors).

    Args:
        g (dict, array or gvar.GVar): The |GVar| or array of |GVar|\s,
            or dictionary whose values are |GVar|\s or arrays of |GVar|\s that
            from which the fake data is generated.

        fac (float): Uncertainties are rescaled by ``fac`` in the fake data.

    Returns:
        A collection of |GVar|\s with the same layout as ``g`` but with
        somewhat different means, and standard deviations rescaled by ``fac``.
    """
    if hasattr(g, 'keys'):
        if not isinstance(g, BufferDict):
            g = BufferDict(g)
        return BufferDict(g, buf=make_fake_data(g.buf, fac))
    else:
        g_shape = numpy.shape(g)
        g_flat = numpy.array(g).flat
        zero = numpy.zeros(len(g_flat), float)
        dg = (2. ** -0.5) * gvar(zero, evalcov(g_flat))
        dg *= fac
        noise = gvar(zero, sdev(dg))
        g_flat = mean(g_flat) + dg + noise + next(raniter(dg + noise))
        return g_flat[0] if g_shape == () else g_flat.reshape(g_shape)

def make_fake_sample_iter(g, n, fac=1.0):
    y = make_fake_data(g, fac=fac)
    y.flat = gvar(mean(y.flat), evalcov(y.flat) * n)
    for yi in raniter(y, n=n):
        yield yi

# legacy code support
fmt_partialsdev = fmt_errorbudget
#
