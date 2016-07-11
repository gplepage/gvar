""" Correlated gaussian random variables.

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

    - ``evalcorr(g)`` --- compute correlation matrix.

    - ``tabulate(g)`` --- create a table of GVar values in dict/array g.

    - ``fmt_values(g)`` --- create table of values.

    - ``fmt_errorbudget(g)`` --- create error-budget table.

    - ``fmt_chi2(f)`` --- format chi**2 information in f.

    - ``PDFIntegrator`` --- (class) integrator for probability density functions.

    - ``PDFStatistics`` --- (class) statistical analysis of moments of a random variable.

    - ``PDFHistogramBuilder`` --- (class) tool for building PDF histograms.

    - ``BufferDict`` --- (class) ordered dictionary with data buffer.

    - ``dump(g, outputfile)`` --- pickle |GVar|\s in file.

    - ``dumps(g)`` --- pickle |GVar|s in a string.

    - ``load(inputfile)`` --- read |GVar|\s from a file.

    - ``loads(inputstr)`` --- read |GVar|\s from a string.

    - ``raniter(g,N)`` --- iterator for random numbers.

    - ``bootstrap_iter(g,N)`` --- bootstrap iterator.

    - ``svd(g)`` --- SVD modification of correlation matrix.

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

    - ``powerseries`` --- power series representation
        of functions.

    - ``root`` --- root-finding for one-dimensional functions.
"""

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-15 G. Peter Lepage.
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

import collections
import sys

import numpy

try:
    import scipy.special
    _have_scipy = True
except ImportError:
    _have_scipy = False

from ._gvarcore import *
gvar = GVarFactory()            # order matters for this statement

from ._svec_smat import *
from ._bufferdict import BufferDict, asbufferdict
from ._utilities import *
from ._version import version as __version__

from . import dataset
from . import ode
from . import cspline
from . import linalg
from . import powerseries
from . import root

try:
    # use lsqfit's gammaQ if available; otherwise use one in ._utilities
    from lsqfit._utilities import gammaQ
except:
    pass

_GVAR_LIST = []

def ranseed(seed=None):
    """ Seed random number generators with tuple ``seed``.

    Argument ``seed`` is a :class:`tuple` of integers that is used to seed
    the random number generators used by :mod:`numpy` and
    :mod:`random` (and therefore by :mod:`gvar`). Reusing
    the same ``seed`` results in the same set of random numbers.

    ``ranseed`` generates its own seed when called without an argument
    or with ``seed=None``. This seed is stored in ``ranseed.seed`` and
    also returned by the function. The seed can be used to regenerate
    the same set of random numbers at a later time.

    :param seed: A tuple of integers. Generates a random tuple if ``None``.
    :type seed: tuple or None
    :returns: The seed.
    """
    if seed is None:
        seed = numpy.random.randint(1, int(2e9), size=3)
    seed = tuple(seed)
    numpy.random.seed(seed)
    ranseed.seed = seed
    return seed

def switch_gvar(cov=None):
    """ Switch :func:`gvar.gvar` to new :class:`gvar.GVarFactory`.

    :returns: New :func:`gvar.gvar`.
    """
    global gvar
    _GVAR_LIST.append(gvar)
    gvar = GVarFactory(cov)
    return gvar

def restore_gvar():
    """ Restore previous :func:`gvar.gvar`.

    :returns: Previous :func:`gvar.gvar`.
    """
    global gvar
    try:
        gvar = _GVAR_LIST.pop()
    except IndexError:
        raise RuntimeError("no previous gvar")
    return gvar

def gvar_factory(cov=None):
    """ Return new function for creating |GVar|\s (to replace
    :func:`gvar.gvar`).

    If ``cov`` is specified, it is used as the covariance matrix
    for new |GVar|\s created by the function returned by
    ``gvar_factory(cov)``. Otherwise a new covariance matrix is created
    internally.
    """
    return GVarFactory(cov)

def chi2(g1, g2=None, svdcut=1e-15, nocorr=False, fmt=False):
    """ Compute chi**2 of ``g1-g2``.

    ``chi**2`` is a measure of whether the multi-dimensional
    Gaussian distributions ``g1`` and ``g2`` (dictionaries or arrays)
    agree with each other --- that is, do their means agree
    within errors for corresponding elements. The probability is high
    if ``chi2(g1,g2)/chi2.dof`` is of order 1 or smaller.

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
    overlap are used. Also setting ``g2=None`` is equivalent to replacing its
    elements by zeros.

    ``chi**2`` is computed from the inverse of the covariance matrix
    of ``g1-g2``. The matrix inversion can be sensitive to roundoff
    errors. In such cases, SVD cuts can be applied by setting
    parameters ``svdcut``; see the documentation
    for :func:`gvar.svd`, which is used to apply the cut.

    The return value is the ``chi**2``. Extra attributes attached to this
    value give additional information:

    - **dof** --- Number of degrees of freedom (that is, the number of variables
      compared).

    - **Q** --- The probability that the ``chi**2`` could have been larger,
      by chance, even if ``g1`` and ``g2`` agree. Values smaller than 0.1
      or so suggest that they do not agree. Also called the *p-value*.
    """
    # customized class for answer
    class ans(float):
        def __new__(cls, chi2, dof, Q):
            return float.__new__(cls, chi2)
        def __init__(self, chi2, dof, Q):
            self.dof = dof
            self.Q = Q

    # leaving nocorr (turn off correlations) undocumented because I
    #   suspect I will remove it
    if g2 is None:
        diff = BufferDict(g1).buf if hasattr(g1, 'keys') else numpy.asarray(g1).flatten()
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
    dof = diff.size
    if dof == 0:
        return ans(0.0, 0, 0)
    if nocorr:
        # ignore correlations
        chi2 = numpy.sum(mean(diff) ** 2 / var(diff))
        dof = len(diff)
    else:
        diffmod, i_wgts = svd(diff, svdcut=svdcut, wgts=-1)
        diffmean = mean(diffmod)
        i, wgts = i_wgts[0]
        chi2 = 0.0
        if len(i) > 0:
            chi2 += numpy.sum((diffmean[i] * wgts) ** 2)
        for i, wgts in i_wgts[1:]:
            chi2 += numpy.sum(wgts.dot(diffmean[i]) ** 2)
        dof = numpy.sum(len(wgts) for i, wgts in i_wgts)
    Q = gammaQ(dof/2., chi2/2.)
    return ans(chi2, dof=dof, Q=Q)

def equivalent(g1, g2, rtol=1e-10, atol=1e-10):
    """ Determine whether ``g1`` and ``g2`` contain equivalent |GVar|\s.

    Compares sums and differences of |GVar|\s stored in ``g1``
    and ``g2`` to see if they agree with tolerances. Operationally,
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
        if not numpy.all(numpy.abs(di.der) < (numpy.abs(ai.der) * rtol + atol)):
            return False
    return True


def fmt_chi2(f):
    """ Return string containing ``chi**2/dof``, ``dof`` and ``Q`` from ``f``.

    Assumes ``f`` has attributes ``chi2``, ``dof`` and ``Q``. The
    logarithm of the Bayes factor will also be printed if ``f`` has
    attribute ``logGBF``.
    """
    if hasattr(f, 'logGBF'):
        fmt = "chi2/dof = %.2g [%d]    Q = %.2g    log(GBF) = %.5g"
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt % (chi2_dof, f.dof, f.Q, f.logGBF)
    else:
        fmt = "chi2/dof = %.2g [%d]    Q = %.2g"
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt % (chi2_dof, f.dof, f.Q)

def svd(g, svdcut=1e-15, wgts=False):
    """ Apply svd cuts to collection of |GVar|\s in ``g``.

    Standard usage is, for example, ::

        svdcut = ...
        gmod = svd(g, svdcut=svdcut)

    where ``g`` is an array of |GVar|\s or a dictionary containing |GVar|\s
    and/or arrays of |GVar|\s. When ``svdcut>0``, ``gmod`` is
    a copy of ``g`` whose |GVar|\s have been modified to make
    their correlation matrix less singular than that of the
    original ``g``: each eigenvalue ``eig`` of the correlation matrix is
    replaced by ``max(eig, svdcut * max_eig)`` where ``max_eig`` is
    the largest eigenvalue. This SVD cut, which is applied separately
    to each block-diagonal sub-matrix of the correlation matrix,
    increases the variance of the eigenmodes with eigenvalues smaller
    than ``svdcut * max_eig``.

    When ``svdcut`` is negative, eigenmodes of the correlation matrix
    whose eigenvalues are smaller than ``|svdcut| * max_eig`` are dropped
    from the new matrix and the corresponding components of ``g`` are
    zeroed out (that is, replaced by 0(0)) in ``gmod``.

    There is an additional parameter ``wgts`` in :func:`gvar.svd` whose
    default value is ``False``. Setting ``wgts=1`` or ``wgts=-1`` instead
    causes :func:`gvar.svd` to return a tuple ``(gmod, i_wgts)`` where
    ``gmod``  is the modified copy of ``g``, and ``i_wgts`` contains a
    spectral  decomposition of the covariance matrix corresponding to
    the  modified correlation matrix if ``wgts=1``, or a decomposition of its
    inverse if ``wgts=-1``. The first entry ``i, wgts = i_wgts[0]``  specifies
    the diagonal part of the matrix: ``i`` is a list of the indices in
    ``gmod.flat`` corresponding to diagonal elements, and ``wgts ** 2``
    gives the corresponding matrix elements. The second and subsequent
    entries, ``i, wgts = i_wgts[n]`` for ``n > 0``, each correspond
    to block-diagonal sub-matrices, where ``i`` is the list of
    indices corresponding to the block, and ``wgts[j]`` are eigenvectors of
    the sub-matrix rescaled so that ::

        numpy.sum(numpy.outer(wi, wi) for wi in wgts[j]

    is the sub-matrix (``wgts=1``) or its inverse (``wgts=-1``).

    To compute the inverse of the covariance matrix from ``i_wgts``,
    for example, one could use code like::

        gmod, i_wgts = svd(g, svdcut=svdcut, wgts=-1)

        inv_cov = numpy.zeros((n, n), float)
        i, wgts = i_wgts[0]                       # 1x1 sub-matrices
        if len(i) > 0:
            inv_cov[i, i] = numpy.array(wgts) ** 2
        for i, wgts in i_wgts[1:]:                # nxn sub-matrices (n>1)
            for w in wgts:
                inv_cov[i, i[:, None]] += numpy.outer(w, w)

    This sets ``inv_cov`` equal to the inverse of the covariance matrix of
    the ``gmod``\s. Similarly, we can  compute the expectation value,
    ``u.dot(inv_cov.dot(v))``, between two vectors (:mod:`numpy` arrays) using::

        result = 0.0
        i, wgts = i_wgts[0]                       # 1x1 sub-matrices
        if len(i) > 0:
            result += numpy.sum((u[i] * wgts) * (v[i] * wgts))
        for i, wgts in i_wgts[1:]:                # nxn sub-matrices (n>1)
            result += numpy.sum(wgts.dot(u[i]) * wgts.dot(v[i]))

    where ``result`` is the desired expectation value.

    The input parameters are :

    :param g: An array of |GVar|\s or a dicitionary whose values are
        |GVar|\s and/or arrays of |GVar|\s.
    :param svdcut: If positive, replace eigenvalues ``eig`` of the correlation
        matrix with ``max(eig, svdcut * max_eig)`` where ``max_eig`` is
        the largest eigenvalue; if negative,
        discard eigenmodes with eigenvalues smaller
        than ``|svdcut| * max_eig``. Default is 1e-15.
    :type svdcut: ``None`` or number ``(|svdcut|<=1)``.
    :param wgts: Setting ``wgts=1`` causes :func:`gvar.svd` to compute
        and return a spectral decomposition of the covariance matrix of
        the modified |GVar|\s, ``gmod``. Setting ``wgts=-1`` results in
        a decomposition of the inverse of the covariance matrix. The
        default value is ``False``, in which case only ``gmod`` is returned.
    :returns: A copy ``gmod`` of ``g`` whose correlation matrix is modified by
        *svd* cuts. If ``wgts`` is not ``False``,
        a tuple ``(g, i_wgts)`` is returned where ``i_wgts``
        contains a spectral decomposition of ``gmod``'s
        covariance matrix or its inverse.

    Data from the *svd* analysis of ``g``'s covariance matrix is stored in
    ``svd`` itself:

    .. attribute:: svd.dof

        Number of independent degrees of freedom left after the
        *svd* cut. This is the same as the number initially unless
        ``svdcut < 0`` in which case it may be smaller.

    .. attribute:: svd.nmod

        Number of modes whose eignevalue was modified by the
        *svd* cut.

    .. attribute:: svd.nblocks

        A dictionary where ``svd.nblocks[s]`` contains the number of
        block-diagonal ``s``-by-``s`` sub-matrices in the correlation
        matrix.

    .. attribute:: svd.eigen_range

        Ratio of the smallest to largest eigenvalue before *svd* cuts are
        applied (but after rescaling).

    .. attribute:: svd.logdet

        Logarithm of the determinant of the covariance matrix after *svd*
        cuts are applied (excluding any omitted modes when
        ``svdcut < 0``).

    .. attribute:: svd.correction

        Array containing the *svd* corrections that were added to ``g.flat``
        to create the modified ``g``\s.
    """
    # replace g by a copy of g
    if hasattr(g,'keys'):
        g = BufferDict(g)
    else:
        g = numpy.array(g)
    cov = evalcov(g.flat)
    block_idx = find_diagonal_blocks(cov)
    svd.logdet = 0.0
    svd.correction = numpy.zeros(cov.shape[0], object)
    svd.correction[:] = gvar(0, 0)
    svd.eigen_range = 1.
    svd.nmod = 0
    if wgts is not False:
        i_wgts = [([], [])] # 1st entry for all 1x1 blocks
    lost_modes = 0
    svd.nblocks = {}
    for idx in block_idx:
        svd.nblocks[len(idx)] = svd.nblocks.get(len(idx), 0) + 1
        if len(idx) == 1:
            i = idx[0]
            svd.logdet += numpy.log(cov[i, i])
            if wgts is not False:
                i_wgts[0][0].append(i)
                i_wgts[0][1].append(cov[i, i] ** (wgts * 0.5))
        else:
            idxT = idx[:, numpy.newaxis]
            block_cov = cov[idx, idxT]
            s = SVD(block_cov, svdcut=svdcut, rescale=True, compute_delta=True)
            if s.D is not None:
                svd.logdet -= 2 * numpy.sum(numpy.log(di) for di in s.D)
            svd.logdet += numpy.sum(numpy.log(vali) for vali in s.val)
            if s.delta is not None:
                svd.correction[idx] = s.delta
                g.flat[idx] += s.delta
            if wgts is not False:
                i_wgts.append(
                    (idx, [w for w in s.decomp(wgts)[::-1]])
                    )
            if svdcut is not None and svdcut < 0:
                newg = numpy.zeros(len(idx), object)
                for w in s.vec:
                    newg += (w / s.D) * (w.dot(s.D * g.flat[idx]))
                lost_modes += len(idx) - len(s.vec)
                g.flat[idx] = newg
            if s.eigen_range < svd.eigen_range:
                svd.eigen_range = s.eigen_range
            svd.nmod += s.nmod
    svd.dof = len(g.flat) - lost_modes
    svd.nmod += lost_modes
    # svd.blocks = block_idx

    # repack into numpy arrays
    if wgts is not False:
        tmp = []
        for iw, wgts in i_wgts:
            tmp.append(
                (numpy.array(iw, numpy.intp), numpy.array(wgts, numpy.double))
                )
        i_wgts = tmp
        return (g, i_wgts)
    else:
        return g

def find_diagonal_blocks(m):
    """ Find block-diagonal components of matrix m.

    Returns a list of index arrays identifying the blocks. The 1x1
    blocks are listed first.

    Used by svd.
    """
    unassigned_indices = set(range(m.shape[0]))
    non_zero = []
    blocks = []
    for i in range(m.shape[0]):
        non_zero.append(set(m[i].nonzero()[0]))
        non_zero[i].add(i)
        if len(non_zero[i]) == 1:
            # diagonal element
            blocks.append(non_zero[i])
            unassigned_indices.remove(i)
    while unassigned_indices:
        new_block = non_zero[unassigned_indices.pop()]
        for j in unassigned_indices:
            if not new_block.isdisjoint(non_zero[j]):
                new_block.update(non_zero[j])
        unassigned_indices.difference_update(new_block)
        blocks.append(new_block)
    for i in range(len(blocks)):
        blocks[i] = numpy.array(sorted(blocks[i]))
    return blocks

def tabulate(g, ncol=1, headers=True, offset='', ndecimal=None):
    """ Tabulate contents of an array or dictionary of |GVar|\s.

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
    """
    entries = []
    if hasattr(g, 'keys'):
        if headers is True:
            headers = ('key/index', 'value')
        g = asbufferdict(g, keylist=g.keys())
        for k in g:
            if g.isscalar(k):
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
    if len(headers) != 2:
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

class ExtendedDict(BufferDict):
    """ |BufferDict| that supports variables from extended distributions.

    Used for parameters when there may be log-normal/sqrt-normal/...  variables.
    The exponentiated/squared/... values of those variables are included in the
    BufferDict, together with  the original versions. Method
    :meth:`ExtendedDict.refill_buf` refills the buffer with  a 1-d array and
    then fills in the exponentiated/squared values of  the log-normal/sqrt-
    normal variables ---  that is, ``p.refill_buf(newbuf)``
    replaces ``p.buf = newbuf``.

    Use function :meth:`gvar.add_parameter_distribution` to add distributions.

    N.B. ExtendedDict is *not* part of the public api yet (or maybe ever).
    """

    extension_pattern = re.compile('^([^()]+)\((.+)\)$')
    extension_fcn = {}

    def __init__(self, p0, buf=None):
        super(ExtendedDict, self).__init__(p0, buf=buf)
        extensions = []
        newkeys = []
        for k in p0:
            k_stripped, k_fcn = ExtendedDict.stripkey(k)
            if k_fcn is not None:
                self[k_stripped] = k_fcn(self[k])
                extensions.append(
                    (self.slice(k_stripped), k_fcn, self.slice(k))
                    )
                newkeys.append(k_stripped)
        self.extensions = extensions
        self._newkeys = newkeys

    def refill_buf(self, newbuf):
        if len(newbuf) != len(self.buf):
            self.buf = numpy.resize(newbuf, len(self.buf))
        else:
            self.buf = newbuf
        for s1, f, s2 in self.extensions:
            self.buf[s1] = f(self.buf[s2])

    def newkeys(self):
        " Iterator containing new keys generated by :class:`ExtendedDict`. "
        return iter(self._newkeys)

    @staticmethod
    def basekey(prior, k):
        """ Find base key in ``prior`` corresponding to ``k``. """
        if not isinstance(k, str):
            return k
        for f in ExtendedDict.extension_fcn:
            newk = f + '(' + k + ')'
            if newk in prior:
                return newk
        return k

    @staticmethod
    def stripkey(k):
        """ Return (stripped key, fcn) where fcn is exp or square or ...

        Strip off any ``"log"`` or ``"sqrt"`` or ... prefix.
        """
        if not isinstance(k, str):
            return k, None
        m = re.match(ExtendedDict.extension_pattern, k)
        if m is None:
            return k, None
        k_fcn, k_stripped = m.groups()
        if k_fcn not in ExtendedDict.extension_fcn:
            return k, None
        return k_stripped, ExtendedDict.extension_fcn[k_fcn]

def add_parameter_parentheses(p):
    """ Return dictionary with proper keys for parameter distributions (legacy code).

    This utility function helps fix legacy code that uses
    parameter keys like ``logp`` or ``sqrtp`` instead of
    ``log(p)`` or ``sqrt(p)``, as now required. This method creates a
    copy of  dictionary ``p'' but with keys like ``logp`` or ``sqrtp``
    replaced by ``log(p)`` or ``sqrt(p)``. So setting ::

        p = add_parameter_parentheses(p)

    fixes the keys in ``p`` for log-normal and sqrt-normal parameters.
    """
    newp = BufferDict()
    for k in p:
        if isinstance(k, str):
            if k[:3] == 'log' and ExtendedDict.stripkey(k)[1] is None:
                newk = 'log(' + k[3:] + ')'
            elif k[:4] == 'sqrt' and ExtendedDict.stripkey(k)[1] is None:
                newk = 'sqrt(' + k[4:] + ')'
            else:
                newk = k
            newp[newk] = p[k]
    return newp


def add_parameter_distribution(name, invfcn):
    """ Add new parameter distribution for use in fits.

    This function adds new distributions for the parameters used in
    :class:`lsqfit.nonlinear_fit`. For example, the code ::

        import gvar as gv
        gv.add_parameter_distribution('log', gv.exp)

    enables the use of log-normal distributions for parameters. The log-normal
    distribution is invoked for a parameter ``p`` by including ``log(p)``
    rather than ``p`` itself in the fit prior. log-normal and sqrt-normal
    distributions are included by default.

    These distributions are implemented by replacing a fit parameter ``p``
    by a new fit parameter ``fcn(p)`` where ``fcn`` is some function. ``fcn(p)``
    is assumed to have a Gaussian distribution, and parameter ``p`` is
    recovered using the inverse function ``invfcn`` where ``p=invfcn(fcn(p))``.

    :param name: Distribution's name.
    :type name: str
    :param invfcn: Inverse of the transformation function.
    """
    ExtendedDict.extension_fcn[name] = invfcn

def del_parameter_distribution(name):
    """ Delete parameter distribution ``name``. """
    del ExtendedDict.extension_fcn[name]

# default extensions
add_parameter_distribution('log', numpy.exp)
add_parameter_distribution('sqrt', numpy.square)

def trim_redundant_keys(p):
    """ Remove redundant keys from dictionary ``p``.

    A key ``'c'`` is redundant if either of ``'log(c)'``
    or ``'sqrt(c)'`` is also a key. (There are additional redundancies
    if :meth:`gvar.add_parameter_distribution` has been used to add
    extra distributions.) This function creates a copy of ``p`` but with
    the redundant keys removed.
    """
    keys = list(p.keys())
    for k in p:
        k_stripped, k_fcn = ExtendedDict.stripkey(k)
        if k_fcn is not None:
            try:
                keys.remove(k_stripped)
            except ValueError:
                pass
    return BufferDict(p, keys=keys)

try:
    import vegas
    class PDFIntegrator(vegas.Integrator):
        """ :mod:`vegas` integrator for PDF expectation values.

        ``PDFIntegrator(g)`` is a :mod:`vegas` integrator that evaluates
        expectation values for the multi-dimensional Gaussian distribution
        specified by with ``g``, which is a |GVar| or an array of |GVar|\s or a
        dictionary whose values are |GVar|\s or arrays of |GVar|\s.

        ``PDFIntegrator`` reformulates integrals over the variables in ``g``
        in terms of new variables that diagonalize ``g``'s covariance matrix.
        This greatly facilitates integration over these variables using the
        :mod:`vegas` module for multi-dimensional integration. (The :mod:`vegas`
        module must be installed in order to use ``PDFIntegrator``.)

        A simple illustration of ``PDFIntegrator`` is given by the following
        code::

            import gvar as gv

            # multi-dimensional Gaussian distribution
            g = gv.BufferDict()
            g['a'] = gv.gvar([0., 1.], [[1., 0.9], [0.9, 1.]])
            g['b'] = gv.gvar('1(1)')

            # integrator for expectation values in distribution g
            g_expval = gv.PDFIntegrator(g)

            # want expectation value of [f(p), f(p)**2]
            def f(p):
                return p['a'][0] * p['a'][1] + p['b']

            def f_f2(p):
                fp = f(p)
                return [fp, fp ** 2]

            # adapt g_expval to f; warmup = <f(p)> in distribution g
            warmup = g_expval(f, neval=1000, nitn=5)

            # results = <f_f2> in distribution g
            results = g_expval(f_f2, neval=1000, nitn=5, adapt=False)
            print (results.summary())
            print ('results =', results, '\\n')

            # mean and standard deviation of f(p)'s distribution
            fmean = results[0]
            fsdev = gv.sqrt(results[1] - results[0] ** 2)
            print ('f.mean =', fmean, '   f.sdev =', fsdev)
            print ("Gaussian approx'n for f(g) =", f(g))

        where the ``warmup`` calls to the integrator are used to
        adapt it to ``f(p)``, and the final results are in ``results``.
        Here ``neval`` is the (approximate) number of function calls
        per iteration of the :mod:`vegas` algorithm and ``nitn`` is the
        number of iterations. We adapt the integrator to ``f(p)`` and
        then use it to calculated the expectation value of ``f(p)`` and
        ``f(p)**2``, so we can compute the standard deviation for the
        distribution of ``f(p)``\s. The output from this code shows that
        the Gaussian approximation (1.0(1.4)) for the mean and
        standard deviation of the ``f(p)`` distribution is not particularly
        accurate here (correct value is 1.9(2.0)), because of the large
        uncertainties in ``g``::

            itn   integral        average         chi2/dof        Q
            -------------------------------------------------------
              1   1.880(15)       1.880(15)           0.00     1.00
              2   1.912(19)       1.896(12)           2.66     0.07
              3   1.892(18)       1.8947(99)          2.31     0.06
              4   1.918(17)       1.9006(85)          2.02     0.06
              5   1.910(19)       1.9026(78)          1.45     0.17

            results = [1.9026(78) 7.479(84)]

            f.mean = 1.9026(78)    f.sdev = 1.965(19)
            Gaussian approx'n for f(g) = 1.0(1.4)

        In general functions being integrated can return a number, or an array of
        numbers, or a dictionary whose values are numbers or arrays of numbers.
        This allows multiple expectation values to be evaluated simultaneously.

        See the documentation with the :mod:`vegas`module for more details on its
        use. The example sets ``adapt=False`` when  computing final results. This
        gives more reliable error estimates  when ``neval`` is small, as it is
        here. Note that ``neval`` may need to be much larger (tens or hundreds of
        thousands) for difficult high-dimension integrals.

        Args:
            g : |GVar|, array of |GVar|\s, or dictionary whose values
                are |GVar|\s or arrays of |GVar|\s that specifies the
                multi-dimensional Gaussian distribution used to construct
                the probability density function.

            svdcut (non-negative float or None): If not ``None``, replace
                covariance matrix of ``g`` with a new matrix whose
                small eigenvalues are modified: eigenvalues smaller than
                ``svdcut`` times the maximum eigenvalue ``eig_max`` are
                replaced by ``svdcut*eig_max``. This can ameliorate
                problems caused by roundoff errors when inverting the
                covariance matrix. It increases the uncertainty associated
                with the modified eigenvalues and so is conservative.
                Setting ``svdcut=None`` or ``svdcut=0`` leaves the
                covariance matrix unchanged. Default is ``1e-15``.

            limit (positive float): Limits the integrations to a finite
                region of size ``limit`` times the standard deviation on
                either side of the mean. This can be useful if the
                functions being integrated misbehave for large parameter
                values (e.g., ``numpy.exp`` overflows for a large range of
                arguments). Default is ``1e15``.
        """
        def __init__(self, g, limit=1e15, svdcut=1e-15, scale=1.):
            self.extend = False
            if hasattr(g, 'keys'):
                # g is a dict
                if isinstance(g, ExtendedDict):
                    self.extend = True
                    g = trim_redundant_keys(g)
                elif not isinstance(g, BufferDict):
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
            self.pjac = s.val ** 0.5
            for i, sigi in enumerate(s.val ** 0.5):
                self.vec_sig[i] *= sigi
                self.vec_isig[i] /= sigi
            self.scale = scale
            self.gmean = mean(gflat)
            self.g = g
            self.log_gnorm = numpy.sum(0.5 * numpy.log(2 * numpy.pi * s.val))
            self.limit = abs(limit)
            if _have_scipy and limit <= 8.:
                limit = scipy.special.ndtr(self.limit)
                super(PDFIntegrator, self).__init__(self.gmean.size * [(1. - limit, limit)])
                self._expval = self._expval_ndtri
            else:
                integ_map = self._make_map(self.limit)
                super(PDFIntegrator, self).__init__(self.gmean.size * [integ_map])
                self._expval = self._expval_tan

        def _make_map(self, limit):
            """ Make vegas grid that is adapted to the pdf. """
            ny = 2000
            y = numpy.random.uniform(0., 1., (ny,1))
            limit = numpy.arctan(limit)
            m = vegas.AdaptiveMap([[-limit, limit]], ninc=100)
            theta = numpy.empty(y.shape, float)
            jac = numpy.empty(y.shape[0], float)
            for itn in range(10):
                m.map(y, theta, jac)
                tan_theta = numpy.tan(theta[:, 0])
                x = self.scale * tan_theta
                fx = (tan_theta ** 2 + 1) * numpy.exp(-(x ** 2) / 2.)
                m.add_training_data(y, (jac * fx) ** 2)
                m.adapt(alpha=1.5)
            return numpy.array(m.grid[0])

        def __call__(self, f=None, nopdf=False, mpi=False, **kargs):
            """ Estimate expectation value of function ``f(p)``.

            Uses module :mod:`vegas` to estimate the integral of
            ``f(p)`` multiplied by the probability density function
            associated with ``g`` (i.e., ``pdf(p)``). The probability
            density function is omitted if ``nopdf=True`` (default
            setting is ``False``). Setting ``mpi=True`` configures vegas
            for multi-processor running using MPI.

            Args:
                f (function): Function ``f(p)`` to integrate. Integral is
                    the expectation value of the function with respect
                    to the distribution. The function can return a number,
                    an array of numbers, or a dictionary whose values are
                    numbers or arrays of numbers.

                nopdf (bool): If ``True`` drop the probability density function
                    from the integrand (so no longer an expectation value).
                    Default is ``False``.

                mpi (bool): If ``True`` configure for use with multiple processors
                    and MPI. This option requires module :mod:`mpi4py`. A
                    script ``xxx.py`` using an MPI integrator is run
                    with ``mpirun``: e.g., ::

                        mpirun -np 4 python xxx.py

                    runs on 4 processors. Setting ``mpi=False`` (default) does
                    not support multiple processors.

            All other keyword arguments are passed on to a :mod:`vegas`
            integrator; see the :mod:`vegas` documentation for further information.
            """
            integrand = self._expval(f, nopdf)
            if mpi:
                integrand = vegas.MPIintegrand(integrand)
            else:
                integrand = vegas.batchintegrand(integrand)
            return super(PDFIntegrator, self).__call__(integrand, **kargs)

        def logpdf(self, p):
            """ Logarithm of the probability density function evaluated at ``p``. """
            if hasattr(p, 'keys'):
                dp = BufferDict(p).buf[:len(self.gmean)] - self.gmean
            else:
                dp = numpy.asarray(p).reshape(-1) - self.gmean
            x2 = self.vec_isig.dot(dp) ** 2
            return -0.5 * numpy.sum(x2) - self.log_gnorm

        def pdf(self, p):
            """ Probability density function associated with ``g`` evaluated at ``p``."""
            return numpy.exp(self.logpdf(p))

        def _expval_ndtri(self, f, nopdf):
            """ Return integrand using ndtr mapping. """
            def ff(theta, nopdf=nopdf):
                x = scipy.special.ndtri(theta)
                dp = x.dot(self.vec_sig)
                if nopdf:
                    # must remove built in pdf
                    pdf = (
                        numpy.sqrt(2 * numpy.pi) * numpy.exp((x ** 2) / 2.)
                        * self.pjac[None,:]
                        )
                else:
                    pdf = numpy.ones(numpy.shape(x), float)
                ans = []
                ans_dict = collections.OrderedDict()
                for dpi, pdfi in zip(dp, pdf):
                    p = self.gmean + dpi
                    if self.g.shape is None:
                        if self.extend:
                            p = ExtendedDict(self.g, buf=p)
                        else:
                            p = BufferDict(self.g, buf=p)
                    else:
                        p = p.reshape(self.g.shape)
                    fp = 1. if f is None else f(p)
                    if hasattr(fp, 'keys'):
                        if not isinstance(fp, BufferDict):
                            fp = BufferDict(fp)
                        fp = BufferDict(fp, buf=fp.buf * numpy.prod(pdfi))
                        for k in fp:
                            if k in ans_dict:
                                ans_dict[k].append(fp[k])
                            else:
                                ans_dict[k] = [fp[k]]
                    else:
                        fp = numpy.asarray(fp) * numpy.prod(pdfi)
                        ans.append(fp)
                return numpy.array(ans) if len(ans) > 0 else BufferDict(ans_dict)
            return ff

        def _expval_tan(self, f, nopdf):
            """ Return integrand using the tan mapping. """
            def ff(theta, nopdf=nopdf):
                tan_theta = numpy.tan(theta)
                x = self.scale * tan_theta
                jac = self.scale * (tan_theta ** 2 + 1.)
                if nopdf:
                    pdf = jac * self.pjac[None, :]
                else:
                    pdf = jac * numpy.exp(-(x ** 2) / 2.) / numpy.sqrt(2 * numpy.pi)
                dp = x.dot(self.vec_sig)
                ans = []
                ans_dict = collections.OrderedDict()
                for dpi, pdfi in zip(dp, pdf):
                    p = self.gmean + dpi
                    if self.g.shape is None:
                        if self.extend:
                            p = ExtendedDict(self.g, buf=p)
                        else:
                            p = BufferDict(self.g, buf=p)
                    else:
                        p = p.reshape(self.g.shape)
                    fp = 1. if f is None else f(p)
                    if hasattr(fp, 'keys'):
                        if not isinstance(fp, BufferDict):
                            fp = BufferDict(fp)
                        fp = BufferDict(fp, buf=fp.buf * numpy.prod(pdfi))
                        for k in fp:
                            if k in ans_dict:
                                ans_dict[k].append(fp[k])
                            else:
                                ans_dict[k] = [fp[k]]
                    else:
                        fp = numpy.asarray(fp) * numpy.prod(pdfi)
                        ans.append(fp)
                return numpy.array(ans) if len(ans) > 0 else BufferDict(ans_dict)
            return ff
except:
    pass

class PDFHistogramBuilder(object):
    """ Utility class for creating PDF histograms using :class:`PDFIntegrator`.

    This class is used to create a histogram of the probability
    distribution function (PDF) for some function ``f(p)`` where ``p``
    represents a (possibly multi-dimensional) Gaussian distribution.
    A trivial example would be to compute and display the PDF
    for ``(p + 2) ** 2 / 4`` where ``p`` is Gaussian variable::

        import gvar as gv
        import numpy as np

        p = gv.gvar('1(1)')

        # function whose PDF will be histogrammed
        def f(p):
            return (p + 2) ** 2 / 4.

        # use f(p) to set up histogram bins
        p2hist = gv.PDFHistogramBuilder(f(p))

        # integrand used to build histogram
        def fhist(p):                               # histogram of f(p)
            return p2hist.integrand(f(p))

        # integrate in two steps
        integ = gv.PDFIntegrator(p)

        # step 1 - adapt integrator to the pdf
        integ(neval=1000, nitn=5)

        # step 2 - integrate the histogram
        results = integ(fhist, neval=1000, nitn=5, adapt=False)

        # print statistical analysis of histogram and plot it
        print(p2hist.histogram(results).stats)  # statistics for p**2 PDF
        plt = p2hist.make_plot(results)
        plt.xlabel('$(p + 2)^2/4$')
        plt.ylabel('probability')
        plt.show()                              # display plot

    The output lists approximate statistics based on the histogram (see
    :class:`PDFStats`):

        mean = 2.4656(12)   sdev = 1.5715(14)   skew = 0.6537(40)   ex_kurt = 0.334(13)

    It also displays a plot showing the probability of ``(p + 2) ** 2 / 4``
    being in each bin. More typically the distribution would be
    multi-dimensional, with ``p`` being an array of |GVar|\s or a dictionary
    whose values are |GVar|\s or arrays of |GVar|\s.

    Args:
        g (|GVar|): The mean and standard deviation of ``g`` are used to
            design the histogram bins, which are centered on the mean.
        nbin (int): The number of histogram bins. Set equal to
            ``PDFHistogramBuilder.default_nbin'' if ``None`` (initial
            value is 8.)
        binwidth(float): The width of each bin is ``binwidth * g.sdev``.
            Set equal to ``PDFHistogramBuilder.default_binwidth'' if ``None``
            (initial value is 1.)
        bins (array): Bin edges for the histogram. ``len(bins)`` is one
            larger than the number of bins. If specified it overrides
            the default bin design specified by ``g``. (Default is ``None``.)

    Attributes:
        bins (array): Same as above.
        midpoints (array): Bin midpoints.
        widths (array): Bin widths.
    """

    Histogram = collections.namedtuple('Histogram', 'bins, prob, stats, norm ')
    default_nbin = 8
    default_binwidth = 1.

    def __init__(self, g=None, nbin=None, binwidth=None, bins=None):
        self.g = g
        if nbin is None:
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
        self.hist = numpy.zeros(len(self.bins) - 1, float)
        self.above = 0
        self.below = 0

    def integrand(self, val):
        """ Integrand for creating histogram using :mod:`vegas`.

        Adds ``[0...1...0]`` to the bins where the 1 is the bin
        into which ``val`` falls. This array-valued integrand
        corresponds to a theta function for each bin. It is
        for use inside an integrand for :meth:`PDFIntegrator.expval`.
        The integrated result is turned into histogram data using
        :meth:`PDFHistogramBuilder.histogram(...)`.

        Args:
            val (float): Value of the quantity being histogrammed.
        """
        self.hist[:] = 0.0
        if val > self.bins[-1]:
            self.above += 1
        elif val <= self.bins[0]:
            self.below += 1
        else:
            self.hist[numpy.searchsorted(self.bins, val) - 1] = 1.
        return numpy.array([1.] + self.hist.tolist())

    def histogram(self, data):
        """ Convert :mod:`vegas` output into histogram data.

        Extracts histogram from integration results obtained
        from an instance of :meth:`PDFIntegrator` applied to
        integrand :meth:`PDFHistogramBuilder.integrand`. Returns
        a named tuple containing the following information (in order).

        Returns (named tuple):
            bins (array): Bin edges for histogram.
            prob (array): Probability in each bin.
            stats (PDFStats): Statistical data about histogram.
            norm (float): Normalization used to normalize histogram.
        """
        if len(data) == len(self.midpoints) + 1:
            norm = data[0]
            data = data[1:] / norm
        elif len(data) != len(self.midpoints):
            raise ValueError(
                'wrong data length: %s != %s'
                    % (len(data), len(self.midpoints))
                )
        else:
            norm = 1.
        mid = self.midpoints
        stats = PDFStatistics(numpy.sum(
            [mid * data, mid ** 2 * data, mid **3 * data, mid ** 4 * data],
            axis=1
            ))
        return PDFHistogramBuilder.Histogram(self.bins, data, stats, norm)

    @staticmethod
    def gaussian_pdf(x, g):
        return (
            numpy.exp(-(x - g.mean) ** 2 / 2. /g.var) /
            numpy.sqrt(g.var * 2 * numpy.pi)
            )

    def make_plot(
        self, data, plot=None, show=False, density=False,
        bar=dict(alpha=0.2, color='b'),
        errorbar=dict(fmt='b.'),
        gaussian=dict(ls='--', c='r')
        ):
        """ Convert :mod:`vegas` output into histogram plot.

        Args:
            data (array): Integration results from :meth:`PDFIntegrator.expval`,
                or the array of probabilities returned by
                :meth:`PDFHistogramBuilder.histogram`.
            plot (plotter): :mod:`matplotlib` plotting window. If ``None``
                uses the default window. Default is ``None``.
            show (boolean): Displayes plot if ``True``; otherwise returns
                the plot. Default is ``False``.
            density (booleam): Display probability density if ``True``;
                otherwise display total probability in each bin. Default is
                ``False``.
            kargs (dictionary): Additional plotting arguments for the
                bar graph.
        """
        if plot is None:
            import matplotlib.pyplot as plot
        if len(data) == len(self.midpoints) + 1:
            data = data[1:] / data[0]
        elif len(data) != len(self.midpoints):
            raise ValueError(
                'wrong data length: %s != %s'
                    % (len(data), len(self.midpoints))
                )
        if density:
            data = data / self.widths
        if errorbar:
            plot.errorbar(self.midpoints, mean(data), sdev(data), **errorbar)
        if bar:
            plot.bar(self.bins[:-1], mean(data), width=self.widths, **bar)
        if gaussian and self.g is not None:
            x = self.midpoints
            y = self.gaussian_pdf(x, self.g)
            if not density:
                y = y * self.widths
            ys = cspline.CSpline(x, y)
            x = numpy.linspace(x[0], x[-1], 100)
            plot.plot(x, ys(x), **gaussian)
        if show:
            plot.show()
        else:
            return plot

class PDFStatistics(object):
    """ Compute statistical information about a distribution.

    Given moments ``m[i]`` of a random variable, computes
    mean, standard deviation, skewness, and excess kurtosis.

    Args:
        m (array of floats): ``m[i]`` is the (i+1)-th moment.

        norm (float or |GVar|): The expectation value of 1. Moments
            are divided by ``norm`` before use. (Default is 1.)

    Attributes:
        mean: mean value
        sdev: standard deviation
        skew: skewness coefficient
        ex_kurt: excess kurtosis
    """
    def __init__(self, m, norm=1.):
        x = numpy.array(m) / norm
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
    def __str__(self):
        ans = 'mean = %s' % self.mean
        for attr in ['sdev', 'skew', 'ex_kurt']:
            if hasattr(self, attr):
                ans += '   %s = %s' % (attr, getattr(self, attr))
        return ans

    @staticmethod
    def moments(f, exponents=numpy.arange(1,5)):
        """ Compute 1st-4th moments of f, returned in an array. """
        return f ** exponents




# legacy code support
fmt_partialsdev = fmt_errorbudget
#
