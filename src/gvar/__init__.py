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

    - ``cov(g1, g2)`` --- covariance of :class:`gvar.GVar` ``g1`` with ``g2``.

    - ``evalcov_blocks(g)`` --- compute diagonal blocks of covariance matrix.

    - ``evalcorr(g)`` --- compute correlation matrix.

    - ``corr(g1, g2)`` --- correlation between :class:`gvar.GVar` ``g1`` and ``g2``.

    - ``tabulate(g)`` --- create a table of GVar values in dict/array g.

    - ``fmt_values(g)`` --- create table of values.

    - ``fmt_errorbudget(g)`` --- create error-budget table.

    - ``fmt_chi2(f)`` --- format chi**2 information in f.

    - ``PDF(g)`` --- (class) probability density function.

    - ``PDFStatistics`` --- (class) statistical analysis of moments of a random variable.

    - ``PDFHistogram`` --- (class) tool for building PDF histograms.

    - ``BufferDict`` --- (class) ordered dictionary with data buffer.

    - ``disassemble(g)`` --- disassemble |GVar|\s in ``g``.

    - ``reassemble(data, cov)`` --- reassemble into |GVar|\s.

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
# Copyright (c) 2012-17 G. Peter Lepage.
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
import math
import sys

import numpy

from ._gvarcore import *
gvar = GVarFactory()            # order matters for this statement

from ._svec_smat import *
from ._bufferdict import BufferDict, asbufferdict
from ._bufferdict import ExtendedDict, trim_redundant_keys
from ._bufferdict import del_parameter_distribution, add_parameter_distribution
from ._bufferdict import add_parameter_parentheses, nonredundant_keys
from ._utilities import *
from ._version import version as __version__

from . import dataset
from . import ode
from . import cspline
from . import linalg
from . import powerseries
from . import root

# try:
#     # use lsqfit's gammaQ if available; otherwise use one in ._utilities
#     from lsqfit._utilities import gammaQ
# except:
#     pass

_GVAR_LIST = []

def ranseed(seed=None):
    """ Seed random number generators with tuple ``seed``.

    Argument ``seed`` is an integer or
    a :class:`tuple` of integers that is used to seed
    the random number generators used by :mod:`numpy` and
    :mod:`random` (and therefore by :mod:`gvar`). Reusing
    the same ``seed`` results in the same set of random numbers.

    ``ranseed`` generates its own seed when called without an argument
    or with ``seed=None``. This seed is stored in ``ranseed.seed`` and
    also returned by the function. The seed can be used to regenerate
    the same set of random numbers at a later time.

    Args:
        seed (int, tuple, or None): Seed for generator. Generates a
            random tuple if ``None``.
    Returns:
        The seed used to reseed the generator.
    """
    if seed is None:
        seed = numpy.random.randint(1, int(2e9), size=3)
    try:
        seed = tuple(seed)
    except TypeError:
        pass
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
            self.chi2 = chi2

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
                inv_cov[i[:, None], i] += numpy.outer(w, w)

    This sets ``inv_cov`` equal to the inverse of the covariance matrix of
    the ``gmod``\s. Similarly, we can  compute the expectation value,
    ``u.dot(inv_cov.dot(v))``, between two vectors (:mod:`numpy` arrays)
    using::

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
    idx_bcov = evalcov_blocks(g.flat)
    svd.logdet = 0.0
    svd.correction = numpy.zeros(len(g.flat), object)
    svd.correction[:] = gvar(0, 0)
    svd.eigen_range = 1.
    svd.nmod = 0
    if wgts is not False:
        i_wgts = [([], [])] # 1st entry for all 1x1 blocks
    lost_modes = 0
    svd.nblocks = {}
    for idx, block_cov in idx_bcov:
        svd.nblocks[len(idx)] = svd.nblocks.get(len(idx), 0) + 1
        if len(idx) == 1:
            i = idx[0]
            svd.logdet += numpy.log(block_cov[0, 0])
            if wgts is not False:
                i_wgts[0][0].append(i)
                i_wgts[0][1].append(block_cov[0, 0] ** (wgts * 0.5))
        else:
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

def erf(x):
    """ Error function.

    Works for floats, |GVar|\s, and :mod:`numpy` arrays.
    """
    try:
        return math.erf(x)
    except TypeError:
        pass
    if isinstance(x, GVar):
        f = math.erf(x.mean)
        dfdx = 2. * math.exp(- x.mean ** 2) / math.sqrt(math.pi)
        return gvar_function(x, f, dfdx)
    else:
        x = numpy.asarray(x)
        ans = numpy.empty(x.shape, x.dtype)
        for i in range(x.size):
            try:
                ans.flat[i] = erf(x.flat[i])
            except TypeError:
                xi = x.flat[i]
                f = math.erf(xi.mean)
                dfdx = 2. * math.exp(- xi.mean ** 2) / math.sqrt(math.pi)
                ans.flat[i] = gvar_function(xi, f, dfdx)
        return ans

# default extensions
add_parameter_distribution('log', numpy.exp)
add_parameter_distribution('sqrt', numpy.square)
add_parameter_distribution('erfinv', erf)

class PDF(object):
    """ Probability density function (PDF) for ``g``.

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
            covariance matrix unchanged. Default is ``1e-15``.
    """
    def __init__(self, g, svdcut=1e-15):
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
        self.meanflat = mean(gflat)
        self.size = g.size
        self.shape = g.shape
        self.g = g
        self.log_gnorm = numpy.sum(0.5 * numpy.log(2 * numpy.pi * s.val))

    def x2dpflat(self, x):
        """ Map vector ``x`` in x-space into the displacement from ``g.mean``.

         x-space is a vector space of dimension ``p.size``. Its axes are
        in the directions specified by the eigenvectors of ``p``'s covariance
        matrix, and distance along an axis is in units of the standard
        deviation in that direction.
        """
        return x.dot(self.vec_sig)

    def p2x(self, p):
        """ Map parameters ``p`` to vector in x-space.

        x-space is a vector space of dimension ``p.size``. Its axes are
        in the directions specified by the eigenvectors of ``p``'s covariance
        matrix, and distance along an axis is in units of the standard
        deviation in that direction.
        """
        if hasattr(p, 'keys'):
            dp = BufferDict(p, keys=self.g.keys())._buf[:self.meanflat.size] - self.meanflat
        else:
            dp = numpy.asarray(p).reshape(-1) - self.meanflat
        return self.vec_isig.dot(dp)

    def logpdf(self, p):
        """ Logarithm of the probability density function evaluated at ``p``. """
        x2 = self.p2x(p) ** 2
        return -0.5 * numpy.sum(x2) - self.log_gnorm

    def __call__(self, p):
        """ Probability density function evaluated at ``p``."""
        return numpy.exp(self.logpdf(p))

class PDFHistogram(object):
    r""" Utility class for creating PDF histograms.

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
        """ Compute histogram of data.

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
        """ Analyze count data from :meth:`PDFHistogram.count`.

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
                    % (len(data), len(self.midpoints))
                )
        else:
            data = count
            norm = 1.
        mid = self.midpoints
        stats = PDFStatistics(histogram=(self.bins, count))
        return PDFHistogram.Histogram(self.bins, data, stats, norm)

    @staticmethod
    def gaussian_pdf(x, g):
        """ Gaussian probability density function at ``x`` for |GVar| ``g``. """
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
        """ Convert histogram counts in array ``count`` into a plot.

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
                yspline = cspline.CSpline(x, y)
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
                yspline = cspline.CSpline(x, y)
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


class PDFStatistics(object):
    """ Compute statistical information about a distribution.

    Given moments ``m[i]`` of a random variable, computes
    mean, standard deviation, skewness, and excess kurtosis.

    Args:
        moments (array of floats): ``moments[i]`` is the (i+1)-th moment.
            Optional argument unless ``histgram=None``.

        histogram (tuple): Tuple ``(bins,prob)`` where ``prob[i]`` is
            the probability in the bin between ``bins[i-1]`` and ``bins[i]``.
            ``prob[0]`` is the probability below ``bins[0]`` and ``prob[-1]``
            is the probability above ``bins[-1]``. Array ``bins`` is ordered.
            The format for ``prob`` is what is returned by accumulating calls
            to :meth:`gvar.PDFHistogram.count`. Optional argument unless
            ``moments=None``.

    The attributes are as follows:

    Attributes:
        mean: mean value
        sdev: standard deviation
        skew: skewness coefficient
        ex_kurt: excess kurtosis
        median: median (if ``histogram`` provided)
        plus: interval ``(median, median+plus)`` contains 34.1% of probability
        minus: interval ``(median-minus, median)`` contains 34.1% of probability
        gvar: ``gvar.gvar(mean, sdev)``
    """
    def __init__(self, moments=None, histogram=None, prefix='   '):
        self.prefix = prefix
        if histogram is not None:
            bins, prob = histogram
            prob = prob / sum(prob)
            cumprob = numpy.cumsum(prob)[:-1]
            probspline = cspline.CSpline(bins, cumprob)
            x0 = []
            for p0 in [0.317310507863 / 2., 0.5, 1 - 0.317310507863 / 2.]:
                if cumprob[0] < p0 and cumprob[-1] > p0:
                    def f(x):
                        return probspline(x) - p0
                    x0.append(root.refine(f, (bins[0], bins[-1])))
                else:
                    x0.append(None)
            self.minus, self.median, self.plus = x0
            if self.median is None:
                self.minus = None
                self.plus = None
            else:
                if self.minus is not None:
                    self.minus = self.median - self.minus
                if self.plus is not None:
                    self.plus = self.plus - self.median
            if moments is None:
                mid = (bins[1:] + bins[:-1]) / 2.
                moments = numpy.sum(
                    [mid * prob[1:-1], mid**2 * prob[1:-1],
                    mid**3 * prob[1:-1], mid**4 * prob[1:-1],
                    ],
                    axis=1
                    )
                if self.minus is not None and self.plus is not None:
                    self.gvar = (
                        self.median + (self.plus - self.minus) / 2. +
                        gvar(0., mean(self.plus + self.minus) / 2.)
                        )
                elif self.minus is not None:
                    self.gvar = self.median + gvar(0, self.minus.mean)
                elif self.plus is not None:
                    self.gvar = self.median + gvar(0, self.plus.mean)
                else:
                    self.gvar = None
        if moments is not None:
            x = numpy.array(moments)
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
                self.gvar = self.mean + gvar(0 * mean(self.mean), mean(self.sdev))
        else:
            raise ValueError('need moments and/or histogram')

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
        if hasattr(self, 'median'):
            ans += '\n' + self.prefix
            ans += 'median = {}'.format(self.median)
            ans += '   plus = {}'.format(self.plus)
            ans += '   minus = {}'.format(self.minus)
        return ans

    @staticmethod
    def moments(f, exponents=numpy.arange(1,5)):
        """ Compute 1st-4th moments of f, returned in an array. """
        return f ** exponents


def make_fake_data(g, fac=1.0):
    """ Make fake data based on ``g``.

    This function replaces the |GVar|\s in ``g`` by  new |GVar|\s with similar
    means and a similar covariance matrix, but multiplied by ``fac**2`` (so
    standard deviations are ``fac`` times smaller). The changes are random.
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
        g_flat = numpy.asarray(g).flat
        zero = numpy.zeros(len(g_flat), float)
        dg = 0.5 * gvar(zero, evalcov(g_flat))
        g_flat = mean(g_flat) +  next(raniter(dg))
        dg *=  fac
        noise = gvar(zero, sdev(dg))
        g_flat = g_flat + dg + noise + next(raniter(noise))
        return g_flat[0] if g_shape == () else g_flat.reshape(g_shape)

# legacy code support
fmt_partialsdev = fmt_errorbudget
#
