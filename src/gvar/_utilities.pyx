# cython: language_level=3str, binding=True
# Created by Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-21 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version  (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import gvar as _gvar
from gvar._gvarcore import GVar
from gvar._gvarcore cimport GVar

from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.linalg import solve_triangular as _solve_triangular
from scipy.linalg import cholesky as _cholesky
from scipy.linalg import eigh as _scipy_eigh
from scipy.special import erf as _scipy_erf
import numpy
cimport numpy
import warnings
import pickle
import json
import collections
import copy
from math import lgamma
import sys
cimport cython

try:
    # needed for oldload (optionally)
    import yaml
    from yaml import FullLoader as yaml_Loader, Dumper as yaml_Dumper
except ImportError:
    yaml = None

if sys.version_info > (3, 0):
    # python 3
    from io import BytesIO 
    from io import StringIO
else:
    # python 2
    from StringIO import StringIO 
    class BytesIO(StringIO):
        pass

cdef double EPSILON = sys.float_info.epsilon * 10.         # roundoff error threshold

from libc.math cimport  log, exp  # don't put lgamma here -- old C compilers don't have it

from ._svec_smat import svec, smat, smask
from ._svec_smat cimport svec, smat, smask

from ._bufferdict import BufferDict

from numpy cimport npy_intp as INTP_TYPE
cimport cython

cdef extern from "math.h":
    double c_pow "pow" (double x,double y)
    double c_sin "sin" (double x)
    double c_cos "cos" (double x)
    double c_tan "tan" (double x)
    double c_sinh "sinh" (double x)
    double c_cosh "cosh" (double x)
    double c_tanh "tanh" (double x)
    double c_log "log" (double x)
    double c_exp "exp" (double x)
    double c_sqrt "sqrt" (double x)
    double c_asin "asin" (double x)
    double c_acos "acos" (double x)
    double c_atan "atan" (double x)

# utility functions
def rebuild(g, corr=0.0, gvar=_gvar.gvar):
    """  Rebuild ``g`` stripping correlations with variables not in ``g``.

    ``g`` is either an array of |GVar|\s or a dictionary containing
    |GVar|\s and/or arrays of |GVar|\s. ``rebuild(g)`` creates a new
    collection |GVar|\s with the same layout, means and covariance matrix
    as those in ``g``, but discarding all correlations with variables not
    in ``g``.

    If ``corr`` is nonzero, ``rebuild`` will introduce correlations
    wherever there aren't any using ::

        cov[i,j] -> corr * sqrt(cov[i,i]*cov[j,j])

    wherever ``cov[i,j]==0.0`` initially. Positive values for ``corr``
    introduce positive correlations, negative values anti-correlations.

    Parameter ``gvar`` specifies a function for creating new |GVar|\s that
    replaces :func:`gvar.gvar` (the default).

    :param g: |GVar|\s to be rebuilt.
    :type g: array or dictionary
    :param gvar: Replacement for :func:`gvar.gvar` to use in rebuilding.
        Default is :func:`gvar.gvar`.
    :type gvar: :class:`gvar.GVarFactory` or ``None``
    :param corr: Size of correlations to introduce where none exist
        initially.
    :type corr: number
    :returns: Array or dictionary (gvar.BufferDict) of |GVar|\s  (same
        layout as ``g``) where all correlations with variables other than
        those in ``g`` are erased.
    """
    cdef numpy.ndarray[numpy.float_t,ndim=2] gcov
    cdef INTP_TYPE i,j,ng
    cdef float cr
    if hasattr(g,'keys'):
        ## g is a dict
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        buf = rebuild(g.flat,corr=corr,gvar=gvar)
        return BufferDict(g,buf=buf)

    else:
        ## g is an array
        g = numpy.asarray(g)
        if corr!=0.0:
            ng = g.size
            gcov = evalcov(g).reshape(ng,ng)
            cr = corr
            for i in range(ng):
                for j in range(i+1,ng):
                    if gcov[i,j]==0:
                        gcov[i,j] = cr*c_sqrt(gcov[i,i]*gcov[j,j])
                        gcov[j,i] = gcov[i,j]
            return gvar(mean(g),gcov.reshape(2*g.shape))
        else:
            return gvar(mean(g),evalcov(g))


def mean(g):
    """ Extract means from :class:`gvar.GVar`\s in ``g``.

    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Result has the same layout as ``g``.

    Elements of ``g`` that are not |GVar|\s are left unchanged.
    """
    cdef INTP_TYPE i
    cdef GVar gi
    cdef numpy.ndarray[numpy.float_t,ndim=1] buf
    if isinstance(g,GVar):
        return g.mean
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = numpy.zeros(g.size,numpy.float_)
    try:
        for i, gi in enumerate(g.flat):
            if gi == None:
                raise ValueError('g contains None')
            buf[i] = gi.v
    except TypeError:
        for i, ogi in enumerate(g.flat):
            buf[i] = ogi.mean if isinstance(ogi, GVar) else ogi
    return BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)

def fmt(g, ndecimal=None, sep='', d=None):
    """ Format :class:`gvar.GVar`\s in ``g``.

    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Each |GVar| ``gi`` in ``g`` is replaced
    by the string generated by ``gi.fmt(ndecimal,sep)``. Result has same
    structure as ``g``.
    """
    cdef INTP_TYPE i
    cdef GVar gi
    if d is not None:
        ndecimal = d        # legacy name
    if isinstance(g,GVar):
        return g.fmt(ndecimal=ndecimal,sep=sep)
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = []
    for i,gi in enumerate(g.flat):
        if gi == None:
            raise ValueError('g contains None')
        buf.append(gi.fmt(ndecimal=ndecimal,sep=sep))
    return BufferDict(g,buf=buf) if g.shape is None else numpy.reshape(buf,g.shape)

def sdev(g):
    """ Extract standard deviations from :class:`gvar.GVar`\s in ``g``.

    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Result has the same layout as ``g``.

    The deviation is set to 0.0 for elements of ``g`` that are not |GVar|\s.
    """
    cdef numpy.ndarray[numpy.float_t, ndim=1] buf
    if isinstance(g, GVar):
        return g.sdev
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        return BufferDict(g, buf=var(g.buf) ** 0.5)
    else:
        g = numpy.asarray(g)
        if g.size == 0:
            return numpy.array([])
        buf = var(g.flat) ** 0.5
        return buf.reshape(g.shape)

def deriv(g, x):
    """ Compute partial derivatives of |GVar|\s in ``g`` w.r.t. primary |GVar|\s in ``x``.

    Primary |GVar|\s are those created using :func:`gvar.gvar` (or any function of 
    such a variable); see :func:`gvar.is_primary`.

    Args:
        g: A |GVar| or array of |GVar|\s, or a dictionary whose values are 
            |GVar|\s or arrays of |GVar|\s.
        x: A |GVar| or array of primary |GVar|\s. Arrays are typically one 
            dimensional but can have any shape.
    
    Returns:
        ``g`` but with each |GVar| in it replaced by an array of
        derivatives with respect to the primary |GVar|\s in ``x``.
        The derivatives have the same shape as ``x``.
    """
    cdef INTP_TYPE i    
    cdef GVar gi
    cdef numpy.ndarray ans
    x = numpy.asarray(x)
    if hasattr(g, 'keys'):
        ansdict = BufferDict()
        for k in g:
            ansdict[k] = deriv(g[k], x)
        return ansdict
    else:
        g = numpy.asarray(g)
        if g.shape == ():
            return g.flat[0].deriv(x)
        ans = numpy.zeros((g.size,  ) + x.shape, dtype=float)
        for i in range(g.size):
            if g.flat[i] == None:
                raise ValueError('g contains None')
            gi = g.flat[i]
            ans[i] = gi.deriv(x)
        return ans.reshape(g.shape + x.shape)

def is_primary(g):
    """ Determine whether or not the |GVar|\s in ``g`` are primary |GVar|\s.
    
    A *primary* |GVar| is one created using :func:`gvar.gvar` (or a 
    function of such a variable). A *derived* |GVar| is one that 
    is constructed from arithmetic expressions and functions that 
    combine multiple primary |GVar|\s. The standard deviations for 
    all |GVar|\s originate with the primary |GVar|\s. 
    In particular, :: 

        z = z.mean + sum_p (p - p.mean) * dz/dp

    is true for any |GVar| ``z``, where the sum is over all primary 
    |GVar|\s ``p``.

    Here ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. The result has the same layout as ``g``. 
    Each |GVar| is replaced by ``True`` or ``False`` depending upon whether
    it is primary or not.
    
    When the same |GVar| appears more than once in ``g``, only the first 
    appearance is marked as a primary |GVar|. This avoids double counting 
    the same primary |GVar| --- each primary |GVar| in the list is unique. 
    """
    cdef INTP_TYPE i, j
    cdef GVar gi
    cdef numpy.ndarray buf
    if isinstance(g, GVar):
        return g.is_primary()
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = numpy.zeros(g.size, dtype=bool)
    done = set()
    try:
        for i, gi in enumerate(g.flat):
            if gi == None:
                raise ValueError('g contains None')
            if gi.d.size == 1 and gi.d.v[0].i not in done:
                buf[i] = True 
                done.add(gi.d.v[0].i)
    except TypeError:
        for i, ogi in enumerate(g.flat):
            if isinstance(ogi, GVar) and ogi.d.size == 1 and gi.d.v[0].i not in done:
                buf[i] = True 
                done.add(ogi.d.v[0].i)
    return BufferDict(g, buf=buf) if g.shape is None else buf.reshape(g.shape)

def dependencies(g, all=False):
    """ Collect primary |GVar|\s contributing to the covariances of |GVar|\s in ``g``.

    Args:
        g: |GVar| or a dictionary, array, etc. containing |GVar|\s.
        all (bool): If ``True`` the result includes all primary |GVar|\s including 
            those that are in ``g``; otherwise it only includes those not in ``g``.
            Default is ``False``.

    Returns:
        An array containing the primary |GVar|\s contributing to covariances of 
        |GVar|\s in ``g``. 
    """
    try:
        return _dependencies(g, all=all)
    except:
        gvlist = []
        collect_gvars(g, gvlist)
        return _dependencies(gvlist, all=all) if gvlist else []

def _dependencies(g, all=False):
    " Same as dependencies when ``g`` is all |GVar|\s. "
    cdef INTP_TYPE i
    cdef GVar gi
    cdef numpy.ndarray[INTP_TYPE, ndim=1] idx 
    cdef numpy.ndarray[numpy.float_t, ndim=1] val
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    dep = set()
    pri = set()
    for gi in g.flat:
        if gi == None:
            raise ValueError('g contains None')
        if gi.d.size == 1:
            pri.add(gi.d.v[0].i)
            if not all:
                continue
        dep.update(gi.d.indices())
    if not all:
        dep = dep - pri
    ans = []
    idx = numpy.zeros(1, numpy.intp)
    val = numpy.ones(1, numpy.float_)
    sv = _gvar.svec(1)
    for i in dep:
        idx[0] = i
        sv = _gvar.svec(1)
        sv._assign(val, idx)
        ans.append(_gvar.gvar(0.0, sv, _gvar.gvar.cov))
    return numpy.array(ans)

def missing_dependencies(g):
    """ ``True`` if ``len(gvar.dependencies(g))!=0``; ``False`` otherwise.

    Args:
        g: |GVar| or array of |GVar|\s, or a dictionary whose values are |GVar|\s or 
            arrays of |GVar|\s.
    """
    cdef GVar gi
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    dep = set()
    pri = set()
    for gi in g.flat:
        if gi == None:
            raise ValueError('g contains None')
        if gi.d.size == 1:
            pri.add(gi.d.v[0].i)
            continue
        dep.update(gi.d.indices())
    return len(dep - pri) > 0

def uncorrelated(g1,g2):
    """ Return ``True`` if |GVar|\s in ``g1`` uncorrelated with those in ``g2``.

    ``g1`` and ``g2`` can be |GVar|\s, arrays of |GVar|\s, or dictionaries
    containing |GVar|\s or arrays of |GVar|\s. Returns ``True`` if either
    of ``g1`` or ``g2`` is ``None``.
    """
    cdef GVar g
    cdef smat cov
    cdef INTP_TYPE i
    if g1 is None or g2 is None:
        return True
    # collect indices from g1 and g2 separately
    s = [set(),set()]
    for i,gi in enumerate([g1,g2]):
        if not hasattr(gi,'flat'):
            if isinstance(gi,GVar):
                gi = numpy.array([gi])
            elif hasattr(gi,'keys'):
                gi = BufferDict(gi)
            else:
                gi = numpy.asarray(gi)
        for g in gi.flat:
            if g == None:
                continue
            s[i].update(g.d.indices())
    if not s[0].isdisjoint(s[1]):
        # index sets overlap, so g1 and g2 not orthogonal
        return False
    # collect indices connected to g1 by the covariance matrix
    cov = g.cov
    s0 = set()
    for i in s[0]:
        # s0.update(cov.rowlist[i].indices())
        s0.update(cov.row[i].indices())
    # orthogonal if indices in g1 not connected to indices in g2 by cov
    return s0.isdisjoint(s[1])

def evalcorr(g):
    """ Compute correlation matrix for elements of
    array/dictionary ``g``.

    If ``g`` is an array of |GVar|\s, ``evalcorr`` returns the
    correlation matrix as an array with shape ``g.shape+g.shape``.
    If ``g`` is a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s, the result is a doubly-indexed dictionary where
    ``corr[k1,k2]`` is the correlation for ``g[k1]`` and ``g[k2]``.

    The correlation matrix is related to the covariance matrix by::

        corr[i,j] = cov[i,j] / (cov[i,i] * cov[j,j]) ** 0.5
    """
    if hasattr(g, 'keys'):
        g = _gvar.asbufferdict(g)
        covflat = evalcov(g.buf)
        idx = numpy.arange(covflat.shape[0])
        sdevflat = covflat[idx, idx] ** 0.5
        sdevflat[sdevflat == 0.0] = 1.          # don't rescale these rows/cols
        corrflat = covflat / numpy.outer(sdevflat, sdevflat)
        ans = BufferDict()
        for i in g:
            i_sl, i_sh = g.slice_shape(i)
            if i_sh == ():
                i_sl = slice(i_sl, i_sl + 1)
            for j in g:
                j_sl, j_sh = g.slice_shape(j)
                if j_sh == ():
                    j_sl = slice(j_sl, j_sl + 1)
                ans[i, j] = corrflat[i_sl, j_sl]
        return ans
    else:
        g = numpy.asarray(g)
        if g.size == 0:
            return numpy.array([], float).reshape(0, 0)
        g_shape = g.shape
        cov = evalcov(g.flat)
        idx = numpy.arange(cov.shape[0])
        sdev = cov[idx, idx] ** 0.5
        sdev[sdev == 0.0] = 1.                  # don't rescale these rows/cols
        ans = (cov / numpy.outer(sdev, sdev)) # .reshape(2 * g.shape)
        return ans.reshape(2*g_shape) if g_shape != () else ans.reshape(1,1)

def corr(g1, g2):
    """ Correlation between :class:`gvar.GVar`\s ``g1`` and ``g2``. """
    if not isinstance(g1, GVar) and not isinstance(g2, GVar):
        raise ValueError('g1 and g2 must be GVars')
    return evalcorr([g1, g2])[0,1]

def cov(g1, g2):
    """ Covariance of :class:`gvar.GVar` ``g1`` with ``g2``. """
    if not isinstance(g1, GVar) and not isinstance(g2, GVar):
        raise ValueError('g1 and g2 must be GVars')
    return evalcov([g1, g2])[0,1]


def correlate(g, corr, upper=False, lower=False, verify=False):
    """ Add correlations to uncorrelated |GVar|\s in ``g``.

    This method creates correlated |GVar|\s from uncorrelated |GVar|\s ``g``,
    using the correlations specified in ``corr``.

    Note that correlations initially present in ``g``, if any, are ignored.

    Examples:
        A typical application involves the construction of correlated
        |GVar|\s give the means and standard deviations, together with
        a correlation matrix:

            >>> import gvar as gv
            >>> g = gv.gvar(['1(1)', '2(10)'])
            >>> print(gv.evalcorr(g))           # uncorrelated
            [[ 1.  0.]
             [ 0.  1.]]
            >>> g =  gv.correlate(g, [[1. , 0.1], [0.1, 1.]])
            >>> print(gv.evalcorr(g))           # correlated
            [[ 1.   0.1]
             [ 0.1  1. ]]

        This also works when ``g`` and ``corr`` are dictionaries::

            >>> g = gv.gvar(dict(a='1(1)', b='2(10)'))
            >>> print(gv.evalcorr(g))
            {('a', 'a'): array([[ 1.]]),('a', 'b'): array([[ 0.]]),('b', 'a'): array([[ 0.]]),('b', 'b'): array([[ 1.]])}
            >>> corr = {}
            >>> corr['a', 'a'] = 1.0
            >>> corr['a', 'b'] = 0.1
            >>> corr['b', 'a'] = 0.1
            >>> corr['b', 'b'] = 1.0
            >>> g = correlate(g, corr)
            >>> print(gv.evalcorr(g))
            {('a', 'a'): array([[ 1.]]),('a', 'b'): array([[ 0.1]]),('b', 'a'): array([[ 0.1]]),('b', 'b'): array([[ 1.]])}

    Args:
        g: An array of |GVar|\s or a dictionary whose values are
            |GVar|\s or arrays of |GVar|\s.
        corr: Correlations between |GVar|\s: ``corr[i, j]``
            is the correlation between ``g[i]`` and ``g[j]``. Should 
            be a symmetric and positive-definite (unless ``upper`` 
            or ``lower`` is specified).
        upper (bool): If ``True``, replaces lower triangular part 
            of ``corr`` with transpose of the upper triangular part.
            The diagonal is set to one. Default is ``False``.
        lower (bool): If ``True``, replaces upper triangular part 
            of ``corr`` with transpose of the lower triangular part.
            The diagonal is set to one. Default is ``False``.
        verify (bool): If ``True``, verifies that ``corr`` 
            is symmetric and positive definite. Default is 
            ``False`` because verification is costly for large
            matrices.
    """
    cdef INTP_TYPE ni, nj
    cdef numpy.ndarray[numpy.float_t, ndim=2] corrflat
    cdef numpy.ndarray[numpy.float_t, ndim=1] sdevflat, meanflat
    if hasattr(g, 'keys'):
        g = _gvar.asbufferdict(g)
        corrflat = numpy.empty((len(g.buf), len(g.buf)), numpy.float_)
        for i in g:
            i_sl, i_sh = g.slice_shape(i)
            if i_sh == ():
                i_sl = slice(i_sl, i_sl + 1)
                ni = 1
            else:
                ni = numpy.prod(i_sh)
            for j in g:
                j_sl, j_sh = g.slice_shape(j)
                if j_sh == ():
                    j_sl = slice(j_sl, j_sl + 1)
                    nj = 1
                else:
                    nj = numpy.prod(j_sh)
                corrflat[i_sl, j_sl] = numpy.asarray(corr[i, j]).reshape(ni, nj) 
        return BufferDict(g, buf=correlate(g.buf, corrflat, upper=upper, lower=lower, verify=verify))
    else:
        g = numpy.asarray(g)
        meanflat = _gvar.mean(g.flat)
        sdevflat = _gvar.sdev(g.flat)
        corrflat = numpy.asarray(corr).reshape(meanflat.shape[0], -1)
        if upper:
            corrflat = numpy.triu(corrflat)
            corrflat += corrflat.T 
            numpy.fill_diagonal(corrflat, 1)
        elif lower:
            corrflat = numpy.tril(corrflat)
            corrflat += corrflat.T 
            numpy.fill_diagonal(corrflat, 1)
        return _gvar.gvar(
            meanflat, 
            sdevflat[:, None] * corrflat * sdevflat[None, :], 
            verify=verify
            ).reshape(g.shape)


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
@cython.initializedcheck(False) # memory views initialized?
def evalcov_blocks_dense(g, compress=False):
    """ Evaluate covariance matrix for elements of ``g``.

    Same as :func:`gvar.evalcov_blocks` but optimized for 
    large, dense covariance matrices.
    """
    cdef INTP_TYPE nvar, i, j, nb, ib
    cdef object[:] varlist
    cdef double sdev
    cdef smat cov 
    if hasattr(g, 'flat'):
        varlist = g.flat[:]
    elif hasattr(g, 'keys'):
        varlist = BufferDict(g).flat[:]
    else:
        varlist = numpy.array(g).flat[:]
    nvar = len(varlist)
    if nvar <= 0:
        return (
            [(numpy.array([]), numpy.array([]))] if compress else 
            [(numpy.array([]), numpy.reshape([], (0,0)))] 
            )
    allcov = evalcov(varlist)
    nb, key = _connected_components(allcov != 0, directed=False)
    allvar = numpy.arange(nvar, dtype=numpy.intp)
    blocks = [([], [])]
    for ib in range(nb):
        idx = allvar[key == ib]
        if len(idx) == 1:
            i = idx[0]
            blocks[0][0].append(i)
            blocks[0][1].append(varlist[i].sdev)
        else:
            blocks.append((numpy.array(idx, dtype=numpy.intp), allcov[idx[:, None], idx]))
    blocks[0] = (numpy.array(blocks[0][0], dtype=numpy.intp), numpy.array(blocks[0][1], dtype=float))
    if not compress: 
        if len(blocks[0][0]) > 0:
            for i,sdev in zip(*blocks[0]):
               blocks.append((numpy.array([i], dtype=numpy.intp), numpy.array([[sdev**2]])))
        blocks = blocks[1:]
    return blocks

@cython.boundscheck(False) # turn off bounds-checking 
@cython.wraparound(False)  # turn off negative index wrapping
@cython.initializedcheck(False) # memory views initialized?
def evalcov_blocks(g, compress=False):
    """ Evaluate covariance matrix for elements of ``g``.

    Evaluates the covariance matrices for |GVar|\s stored in
    array or dictionary of arrays/|GVar|\s ``g``. The covariance matrix is
    decomposed into its block diagonal components, and a list of
    tuples ``(idx,bcov)`` is returned where ``bcov`` is a diagonal
    block of the covariance matrix and ``idx`` an array containing the
    corresponding indices in ``g.flat`` for that block. So to reassemble
    the blocks into a single matrix ``cov``, for example, one would use::

        import numpy as np
        import gvar as gv
        cov = np.zeros((len(g.flat), len(g.flat)), float)
        for idx, bcov in gv.evalcov_blocks(g):
            cov[idx[:, None], idx] = bcov

    :func:`gvar.evalcov_blocks` is particularly useful when the covariance
    matrix is sparse; only nonzero elements are retained.

    Setting argument ``compress=True`` (default is ``False``) changes 
    the meaning of the first element in the list: ``idx`` for the first 
    element contains the indices of all |GVar|\s in ``g.flat`` that 
    are uncorrelated from other |GVar|\s in ``g.flat``, and ``bcov`` 
    contains the standard deviations for those |GVar|\s. Thus the full
    covariance matrix is reconstructed using the following code::

        import numpy as np
        import gvar as gv
        cov = np.zeros((len(g.flat), len(g.flat)), float)
        blocks = gv.evalcov_blocks(g, compress=True)
        # uncorrelated pieces are diagonal
        idx, sdev = blocks[0]
        cov[idx, idx] = sdev ** 2
        # correlated pieces
        for idx, bcov in blocks[1:]:
            cov[idx[:, None], idx] = bcov
    
    The code with ``compress=True`` should be slightly faster if 
    there are many uncorrelated |GVar|\s.

    It is easy to create an array of |GVar|\s having the covariance
    matrix from ``g.flat``: for example, ::

        import numpy as np 
        import gvar as gv
        new_g = np.empty(len(g.flat), dtype=object)
        for idx, bcov in evalcov_blocks(g):
            new_g[idx] = gv.gvar(new_g[idx], bcov)

    creates an array of |GVar|\s with zero mean and the same covariance
    matrix as ``g.flat``. This works with either value for argument ``compress``.

    Args:
        g: A |GVar|, an array of |GVar|\s, or a dictionary whose values 
            are |GVar|\s and/or arrays of |GVar|\s.
        compress (bool): Setting ``compress=True`` collects all of the 
            uncorrelated |GVar|\s in ``g.flat`` into the first element of 
            the returned list (see above). Default is ``False``.
    """
    cdef INTP_TYPE nvar, iv, i, j, id, nb, ib, sib, snb, nval, nvalmax, n, nzeros
    cdef double sdev
    cdef smat cov 
    cdef GVar gi
    cdef INTP_TYPE[::1] rows, cols
    cdef numpy.int8_t[::1] vals
    cdef object[::1] ivset_iv
    if hasattr(g, 'flat'):
        varlist = g.flat[:]
    elif hasattr(g, 'keys'):
        varlist = BufferDict(g).flat[:]
    else:
        varlist = numpy.array(g).flat[:]
    nvar = len(varlist)
    if nvar <= 0:
        return (
            [(numpy.array([]), numpy.array([]))] if compress else 
            [(numpy.array([]), numpy.reshape([], (0,0)))] 
            )
    elif nvar == 1:
        if compress:
            return [(numpy.array([0]), numpy.array([varlist[0].sdev]))]
        else:
            return [(numpy.array([0]), numpy.array([[varlist[0].var]]))]
    cov = varlist[0].cov
    ivlist_id = {} 
    ivlist_idset = {}
    ivset_iv = numpy.array([set() for i in range(nvar)])
    nzeros = 0
    for iv, gi in enumerate(varlist):
        idset = set()
        for i in range(gi.d.size):
            idset.add(cov.block[gi.d.v[i].i])
        idset = frozenset(idset)
        if idset not in ivlist_idset:
            for id in idset:
                if id not in ivlist_id:
                    # new id, not used by smaller ivs
                    ivlist_id[id] = [iv]
                else:
                    ivlist_id[id].append(iv)
                ivset_iv[iv].update(ivlist_id[id])
            ivlist_idset[idset] = list(ivset_iv[iv])
        else:
            # ivlist_idset[idset] only has ivs smaller than iv
            ivlist_idset[idset].append(iv)
            ivset_iv[iv] = set(ivlist_idset[idset])
        nzeros += iv - len(ivset_iv[iv]) + 1
    nvalmax = nvar * (nvar + 1) // 2
    if nzeros <= nvalmax / 2:
        # probably not sparse
        # nzeros is the minimum number of zeros; could actually be larger
        # so not foolproof
        return evalcov_blocks_dense(varlist, compress=compress)
    
    # build graph showing which pairs of variables share a block (or blocks)
    n = min(_gvar._CONFIG['evalcov_blocks'], nvalmax)
    rows = numpy.empty(n, dtype=numpy.intp)
    cols = numpy.empty(n, dtype=numpy.intp)
    vals = numpy.empty(n, dtype=numpy.int8)
    nval = 0
    for i in range(nvar):
        for j in ivset_iv[i]:
            if nval == len(rows):
                # need more space
                n = min(2 * nval, nvalmax)
                tmp = numpy.empty(n, dtype=numpy.intp)
                tmp[:nval], rows = rows, tmp
                tmp = numpy.empty(n, dtype=numpy.intp)
                tmp[:nval], cols = cols, tmp
                tmp = numpy.empty(n, dtype=numpy.int8)
                tmp[:nval], vals = vals, tmp
                # print(n, nval, nvalmax, len(rows), i, ivset_iv[i])
            rows[nval] = i 
            cols[nval] = j 
            vals[nval] = True
            nval += 1
    assert nval <= nvalmax
    graph = _csr_matrix((vals[:nval], (rows[:nval], cols[:nval])))
    # find and collect the sub-blocks
    nb, key = _connected_components(graph, directed=False)
    allvar = numpy.arange(nvar, dtype=numpy.intp)
    blocks = [([], [])]
    for ib in range(nb):
        idx = allvar[key == ib]
        if len(idx) == 1:
            i = idx[0]
            blocks[0][0].append(i)
            blocks[0][1].append(varlist[i].sdev)
        else:
            # evaluate cov for sub-block
            bcov = evalcov(varlist[idx])
            # check for sub-blocks within the sub-blocks
            allbcov =  numpy.arange(bcov.shape[0])
            snb, skey = _connected_components(bcov != 0, directed=False, connection='strong')
            for sib in range(snb):
                sidx = idx[skey == sib]
                if len(sidx) == 1:
                    i = sidx[0]
                    blocks[0][0].append(i)
                    blocks[0][1].append(varlist[i].sdev)
                else:
                    bidx = allbcov[skey == sib]
                    sbcov = numpy.array(bcov[bidx[:, None], bidx])
                    blocks.append((numpy.array(sidx, dtype=numpy.intp), sbcov))
    blocks[0] = (numpy.array(blocks[0][0], dtype=numpy.intp), numpy.array(blocks[0][1], dtype=float))
    if not compress: 
        if len(blocks[0][0]) > 0:
            for i,sdev in zip(*blocks[0]):
               blocks.append((numpy.array([i], dtype=numpy.intp), numpy.array([[sdev**2]])))
        blocks = blocks[1:]
    return blocks

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
@cython.initializedcheck(False) # memory views initialized?
def mock_evalcov(numpy.ndarray[numpy.float_t, ndim=2] cov, numpy.ndarray[numpy.float_t, ndim=2] d):
    """ Mock evalcov for testing.

    Assumes maximum density. It also has no overhead for collecting either ``cov`` or 
    ``d``, unlike the real ``evalcov``. The focus is on the N**3 part of the algorithm.

    Args:
        cov: Covariance matrix of primary |GVar|\s.
        d: ``d[a]`` is the derivative vector for |GVar| ``g[a]`` (i.e., 
            ``dg[a]/dx[i]`` for |GVar| ``ga[a]`` where ``x[i]`` are primary |GVar|\s.)
    """
    return d.T @ cov @ d

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
@cython.initializedcheck(False) # memory views initialized?
def var(g):
    """ Compute variances for elements of 
    array/dictionary ``g`` or a single |GVar|.

    If ``g`` is an array of |GVar|\s, ``var`` returns the
    variances as an array with shape ``g.shape``.
    If ``g`` is a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s, the result is a dictionary where
    ``cov[k]`` is the variance for ``g[k]``.
    """
    cdef INTP_TYPE a,b,ng,i,j,nc, vec_zeros
    cdef INTP_TYPE cov_zeros, previousid, bsize, ni, id
    cdef numpy.ndarray[numpy.float_t, ndim=1] ans
    cdef numpy.ndarray[numpy.float_t, ndim=2] np_gd
    cdef numpy.int8_t[::1] imask
    cdef numpy.ndarray[numpy.int8_t, ndim=1] np_imask
    cdef numpy.ndarray[numpy.float_t, ndim=2] mcov, mcov_gd
    cdef numpy.ndarray[numpy.float_t, ndim=1] varlist
    cdef numpy.int8_t is_dense, ib, only_gvars
    cdef GVar ga
    cdef svec da
    cdef smat cov
    cdef smask mask
    cdef svec[::1] gdlist
    cdef svec covd

    if isinstance(g, _gvar.GVar):
        return g.var
    if hasattr(g, "keys"):
        # convert g to list and call evalcov; repack as double dict
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        varlist = var(g.flat)
        ansd = BufferDict()
        for k1 in g:
            k1_sl, k1_sh = g.slice_shape(k1)
            if k1_sh == ():
                k1_sl = slice(k1_sl, k1_sl+1)
            ansd[k1] = varlist[k1_sl].reshape(k1_sh)
        return ansd
    g = numpy.asarray(g)
    if g.size == 0:
        return numpy.array([])
    g_shape = g.shape
    g = g.flat
    ng = len(g)
    if ng <= 0:
        return numpy.array(None)
    ans = numpy.zeros(ng, dtype=float)
    gdlist = numpy.empty(ng, object)
    cov = None
    only_gvars = True
    for a in range(ng):
        if isinstance(g[a], GVar):
            ga = g[a]
            gdlist[a] = ga.d 
            if cov is None:
                cov = ga.cov
        else:
            only_gvars = False
            gdlist[a] = None
    nc = cov.nrow

    if ng > _gvar._CONFIG['var'] and only_gvars:
        # only efficient for larger systems
        # create a mask indentifying relevant primary GVars 
        imask = numpy.zeros(nc, numpy.int8)
        vec_zeros = 0
        for i in range(ng):
            da = gdlist[i]
            if da is None:
                continue
            vec_zeros += da.size
            for j in range(da.size):
                imask[da.v[j].i] = True
        # estimate how many zeros in the primary GVars' cov
        cov_zeros = 0
        previousid = -1
        bsize = 0
        ni = 0
        for i in range(nc):
            if imask[i]:
                ni += 1
                id = cov.block[i]
                if id > previousid:
                    bsize = cov.row[i].size
                    cov_zeros += bsize * bsize 
                    previousid = id 
        # convert to zeros
        vec_zeros = ni - vec_zeros // ng
        cov_zeros = ni * ni - cov_zeros 
        # less than 50% zero is defined as dense (ad hoc)
        is_dense = cov_zeros / ni / ni < 0.5 and vec_zeros / ni < 0.5  # sparse vecs ok 
        if is_dense:
            # collect cov matrix for primary GVars and
            # derivative vectors for g[a]; form dot products
            mask = smask(imask)
            np_gd = numpy.zeros((ng, mask.len), dtype=float)
            mcov = cov.masked_mat(mask)
            for a in range(ng):
                da = gdlist[a]
                if da is None:
                    continue
                da.masked_vec(mask, out=np_gd[a])
            mcov_gd = mcov.dot(np_gd.T)
            for a in range(ng):
                if gdlist[a] is None:
                    continue
                ans[a] = np_gd[a].dot(mcov_gd[:, a])
    else:
        is_dense = False
    if not is_dense:
        for a in range(ng):
            da = gdlist[a]
            if da is None:
                continue
            ans[a] = cov.expval(da)
    return ans.reshape(g_shape)


@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
@cython.initializedcheck(False) # memory views initialized?
def evalcov(g):
    """ Compute covariance matrix for elements of 
    array/dictionary ``g``.

    If ``g`` is an array of |GVar|\s, ``evalcov`` returns the
    covariance matrix as an array with shape ``g.shape+g.shape``.
    If ``g`` is a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s, the result is a doubly-indexed dictionary where
    ``cov[k1,k2]`` is the covariance for ``g[k1]`` and ``g[k2]``.
    """
    cdef INTP_TYPE a,b,ng,i,j,nc
    cdef INTP_TYPE cov_zeros, previousid, bsize, ni, id
    cdef numpy.ndarray[numpy.float_t, ndim=2] ans
    cdef numpy.ndarray[numpy.float_t, ndim=2] np_gd
    cdef numpy.int8_t[::1] imask
    cdef numpy.ndarray[numpy.int8_t, ndim=1] np_imask
    cdef numpy.ndarray[numpy.float_t, ndim=2] mcov 
    cdef numpy.int8_t is_dense, ib
    cdef GVar ga
    cdef svec da
    cdef smat cov
    cdef smask mask
    cdef svec[::1] gdlist
    cdef svec[::1] covd
    if hasattr(g, "keys"):
        # convert g to list and call evalcov; repack as double dict
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        gcov = evalcov(g.flat)
        ansd = BufferDict()
        for k1 in g:
            k1_sl, k1_sh = g.slice_shape(k1)
            if k1_sh == ():
                k1_sl = slice(k1_sl, k1_sl+1)
                k1_sh = (1,)
            for k2 in g:
                k2_sl, k2_sh = g.slice_shape(k2)
                if k2_sh == ():
                    k2_sl = slice(k2_sl, k2_sl+1)
                    k2_sh = (1,)
                ansd[k1, k2] = gcov[k1_sl, k2_sl].reshape(k1_sh + k2_sh)
        return ansd
    g = numpy.asarray(g)
    g_shape = g.shape
    g = g.flat
    ng = len(g)
    if ng <= 0:
        return numpy.array([], float).reshape(0,0)
    if hasattr(g[0], 'cov'):
        cov = g[0].cov
    else:
        raise ValueError("g does not contain GVar's")
    ####
    nc = cov.nrow 
    gdlist = numpy.array([ga.d for ga in g])
    # create a mask indentifying relevant primary GVars 
    imask = numpy.zeros(nc, numpy.int8)
    # vec_zeros = 0
    for i in range(ng):
        da = gdlist[i]
        for j in range(da.size):
            imask[da.v[j].i] = True
    if ng > _gvar._CONFIG['evalcov']:
        # only efficient for larger matrices
        # estimate how many zeros in the primary GVars' cov
        cov_zeros = 0
        previousid = -1
        bsize = 0
        ni = 0
        for i in range(nc):
            if imask[i]:
                ni += 1
                id = cov.block[i]
                if id > previousid:
                    bsize = cov.row[i].size
                    cov_zeros += bsize * bsize 
                    previousid = id 
        # finish last block
        cov_zeros += bsize * bsize
        # convert to zeros
        cov_zeros = ni * ni - cov_zeros 
        if ni <= 0:
            # GVars don't depend on other GVars (eg, 0*gvar('1(1)') gives such a thing)
            is_dense = True 
            ans = numpy.zeros((ng, ng), numpy.float_)
        else:
            # less than 50% zero is defined as dense (ad hoc)
            is_dense = cov_zeros / ni / ni < 0.5  # N.B. sparse vecs ok 
            if is_dense:
                # collect cov matrix for primary GVars and
                # derivative vectors for g[a]; form dot products
                mask = smask(imask)
                np_gd = numpy.zeros((ng, mask.len), dtype=float)
                mcov = cov.masked_mat(mask)
                for a in range(ng):
                    da = gdlist[a]
                    da.masked_vec(mask, out=np_gd[a])
                # ans = numpy.matmul(np_gd, numpy.matmul(mcov,np_gd.T))
                ans = np_gd.dot(mcov.dot(np_gd.T))
    else:
        is_dense = False
    if not is_dense:
        ans = numpy.empty((ng, ng),numpy.float_)
        covd = numpy.zeros(ng, object)
        np_imask = numpy.asarray(imask)
        for a in range(ng):
            da = gdlist[a]
            covd[a] = cov.masked_dot(da, np_imask)
            ans[a, a] = da.dot(covd[a])
            for b in range(a):
                ans[a, b] = da.dot(covd[b])
                ans[b, a] = ans[a, b]
    return ans.reshape(2*g_shape) if g_shape != () else ans.reshape(1,1)

def evalcov_old(g):
    """ Compute covariance matrix for elements of
    array/dictionary ``g``. Old version, for testing.

    If ``g`` is an array of |GVar|\s, ``evalcov`` returns the
    covariance matrix as an array with shape ``g.shape+g.shape``.
    If ``g`` is a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s, the result is a doubly-indexed dictionary where
    ``cov[k1,k2]`` is the covariance for ``g[k1]`` and ``g[k2]``.
    """
    cdef INTP_TYPE a,b,ng,i,j,nc
    cdef numpy.ndarray[numpy.float_t, ndim=2] ans
    cdef numpy.ndarray[object, ndim=1] covd
    cdef numpy.ndarray[numpy.int8_t, ndim=1] imask
    cdef GVar ga,gb
    cdef svec da,db
    cdef smat cov
    if hasattr(g,"keys"):
        # convert g to list and call evalcov; repack as double dict
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        gcov = evalcov(g.flat)
        ansd = BufferDict()
        for k1 in g:
            k1_sl, k1_sh = g.slice_shape(k1)
            if k1_sh == ():
                k1_sl = slice(k1_sl, k1_sl+1)
                k1_sh = (1,)
            for k2 in g:
                k2_sl, k2_sh = g.slice_shape(k2)
                if k2_sh == ():
                    k2_sl = slice(k2_sl, k2_sl+1)
                    k2_sh = (1,)
                ansd[k1, k2] = gcov[k1_sl, k2_sl].reshape(k1_sh + k2_sh)
        return ansd
    g = numpy.asarray(g)
    g_shape = g.shape
    g = g.flat
    ng = len(g)
    ans = numpy.zeros((ng,ng),numpy.float_)
    if hasattr(g[0], 'cov'):
        cov = g[0].cov
    else:
        raise ValueError("g does not contain GVar's")
    nc = cov.nrow 
    # mask restricts calculation to relevant primary GVars
    imask = numpy.zeros(nc, numpy.int8)
    for a in range(ng):
        ga = g[a]
        da = ga.d
        for i in range(da.size):
            imask[da.v[i].i] = True
    covd = numpy.zeros(ng, object)
    for a in range(ng):
        ga = g[a]
        covd[a] = cov.masked_dot(ga.d, imask)
        ans[a,a] = ga.d.dot(covd[a])
        for b in range(a):
            ans[a,b] = ga.d.dot(covd[b])
            ans[b,a] = ans[a,b]
    return ans.reshape(2*g_shape) if g_shape != () else ans.reshape(1,1)

###### dump/load
######
class GVarRef:
    """ Placeholder for a |GVar|, used by :func:`gvar.remove_gvars`. 
    
    Typical usage, when ``g`` is a dictionary containing |GVar|\s::

        import gvar as gv 

        gvlist = []
        for k in g:
            if isinstance(g[k], gv.GVar):
                g[k] = gv.GVarRef(g[k], gvlist)

    where at the end ``gvlist`` contains the |GVar|\s removed 
    from ``g``.
    
    The |GVar|\s are restored using::

        for k in g:
            if isinstance(g[k], gv.GVarRef):
                g[k] = g[k](gvlist)

    where the |GVar|\s are drawn from list ``gvlist``.
    """
    def __init__(self, g, gvlist):
        self.loc = len(gvlist)
        gvlist.append(g)
    
    def __call__(self, gvlist):
        return gvlist[self.loc]

def distribute_gvars(g, gvlist):
    """ Distribute |GVar|\s from ``gvlist`` in structure g, replacing :class:`gvar.GVarRef`\s. 
    
    :func:`distribute_gvars` undoes what :func:`remove_gvars` does to ``g``. See discussion
    of :func:`remove_gvars` for more details.

    Args:
        g: Object containing :class:`GVarRef`\s created by :func:`remove_gvars`.
        gvlist: List of |GVar|\s created by :func:`remove_gvars`.

    Returns:
        Object ``g`` with :class:`GVarRef`\s replaced by corresponding |GVar|\s
        from list ``gvlist``.
    """
    if hasattr(g, '_distribute_gvars'):
        return g._distribute_gvars(gvlist)
    elif isinstance(g, GVarRef):
        return g(gvlist)
    elif hasattr(g, 'keys'):
        return type(g)([(k, distribute_gvars(g[k], gvlist)) for k in g])
    elif type(g) in [collections.deque, list]:
        return type(g)([distribute_gvars(x, gvlist) for x in g])
    elif isinstance(g, tuple):
        if type(g) != tuple and hasattr(g, '_fields'):
            # named tuple 
            return type(g)(**dict(zip(g._fields, [distribute_gvars(x, gvlist) for x in g])))
        elif type(g) == tuple:
            return type(g)([distribute_gvars(x, gvlist) for x in g])
        else:
            return g
    elif type(g) == numpy.ndarray:
        return numpy.array([distribute_gvars(x, gvlist) for x in g])
    elif hasattr(g, '__slots__'):
        try:
            for k in g.__slots__:
                setattr(g, k, _gvar.distribute_gvars(getattr(g, k), gvlist))
            return g
        except:
            return g
    elif hasattr(g, '__dict__') and not hasattr(g, '__slots__'):
        try:
            g.__dict__ = _gvar.distribute_gvars(g.__dict__, gvlist)
            return g
        except:
            return g
    else:
        return g

def remove_gvars(g, gvlist):
    """ Remove |GVar|\s from structure g, replacing them with :class:`gvar.GVarRef`\s and collecting them in ``gvlist``. 
    
    :func:`remove_gvars` searches object ``g`` (recursively) 
    for |GVar|\s and replaces them with :class:`GVarRef` objects. 
    The |GVar|\s are collected in list ``gvlist``. ``g`` can 
    be a standard container object (dictionary, list, etc.) or an 
    object ``obj`` whose contents can be accessed through ``obj.__dict__``
    or ``obj.__slots__``. An object that defines method ``obj._remove_gvars``
    is replaced by ``obj._remove_gvars(gvlist)``.

    The |GVar|\s are restored using ``gvar.distribute_gvars``: e.g., ::

        gvlist = []
        new_g = gvar.distribute_gvars(gvar.remove_gvars(g, gvlist), gvlist)

    creates a copy ``new_g`` of ``g`` with the |GVar|\s restored 
    (and preserving correlations between different |GVar|\s). ``gvlist``
    contains copies of the restored |GVar|\s.

    The default treatment of a class instance ``obj`` without an 
    ``obj._remove_gvars(gvlist)`` method is equivalent to adding 
    the following method to the object's class ::

        def _remove_gvars(self, gvlist):
            tmp = copy.copy(self)
            tmp.__dict__ = gvar.remove_gvars(tmp.__dict__, gvlist)
            return tmp 

    assuming the class does not use ``__slots__``. A class that has 
    method ``obj._remove_gvars(gvlist)`` should have a corresponding method 
    ``obj._distribute_gvars(gvlist)``, for use by :func:`gvar.distribute_gvars`. 
    The default behavior when this method is undefined is equivalent to::

        def _distribute_gvars(self, gvlist):
            self.__dict__ = gvar.distribute_gvars(self.__dict__, gvlist)

    There are analogous defaults for classes that uses ``__slots__``.

    These routines are used by ``gvar.dump`` and ``gvar.load`` to facilitate
    pickling of objects containing |GVar|\s.

    Args:
        g: Object containing |GVar|\s.
        gvlist: |GVar|\s removed from ``g`` are appended to list ``gvlist``.

    Returns:
        Copy of object ``g`` with |GVar|\s replaced by :class:`GVarRef`\s. 
        The |GVar|\s are appended to ``gvlist``.
    """
    if hasattr(g, '_remove_gvars'):  # must be first
        return g._remove_gvars(gvlist)
    elif isinstance(g, _gvar.GVar):
        return GVarRef(g, gvlist)
    elif hasattr(g, 'keys'):
        return type(g)([(k, remove_gvars(g[k], gvlist)) for k in g])
    elif type(g) in [collections.deque, list]:
        return type(g)([remove_gvars(x, gvlist) for x in g])
    elif isinstance(g, tuple):
        if type(g) != tuple and hasattr(g,'_fields'):
            # named tuple 
            return type(g)(**dict(zip(g._fields, [remove_gvars(x, gvlist) for x in g])))
        elif type(g) == tuple:
            return type(g)([remove_gvars(x, gvlist) for x in g])
        else:
            return g
    elif type(g) == numpy.ndarray:
        return numpy.array([remove_gvars(x, gvlist) for x in g])
    elif hasattr(g, '__slots__'):
        try:
            tmp = copy.copy(g)
            for k in tmp.__slots__:
                setattr(tmp, k, _gvar.remove_gvars(getattr(tmp, k), gvlist))
            return tmp
        except:
            return g
    elif hasattr(g, '__dict__') and not hasattr(g, '__slots__'):
        try:
            tmp = copy.copy(g)
            tmp.__dict__ = _gvar.remove_gvars(tmp.__dict__, gvlist)
            return tmp 
        except:
            return g
    else:
        return g

def collect_gvars(g, gvlist):
    " Collect |GVar|\s into ``gvlist`` from container object g. "
    if isinstance(g, _gvar.GVar):
        gvlist.append(g)
    elif hasattr(g, 'keys'):
        for k in g:
            collect_gvars(g[k], gvlist)
    elif type(g) in [collections.deque, list, tuple, numpy.ndarray]:
        for x in g:
            collect_gvars(x, gvlist)
    elif hasattr(g, '_remove_gvars'):
        g._remove_gvars(gvlist)
    elif hasattr(g, '__slots__'):
        for k in g.__slots__:
            try:
                collect_gvars(getattr(g, k), gvlist)
            except:
                pass
    elif hasattr(g, '__dict__') and not hasattr(g, '__slots__'):
        try:
            collect_gvars(g.__dict__, gvlist)
        except:
            pass

def filter(g, f, *args, **kargs):
    """ Filter |GVar|\s in ``g`` through function ``f``. 
    
    Sample usage::

        import gvar as gv
        g_mod = gv.filter(g, gv.mean)

    replaces every |GVar| in ``g`` by its mean. This is useful 
    when ``gv.mean(g)`` doesn't work --- for example, because 
    ``g`` contains data types other than |GVar|\s,
    or consists of nested dictionaries.

    Args:
        g: Object consisting of (possibly nested) dictionaries,
            deques, lists, ``numpy.array``\s, tuples, and/or 
            other data types that contain |GVar|\s and other 
            types of data.
        f: Function that takes an array of |GVar|\s as an argument
            and returns an array of results having the same shape.
            Given an array ``gvar_array`` containing all 
            the |GVar|\s from ``g``, the function call is 
            ``f(gvar_array, *args, **kargs)``. Results 
            from the returned array replace the |GVar|\s
            in the original object.
        args: Additional arguments for ``f``.
        kargs: Additional keyword arguments for ``f``.

    Returns:
        An object like ``g``  but with all the |GVar|\s 
        replaced by objects generated by function ``f``.
    """
    gvlist = []
    new_g = remove_gvars(g, gvlist)
    gvlist = f(gvlist, *args, **kargs)
    return distribute_gvars(new_g, gvlist)

def dumps(g, add_dependencies=False, **kargs):
    """ Return a serialized representation of ``g``.

    This function is shorthand for::
    
        gvar.dump(g).getvalue()
    
    Typical usage is::

        # create bytes object containing GVars in g
        import gvar as gv 
        gbytes = gv.dumps(g)

        # convert bytes object back into g
        new_g = gv.loads(gbytes)
    
    Args:
        g: Object to be serialized. Consists of (possibly nested) 
            dictionaries, deques, lists, ``numpy.array``\s, 
            and/or tuples that contain |GVar|\s and other types of data.
        add_dependencies (bool): If ``True``, automatically includes 
            all primary |GVar|\s that contribute to the covariances 
            of the |GVar|\s in ``g`` but are not already in ``g``.
            Default is ``False``.
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying serializer (:mod:`pickle`).
    
    Returns:
        A bytes object containing a serialized representation 
        of object ``g``.
    """
    return dump(g, add_dependencies=add_dependencies).getvalue()

def loads(inputbytes, **kargs):
    """ Read and return object serialized in ``inputbytes`` by :func:`gvar.dumps`.

    This function recovers data serialized with :func:`gvar.dumps`. It is 
    shorthand for::

        gvar.load(BytesIO(inputbytes))

    Typical usage is::

        # create bytes object containing data in g
        import gvar as gv 
        gbytes = gv.dumps(g)

        # recreate g from bytes object gbytes
        new_g = gv.gloads(gbytes)

    Args:
        inputbytes (bytes): Bytes object created by :func:`gvar.dumps`.
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying serializer (:mod:`pickle`).

    Returns:
        The reconstructed data.
    """
    return load(BytesIO(inputbytes), **kargs)

def dump(g, outputfile=None, add_dependencies=False, **kargs):
    """ Write a representation of ``g``  to file ``outputfile``.

    Object ``g`` consists of (possibly nested) dictionaries, deques,
    lists, ``numpy.array``\s, and/or tuples that contain |GVar|\s and 
    other types of data. Calling ``gvar.dump(g, 'filename')`` writes a 
    serialized representation of ``g`` into the file named ``filename``. 
    Object ``g`` can be recovered later using ``gvar.load('filename')``.

    Typical usage is::

        # create file xxx.pickle containing GVars in g
        import gvar as gv 
        gv.dump(g, 'xxx.pickle')

        # read file xxx.pickle to recreate g
        new_g = gv.load('xxx.pickle')

    :func:`gvar.dump` is an alternative to ``pickle.dump()`` which,
    unlike the latter, works when there are |GVar|\s in ``g``.
    In particular it preserves correlations between different
    |GVar|\s, as well as relationships (i.e., derivatives) between
    derived |GVar|\s and primary |GVar|\s in ``g``.
    :func:`gvar.dump` uses :func:`gvar.remove_gvars` to search
    (recursively) for the |GVar|\s in ``g``.

    The partial variances for derived |GVar|\s in ``g`` coming from 
    primary |GVar|\s in ``g`` are preserved by :func:`gvar.dump`.
    (These are used, for example, to calculate error budgets.)
    Partial variances coming from derived (rather than 
    primary) |GVar|\s, however, are unreliable unless 
    every primary |GVar| that contributes to the covariances
    in ``g`` is included in ``g``. To guarantee that
    this is the case, set keyword ``add_dependencies=True``.
    This can greatly increase the size of the output file,
    and so should only be done if error budgets, etc. are needed. 
    (Also the cost of evaluating covariance matrices 
    for the reconstituted |GVar|\s is increased if there 
    are large numbers of primary |GVar|\s.) The default is 
    ``add_dependencies=False``.

    Args:
        g: Object to be serialized. Consists of (possibly nested) 
            dictionaries, deques, lists, ``numpy.array``\s, 
            and/or tuples that contain |GVar|\s and other types of data.
        outputfile: The name of a file or a file object in which the
            serialized ``g`` is stored. If ``outputfile=None`` (default),
            the results are written to a :class:`BytesIO`. 
        add_dependencies (bool): If ``True``, automatically includes 
            all primary |GVar|\s that contribute to the covariances 
            of the |GVar|\s in ``g`` but are not already in ``g``.
            Default is ``False``.
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying serializer (:mod:`pickle`).

    Returns:
        The :class:`BytesIO` object containing the serialized data, 
        if ``outputfile=None``; otherwise ``outputfile``.
    """    
    if outputfile is None:
        ofile = BytesIO()
    elif isinstance(outputfile, str):
        ofile = open(outputfile, 'wb')
    else:
        ofile = outputfile 
    datadict = {}
    gvlist = []
    datadict['data'] = remove_gvars(g, gvlist)
    if gvlist:
        datadict['gvlist'] = _gvar.gdumps(
            gvlist, method='pickle', 
            add_dependencies=add_dependencies
            )
    pickle.dump(datadict, ofile, **kargs)
    # cleanup and return
    if outputfile is None:
        ofile.seek(0)
        return ofile
    elif isinstance(outputfile, str):
        ofile.close()
    return outputfile

def load(inputfile, method=None, **kargs):
    """ Read and return object serialized in ``inputfile`` by :func:`gvar.dump`.

    This function recovers data serialized with :func:`gvar.dump`.
    Typical usage is::

        # create file xxx.pickle containing data in g
        import gvar as gv 
        gv.dump(g, 'xxx.pickle')

        # read file xxx.pickle to recreate g
        new_g = gv.gload('xxx.pickle')

    Note that the format used by :func:`gvar.dump` changed with 
    version |~| 11.0 of :mod:`gvar`. :func:`gvar.load` will 
    attempt to read the old formats if they are encountered, but 
    old data should be converted to the new format (by reading 
    it in with :func:`load` and them writing it out again 
    with :func:`dump`).

    Args:
        inputfile: The name of the file or a file object in which the
            serialized |GVar|\s are stored (created by :func:`gvar.dump`).
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying de-serializer (:mod:`pickle`).

    Returns:
        The reconstructed data object.
    """
    # N.B. method included for legacy code only
    if isinstance(inputfile, BytesIO): 
        ifile = inputfile
        iloc = inputfile.tell()
    elif isinstance(inputfile, StringIO):
        return _gvar.gload(inputfile, method)
    elif isinstance(inputfile, str):
        ifile = open(inputfile, 'rb')
    else:
        ifile = inputfile 
    try:
        datadict = pickle.load(ifile, **kargs)
        if isinstance(inputfile, str):
            ifile.close()
        if 'gvlist' in datadict:
            return distribute_gvars(
                datadict['data'], 
                _gvar.gloads(datadict['gvlist'])
                )   
        else:
            return datadict['data']
    except (KeyError, TypeError, pickle.PickleError):
        if isinstance(inputfile, BytesIO):
            inputfile.seek(iloc)
        return gload(inputfile, method)

##### gdump/gload
#####
def gdumps(g, method='json', add_dependencies=False):
    """ Return a serialized representation of ``g``.

    This function is shorthand for::
    
        gvar.gdump(g, method='json').getvalue()
    
    Typical usage is::

        # create string containing GVars in g
        import gvar as gv 
        gstr = gv.gdumps(g)

        # convert string back into GVars
        new_g = gv.gloads(gstr)
    
    Args:
        g: A |GVar|, array of |GVar|\s, or dictionary whose values
            are |GVar|\s and/or arrays of |GVar|\s.
        method: Serialization method, which should be either
            ``'json'`` or ``'pickle'``. Default is ``'json'``.
        add_dependencies (bool): If ``True``, automatically includes 
            all primary |GVar|\s that contribute to the covariances 
            of the |GVar|\s in ``g`` but are not already in ``g``.
            Default is ``False``.
    
    Returns:
        A string or bytes object containing a serialized 
        representation of object ``g``.
    """
    if method is None or method == 'json':
        return gdump(g, method='json', add_dependencies=add_dependencies).getvalue()
    elif method == 'pickle':
        return gdump(g, method='pickle', add_dependencies=add_dependencies).getvalue()
    else:
        raise ValueError('unknown method: ' + str(method))

def gdump(g, outputfile=None, method=None, add_dependencies=False, **kargs):
    """ Write a representation of ``g``  to file ``outputfile``.

    Object ``g`` is a |GVar|, an array of |GVar|\s (any shape), or 
    a dictionary whose values are |GVar|\s and/or arrays of |GVar|\s;
    it describes a general (multi-dimensional) Gaussian distribution.
    Calling ``gvar.gdump(g, 'filename')`` writes a serialized representation
    of ``g`` into the file named ``filename``. The Gaussian distribution
    described by ``g`` can be recovered later using ``gvar.load('filename')``.
    Correlations between different |GVar|\s in ``g`` are preserved, as
    are relationships (i.e., derivatives) between derived |GVar|\s and 
    any primary |GVar|\s in ``g``.

    :func:`gvar.gdump` differs from :func:`gvar.dump` in that the elements 
    in ``g`` must all be |GVar|\s for the former, whereas |GVar|\s may be
    mixed in with other data types for the latter. The structure of ``g``
    is also more restricted for :func:`gvar.gdump`, but it is 
    typically faster than :func:`gvar.dump`.

    Typical usage is::

        # create file xxx.pickle containing GVars in g
        import gvar as gv 
        gv.gdump(g, 'xxx.pickle')

        # read file xxx.pickle to recreate g
        new_g = gv.gload('xxx.pickle')

    Object ``g`` is serialized using one of the :mod:`json` (text
    file) or :mod:`pickle` (binary file) Python modules. 
    Method ``'json'`` can produce smaller files, but
    method ``'pickle'`` can be significantly faster. ``'json'`` 
    is more secure than ``'pickle'``, if that is an issue.
    
    :mod:`json` can have trouble with dictionaries whose keys are
    not strings. A workaround is used here that succeeds provided
    ``eval(repr(k)) == k`` for every key ``k``. This works for 
    a wide variety of standard key types including strings, integers, 
    and tuples of strings and/or integers. Try :mod:`pickle` if the 
    workaround fails.

    The partial variances for derived |GVar|\s in ``g`` coming from 
    primary |GVar|\s in ``g`` are preserved by :func:`gvar.gdump`.
    (These are used, for example, to calculate error budgets.)
    Partial variances coming from derived (rather than 
    primary) |GVar|\s, however, are unreliable unless 
    every primary |GVar| that contributes to the covariances
    in ``g`` is included in ``g``. To guarantee that
    this is the case set keyword ``add_dependencies=True``.
    This can greatly increase the size of the output file,
    and so should only be done if error budgets, etc. are needed. 
    (Also the cost of evaluating covariance matrices 
    for the reconstituted |GVar|\s is increased if there 
    are large numbers of primary |GVar|\s.) The default is 
    ``add_dependencies=False``.

    Args:
        g: A |GVar|, array of |GVar|\s, or dictionary whose values
            are |GVar|\s and/or arrays of |GVar|\s.
        outputfile: The name of a file or a file object in which the
            serialized |GVar|\s are stored. If ``outputfile=None`` (default),
            the results are written to a :class:`StringIO` (for 
            ``method='json'``) or :class:`BytesIO` (for
            ``method='pickle'``) object. 
        method (str): Serialization method, which should be either
            ``'json'`` or ``'pickle'``. If ``method=None`` (default),
            the method is inferred from the filename's extension:
            ``.json`` for ``'json'``; and ``.pickle`` or ``.pkl`` 
            or ``.p`` for ``'pickle'``. If that fails the method 
            defaults to ``'json'``.
        add_dependencies (bool): If ``True``, automatically includes 
            all primary |GVar|\s that contribute to the covariances 
            of the |GVar|\s in ``g`` but are not already in ``g``.
            Default is ``False``.
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying serializer (:mod:`pickle` or :mod:`json`).

    Returns:
        if ``outputfile=None``, the :class:`StringIO` or :class:`BytesIO` 
        object containing the serialized data is returned. Otherwise
        ``outputfile`` is returned.
    """
    sg = _gdump(g, add_dependencies=add_dependencies)
    if outputfile is None:
        if method is None:
            method = 'json' 
        elif method == 'dict':
            return sg
        ofile = StringIO() if method == 'json' else BytesIO()
    elif isinstance(outputfile, str):
        if method is None:
            method = (
                'json' if '.' not in outputfile else
                _DUMPMETHODS.get(outputfile.split('.')[-1], 'json')
                )
        ofile = open(outputfile, 'w' if method == 'json' else 'wb')
    else:
        ofile = outputfile 
        if method is None:
            method = 'json'
    if method == 'json':
        if 'keys' in sg:
            sg['keys'] = repr(sg['keys'])
        json.dump(sg, ofile, **kargs)
    elif method == 'pickle':
        pickle.dump(sg, ofile, **kargs)
    else:
        if isinstance(outputfile, str):
            ofile.close()
        raise ValueError('unknown method: ' + str(method))
    if outputfile is None:
        ofile.seek(0)
        return ofile
    elif isinstance(outputfile, str):
        ofile.close()
    return outputfile
    
_DUMPMETHODS = dict(json='json', p='pickle', pkl='pickle', pickle='pickle')

def gloads(inputstr):
    """ Return |GVar|\s that are serialized in ``inputstr``.

    This function recovers the |GVar|\s serialized with :func:`gvar.gdumps(g)`.
    It is shorthand for::

        gvar.gload(StringIO(inputstr), method='json')
    
    Typical usage is::

        # create string containing GVars in g
        import gvar as gv 
        gstr = gv.gdumps(g)

        # convert string back into GVars
        new_g = gv.gloads(gstr)

    Args:
        inputstr (str or bytes): String or bytes object 
            created by :func:`gvar.dumps`. 

    Returns:
        The reconstructed |GVar|, array of |GVar|\s, or dictionary 
        whose values are |GVar|\s and/or arrays of |GVar|\s.
    """
    try:
        return gload(StringIO(inputstr), method='json')
    except (TypeError, ValueError):
        return gload(BytesIO(inputstr), method='pickle')

def gload(inputfile, method=None, **kargs):
    """ Read and return |GVar|\s that are serialized in ``inputfile``.

    This function recovers |GVar|\s serialized with :func:`gvar.gdump`.
    Typical usage is::

        # create file xxx.pickle containing GVars in g
        import gvar as gv 
        gv.gdump(g, 'xxx.pickle')

        # read file xxx.pickle to recreate g
        new_g = gv.gload('xxx.pickle')

    Args:
        inputfile: The name of the file or a file object in which the
            serialized |GVar|\s are stored (created by :func:`gvar.gdump`).
        method (str or None): Serialization method, which should be either
            ``'pickle'`` or ``'json'``. If ``method=None`` (default),
            the method is inferred from the filename's extension:
            ``.json`` for ``'json'``; and ``.pickle`` or ``.pkl`` 
            or ``.p`` for ``'pickle'``. If that fails the method 
            defaults to ``'json'``. Argument ``method`` is ignored 
            if ``inputfile`` is either a :class:`StringIO` or 
            :class:`BytesIO` object, with the method being set 
            to ``'json'`` or ``'pickle'``, respectively.
        kargs (dict): Additional arguments, if any, that are passed to 
            the underlying de-serializer (:mod:`pickle` or :mod:`json`).

    Returns:
        The reconstructed |GVar|, array of |GVar|\s, or dictionary 
        whose values are |GVar|\s and/or arrays of |GVar|\s.
    """
    if isinstance(inputfile, dict):
        return _gload(inputfile)
    elif isinstance(inputfile, BytesIO): # check for this before StringIO
        ifile = inputfile
        iloc = inputfile.tell()
        method = 'pickle'
    elif isinstance(inputfile, StringIO):
        ifile = inputfile 
        iloc = inputfile.tell()
        method = 'json'
    elif isinstance(inputfile, str):
        if method is None:
            method = (
                'json' if '.' not in inputfile else
                _DUMPMETHODS.get(inputfile.split('.')[-1], 'json')
                )
        ifile = open(inputfile, 'r' if method == 'json' else 'rb')
    else:
        ifile = inputfile 
        if method is None:
            method = 'json'  
    if method == 'json':
        sg = json.load(ifile, **kargs) 
    elif method == 'pickle':
        sg = pickle.load(ifile, **kargs)
    else:
        if isinstance(inputfile, str):
            ifile.close()
        raise ValueError('invalid method: ' + str(method))
    if isinstance(inputfile, str):
        ifile.close()
    try:
        return _gload(sg)
    except KeyError, TypeError:
        if isinstance(inputfile, BytesIO) or isinstance(inputfile, StringIO):
            inputfile.seek(iloc)
        return _oldload1(inputfile, method)

def _gdump(g, add_dependencies=False):
    """ Repack ``g`` in dictionary that can be serialized. Used by :func:`gdump`.
    """
    cdef numpy.ndarray[INTP_TYPE, ndim=1] idx 
    cdef numpy.ndarray bcov
    cdef numpy.ndarray primary, derived
    data = {}
    if hasattr(g, 'keys'):
        if not isinstance(g, _gvar.BufferDict):
            g = _gvar.BufferDict(g)
        data['keys'] = list(g.keys())
        data['layouts'] = []
        for k in data['keys']:
            slice, shape = g.slice_shape(k)
            if shape == ():
                data['layouts'].append([slice, []])
            else:
                data['layouts'].append([[slice.start, slice.stop, slice.step], list(shape)])
    else:
        g = numpy.asarray(g)
        data['shape'] = list(numpy.shape(g))
    buf = g.flat[:]
    data['bufsize'] = len(buf)
    if add_dependencies:
        buf = numpy.concatenate([buf, _gvar.dependencies(buf)])
        data['fix_cov'] = False
    elif not _gvar.missing_dependencies(buf):
        data['fix_cov'] = False 
    else:
        data['fix_cov'] = True
    data['means'] = _gvar.mean(buf).tolist()
    data['bcovs'] = []
    first_pass = True
    for idx, bcov in _gvar.evalcov_blocks(buf, compress=True):
        data['bcovs'] += [[idx.tolist(), bcov.tolist()]]
        if first_pass:
            data['bcovs'][0] += [[], []]
            first_pass = False
        else:
            primary = _gvar.is_primary(buf[idx])
            derived = numpy.logical_not(primary)
            if numpy.all(primary) or numpy.all(derived):
                data['bcovs'][-1] += [[], []]
            else:
                derivs = _gvar.deriv(buf[idx][derived], buf[idx][primary])
                data['bcovs'][-1] += [primary.tolist(), derivs.tolist()]
    return data 

def _gload(data):
    """ Inverse of :func:`_gdump`.
    """
    # reconstitute the GVars
    buf = numpy.array(data['means'], dtype=object)
    try:
        for idx, bcov, primary, derivs in data['bcovs']:
            buf[idx] = _rebuild_gvars(
                _gvar.gvar(buf[idx], bcov), numpy.array(bcov), 
                primary, numpy.array(derivs), 
                fix_cov = data['fix_cov'],
                )
        buf = numpy.array(buf[:data['bufsize']])
    except ValueError:
        # old format for legacy code
        for idx, bcov in data['bcovs']:
            buf[idx] = _gvar.gvar(buf[idx], bcov)

    # rebuild the data structures
    if 'keys' in data:
        keys = data['keys']
        if not isinstance(keys, list):
            keys = eval(keys, {}, {})
        g = collections.OrderedDict()
        for k, (sl, sh) in zip(keys, data['layouts']):
            sh = tuple(sh)
            if sh == ():
                g[k] = buf[sl]
            else:
                sl = slice(*sl)
                g[k] = numpy.reshape(buf[sl], sh)
        return _gvar.BufferDict(g)
    else:
        buf.shape = tuple(data['shape'])
        return buf

def _rebuild_gvars(buf, cov, primary, derivs, fix_cov):
    " reconnect derived GVars to primary GVars; used by :func:`_gload` "
    if primary == [] or derivs.size == 0:
        return buf
    idx_primary = numpy.arange(buf.size)[primary]
    idx_derived = numpy.arange(buf.size)[numpy.logical_not(primary)]
    # contributions from primaries
    tmp = _gvar.mean(buf[idx_derived]) + numpy.sum(
        (buf[idx_primary] - _gvar.mean(buf[idx_primary]))[None, :] * derivs,
        axis=1
        )
    # contributions from missing primaries
    if fix_cov:
        buf[idx_derived] = tmp + _gvar.gvar(
            numpy.zeros(len(idx_derived), dtype=float),
            cov[idx_derived[:, None], idx_derived] * (1 + EPSILON) - _gvar.evalcov(tmp)
            )
    else:
        buf[idx_derived] = tmp
    return buf

####### old versions kept for legacy purposes (for now)
#######
def olddump(g, outputfile, method='pickle', use_json=False):
    """ Serialize a collection ``g`` of |GVar|\s into file ``outputfile``.

    Old verion, here for testing purposes only.

    The |GVar|\s are recovered using :func:`gvar.load`.

    Three serialization methods are available: :mod:`pickle`, :mod:`json`,
    and :mod:`yaml` (provided the :mod:`yaml` module is installed).

    :mod:`json` can have trouble with dictionaries whose keys are not
    strings. A workaround is used here that succeeds provided
    ``eval(repr(k)) == k`` for every key ``k``, which is true for strings and
    lots of other types of key. Use :mod:`pickle` where the workaround fails.

    Args:
        g: A |GVar|, array of |GVar|\s, or dictionary whose values
            are |GVar|\s and/or arrays of |GVar|\s.
        outputfile: The name of a file or a file object in which the
            serialized |GVar|\s are stored.
        method (str): Serialization method, which should be one of
            ``['pickle', 'json', 'yaml']``. Default is ``'pickle'``.
    """
    if use_json is True:  # for legacy code
        method = 'json'
    if yaml is None and method == 'yaml':
        raise RuntimeError('yaml module not installed')
    if isinstance(outputfile, str):
        with open(outputfile, 'w' if method in ['json', 'yaml'] else 'wb') as ofile:
            return olddump(g, ofile, method=method)
    else:
        ofile = outputfile
    if method in ['json', 'yaml']:
        if hasattr(g, 'keys'):
            if not isinstance(g, _gvar.BufferDict):
                g = _gvar.BufferDict(g)
            tag = method, 'dict'
            gmean = [
                    (repr(k) if method == 'json' else k, d.tolist())
                    for k,d in _gvar.mean(g).items()
                    ]
            gcov = _gvar.evalcov(g.buf).tolist()
        else:
            tag = method, 'array'
            gmean = numpy.array(_gvar.mean(g)).tolist()
            gcov = _gvar.evalcov(g).tolist()
        data = dict(tag=tag, gmean=gmean, gcov=gcov)
        return (
            json.dump(data, ofile) if method == 'json' else
            yaml.dump(data, ofile, Dumper=yaml_Dumper)
            )
    elif method == 'pickle':
        pickle.dump(
            dict(tag=('pickle', None), gmean=_gvar.mean(g), gcov=_gvar.evalcov(g)), ofile
            )
    else:
        raise ValueError('unknown method: ' + str(method))

def _oldload1(inputfile, method=None, use_json=None):
    """ Load and return serialized |GVar|\s from file ``inputfile``.

    This is the version of :func:`gvar.load` used before  
    version |~| 10.0 of :mod:`gvar`.

    This function recovers |GVar|\s pickled with :func:`gvar.dump`.
    It will disappear eventually.

    Args:
        inputfile: The name of the file or a file object in which the
            serialized |GVar|\s are stored.
        method (str or None): Serialization method, which should be one of
            ``['pickle', 'json', 'yaml']``. If ``method=None``, then each
            method is tried in turn.

    Returns:
        The reconstructed |GVar|, or array or dictionary of |GVar|\s.
    """
    warnings.warn("using old dump format", DeprecationWarning)
    if use_json is True: # for legacy code
        method = 'json'
    elif use_json is False:
        method = 'pickle'
    if method is None:
        try:
            return _oldload1(inputfile, method='pickle')
        except:
            pass
        try:
            return _oldload1(inputfile, method='json')
        except:
            pass
        if yaml is not None:
            try:
                return _oldload1(inputfile, method='yaml')
            except:
                pass
        try:
            return _oldload0(inputfile)
        except:
            raise RuntimeError('cannot read file')
    if yaml is None and method == 'yaml':
        raise RuntimeError('yaml module not installed')
    if isinstance(inputfile, str):
        with open(inputfile, 'rb' if method == 'pickle' else 'r') as ifile:
            return _oldload1(ifile, method=method)
    else:
        ifile = inputfile
    if method in ['json', 'yaml']:
        data = json.load(ifile) if method == 'json' else yaml.load(ifile, Loader=yaml_Loader)
        assert data['tag'][0] == method
        if data['tag'][1] == 'dict':
            if method == 'json':
                data['gmean'] = [(eval(k, {}, {}), d) for k, d in data['gmean']]
            ans = _gvar.BufferDict(data['gmean'])
            ans.buf = _gvar.gvar(ans._buf, data['gcov'])
        else:
            ans = _gvar.gvar(data['gmean'], data['gcov'])
    elif method == 'pickle':
        data = pickle.load(ifile)
        assert data['tag'][0] == 'pickle'
        ans = _gvar.gvar(data['gmean'], data['gcov'])
    else:
        raise ValueError('unknown method: ' + str(method))
    return ans

def _oldload0(inputfile, use_json=None):
    """
    Older version of :func:`load`, included to allow loading
    of previously dumped data.
    """
    if use_json is None:
        try:
            return _oldload0(inputfile, use_json=False)
        except:
            return _oldload0(inputfile, use_json=True)
    if isinstance(inputfile, str):
        with open(inputfile, 'r' if use_json else 'rb') as ifile:
            return _oldload0(ifile, use_json=use_json)
    else:
        ifile = inputfile
    if use_json:
        data = json.load(ifile)
        if hasattr(data, 'keys'):
            ans = _gvar.BufferDict([(eval(k), d) for k, d in data['items']])
            ans.buf = _gvar.gvar(ans._buf, data['cov'])
        else:
            ans = _gvar.gvar(*data)  # need * with json since it doesn't have tuples
    else:
        ans = _gvar.gvar(pickle.load(ifile))
    return ans

def disassemble(g):
    """ Disassemble collection ``g`` of |GVar|\s.

    Args:
        g (dict, array, or gvar.GVar): Collection of |GVar|\s to be
            disassembled.
    """
    cdef INTP_TYPE i, gsize
    cdef GVar gi
    cdef numpy.ndarray[object, ndim=1] newbuf
    if hasattr(g, 'keys'):
        if not isinstance(g, BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    gsize = g.size
    newbuf = numpy.empty(gsize, object)
    for i,gi in enumerate(g.flat):
        if gi == None:
            raise ValueError('g contains None')
        newbuf[i] = (gi.v, gi.d)
    return BufferDict(g, buf=newbuf) if g.shape is None else newbuf.reshape(g.shape)

def reassemble(data, smat cov=_gvar.gvar.cov):
    """ Convert data from :func:`gvar.disassemble` back into |GVar|\s.

    Args:
        data (BufferDict, array): Disassembled collection of |GVar|\s
            from :func:`gvar.disassemble` that are to be reassembled.
        cov (gvar.smat): Covariance matrix corresponding to the |GVar|\s
            in ``data``. (Default is ``gvar.gvar.cov``.)
    """
    cdef INTP_TYPE i, datasize
    cdef object datai
    cdef svec der
    cdef double val
    cdef numpy.ndarray[object, ndim=1] newbuf
    if hasattr(data, 'keys'):
        if not isinstance(data, BufferDict):
            data = BufferDict(data)
    else:
        data = numpy.asarray(data)
    datasize = data.size
    newbuf = numpy.empty(datasize, object)
    for i,(val,der) in enumerate(data.flat):
        newbuf[i] = GVar(val, der, cov)
    return (
        BufferDict(data, buf=newbuf) if data.shape is None else
        newbuf.reshape(data.shape)
        )


def wsum_der(numpy.float_t[:] wgt, GVar[:] glist):
    """ weighted sum of |GVar| derivatives """
    cdef GVar g
    cdef smat cov
    cdef double w
    cdef INTP_TYPE ng,i,j
    cdef numpy.ndarray[numpy.float_t,ndim=1] ans
    ng = len(glist)
    assert ng==len(wgt),"wgt and glist have different lengths."
    cov = glist[0].cov
    ans = numpy.zeros(len(cov),numpy.float_)
    for i in range(wgt.shape[0]):
        w = wgt[i]
        g = glist[i]
        assert g.cov is cov,"Incompatible |GVar|\s."
        for j in range(g.d.size):
            ans[g.d.v[j].i] += w*g.d.v[j].v
    return ans

def msum_gvar(numpy.float_t[:, :] wgt, GVar[:] glist, GVar[:] out):
    cdef INTP_TYPE i
    for i in range(wgt.shape[0]):
        out[i] = wsum_gvar(wgt[i], glist)

cpdef GVar wsum_gvar(numpy.float_t[:] wgt, GVar[:] glist):
    """ weighted sum of |GVar|\s """
    cdef svec wd
    cdef double wv, w
    cdef GVar g
    cdef smat cov
    cdef INTP_TYPE ng, i, j, nd, size
    cdef double[:] der
    cdef INTP_TYPE[:] idx
    ng = len(glist)
    assert ng==len(wgt),"wgt and glist have different lengths."
    cov = glist[0].cov
    der = numpy.zeros(len(cov),numpy.float_)
    wv = 0.0
    for i in range(ng): #w,g in zip(wgt,glist):
        w = wgt[i]
        g = glist[i]
        assert g.cov is cov,"Incompatible |GVar|\s."
        wv += w * g.v
        for j in range(g.d.size):
            der[g.d.v[j].i] += w * g.d.v[j].v
    idx = numpy.zeros(len(cov), numpy.intp) # der.nonzero()[0]
    nd = 0
    for i in range(der.shape[0]):
        if der[i]!=0:
            idx[nd] = i
            nd += 1
    wd = svec(nd)
    for i in range(nd):
        wd.v[i].i = idx[i]
        wd.v[i].v = der[idx[i]]
    return GVar(wv, wd, cov)

def fmt_values(outputs, ndecimal=None, ndigit=None):
    """ Tabulate :class:`gvar.GVar`\s in ``outputs``.

    Args:
        outputs: A dictionary of :class:`gvar.GVar` objects.
        ndecimal (int): Format values ``v`` using ``v.fmt(ndecimal)``.
    Returns:
        A table (``str``) containing values and standard
        deviations for variables in ``outputs``, labeled by the keys
        in ``outputs``.
    """
    if ndigit is not None:
        ndecimal = ndigit
    ans = "Values:\n"
    for vk in outputs:
        ans += "%19s: %-20s\n" % (vk,outputs[vk].fmt(ndecimal))
    return ans

def fmt_errorbudget(
    outputs, inputs, ndecimal=2, percent=True, colwidth=None,
    verify=False, ndigit=None
    ):
    """ Tabulate error budget for ``outputs[ko]`` due to ``inputs[ki]``.

    For each output ``outputs[ko]``, ``fmt_errorbudget`` computes the
    contributions to ``outputs[ko]``'s standard deviation coming from the
    |GVar|\s collected in ``inputs[ki]``. This is done for each key
    combination ``(ko,ki)`` and the results are tabulated with columns and
    rows labeled by ``ko`` and ``ki``, respectively. If a |GVar| in
    ``inputs[ki]`` is correlated with other |GVar|\s, the contribution from
    the others is included in the ``ki`` contribution as well (since
    contributions from correlated |GVar|\s cannot be distinguished). The table
    is returned as a string.

    Args:
        outputs: Dictionary of |GVar|\s for which an error budget
            is computed.
        inputs: Dictionary of: |GVar|\s, arrays/dictionaries of
            |GVar|\s, or lists of |GVar|\s and/or arrays/dictionaries of
            |GVar|\s. ``fmt_errorbudget`` tabulates the parts of the standard
            deviations of each ``outputs[ko]`` due to each ``inputs[ki]``.
        ndecimal (int): Number of decimal places displayed in table.
        percent (bool): Tabulate % errors if ``percent is True``;   
            otherwise tabulate the errors themselves.
        colwidth (int): Width of each column. This is set automatically,
            to accommodate label widths, if ``colwidth=None`` (default).
        verify (bool): If ``True``, a warning is issued if: 1) different 
            inputs are correlated (and therefore double count errors); or
            2) the sum (in quadrature) of partial errors is not equal to 
            the total error to within 0.1% of the error (and the error 
            budget is incomplete or overcomplete). No checking is done 
            if ``verify==False`` (default).

    Returns:
        A table (``str``) containing the error budget.
        Output variables are labeled by the keys in ``outputs``
        (columns); sources of uncertainty are labeled by the keys in
        ``inputs`` (rows).
    """
    # collect partial errors
    if ndigit is not None:
        ndecimal = ndigit       # legacy name
    err = {}
    outputs_keys = []
    for ko in outputs:
        outputs_keys.append(str(ko))
        for ki in inputs:
            inputs_ki = inputs[ki]
            if hasattr(inputs_ki,'keys') or not hasattr(inputs_ki,'__iter__'):
                inputs_ki = [inputs_ki]
            err[ko,ki] = outputs[ko].partialvar(*inputs_ki)**0.5

    # verify?
    if verify:
        # correlated inputs?
        unprocessed_keys = set(inputs.keys())
        for ki1 in inputs.keys():
            unprocessed_keys.discard(ki1)
            for ki2 in unprocessed_keys:
                if not uncorrelated(inputs[ki1], inputs[ki2]):
                    warnings.warn("{} and {} double count errors".format(
                        ki1, ki2
                        ))
        # anything missing?
        for ko in outputs:
            totvar = 0.0
            for ki in inputs:
                totvar += err[ko, ki] ** 2
            if abs(totvar - outputs[ko].var) > 0.001 * outputs[ko].var:
                warnings.warn("{} partial error {}  !=  total error {}".format(
                    ko, totvar ** 0.5, outputs[ko].sdev
                    ))
    # form table
    # determine column widths
    if colwidth is None:
        # find it by hand: w0 for 1st col, w for rest
        w = 10
        for ko in outputs_keys:
            ko = str(ko)
            if len(ko) >= w:
                w = len(ko) + 1
        w0 = 10
        for ki in inputs:
            ki = str(ki)
            if len(ki) >= w0:
                w0 = len(ki) + 1
    else:
        w = colwidth
        w0 = w if w > 20 else 20
    lfmt = (
        "%" + str(w0 - 1) + "s:" +
        len(outputs) * ( "%" + str(w) + "." + str(ndecimal) + "f") + "\n"
        )
    hfmt = (
        "%" + str(w0) + "s" + len(outputs) * ("%" + str(w) + "s") + "\n"
        )
    if percent:
        val = numpy.array([abs(outputs[vk].mean)
                                for vk in outputs])/100.
        ans = "Partial % Errors:\n"
    else:
        val = 1.
        ans = "Partial Errors:\n"
    ans += hfmt % (("",)+tuple(outputs_keys))
    ans += (w0 +len(outputs) * w) * '-' + "\n"
    for ck in inputs:
        verr = numpy.array([err[vk,ck] for vk in outputs])/val
        ans += lfmt%((ck,)+tuple(verr))
    ans += (w0 +len(outputs) * w) * '-' + "\n"
    ans += lfmt%(("total",)+tuple(numpy.array([outputs[vk].sdev
                                    for vk in outputs])/val))
    return ans

# bootstrap_iter, raniter, svd, valder
def bootstrap_iter(g, n=None, eps=None, svdcut=None):
    """ Return iterator for bootstrap copies of ``g``.

    The gaussian variables (|GVar| objects) in array (or dictionary) ``g``
    collectively define a multidimensional gaussian distribution. The
    iterator created by :func:`bootstrap_iter` generates an array (or
    dictionary) of new |GVar|\s whose covariance matrix is the same as
    ``g``'s but whose means are drawn at random from the original ``g``
    distribution. This is a *bootstrap copy* of the original distribution.
    Each iteration of the iterator has different means (but the same
    covariance matrix).

    :func:`bootstrap_iter` also works when ``g`` is a single |GVar|, in
    which case the resulting iterator returns bootstrap copies of the
    ``g``.

    Args:
        g: An array (or dictionary) of objects of type |GVar|.
        n: Maximum number of random iterations. Setting ``n=None``
            (the default) implies there is no maximum number.
        eps (float): If positive, singularities in the correlation matrix 
            for ``g`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. Ignored if ``svdcut`` is specified (and 
            not ``None``).
        svdcut (float): If nonzero, singularities in the correlation
            matrix are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. Default is ``svdcut=1e-12``.

    Returns:
        An iterator that returns bootstrap copies of ``g``.
    """
    g, i_wgts = _gvar.regulate(g, eps=eps, svdcut=svdcut, wgts=1.)
    g_flat = g.flat
    nwgt = sum(len(wgts) for i, wgts in i_wgts)
    count = 0
    while (n is None) or (count < n):
        count += 1
        buf = numpy.array(g.flat)
        z = numpy.random.normal(0.0, 1.0, nwgt)
        i, wgts = i_wgts[0]
        if len(i) > 0:
            buf[i] += z[i] * wgts
        for i, wgts in i_wgts[1:]:
            buf[i] += sum(zi * wi for zi, wi in zip(z[i], wgts))
        if g.shape is None:
            yield BufferDict(g, buf=buf)
        elif g.shape == ():
            yield next(buf.flat)
        else:
            yield buf.reshape(g.shape)
    # raise StopIteration

def sample(g, eps=None, svdcut=None):
    """ Generate random sample from distribution ``g``.

    Equivalent to ``next(gvar.raniter(g, svdcut=svdcut, eps=eps))``.

    Args:
        g: An array or dictionary of objects of type |GVar|; or a |GVar|.
        eps (float): If positive, singularities in the correlation matrix 
            for ``g`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. Ignored if ``svdcut`` is specified (and 
            not ``None``).
        svdcut (float): If nonzero, singularities in the correlation
            matrix are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. Default is ``svdcut=1e-12``.

    Returns:
        A random array or dictionary, with the same shape as ``g``,
        drawn from the Gaussian distribution defined by ``g``.
    """
    return next(raniter(g, svdcut=svdcut, eps=eps))

def raniter(g, n=None, eps=None, svdcut=None, uniform=None):
    """ Return iterator for random samples from distribution ``g``

    The Gaussian variables (|GVar| objects) in array (or dictionary) ``g``
    collectively define a multidimensional gaussian distribution. The
    iterator defined by :func:`raniter` generates an array (or dictionary)
    containing random numbers drawn from that distribution, with
    correlations intact.

    The layout for the result is the same as for ``g``. So an array of the
    same shape is returned if ``g`` is an array. When ``g`` is a dictionary,
    individual entries ``g[k]`` may be |GVar|\s or arrays of |GVar|\s,
    with arbitrary shapes.

    :func:`raniter` also works when ``g`` is a single |GVar|, in which case
    the resulting iterator returns random numbers drawn from the
    distribution specified by ``g``.

    Args:
        g: An array (or dictionary) of objects of type |GVar|; or a |GVar|.
        n (int or ``None``): Maximum number of random iterations.
            Setting ``n=None`` (the default) implies there is
            no maximum number.
        eps (float): If positive, singularities in the correlation matrix 
            for ``g`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. Ignored if ``svdcut`` is specified (and 
            not ``None``).
        svdcut (float): If nonzero, singularities in the correlation
            matrix are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. Default is ``svdcut=1e-12``.
        uniform (float or None): Replace Gaussian distribution specified 
            by ``g`` with a uniform distribution covering the interval 
            ``[-uniform, uniform]`` times the standard deviation centered 
            on the mean (along each principal axis of the error ellipse).
            Ignored if ``None`` (default).

    Returns:
        An iterator that returns random arrays or dictionaries
        with the same shape as ``g`` drawn from the Gaussian distribution
        defined by ``g``.
    """
    g, i_wgts = _gvar.regulate(g, eps=eps, svdcut=svdcut, wgts=1.)
    g_mean = mean(g.flat)
    nwgt = sum(len(wgts) for i, wgts in i_wgts)
    count = 0
    while (n is None) or (count < n):
        count += 1
        z = numpy.random.normal(0.0, 1.0, nwgt) if uniform is None else numpy.random.uniform(-uniform, uniform, nwgt)
        zstart = 0 
        zstop = 0
        buf = numpy.array(g_mean)
        i, wgts = i_wgts[0]
        if len(i) > 0:
            zstop += len(i)
            buf[i] += z[zstart:zstop] * wgts
            zstart = zstop
        for i, wgts in i_wgts[1:]:
            zstop += len(i)
            buf[i] += z[zstart:zstop].dot(wgts) 
            zstart = zstop
        if g.shape is None:
            yield BufferDict(g, buf=buf)
        elif g.shape == ():
            yield next(buf.flat)
        else:
            yield buf.reshape(g.shape)
    # raise StopIteration

class SVD(object):
    """ SVD decomposition of a pos. sym. matrix.

    :class:`SVD` is a function-class that computes the eigenvalues and
    eigenvectors of a symmetric matrix ``mat``. Eigenvalues that
    are small (or negative, because of roundoff) can be eliminated or
    modified using *svd* cuts. Typical usage is::

        >>> mat = [[1.,.25],[.25,2.]]
        >>> s = SVD(mat)
        >>> print(s.val)             # eigenvalues
        [ 0.94098301  2.05901699]
        >>> print(s.vec[0])          # 1st eigenvector (for s.val[0])
        [ 0.97324899 -0.22975292]
        >>> print(s.vec[1])          # 2nd eigenvector (for s.val[1])
        [ 0.22975292  0.97324899]

        >>> s = SVD(mat,svdcut=0.6)  # force s.val[i]>=s.val[-1]*0.6
        >>> print(s.val)
        [ 1.2354102   2.05901699]
        >>> print(s.vec[0])          # eigenvector unchanged
        [ 0.97324899 -0.22975292]

        >>> s = SVD(mat)
        >>> w = s.decomp(-1)         # decomposition of inverse of mat
        >>> invmat = sum(numpy.outer(wj,wj) for wj in w)
        >>> print(numpy.dot(mat,invmat))    # should be unit matrix
        [[  1.00000000e+00   2.77555756e-17]
         [  1.66533454e-16   1.00000000e+00]]

    Args:
        mat: Positive, symmetric matrix.
        svdcut: If positive, replace eigenvalues of ``mat`` with
            ``svdcut*(max eigenvalue)``; if negative, discard 
            eigenmodes with eigenvalues smaller than ``svdcut`` 
            times the maximum eigenvalue.
        svdnum: If positive, keep only the modes with the largest
            ``svdnum`` eigenvalues; ignore if set to ``None``.
        compute_delta (bool): Compute ``delta`` (see below) 
            if ``True``; set ``delta=None`` otherwise.
        rescale: Rescale the input matrix to make its diagonal 
            elements equal to +-1.0 before diagonalizing.

    The results are accessed using:

    ..  attribute:: val

        An ordered array containing the eigenvalues of ``mat``. Note
        that ``val[i]<=val[i+1]``.

    ..  attribute:: vec

        Eigenvectors ``vec[i]`` corresponding to the eigenvalues
        ``val[i]``.

    ..  attribute:: valmin

        Minimum eigenvalue allowed in the modified matrix.

    ..  attribute:: valorig

        Eigenvalues of original matrix.

    ..  attribute:: D

        The diagonal matrix used to precondition the input matrix if
        ``rescale==True``. The matrix diagonalized is ``D M D`` where ``M``
        is the input matrix. ``D`` is stored as a one-dimensional vector of
        diagonal elements. ``D`` is ``None`` if ``rescale==False``.

    ..  attribute:: nmod

        The first ``nmod`` eigenvalues in ``self.val`` were modified by
        the SVD cut (equals 0 unless ``svdcut > 0``).

    ..  attribute:: eigen_range

        Ratio of the smallest to the largest eigenvector in the
        unconditioned matrix (after rescaling if ``rescale=True``)

    ..  attribute:: delta

        A vector of ``gvar``\s whose means are zero and whose
        covariance matrix is what was added to ``mat`` to condition
        its eigenvalues. Is ``None`` if ``svdcut<0`` or
        ``compute_delta==False``.
    """
    def __init__(
        self, mat, svdcut=None, svdnum=None, compute_delta=False, rescale=False
        ):
        super(SVD,self).__init__()
        mat = numpy.asarray(mat)
        if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError('mat is not a square')
        self.svdcut = svdcut
        self.svdnum = svdnum
        if rescale:
            diag = numpy.fabs(mat.diagonal())
            diag[diag==0.0] = 1.
            D = (diag)**(-0.5)
            # DmatD = mat*D
            # DmatD = (DmatD.transpose()*D).transpose()
            DmatD = D[:, None] * mat * D[None, :]
            self.D = D
        else:
            DmatD = mat
            self.D = None
        val = None
        for key in SVD._analyzers:
            try:
                val, vec = SVD._analyzers[key](DmatD)
            except numpy.linalg.LinAlgError:
                continue
            break
        if val is None:
            raise numpy.linalg.LinAlgError('eigen analysis failed')
        self.kappa = val[0]/val[-1] if val[-1]!=0 else None  # min/max eval
        self.eigen_range = self.kappa
        self.delta = None
        self.nmod = 0
        self.valorig = numpy.array(val)
        # svd cuts
        if (svdcut is None or svdcut==0.0) and (svdnum is None or svdnum<=0):
            self.val = val
            self.vec = vec
            return
        # restrict to svdnum largest eigenvalues
        if svdnum is not None and svdnum>0:
            val = val[-svdnum:]
            vec = vec[-svdnum:]
        # impose svdcut on eigenvalues
        if svdcut is None or svdcut==0:
            self.val = val
            self.vec = vec
            return
        valmin = abs(svdcut)*val[-1]
        if svdcut>0:
            # force all eigenvalues >= valmin
            dely = None
            for i in range(len(val)):
                if val[i]<valmin:
                    self.nmod += 1
                    if compute_delta:
                        if dely is None:
                            dely = vec[i]*_gvar.gvar(0.0,(valmin-val[i])**0.5)
                        else:
                            dely += vec[i]*_gvar.gvar(0.0,(valmin-val[i])**0.5)
                    val[i] = valmin
                else:
                    break
            self.val = val
            self.vec = vec
            self.valmin = valmin
            self.delta = dely if (self.D is None or dely is None) else dely/self.D
            return
        else:
            # discard modes with eigenvalues < valmin
            for i in range(len(val)):
                if val[i]>=valmin:
                    break
            self.val = val[i:]
            self.vec = vec[i:]
            return  # val[i:],vec[i:],kappa,None

    @staticmethod 
    def _numpy_eigh(DmatD):
        val, vec = numpy.linalg.eigh(DmatD)
        vec = numpy.transpose(vec) # now 1st index labels eigenval
        val = numpy.fabs(val)
        # guarantee that sorted, with smallest val[i] first
        indices = numpy.arange(val.size) # in case val[i]==val[j]
        val, indices, vec = zip(*sorted(zip(val, indices, vec)))
        val = numpy.array(val, dtype=float)
        vec = numpy.array(vec, dtype=float)
        return val, vec 
    
    @staticmethod 
    def _scipy_eigh(DmatD):
        val, vec = _scipy_eigh(DmatD)
        vec = numpy.transpose(vec) # now 1st index labels eigenval
        val = numpy.fabs(val)
        # guarantee that sorted, with smallest val[i] first
        indices = numpy.arange(val.size) # in case val[i]==val[j]
        val, indices, vec = zip(*sorted(zip(val, indices, vec)))
        val = numpy.array(val, dtype=float)
        vec = numpy.array(vec, dtype=float)
        return val, vec 

    @staticmethod
    def _numpy_svd(DmatD):
        # warnings.warn('numpy.linalg.eigh failed; trying numpy.linalg.svd')
        DmatD = (DmatD.T + DmatD) / 2.
        # different algorithm from eigh if hermitian=False
        vec,val,dummy = numpy.linalg.svd(DmatD)  
        vec = vec.T # numpy.transpose(vec) # now 1st index labels eigenval
        # guarantee that sorted, with smallest val[i] first
        vec = numpy.array(vec[-1::-1], dtype=float)
        val = numpy.array(val[-1::-1], dtype=float)
        return val, vec 

    def decomp(self,n=1):
        """ Vector decomposition of input matrix raised to power ``n``.

        Computes vectors ``w[i]`` such that

            mat**n = sum_i numpy.outer(w[i],w[i])

        where ``mat`` is the original input matrix to :class:`svd`. This
        decomposition cannot be computed if the input matrix was rescaled
        (``rescale=True``) except for ``n=1`` and ``n=-1``.

        :param n: Power of input matrix.
        :type n: number
        :returns: Array ``w`` of vectors.
        """
        if self.D is None:
            w = numpy.array(self.vec)
            for j,valj in enumerate(self.val):
                w[j] *= valj**(n/2.)
        else:
            if n!=1 and n!=-1:
                raise ValueError(           #
                    "Can't compute decomposition for rescaled matrix.")
            w = numpy.array(self.vec)
            Dfac = self.D**(-n)
            for j,valj in enumerate(self.val):
                w[j] *= Dfac*valj**(n/2.)
        return w

# use ordered dict for python2
SVD._analyzers = collections.OrderedDict()
SVD._analyzers['scipy_eigh']  = SVD._scipy_eigh
SVD._analyzers['numpy_eigh']  = SVD._numpy_eigh
SVD._analyzers['numpy_svd']  = SVD._numpy_svd

def valder(v):
    """ Convert array ``v`` of numbers into an array of |GVar|\s.

    The |GVar|\s created by ``valder(v)`` have means equal to the
    values ``v[i]`` and standard deviations of zero. If ``v`` is
    one-dimensional, for example, ``vd = valder(v)`` is functionally
    equivalent to::

        newgvar = gvar.gvar_factory()
        vd = numpy.array([newgvar(vi,0.0) for vi in v])

    The use of ``newgvar`` to create the |GVar|\s means that these
    variables are incompatible with those created by ``gvar.gvar``.
    More usefully, it also means that the vector of derivatives ``x.der``
    for any |GVar| ``x`` formed from elements of ``vd = valder(v)``
    correspond to derivatives with respect to ``vd``: that is, ``x.der[i]``
    is the derivative of ``x`` with respect to ``vd.flat[i]``.

    In general, the shape of the array returned by ``valder`` is the
    same as that of ``v``.
    """
    try:
        v = numpy.asarray(v,numpy.float_)
    except ValueError:
        raise ValueError("Bad input.")
    gv_gvar = _gvar.gvar_factory()
    return gv_gvar(v,numpy.zeros(v.shape,numpy.float_))


# ## miscellaneous functions ##
# def gammaQ(double a, double x):
#     """ Return the incomplete gamma function ``Q(a,x) = 1-P(a,x)``. Y

#     Note that ``gammaQ(ndof/2., chi2/2.)`` is the probabilty that one could
#     get a ``chi**2`` larger than ``chi2`` with ``ndof`` degrees
#     of freedom even if the model used to construct ``chi2`` is correct.
#     """
#     cdef gsl_sf_result_struct res
#     cdef int status
#     status = gsl_sf_gamma_inc_Q_e(a, x, &res)
#     assert status==GSL_SUCCESS, status
#     return res.val

# following are substitues for GSL's routine if gsl is not present (via lsqfit)
cdef double gammaP_ser(double a, double x, double rtol, int itmax):
    """ Power series expansion for P(a, x) (for x < a+1).

    P(a, x) = 1/Gamma(a) * \int_0^x dt exp(-t) t ** (a-1) = 1 - Q(a, x)
    """
    cdef int n
    cdef double ans, term
    if x == 0:
        return 0.
    ans = 0.
    term = 1. / x
    for n in range(itmax):
        term *= x / float(a + n)
        ans += term
        if abs(term) < rtol * abs(ans):
            break
    else:
        warnings.warn(
            'gammaP convergence not complete -- want: %.3g << %.3g'
            % (abs(term), rtol * abs(ans))
            )
    log_ans = log(ans) - x + a * log(x) - lgamma(a)
    return exp(log_ans)

cdef double gammaQ_cf(double a, double x, double rtol, int itmax):
    """ Continuing fraction expansion for Q(a, x) (for x > a+1).

    Q(a, x) = 1/Gamma(a) * \int_x^\infty dt exp(-t) t ** (a-1) = 1 - P(a, x)
    Uses Lentz's algorithm for continued fractions.
    """
    cdef double tiny = 1e-30
    cdef double den, Cj, Dj, fj
    cdef int j
    den = x + 1. - a
    if abs(den) < tiny:
        den = tiny
    Cj = x + 1. - a + 1. / tiny
    Dj = 1 / den
    fj = Cj * Dj * tiny
    for j in range(1, itmax):
        aj = - j * (j - a)
        bj = x + 2 * j + 1. - a
        Dj = bj + aj * Dj
        if abs(Dj) < tiny:
            Dj = tiny
        Dj = 1. / Dj
        Cj = bj + aj / Cj
        if abs(Cj) < tiny:
            Cj = tiny
        fac = Cj * Dj
        fj = fac * fj
        if abs(fac-1) < rtol:
            break
    # else:
    #     warnings.warn(
    #         'gammaQ convergence not complete -- want: %.3g << %.3g'
    #         % (abs(fac-1), rtol)
    #         )
    return exp(log(fj) - x + a * log(x) - lgamma(a))

def gammaQ(double a, double x, double rtol=1e-5, int itmax=10000):
    """ Complement of normalized incomplete gamma function, Q(a,x).

    Q(a, x) = 1/Gamma(a) * \int_x^\infty dt exp(-t) t ** (a-1) = 1 - P(a, x)
    """
    if x < 0 or a < 0:
        raise ValueError('negative argument: %g, %g' % (a, x))
    if x == 0:
        return 1.
    elif a == 0:
        return 0.
    if x < a + 1.:
        return 1. - gammaP_ser(a, x, rtol=rtol, itmax=itmax)
    else:
        return gammaQ_cf(a, x, rtol=rtol, itmax=itmax)

def gammaP(double a, double x, double rtol=1e-5, itmax=10000):
    """ Normalized incomplete gamma function, P(a,x).

    P(a, x) = 1/Gamma(a) * \int_0^x dt exp(-t) t ** (a-1) = 1 - Q(a, x)
    """
    if x < 0 or a < 0:
        raise ValueError('negative argument: %g, %g' % (a, x))
    if x == 0:
        return 0.
    elif a == 0:
        return 1.
    if x < a + 1.:
        return gammaP_ser(a, x, rtol=rtol, itmax=itmax)
    else:
        return 1. - gammaQ_cf(a, x, rtol=rtol, itmax=itmax)


def regulate(g, eps=None, svdcut=None, wgts=False, noise=False):
    """ Regulate singularities in the correlation matrix of the |GVar|\s in ``g``.

    Standard usage is, for example, ::

        import gvar as gv
        
        # with eps cutoff
        gmod = gv.regulate(g, eps=1e-6)

        # or with svd cutoff
        gmod = gv.regulate(g, svdcut=1e-6)
    
    where ``g`` is an array of |GVar|\s or a dictionary containing |GVar|\s
    and/or arrays of |GVar|\s. The result ``gmod`` is a copy of ``g`` 
    whose |GVar|\s have been modified to make their correlation matrix 
    less singular than that of the original ``g``. Parameter ``eps`` or
    ``svdcut`` specifies the extent of the modification.
  
    The modification of ``g`` is implemented by adding a set of |GVar|\s
    with zero means::

        gmod = g + gmod.correction

    where ``gmod.correction`` is an array/dictionary containing the |GVar|\s.
    When ``g`` is an array and ``eps`` is specified, for example, 
    ``g[i]`` is modified by adding::

        gmod.correction[i] = gv.gvar(0, (eps * norm) ** 0.5 * g[i].sdev)

    where ``norm = numpy.linalg.norm(corr, numpy.inf)`` is an estimate
    of the largest eigenvalue of the correlation matrix ``corr``. This
    correction typically has a negligible effect on the final standard 
    deviations (relative order ``eps*norm/2``), but can make noticeable 
    changes in the correlation matrix for highly correlated data. Strong 
    correlations lead to small eigenvalues in the correlation matrix, 
    and these are significantly increased by the cutoff, which in effect 
    replaces each eigenvalue ``eig`` by  ``eig + eps * norm``. Only members 
    of ``g`` that are correlated with other members of ``g`` are modified; 
    uncorrelated members are left unchanged (i.e., the correction 
    is ``gv.gvar(0,0)``).

    Adding ``gmod.correction`` to ``g`` increases the uncertainties in ``g``
    but does not affect random fluctuations in its mean values. 
    If parameter ``noise=True``, random noise is included in
    ``gmod.correction``, ::

        gmod.correction += gv.sample(gmod.correction),

    before it is added to ``g``. This adds random fluctuations to the means
    in ``gmod`` that are commensurate with the additions to the uncertainties.
    This is important, for example, when interpreting a ``chi**2`` 
    involving ``gmod``, since ``chi**2`` compares fluctuations in 
    the means with the uncertainties. 

    Specifying ``svdcut`` rather than ``eps`` regulates the correlation
    matrix using :func:`gvar.svd`: calling ``gvar.regulate(g, svdcut=1e-4)`` 
    is the same as calling ``gvar.svd(g, svdcut=1e-4)``. When ``svdcut>=0``,
    each eigenvalue ``eig`` of the correlation matrix is
    replaced by ``max(eig, svdcut * max_eig)`` where ``max_eig`` is
    the largest eigenvalue. See the :func:`gvar.svd` documentation for 
    more details. SVD cuts are numerically more robust, but more costly, 
    especially for large systems.

    There are a variety of reasons for regulating the correlation matrices. 
    Roundoff error, for example, can make the smallest eigenvalues unreliable
    and destabilize calulations involving the correlation matrix. A similar 
    situation arises when the correlation matrix is estimated from a small set
    of random data. This can result in small eigenvalues that are badly 
    underestimated.

    There is an additional parameter ``wgts`` in :func:`gvar.regulate` whose
    default value is ``False``. Setting ``wgts=1`` or ``wgts=-1`` instead
    causes :func:`gvar.regulate` to return a tuple ``(gmod, i_wgts)`` where
    ``gmod``  is the modified copy of ``g``, and ``i_wgts`` contains a
    decomposition of either the modified covariance matrix, when ``wgts=1``, or 
    the inverse of the modified covariance matrix, when ``wgts=-1``. 
    The covariance matrix is decomposed into non-overlapping sub-matrices,
    with ``i_wgts[0]`` containing the contributions from all 1x1 sub-matrices.
    Typical usage, for example to compute the inverse ``inv_cov`` of the 
    ``gmod`` covariance matrix, is::

        gmod, i_wgts = gvar.regulate(g, eps=1e-6, wgts=-1)
        inv_cov = numpy.zeros((gmod.size, gmod.size))
        # 1x1 sub-matrices
        i, wgts = i_wgts[0]
        inv_cov[i, i] = wgts ** 2
        # nxn sub-matrices (n>1)
        for i, wgts in i_wgts[1:]:
            inv_cov[i[:, None], i] = wgts.T @ wgts
    
    Similarly, we can  compute the expectation value, ``u.dot(inv_cov.dot(v))``, 
    between two vectors (:mod:`numpy` arrays) using::

        result = 0.0
        i, wgts = i_wgts[0]                       
        # 1x1 sub-matrices
        if len(i) > 0:
            result += numpy.sum((u[i] * wgts) * (v[i] * wgts))
        # nxn sub-matrices (n>1)
        for i, wgts in i_wgts[1:]:                
            result += numpy.sum(wgts.dot(u[i]) * wgts.dot(v[i]))

    where ``result`` is the desired expectation value. 
    
    These decompositions are useful for least squares fitting and simulating 
    correlated data. A Cholesky decomposition is used when ``eps`` is specified,
    while an SVD decomposition is used with ``svdcut``.

    Args:
        g: An array of |GVar|\s or a dicitionary whose values are
            |GVar|\s and/or arrays of |GVar|\s.
        eps (float): The diagonal elements of the ``g`` covariance matrix
            are multiplied by ``1 + eps*norm`` where ``norm`` is the norm of
            the correlation matrix, ``numpy.linalg.norm(corr, numpy.inf)``. 
            Ignored if ``svdcut`` is specified (and not ``None``).
        svdcut (float): If positive, replaces eigenvalues ``eig``
            of the correlation matrix with ``max(eig, svdcut * max_eig)``
            where ``max_eig`` is the largest eigenvalue; if negative, discards
            eigenmodes with eigenvalues smaller than ``|svdcut| * max_eig``.
            Note ``|svdcut| < 1``. Default is 1e-12.
        wgts: Setting ``wgts=1`` causes :func:`gvar.regulate` to compute
            and return a decomposition of the covariance matrix of
            the modified |GVar|\s, ``gmod``. Setting ``wgts=-1`` results in
            a decomposition of the inverse of the covariance matrix. The
            default value is ``False``, in which case only ``gmod`` is returned.
        noise (bool): If ``True``, noise is added to the correction (see
            above). Default is ``False``.

    Returns:
        A copy ``gmod`` of ``g`` with the modified correlation matrix.
        If ``wgts`` is not ``False``, a tuple ``(g, i_wgts)`` is returned 
        where ``i_wgts`` contains a decomposition of the ``gmod``
        covariance matrix or its inverse (see above).

    Additional information is stored in ``gmod``:

    .. attribute:: gmod.correction
        :noindex:

        Array or dictionary containing the SVD corrections added to ``g``
        to create ``gmod``: ``gmod = g + gmod.correction``.

    .. attribute:: gmod.eps
        :noindex:

        ``eps`` used to create ``gmod`` if set (otherwise ``None``).

    .. attribute:: gmod.svdcut
        :noindex:

        ``svdcut`` used to create ``gmod`` if set (otherwise ``None``).

    .. attribute:: gmod.dof
        :noindex:

        Number of degrees of freedom in ``gmod``.

    .. attribute:: gmod.nmod
        :noindex:

        Number of members of ``gmod`` modified by regulation. 

    .. attribute:: gmod.nblocks
        :noindex:

        A dictionary where ``gmod.nblocks[s]`` contains the number of
        block-diagonal ``s``-by-``s`` sub-matrices in the correlation
        matrix. Useful for debugging.

    .. attribute:: gmod.logdet
        :noindex:

        Logarithm of the determinant of the covariance matrix after ``eps``
        regulation is applied.
    """
    cdef numpy.ndarray[INTP_TYPE, ndim=1] idx 
    cdef numpy.ndarray[numpy.float_t, ndim=2] block_cov 
    cdef numpy.ndarray[numpy.float_t, ndim=1] block_sdev, D, sd, mn, z, Dinv
    cdef numpy.ndarray[object, ndim=1] correction
    cdef double root_eps_norm, logdet
    # which cutoff?
    if eps is None and svdcut is None:
        svdcut = 1e-12   # default
    if svdcut is not None:
        ans = svd(g, wgts=wgts, svdcut=svdcut, noise=noise)
        if wgts is False:
            ans.svdcut = svdcut 
            ans.eps = None
        else:
            ans[0].svdcut = svdcut 
            ans[0].eps = None
        return ans
    # eps must be not None, from above ifs
    if eps < 0:
        raise ValueError('negative eps = ' + str(eps))
    # ... continue with eps cutoff
    # replace g by a copy of g
    if hasattr(g,'keys'):
        is_dict = True
        g = _gvar.BufferDict(g)
    else:
        is_dict = False
        class regarray(numpy.ndarray):
            def __new__(cls, inputarray):
                obj = numpy.array(g).view(cls)
                return obj
        g = regarray(g)
    # begin setup and analysis
    g.dof = g.size
    g.nmod = 0 
    g.eps = eps
    g.svdcut = None
    correction = numpy.empty(len(g.flat), object)
    correction[:] = 0 * _gvar.gvar(0, 1)
    i_wgts = [([], [])]     # 1st entry, for all 1x1 blocks 
    logdet = 0
    g.nblocks = {}
    idx_bcov = evalcov_blocks(g.flat, compress=True)
    # uncorrelated parts
    idx, block_sdev = idx_bcov[0]
    if len(idx) > 0:
        g.nblocks[1] = len(idx)
        if wgts is not False:
            if numpy.any(block_sdev <= 0):
                logdet = -numpy.inf
            else:
                logdet = 2 * numpy.sum(numpy.log(block_sdev))
            i_wgts[0][0].extend(idx)
            i_wgts[0][1].extend(block_sdev ** wgts)
    # correlated parts
    for idx, block_cov in idx_bcov[1:]:
        g.nblocks[len(idx)] = g.nblocks.get(len(idx), 0) + 1
        D = block_cov.diagonal() ** 0.5
        Dinv = 1 / D
        corr = Dinv[:, None] * block_cov * Dinv[None, :]
        if eps > 0:
            root_eps_norm = (eps * numpy.linalg.norm(corr, numpy.inf)) ** 0.5
            if noise:
                sd = root_eps_norm * D
                correction[idx] = _gvar.gvar(numpy.random.normal(0, sd), sd)
            else:
                correction[idx] = _gvar.gvar(numpy.zeros(D.size), root_eps_norm * D)
            g.nmod += len(idx)
            g.flat[idx] += correction[idx] 
        if wgts is not False:
            if eps > 0:
                corr[numpy.diag_indices_from(corr)] += root_eps_norm
            ex = False
            try:
                U = _cholesky(corr, lower=False)
            except numpy.linalg.LinAlgError as e:
                ex = e
            if ex:
                raise numpy.linalg.LinAlgError(
                    'cholesky decomposition failed -- try larger eps'
                    )
            if wgts == 1:
                i_wgts.append((idx, U[:, :] * D[None, :]))
            elif wgts == -1:
                Uinv = _solve_triangular( 
                    U.T, numpy.eye(len(corr)), trans=1, lower=True
                    # U, np.eye(len(corr)),  lower=False # worse for Fortran (!?)
                    )
                i_wgts.append((idx, Uinv.T * Dinv[None, :]))
            else:
                raise ValueError('invalid wgts = ' + str(wgts))
            logdet += 2 * numpy.sum(numpy.log(U.diagonal() * D))
    # format the correction
    if is_dict:
        g.correction = _gvar.BufferDict(g, buf=correction)
    else:
        g.correction = correction.reshape(g.shape)
    if wgts is False:
        return g
    else:
        g.logdet = logdet
        # repack i_wgts into numpy arrays
        tmp = []
        for iw, wgts in i_wgts:
            tmp.append(
                (numpy.array(iw, numpy.intp), numpy.array(wgts, numpy.double))
                )
        i_wgts = tmp
        return g, i_wgts

def svd(g, svdcut=1e-12, wgts=False, noise=False, add_svdnoise=None):
    """ Apply SVD cuts to collection of |GVar|\s in ``g``.

    Standard usage is, for example, ::

        import gvar as gv
        ...
        gmod = gv.svd(g, svdcut=1e-4)

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

    The modification of ``g``'s covariance matrix is implemented by adding
    (to ``g``) a set of |GVar|\s with zero means::

        gmod = g + gmod.correction

    where ``gmod.correction`` is an array/dictionary containing the |GVar|\s.

    Adding ``gmod.correction`` to ``g`` increases the uncertainties in ``g``
    but does not affect random fluctuations in the mean values. 
    If parameter ``noise=True``, random noise is included in
    ``gmod.correction``, ::

        gmod.correction += gv.sample(gmod.correction),

    before it is added to ``g``. This adds random fluctuations to the means
    in ``gmod`` that are commensurate with the additions to the uncertainties.
    This is important, for example, when interpreting a ``chi**2`` 
    involving ``gmod``, since ``chi**2`` tests whether fluctuations in 
    the means are consistent with the uncertainties. 

    When ``svdcut`` is negative, eigenmodes of the correlation matrix
    whose eigenvalues are smaller than ``|svdcut| * max_eig`` are dropped
    from the new matrix and the corresponding components of ``g`` are
    zeroed out (that is, replaced by 0(0)) in ``gmod``.

    There is an additional parameter ``wgts`` in :func:`gvar.svd` whose
    default value is ``False``. Setting ``wgts=1`` or ``wgts=-1`` instead
    causes :func:`gvar.svd` to return a tuple ``(gmod, i_wgts)`` where
    ``gmod``  is the modified copy of ``g``, and ``i_wgts`` contains an SVD
    decomposition of either the modified covariance matrix, when ``wgts=1``, or 
    the inverse of the modified covariance matrix, when ``wgts=-1``. 
    The covariance matrix is decomposed into non-overlapping sub-matrices,
    with ``i_wgts[0]`` containing the contributions from all 1x1 sub-matrices.
    Typical usage, for example to compute the inverse ``inv_cov`` of the 
    ``gmod`` covariance matrix, is::

        gmod, i_wgts = gvar.svd(g, svdcut=1e-6, wgts=-1)
        inv_cov = numpy.zeros((gmod.size, gmod.size))
        # 1x1 sub-matrices
        i, wgts = i_wgts[0]
        inv_cov[i, i] = wgts ** 2
        # nxn sub-matrices (n>1)
        for i, wgts in i_wgts[1:]:
            inv_cov[i[:, None], i] = wgts.T @ wgts
    
    Similarly, we can  compute the expectation value, ``u.dot(inv_cov.dot(v))``, 
    between two vectors (:mod:`numpy` arrays) using::

        result = 0.0
        i, wgts = i_wgts[0]                       
        # 1x1 sub-matrices
        if len(i) > 0:
            result += numpy.sum((u[i] * wgts) * (v[i] * wgts))
        # nxn sub-matrices (n>1)
        for i, wgts in i_wgts[1:]:                
            result += numpy.sum(wgts.dot(u[i]) * wgts.dot(v[i]))

    where ``result`` is the desired expectation value. The SVD 
    decompositions are useful for least squares fitting and simulating 
    correlated data.

    Args:
        g: An array of |GVar|\s or a dicitionary whose values are
            |GVar|\s and/or arrays of |GVar|\s.
        svdcut (None or float): If positive, replace eigenvalues ``eig``
            of the correlation matrix with ``max(eig, svdcut * max_eig)``
            where ``max_eig`` is the largest eigenvalue; if negative, discard
            eigenmodes with eigenvalues smaller than ``|svdcut| * max_eig``.
            Note ``|svdcut| < 1``. Default is 1e-12.
        wgts: Setting ``wgts=1`` causes :func:`gvar.svd` to compute
            and return a spectral decomposition of the covariance matrix of
            the modified |GVar|\s, ``gmod``. Setting ``wgts=-1`` results in
            a decomposition of the inverse of the covariance matrix. The
            default value is ``False``, in which case only ``gmod`` is returned.
        noise: If ``True``, noise is added to the SVD correction (see
            above). Default is ``False``.

    Returns:
        A copy ``gmod`` of ``g`` whose correlation matrix is modified by
        SVD cuts. If ``wgts`` is not ``False``,
        a tuple ``(g, i_wgts)`` is returned where ``i_wgts``
        contains an SVD decomposition of ``gmod``'s
        covariance matrix or its inverse.

    Data from the SVD analysis is stored in ``gmod``:

    .. attribute:: gmod.svdcut
        :noindex:

        SVD cut used to create ``gmod``.

    .. attribute:: gmod.dof
        :noindex:

        Number of independent degrees of freedom left after the
        SVD cut. This is the same as the number initially unless
        ``svdcut < 0`` in which case it may be smaller.

    .. attribute:: gmod.nmod
        :noindex:

        Number of modes whose eignevalue was modified by the
        SVD cut.

    .. attribute:: gmod.nblocks
        :noindex:

        A dictionary where ``gmod.nblocks[s]`` contains the number of
        block-diagonal ``s``-by-``s`` sub-matrices in the correlation
        matrix.

    .. attribute:: gmod.eigen_range
        :noindex:

        Ratio of the smallest to largest eigenvalue before SVD cuts are
        applied (but after rescaling).

    .. attribute:: gmod.logdet
        :noindex:

        Logarithm of the determinant of the covariance matrix after SVD
        cuts are applied (excluding any omitted modes when
        ``svdcut < 0`` and any diagonal zero modes).

    .. attribute:: gmod.correction
        :noindex:

        Array or dictionary containing the SVD corrections added to ``g``
        to create ``gmod``: ``gmod = g + gmod.correction``.
    """
    cdef numpy.ndarray[INTP_TYPE, ndim=1] idx 
    cdef numpy.ndarray[numpy.float_t, ndim=2] block_cov 
    cdef numpy.ndarray[numpy.float_t, ndim=1] block_sdev, D
    cdef numpy.ndarray[object, ndim=1] correction
    # legacy keyword
    if add_svdnoise is not None:
        noise = add_svdnoise
    # replace g by a copy of g
    if hasattr(g,'keys'):
        is_dict = True
        g = _gvar.BufferDict(g)
    else:
        is_dict = False
        class svdarray(numpy.ndarray):
            def __new__(cls, inputarray):
                obj = numpy.array(g).view(cls)
                return obj
        g = svdarray(g)
    idx_bcov = _gvar.evalcov_blocks(g.flat, compress=True)
    g.logdet = 0.0
    correction = numpy.zeros(len(g.flat), object)
    correction[:] = 0 * _gvar.gvar(0, 1)
    g.eigen_range = 1.
    g.nmod = 0
    if wgts is not False:
        i_wgts = [([], [])] # 1st entry for all 1x1 blocks
    lost_modes = 0
    g.nblocks = {}
    # uncorrelated parts
    idx, block_sdev = idx_bcov[0]
    if len(idx) > 0:
        g.nblocks[1] = len(idx)
        if numpy.any(block_sdev == 0):
            g.logdet = -numpy.inf 
        else:
            g.logdet += 2 * numpy.sum(numpy.log(block_sdev))
        if wgts is not False:
            i_wgts[0][0].extend(idx)
            i_wgts[0][1].extend(block_sdev ** wgts)
    # correlated parts
    for idx, block_cov in idx_bcov[1:]:
        g.nblocks[len(idx)] = g.nblocks.get(len(idx), 0) + 1
        s = SVD(block_cov, svdcut=svdcut, rescale=True, compute_delta=True)
        if s.D is not None:
            g.logdet -= 2 * sum(numpy.log(di) for di in s.D)
        g.logdet += sum(numpy.log(vali) for vali in s.val)
        g.nmod += s.nmod
        if s.delta is not None:
            if noise:
                for vali, valorigi, veci in zip(s.val, s.valorig, s.vec):
                    if vali > valorigi:
                        # add next(raniter(s.delta)) to s.delta in correction
                        s.delta += (veci / s.D) * (
                            numpy.random.normal(0.0, (vali - valorigi) ** 0.5)
                            )
            correction[idx] = s.delta
            g.flat[idx] += s.delta
        elif svdcut is not None and svdcut < 0:
            newg = numpy.zeros(len(idx), object)
            for veci in s.vec:
                veci_D = veci / s.D
                newg += veci_D * (veci.dot(s.D * g.flat[idx]))
            lost_modes += len(idx) - len(s.vec)
            g.flat[idx] = newg
        if wgts is not False:
            i_wgts.append(
                (idx, numpy.array(s.decomp(wgts)[::-1]))
                # (idx, numpy.array([w for w in s.decomp(wgts)[::-1]]))
                )
        if s.eigen_range < g.eigen_range:
            g.eigen_range = s.eigen_range
    g.nmod += lost_modes
    g.dof = len(g.flat) - lost_modes
    g.svdcut = svdcut

    # repackage correction
    if is_dict:
        g.correction = _gvar.BufferDict(g, buf=correction)
    else:
        g.correction = correction.reshape(g.shape)

    ##### for legacy code (don't use)
    g.svdcorrection = g.correction
    svd.dof = g.dof
    svd.nmod = g.nmod
    svd.eigen_range = g.eigen_range
    svd.logdet = g.logdet
    svd.svdcorrection = g.correction.flat[:]
    svd.nblocks = g.nblocks
    ##### end of legacy code

    # done
    if wgts is not False:
        # repack into numpy arrays
        tmp = []
        for iw, wgts in i_wgts:
            tmp.append(
                (numpy.array(iw, numpy.intp), numpy.array(wgts, numpy.double))
                )
        i_wgts = tmp
        return (g, i_wgts)
    else:
        return g

def erf(x):
    """ Error function.

    Works for floats, |GVar|\s, and :mod:`numpy` arrays.
    """
    cdef int i
    if isinstance(x, GVar):
        f = _scipy_erf(x.mean)
        dfdx = 2. * numpy.exp(- x.mean ** 2) / numpy.sqrt(numpy.pi)
        return _gvar.gvar_function(x, f, dfdx)
    if hasattr(x, 'erf'):
        return x.erf()
    x = numpy.asarray(x)
    if x.dtype != object:
        return _scipy_erf(x)
    ans = numpy.empty(x.shape, x.dtype)
    for i in range(x.size):
        ans.flat[i] = erf(x.flat[i])
    return ans
