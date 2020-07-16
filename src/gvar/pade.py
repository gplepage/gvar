""" Pade approximants for GVars.  """
# Created by G. Peter Lepage on 2009-12-14.
# Copyright (c) 2009-2020 G. Peter Lepage.
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

import sys
import numpy 
import gvar as _gvar 
import scipy.linalg 
from scipy.interpolate import pade as _scipy_pade

class Pade(object):
    """ Pade approximant to ``sum_i f[i] x**i`` for ``GVar``\s.

    The ``order=(m,n)`` Pade approximant to a series given by
    ``sum_i f[i] * x**i`` is the ratio of  polynomials of order ``m``
    (numerator) and ``n`` (denominator) whose  Taylor expansion agrees
    with that of the original series up to order ``m+n``.

    A :class:`Pade` object ``pade`` creates :class:`gvar.powerseries.PowerSeries`
    objects for the numerator (``pade.num``) and the denominator (``pade.den``)
    of the Pade approximant corresponding to the input series ``f[i]``. The 
    approximant can be evaluated for arbitrary ``x`` using ``pade(x)``. The
    coefficients used in the numerator and denominator are given by ``pade.num.c``
    and ``pade.den.c``, respectively.

    Elements in the series ``f[i]`` may be numbers or :class:`gvar.GVar`\s.
    When the latter appear, the code uses an SVD algorithm (see :func:`pade_svd`) 
    to deal with the imprecision in the input data. It automatically reduces
    the order of the approximant if the extraction of Pade coefficients
    is too unstable given the noise in the input data. The actual order used 
    in the approximant is given by ``pade.order``.

    Examples:
        Typical usage is::

            >>> import gvar as gv
            >>> c = gv.gvar(['1(0)', '1.0(1)', '0.500(10)', '0.1667(100)','.04167(100)'])
            >>> pade = gv.pade.Pade(c, order=(2,2))
            >>> print(pade.num.c, pade.den.c)               # num and den coefficients
            [1(0) 0.50(14) 0.08(11)] [1(0) -0.50(14) 0.083(55)]
            >>> print(pade(1))                              # evaluate pade at x=1
            2.714(99)
            >>> print(pade.num(1), pade.den(1), pade.order) # evaluate num and den at x=1
            1.58(21) 0.583(82) (2, 2) 

        When errors are larger (see ``c[0]``), the order may be automatically 
        reduced (here to ``(1,2)``)::

            >>> c = gv.gvar(['1.0(1)', '1.0(1)', '0.500(10)', '0.1667(100)','.04167(100)'])
            >>> pade = gv.pade.Pade(c, order=(2,2))
            >>> print(pade.num.c, pade.den.c)               # num and den coefficients
            [1.00(10) 0.33(29)] [1(0) -0.67(17) 0.17(11)]
            >>> print(pade(1))                              # evaluate pade at x=1
            2.67(20)
            >>> print(pade.num(1), pade.den(1), pade.order) # evaluate num and den at x=1
            1.33(27) 0.500(69) (1, 2)
 
    Args:
        f: Array ``f[i]`` of power series coefficients for ``i=0...n+m``.
        order (tuple): ``order=(m,n)`` specifies that the numerator and 
            denominator of the Pade approximant have maximum order 
            ``m`` and ``n``, respectively.
        rtol (float or str): Relative uncertainty in the coefficients, for
            use by the SVD algorithm. If ``rtol`` is a string, it determines 
            how the relative tolerance is determined from the relative
            uncertainties in the ``f[i]``. Set ``rtol`` equal to:
            ``'gavg'`` for the geometric mean (default); ``'avg'`` for
            the average; ``'min'`` for the minimum; or ``'max'`` for
            the maximum. Otherwise a number can be specified, in which case
            the uncertainties in ``f[i]`` are ignored. The SVD analysis
            is skipped if ``rtol=None``.
    """
    def __init__(self, f, order, rtol='gavg'):
        # check inputs
        m, n = order
        assert m >= 0 and n >= 0
        f = f[:n + m + 1]
        if len(f) < (m + n + 1):
            raise ValueError(
                'not enough f[i]s -- need {} have {}'.format(n + m + 1, len(f))
                )
        if not numpy.any([isinstance(fi, _gvar.GVar) for fi in f]):
            if rtol is None:
                p, q = Pade._scipy_pade(f, n)
            else:
                self.rtol = 1e-14
                p, q = pade_svd(f, m, n, rtol=self.rtol)
            self.num = _gvar.powerseries.PowerSeries(p)
            self.den = _gvar.powerseries.PowerSeries(q)
            self.order = [len(p) - 1, len(q) - 1]
            return
        else:
            c = numpy.array(f)

        # compute tolerance if not specified
        if rtol in ['avg', 'min', 'max', 'gavg']:
            means = numpy.fabs(_gvar.mean(c))
            sdevs = _gvar.sdev(c)
            idx = means > 0.0
            if numpy.any(idx) and numpy.all(sdevs[idx] > 0):
                ratio = sdevs[idx] / means[idx]
                if rtol == 'gavg':
                    # geometric mean
                    rtol = numpy.exp(
                        numpy.average(numpy.log(ratio))
                        )
                elif rtol == 'avg':
                    rtol = numpy.average(ratio)
                elif rtol == 'min':
                    rtol = numpy.min(ratio)
                else:
                    rtol = numpy.max(ratio)
            else:
                rtol = 1e-14
        elif rtol is not None:
            rtol = numpy.fabs(rtol)
        self.rtol = rtol

        # find Pade coefficients
        if self.rtol is None:
            p, q = Pade._scipy_pade(_gvar.mean(c), n)
        else:
            p, q = pade_svd(_gvar.mean(c), m, n, rtol=self.rtol)
        m = len(p) - 1
        n = len(q) - 1

        # add uncertainties
        p = p * _gvar.gvar(len(p) * ['1(0)'])
        q = q[1:] * _gvar.gvar(len(q[1:]) * ['1(0)'])
        num = _gvar.powerseries.PowerSeries(p, order=m + n)
        den = _gvar.powerseries.PowerSeries([1] + list(q), order=m + n)
        pq = numpy.concatenate((p,q))
        cc = (num / den).c
        M = numpy.empty((len(pq), len(pq)), float)
        for i in range(len(pq)):
            for j in range(len(pq)):
                M[i, j] = cc[i].deriv(pq[j])
        pq = pq + _gvar.linalg.solve(M, (c - _gvar.mean(c))[:len(pq)])
        self.num = _gvar.powerseries.PowerSeries(pq[:m + 1])
        self.den = _gvar.powerseries.PowerSeries([_gvar.gvar(1,0)] + list(pq[m + 1:]))
        self.order = (m, n)

    @staticmethod
    def _scipy_pade(c, n):
        p, q = _scipy_pade(c, n)
        return numpy.array(p.c[::-1]), numpy.array(q.c[::-1])

    def __call__(self, x):
        return self.num(x) / self.den(x)

def pade_gvar(f, m, n, rtol='gavg'):  
    """ ``(m,n)`` Pade approximant to ``sum_i f[i] x**i`` for ``GVar``\s.

    The ``(m,n)`` Pade approximant to a series given by
    ``sum_i f[i] * x**i`` is the ratio of  polynomials of order ``m``
    (numerator) and ``n`` (denominator) whose  Taylor expansion agrees
    with that of the original series up to order ``m+n``.

    This code uses an SVD algorithm (see :func:`pade_svd`) to deal with
    imprecision in the input data. It automatically reduces
    the order of the approximant if the extraction of Pade coefficients
    is too unstable given noise in the input data.

    Args:
        f: Array ``f[i]`` of power series coefficients for ``i=0...n+m``.
        m: Maximum order of polynomial in numerator of Pade
            approximant (``m>=0``).
        n: Maximum order of polynomial in denominator of Pade
            approximant (``m>=0``).
        rtol (float or str): If ``rtol`` is a string, it determines how the
            relative tolerance is determined from the relative
            uncertainties in the ``f[i]``. Set ``rtol`` equal to:
            ``'gavg'`` for the geometric mean (default); ``'avg'`` for
            the average; ``'min'`` for the minimum; or ``'max'`` for
            the maximum. Otherwise a number can be specified, in which case
            the uncertainties in ``f[i]`` are ignored. The SVD analysis is 
            skipped if ``rtol=None``.
    Returns:
        Tuple of power series coefficients ``(p, q)`` such that
        ``sum_i p[i] x**i`` is the numerator of the approximant,
        and ``sum_i q[i] x**i`` is the denominator. ``q[0]`` is
        normalized to 1.
    """
    pade = Pade(f, order=(m,n), rtol=rtol)
    return (pade.num.c, pade.den.c)

def pade_svd(f, m, n, rtol=1e-14):
    """ ``(m,n)`` Pade approximant to ``sum_i f[i] x**i``.

    The ``(m,n)`` Pade approximant to a series given by
    ``sum_i f[i] * x**i`` is the ratio of  polynomials of order ``m``
    (numerator) and ``n`` (denominator) whose  Taylor expansion agrees
    with that of the original series up to order ``m+n``.

    This code is adapted from P. Gonnet,  S. Guttel, L. N. Trefethen, SIAM
    Review Vol 55, No. 1, 101 (2013). It uses an SVD algorithm to deal with
    imprecision in the input data,  here specified by the relative tolerance
    ``rtol`` for the  input coefficients ``f[i]``. It automatically reduces
    the order of the approximant if the extraction of Pade coefficients
    is too unstable given tolerance ``rtol``.

    Args:
        f: Array ``f[i]`` of power series coefficients for ``i=0...n+m``.
        m: Maximum order of polynomial in numerator of Pade
            approximant (``m>=0``).
        n: Maximum order of polynomial in denominator of Pade
            approximant (``m>=0``).
        rtol: Relative accuracy of input coefficients. (Default is 1e-14.)
    
    Returns:
        Tuple of power series coefficients ``(p, q)`` such that
        ``sum_i p[i] x**i`` is the numerator of the approximant,
        and ``sum_i q[i] x**i`` is the denominator. ``q[0]`` is
        normalized to 1.
    """
    linalg = scipy.linalg
    mn_save = m,n
    c = numpy.array(f[:n + m + 1], float)
    if len(f) < (m + n + 1):
        raise ValueError(
            'not enough f[i]s -- need {} have {}'.format(n + m + 1, len(f))
            )
    # if USE_SCIPY_PADE:
    #     p, q = scipy_pade(c, n)
    #     return numpy.array(p.c[::-1]), numpy.array(q.c[::-1])
    ts = rtol * linalg.norm(c)
    if linalg.norm(c[:m + 1]) <= rtol * linalg.norm(c):
        # return power series through order m
        a = numpy.array(c[:1])
        b = numpy.array([1.])
    else:
        row = numpy.zeros(n+1)
        row[0] = c[0]
        col = c
        while True:
            if n == 0:
                # return the power series through order m
                a = c[:m + 1]
                b = numpy.array([1.])
                return a, b
            Z = linalg.toeplitz(col[:m + n + 1], row[:n + 1])
            C = Z[m + 1:, :]
            rho = numpy.sum(linalg.svdvals(C) > ts)
            if rho == n:
                break
            m -= n - rho
            n = rho
            if m < 0:
                m = 0
        if n > 0:
            # use svd to get solution b, but only to normalize C
            # then use QR decomposition to get final b
            U, S, V = linalg.svd(C, full_matrices=True)
            b = V.transpose()[:, -1]
            D = numpy.diag(numpy.abs(b) + numpy.sqrt(sys.float_info.epsilon))
            Q,R = linalg.qr(C.dot(D).transpose())
            b = D.dot(Q)[:,-1]
            b = b / linalg.norm(b)
            a = Z[:m + 1, :n + 1].dot(b)
            lam = numpy.where(abs(b) > rtol)[0][0]
            b = b[lam:]
            a = a[lam:]
            b = b[:numpy.where(abs(b) > rtol)[0][-1] + 1]
        idx = abs(a) > ts
        if not numpy.any(idx):
            a = a[:1]
        else:
            a = a[:numpy.where(idx)[0][-1] + 1]
        a = a / b[0]
        b = b / b[0]
    # N.B.: an approximant for non-zero rtol is the
    # same as the reduced-order approximant evaluated with
    # zero rtol; any approximant returned by the algorithm should
    # be an exact approximant to the input. Thus rtol determines the
    # order of the final approximant, but does not affect the values
    # of the approximant's coefficients.
    mfinal = len(a) - 1
    nfinal = len(b) - 1
    return (a,b) if (mfinal,nfinal) == mn_save else pade_svd(f, mfinal, nfinal)
