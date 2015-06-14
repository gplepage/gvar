""" Function root-finding for GVars. """

# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2015 G. Peter Lepage. 
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
import warnings
import numpy
import gvar

def search(fcn, x0, incr=0, fac=1.1, maxit=100, analyzer=None):
    """ Search for and bracket root of one-dimensional function ``fcn(x)``.

    This method searches for an interval in ``x`` that brackets 
    a root of ``fcn(x)=0``. It examines points ::

        x[j + 1] = fac * x[j] + incr

    where ``x[0]=x0`` and ``j=0...maxit-1``, looking for a pair 
    of successive points where ``fcn(x[j])`` changes sign. These
    points bracket a root (assuming the function is continuous),
    providing a coarse estimate of the root. That estimate can
    be refined using :meth:`root.refine`.

    Example:
        The following code seeks to bracket the first zero of
        ``sin(x)`` with ``x>0.1``::

            >>> import math
            >>> import gvar as gv
            >>> interval = gv.root.search(math.sin, 0.1)
            >>> print(interval)
            (3.0912680532870755, 3.4003948586157833)

        The resulting interval correctly brackets the root 
        at ``pi``.

    Args:
        fcn: One dimenionsal function whose root is sought.
        x0 (float): Starting point for search.
        incr (float, optional): Increment used for linear searches. Default 
            value is 0.
        fac (float, optional): Rescaling factor for exponential searches. 
            Default value is 1.1.
        maxit (int, optional): Maximum number of steps allowed for search. An 
            exception is raised if a root is not found in time. Default
            value is 100.
        analyzer: Optional function ``f(x, fcn(x))`` that is called
            for each point ``x`` that is examined. This can be used,
            for example, to monitor the search while debugging. 
            Default is ``None``.

    Returns:
        Tuple ``(a, b)`` where ``fcn(a) * fcn(b) <= 0``, which implies
        that a root occurs between ``a`` and ``b`` (provided the 
        function is continuous). The tuple has extra attributes 
        that provide additional information about the search:

        - **nit** --- Number of iterations used to find interval ``(a,b)``.
        - **fcnval** --- Tuple containing the function values at ``(a,b)``.

    Raises:
        RuntimeError: If unable to find a root in ``maxit`` steps.
    """
    # custom return class for result
    class ans(tuple):
        def __new__(cls, interval, nit, fcnval):
            return tuple.__new__(cls, interval)
        def __init__(self, interval, nit, fcnval):
            self.interval = interval
            self.nit = nit 
            self.fcnval = fcnval
    x = x0
    f = fcn(x)
    for nit in range(maxit):
        xo, fo = x, f
        x = xo * fac + incr
        f = fcn(x)
        if analyzer!=None:
            analyzer(x, f)
        if f*fo<=0:
            if numpy.fabs(fo)<=numpy.fabs(f):
                return ans((xo, x), nit=nit + 1, fcnval=(fo, f))
            else:
                return ans((x, xo), nit=nit+1, fcnval=(f, fo))
    raise RuntimeError("unable to bracket root")


def refine(fcn, interval, rtol=None, maxit=1000, analyzer=None):
    """ Find root ``x`` of one-dimensional function ``fcn`` on an interval.

    This method finds a root ``x`` of ``fcn(x)=0`` inside an ``interval=(a,b)``
    that brackets the root, with ``fcn(a) * fcn(b) <= 0``.

    This method is a pure Python adaptation of an algorithm
    from Richard Brent's book "Algorithms for Minimization 
    without Derivatives" (1973). Being pure Python it works with
    :class:`gvar.GVar`-valued functions and variables.

    Example:
        The following code finds a root of ``sin(x)`` in the interval
        ``1 <= x <= 4``, using 7 iterative refinements of the initial
        interval::

            >>> import math
            >>> import gvar as gv
            >>> root = gv.root.refine(math.sin, (1, 4))
            >>> print(root)
            3.14159265359
            >>> print(root.nit)
            7

    Args:
        fcn: One-dimensional function whose zero/root is sought.
        interval: Tuple ``(a,b)`` specifying an interval containing
            the root, with ``fcn(a) * fcn(b) <= 0``. The search 
            for a root is confined to this interval.
        rtol (float, optional): Relative tolerance for the root. The default 
            value is ``None``, which sets ``rtol`` equal to machine
            precision (``sys.float_info.epsilon``). A larger value 
            usually leads to less precision but is faster.
        maxit (int, optional): Maximum number of iterations used to find 
            a root with the given tolerance. A warning is 
            issued if the algorithm does not converge in time.
            (Default value is 1000.)
        analyzer: Optional function ``f(x, fcn(x))`` that is called
            for each point ``x`` examined by the algorithm. This can 
            be used, for example, to monitor convergence while 
            debugging. Default is ``None``.

    Returns:
        The root, which is either a ``float`` or 
        a :class:`gvar.GVar` but with extra attributes that
        provide additional information about the root: 

        - **nit** --- Number of iterations used to find the root.

        - **interval** --- Smallest interval ``(b,c)`` found containing
          the root, where ``b`` is the root returned by the method.

        - **fcnval** --- Value of ``fcn(x)`` at the root.

    Raises:
        ValueError: If ``fcn(a) * fcn(b) > 0`` for initial 
            interval ``(a,b)``.
        UserWarning: If the algorithm fails to converge 
            after ``maxit`` iterations.
    """
    # Create custom class for answer
    def ans(root, interval, nit, fcnval):
        if isinstance(root, gvar.GVar):
            class _ans(gvar.GVar):
                def __init__(self, root):
                    super(_ans, self).__init__(*root.internaldata)
        else:
            class _ans(float):
                def __new__(cls, root):
                    return float.__new__(cls, root)
        root = _ans(root)
        root.interval = interval
        root.nit = nit 
        root.fcnval = fcnval
        return root

    # Throughout the routine: root is in interval (b,c), 
    # b is the best value, and a is the previous value of b.
    if rtol is None:
        rtol = sys.float_info.epsilon
    if rtol < 0:
        raise ValueError('negative rtol: {}'.format(rtol))
    a, b = interval 
    fa, fb = fcn(a), fcn(b)  
    if (fa > 0) == (fb > 0):
        raise ValueError("fcn(a)*fcn(b) is not negative for (a,b)=interval")
    if analyzer is not None:
        analyzer(a, fa)
        analyzer(b, fb)        
    # put a,b,c into canonical order by swapping as necessary.
    if numpy.fabs(fa) < numpy.fabs(fb):
        a, b = b, a 
        fa, fb = fb, fa
    c, fc = a, fa
    d = b - a 
    e = d
    for nit in range(maxit):
        tol = 2 * sys.float_info.epsilon * numpy.fabs(b) + rtol 
        m = 0.5 * (c - b) 
        if numpy.fabs(m) < tol or fb == 0:
            # found root
            # refine.fcnval = fb
            # refine.nit = nit
            # refine.interval = (b, c)
            return ans(b, fcnval=fb, nit=nit, interval=(b, c))
        # check for bisection
        if numpy.fabs(e) < tol or numpy.fabs(fa) <= numpy.fabs(fb):
            d = m 
            e = m
        else:
            s = fb / fa 
            if a == c:
                # linear interpolation
                p = 2 * m * s
                q = 1. - s
            else:
                # inverse quadratic interpolation
                q = fa / fc 
                r = fb / fc 
                p = s * (2. * m * q * (q-r) - (b - a) * (r - 1.))
                q = (q - 1.) * (r - 1.) * (s - 1.)
            if p > 0:
                q = -q 
            else:
                p = -p
            s = e
            e = d
            if 2 * p < 3 * m * q - numpy.fabs(tol * q) and p < numpy.fabs(0.5 * s * q):
                d = p / q 
            else:
                d = m 
                e = m
        a, fa = b, fb
        if numpy.fabs(d) > tol:
            b = b + d 
        elif m > 0:
            b = b + tol
        else:
            b = b - tol 
        fb = fcn(b)
        if analyzer is not None:
            analyzer(b, fb)
        # reorder a,b,c if necessary
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
            d = b - a 
            e = d
        if numpy.fabs(fc) < numpy.fabs(fb):
            a, b = b, c
            c = a
            fa, fb = fb, fc
            fc = fa 
    warnings.warn(
        "failed to converge in maxit={} iterations".format(maxit)
        ) 
    # refine.fcnval = fb
    # refine.nit = nit
    # refine.interval = (b, c)            
    return ans(b, fcnval=fb, nit=nit, interval=(b, c))
