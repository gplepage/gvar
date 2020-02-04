""" Cubic splines for GVars. """

# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2014-20 G. Peter Lepage.
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

import warnings
import numpy
import gvar as _gvar

class CSpline:
    """ Cubic spline approximation to a function.

    Given ``N`` values of a function ``yknot[i]`` at ``N`` points
    ``xknot[i]`` for ``i=0..N-1`` (the *knots* of the spline),
    the code ::

        from gvar.cspline import CSpline

        f = CSpline(xknot, yknot)

    defines a function ``f`` such that: a) ``f(xknot[i]) = yknot[i]`` 
    for all ``i``; and b) ``f(x)`` and its first derivative are 
    continuous. Function ``f(x)`` is a cubic polynomial 
    between the knots ``xknot[i]`` (determined from the values 
    and first derivatives at the adjacent knots). 
    Argument ``x`` in ``f(x)`` is a number or an array 
    of numbers (any shape). Numbers can be replaced 
    by :mod:`gvar.GVar` objects in ``x``, ``xknot`` 
    and/or ``yknot``.

    Derivatives and integrals of the spline function are also
    available:

        ``f.D(x)`` --- first derivative at ``x``;

        ``f.D2(x)`` --- second derivative at ``x``;

        ``f.D3(x)`` --- third derivative at ``x``;

        ``f.integ(x)`` --- integral from ``xknot[0]`` to ``x``.

    The derivatives are increasingly unreliable with increasing order. 

    Splines can be used outside the range covered by the defining
    ``xknot`` values. As this is often a bad idea, keyword parameter 
    ``warn`` can be set to ``True`` so that warnings are generated
    if the spline is used out of range. This keyword is set
    in the original constructor (``CSpline(x, y, warn=True)``) or in
    calls to a :mod:`CSpline` object (eg, ``f.D(x, warn=True)``).
    The spline value for an out-of-range point is calculated
    using a polynomial whose value and derivatives match those of the spline
    at the knot closest to the out-of-range point. The extrapolation
    polynomial is cubic by default, but lower orders are specified by
    setting parameter ``extrap_order`` to a (non-negative) integer
    less than 3.

    The first derivatives of ``f(x)`` at the knots are determined 
    from the function values in the vicinity of the knot. The derivatives
    at the endpoints (``xknot[0]`` and ``xknot[-1]`` if the knots are 
    ordered) can be specified explicitly, if they are known::

        f = CSpline(xknot, yknot, deriv=[dydx_i, dydx_f])

    where ``dydx_i`` is the derivative at ``xknot[0]`` and ``dydx_f`` 
    is the derivative at ``xknot[-1]``. 

    There are different types of spline available, selected with 
    keyword ``alg``:

        ``alg='steffen'``
            Monotonic cubic spline that has quadratic precision
            where the function is monotonic between knots. This is 
            the default algorithm.

        ``alg='cspline'``
            Cubic spline with continuous second derivatives ``f''(x)``
            (in addition to ``f'(x)`` and ``f(x)``). Sets 
            ``f''(x)`` to zero at the endpoints (*natural* boundary
            conditions) if the first derivatives have not been 
            specified there (using keyword ``deriv``).

        ``alg='pchip'``
            Monotonic cubic spline used in the :mod:`scipy`
            function :class:`scipy.interpolate.PchipInterpolator`.
            This algorithm is usually less accurate than ``'steffen'`` 
            where the function is smooth (and monotonic).

    The ``'cspline'`` algorithm gives the smoothest interpolations 
    and the most accuracy for very smooth functions, but it tends
    to overreact to abrupt jumps in the function, leading
    to unrealistic oscillations around the jump. Monotonic 
    splines are guaranteed to be monotonic between knots and so 
    are unable to overshoot the input data in this way. (The 
    monotonic splines may not be monotonic in the first or last
    intervals if derivatives are supplied for the endpoints,
    using the ``deriv`` keyword. Also third derivatives (``f.D3(x)``) 
    are unreliable for monotonic splines.)

    The algorithm is irrelevant when there are only two knots. In that case
    the spline is linear if no derivatives are specified (using ``deriv``), 
    quadratic if one or the other derivative is specified, or cubic
    if both derivatives are specified. 

    Examples:
        Typical usage is::

            >>> import gvar as gv
            >>> xknot = [0., 0.78539816, 1.57079633, 2.35619449, 3.14159265]
            >>> yknot = [0., 0.70710678, 1.0, 0.70710678, 0.]
            >>> f = gv.cspline.CSpline(xknot, yknot)
            >>> print(f(0.7), f.D(0.7), f.integ(0.7))
            0.644243383100598 0.7655924482958195 0.23496394264839268

        Here the ``yknot`` values were obtained by taking ``sin(xknot)``.
        Tabulating results from the spline together with the exact results
        shows that this 5-knot spline gives a pretty good approximation
        of the function ``sin(x)``, as well as its derivatives and integral::

            x    f(x)    f.D(x)  f.D2(x) f.integ(x) | sin(x)  cos(x)  1-cos(x)
            ------------------------------------------------------------------
            0.3  0.2951  0.9551  -0.2842 0.0446     | 0.2955  0.9553  0.0447
            0.5  0.4791  0.8793  -0.4737 0.1222     | 0.4794  0.8776  0.1224
            0.7  0.6442  0.7656  -0.6632 0.2350     | 0.6442  0.7648  0.2352
            0.9  0.7830  0.6176  -0.7891 0.3782     | 0.7833  0.6216  0.3784
            1.1  0.8902  0.4520  -0.8676 0.5461     | 0.8912  0.4536  0.5464
            1.3  0.9627  0.2706  -0.9461 0.7319     | 0.9636  0.2675  0.7325
            1.5  0.9974  0.0735  -1.0246 0.9286     | 0.9975  0.07074 0.9293

        Using the spline outside
        the range covered by the knots is less good::

            >>> print(f(2 * math.pi))
            1.7618635470106572

        The correct answer is 0.0, of course. Working just outside 
        the knot region is often fine, although it might be
        a good idea to limit the order of the polynomial
        used in such regions: for example, setting ::

            >>> f = gv.cspline.CSpline(xknot, yknot, extrap_order=2)

        implies that quadratic polynomials are used outside the spline range.

    Args:
        xknot (1-d sequence): Location of the 
            spline's knots, where the function values are 
            specified. The locations can be numbers of
            :mod:`gvar.GVar` objects.
        yknot (1-d sequence): Function values at 
            the locations specified by ``xknot[i]``. The 
            values can be numbers or
            :mod:`gvar.GVar` objects.
        deriv (2-component sequence): Derivatives at initial 
            and final boundaries of the  region specified 
            by ``xknot[i]``. The derivatives can be numbers or
            :mod:`gvar.GVar` objects. Default value is ``None`` 
            for each boundary, which implies *natural* boundary
            conditions (vanishing second derivative). 
        extrap_order (int): Order of polynomial used for 
            extrapolations outside of the spline range. 
            The polynomial is constructed from the spline's 
            value and derivatives at the (nearest) knot of the
            spline. The allowed range is ``0 <= extrap_order <= 3``. 
            The default value is 3 although it is common practice 
            to use smaller values.
        warn (bool): If ``True``, warnings are generated
            when the spline function is called for ``x`` values that
            fall outside of the original range of ``xknot``\s used to
            define the spline. Default value is ``False``, 
            which means out-of-range warnings are suppressed.
        alg (str): Spline algorithm used, which is one of:
            ``'steffen'`` (default), ``'cspline'``, and 
            ``'pchip'``. The first and last of these give 
            monotonic splines.

    """
    def __init__(self, xknot, yknot, deriv=(None, None), extrap_order=3, alg='steffen', warn=False):
        # sort and store arguments
        x, y = zip(*sorted(zip(xknot, yknot)))
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.deriv = deriv 
        self.alg = alg 
        self.warn = warn 
        self.extrap_order = extrap_order
        self.n = len(self.y)
        # initial estimates for derivatives
        h = self.x[1:] - self.x[:-1]
        if numpy.any(h == 0):
            raise ValueError('knots must be a different locations: x = ' + str(self.x))
        delta = (self.y[1:] - self.y[:-1]) / h
        # slopes at knots
        if self.n < 2:
            raise ValueError('spline needs more knots than ' + str(self.n))
        elif self.n == 2:
            self.dy = CSpline._2pt_slopes(delta, h, self.deriv)
        else:
            if alg == 'pchip':
                self.dy = CSpline._monotonic_slopes(delta, h, self.deriv)
            elif alg == 'cspline':
                self.dy = CSpline._cspline_slopes(delta, h, self.deriv)
            elif alg == 'steffen':
                self.dy = CSpline._steffen_slopes(delta, h, self.deriv)
            else:
                raise ValueError('undefined algorithm = ' + str(alg))
        # higher derivatives (times 1/n!) for start of each interval
        self.d2y =(3 * delta - 2 * self.dy[:-1] - self.dy[1:]) / h 
        self.d3y = (self.dy[:-1] - 2 * delta + self.dy[1:]) / h**2
        # integrals from x[0] to each x[i]
        self.intydx = numpy.zeros(self.dy.shape, self.dy.dtype)
        self.intydx[1:] = h * (
            self.y[:-1] + h * (self.dy[:-1] / 2. + h * (self.d2y / 3. + h * self.d3y / 4.))
            )
        self.intydx = numpy.cumsum(self.intydx)
        # taylor coefficients for expansions to left or right of range
        self.cleft=numpy.array(
                [self.y[0], self.D(self.x[0]), 0.5 * self.D2(self.x[0], warn=False)]
                )
        self.cright=numpy.array(
                [self.y[-1], self.D(self.x[-1]), 0.5 * self.D2(self.x[-1], warn=False)]
                )

    @staticmethod 
    def _2pt_slopes(delta, h, deriv):
        if deriv[0] == None and deriv[1] == None:
            return numpy.array([delta[0], delta[0]])
        elif deriv[0] != None and deriv[1] == None:
            return numpy.array([deriv[0], 2 * delta[0] - deriv[0]])
        elif deriv[1] != None and deriv[0] == None:
            return numpy.array([2 * delta[0] - deriv[1], deriv[1]])
        else:
            return numpy.array(deriv)

    @staticmethod
    def _cspline_slopes(delta, h, deriv):
        tmp = delta[:1] + (0 if deriv[0] is None else deriv[0]) + (0 if deriv[1] is None else deriv[1])
        dtype = object if tmp.dtype == object else float 
        n = len(delta) + 1
        a = numpy.zeros(n, dtype)
        b = numpy.zeros(n, dtype)
        c = numpy.zeros(n, dtype)
        d = numpy.zeros(n, dtype)
        # midpoints
        a[1:-1] = 1 / h[:-1]
        c[1:-1] = 1 / h[1:]
        b[1:-1] = 2 * (a[1:-1] + c[1:-1])
        d[1:-1] = 3 * (delta[:-1] * a[1:-1] + delta[1:] * c[1:-1])
        # endpoints (set slope or natural bc => d2y/dx2 = 0)
        if deriv[0] is None:
            # m[0, 0]
            b[0] = 2. / h[0]
            # m[0, 1]
            c[0] = 1. / h[0]
            d[0] = 3. * delta[0]  / h[0]
        else:
            b[0] = 1.
            d[0] = deriv[0]
        if deriv[1] is None:
            # m[-1, -2]
            a[-1] = 1 / h[-1]
            # m[-1, -1]
            b[-1] = 2. / h[-1]
            d[-1] = 3. * delta[-1] / h[-1]
        else:
            b[-1] = 1.
            d[-1] = deriv[1]
        return numpy.array(tri_diag_solve(a, b, c, d))

    @staticmethod
    def _monotonic_slopes(delta, h, deriv):
        # c.f. scipy PCHIP implementation
        # build m = dy/dx (default value is 0)
        m = numpy.zeros(len(delta) + 1, object if delta.dtype == object else float)
        # delta,h live on intervals; x,y,m on knots
        # exclude ends, and points where delta[i-1] and delta[i] differ in sign or vanish
        idx = numpy.arange(1, len(m) - 1)[(delta[1:] * delta[:-1]) > 0]
        w1 = 2 * h[idx] + h[idx - 1]
        w2 = h[idx] + 2 * h[idx - 1]
        m[idx] = (w1 + w2) / (w1 / delta[idx-1] + w2 / delta[idx])
        # endpoints
        for i, hi, deltai in [(0,  h[:2], delta[:2]), (-1, h[:-3:-1], delta[:-3:-1])]:
            if deriv[i] is not None:
                m[i] = deriv[i]
                continue
            m[i] = ((2 * hi[0] + hi[1]) * deltai[0] - hi[0] * deltai[1]) / (hi[0] + hi[1])
            if m[i] * deltai[0] <= 0:
                m[i] = 0
            elif deltai[0] * deltai[1] <= 0 and _gvar.fabs(m[i]) > 3 * _gvar.fabs(deltai[0]):
                m[i] =  3 * deltai[0]
        return m

    @staticmethod
    def _steffen_slopes(delta, h, deriv):
        # c.f. scipy PCHIP implementation
        # build m = dy/dx (default value is 0)
        m = numpy.zeros(len(delta) + 1, object if delta.dtype == object else float)
        # delta,h live on intervals; x,y,m on knots
        # interior points
        p = (delta[:-1] * h[1:] + delta[1:] * h[:-1]) / (h[:-1] + h[1:])
        idx = (numpy.sign(delta[:-1]) * numpy.sign(delta[1:]) <= 0)
        p[idx] *= 0 
        idx = _gvar.fabs(p) > 2 * _gvar.fabs(delta[:-1])
        p[idx] = 2 * delta[:-1][idx]
        idx = _gvar.fabs(p) > 2 * _gvar.fabs(delta[1:])
        p[idx] = 2 * delta[1:][idx]
        m[1:-1] = p
        # endpoints
        for i, hi, deltai in [(0,  h[:2], delta[:2]), (-1, h[:-3:-1], delta[:-3:-1])]:
            if deriv[i] is not None:
                m[i] = deriv[i]
                continue
            m[i] = ((2 * hi[0] + hi[1]) * deltai[0] - hi[0] * deltai[1]) / (hi[0] + hi[1])
            if m[i] * deltai[0] <= 0:
                m[i] = 0
            elif deltai[0] * deltai[1] <= 0 and _gvar.fabs(m[i]) > 3 * _gvar.fabs(deltai[0]):
                m[i] =  3 * deltai[0]
        return m
    
    def __call__(self, x, warn=None):
        if warn is None:
            warn = self.warn
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        left = x < self.x[0]
        right = x > self.x[-1]
        out_of_range = numpy.any(left) or numpy.any(right)
        if warn and out_of_range:
            warnings.warn('x outside of spline range: ' + str(x[left]) + ' and ' + str(x[right]))
        if out_of_range and self.extrap_order < 3:
            ans = numpy.empty(len(x), object)
            middle = numpy.logical_not(numpy.logical_or(left, right))
            for coef, idx, x0 in [
                (self.cleft, left, self.x[0]),
                (self.cright, right, self.x[-1]),
                ]:
                t = x[idx] - x0
                if self.extrap_order == 2:
                    ans[idx] = coef[0] + t * (coef[1] + t * coef[2])
                elif self.extrap_order == 1:
                    ans[idx] = coef[0] + t * coef[1] 
                elif self.extrap_order == 0:
                    ans[idx] = coef[0]
                else:
                    raise ValueError('bad value for extrap_order = ' + str(self.extrap_order))
            ans[middle] = self(x[middle])
            try:
                ans = numpy.array(ans, float)
            except TypeError:
                pass          
        else:  
            # x between x[i] and x[j]
            j = numpy.searchsorted(self.x, x)
            j[j <= 0] = 1
            j[j >= self.n] = self.n - 1
            i = j - 1
            t = x - self.x[i]
            ans = self.y[i] + t * (self.dy[i] + t * (self.d2y[i] + t * self.d3y[i]))
        return ans.reshape(xshape) if xshape != () else ans[0]            

    def integ(self, x, warn=None):
        if warn is None:
            warn = self.warn
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        left = x < self.x[0]
        right = x > self.x[-1]
        out_of_range = numpy.any(left) or numpy.any(right)
        if warn and out_of_range:
            warnings.warn('x outside of spline range: ' + str(x[left]) + ' and ' + str(x[right]))
        if out_of_range and self.extrap_order < 3:
            ans = numpy.empty(len(x), object)
            middle = numpy.logical_not(numpy.logical_or(left, right))
            for coef, idx, x0, intydx in [
                (self.cleft, left, self.x[0], self.intydx[0]),
                (self.cright, right, self.x[-1], self.intydx[-1]),
                ]:
                t = x[idx] - x0
                if self.extrap_order == 2:
                    ans[idx] = t * (coef[0] + t * (coef[1] / 2. + t * coef[2] / 3.))
                elif self.extrap_order == 1:
                    ans[idx] = t * (coef[0] + t * coef[1] / 2.)
                elif self.extrap_order == 0:
                    ans[idx] = t * coef[0]
                else:
                    raise ValueError('bad value for extrap_order = ' + str(self.extrap_order))
                ans[idx] += intydx
            ans[middle] = self.integ(x[middle])
            try:
                ans = numpy.array(ans, float)
            except TypeError:
                pass          
        else:  
            j = numpy.searchsorted(self.x, x)
            j[j <= 0] = 1
            j[j >= self.n] = self.n - 1
            i = j - 1
            t = x - self.x[i]
            ans = t * (self.y[i] + t * (self.dy[i] / 2. + t * (self.d2y[i] / 3. + t * self.d3y[i] / 4.)))
            ans += self.intydx[i]
        return ans.reshape(xshape) if xshape != () else ans[0]            

    def D(self, x, warn=None):
        if warn is None:
            warn = self.warn
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        left = x < self.x[0]
        right = x > self.x[-1]
        out_of_range = numpy.any(left) or numpy.any(right)
        if warn and out_of_range:
            warnings.warn('x outside of spline range: ' + str(x[left]) + ' and ' + str(x[right]))
        if out_of_range and self.extrap_order < 3:
            ans = numpy.empty(len(x), object)
            middle = numpy.logical_not(numpy.logical_or(left, right))
            for coef, idx, x0 in [
                (self.cleft, left, self.x[0]),
                (self.cright, right, self.x[-1]),
                ]:
                t = x[idx] - x0
                if self.extrap_order == 2:
                    ans[idx] = coef[1] + 2 * t * coef[2]
                elif self.extrap_order == 1:
                    ans[idx] = coef[1] 
                elif self.extrap_order == 0:
                    ans[idx] = 0 * coef[1]
                else:
                    raise ValueError('bad value for extrap_order = ' + str(self.extrap_order))
            ans[middle] = self.D(x[middle])
            try:
                ans = numpy.array(ans, float)
            except TypeError:
                pass          
        else:  
            j = numpy.searchsorted(self.x, x)
            j[j <= 0] = 1
            j[j >= self.n] = self.n - 1
            i = j - 1
            t = x - self.x[i]
            ans = self.dy[i] + t * (2. * self.d2y[i] + 3. * t * self.d3y[i])
        return ans.reshape(xshape) if xshape != () else ans[0]            

    def D2(self, x, warn=None):
        if warn is None:
            warn = self.warn
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        left = x < self.x[0]
        right = x > self.x[-1]
        out_of_range = numpy.any(left) or numpy.any(right)
        if warn and out_of_range:
            warnings.warn('x outside of spline range: ' + str(x[left]) + ' and ' + str(x[right]))
        if out_of_range and self.extrap_order < 3:
            ans = numpy.empty(len(x), object)
            middle = numpy.logical_not(numpy.logical_or(left, right))
            for coef, idx, x0 in [
                (self.cleft, left, self.x[0]),
                (self.cright, right, self.x[-1]),
                ]:
                t = x[idx] - x0
                if self.extrap_order == 2:
                    ans[idx] = 2 * coef[2]
                elif self.extrap_order in [0, 1]:
                    ans[idx] = 0 * coef[2] 
                else:
                    raise ValueError('bad value for extrap_order = ' + str(self.extrap_order))
            ans[middle] = self.D2(x[middle])
            try:
                ans = numpy.array(ans, float)
            except TypeError:
                pass          
        else:  
            j = numpy.searchsorted(self.x, x)
            j[j <= 0] = 1
            j[j >= self.n] = self.n - 1
            i = j - 1
            t = x - self.x[i]
            ans = 2. * self.d2y[i] + 6. * t * self.d3y[i]
        return ans.reshape(xshape) if xshape != () else ans[0]            

    def D3(self, x, warn=None):
        if warn is None:
            warn = self.warn
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        left = x < self.x[0]
        right = x > self.x[-1]
        out_of_range = numpy.any(left) or numpy.any(right)
        if warn and out_of_range:
            warnings.warn('x outside of spline range: ' + str(x[left]) + ' and ' + str(x[right]))
        if warn and alg!='cspline':
            warnings.warn('3rd derivatives are unreliable for alg = ' + self.alg)
        if out_of_range and self.extrap_order < 3:
            ans = numpy.empty(len(x), object)
            middle = numpy.logical_not(numpy.logical_or(left, right))
            for coef, idx, x0 in [
                (self.cleft, left, self.x[0]),
                (self.cright, right, self.x[-1]),
                ]:
                t = x[idx] - x0
                if self.extrap_order in [0, 1, 2]:
                    ans[idx] = 0 * t * coef[2]
                else:
                    raise ValueError('bad value for extrap_order = ' + str(self.extrap_order))
            ans[middle] = self.D3(x[middle])
            try:
                ans = numpy.array(ans, float)
            except TypeError:
                pass          
        else:
            j = numpy.searchsorted(self.x, x)
            j[j <= 0] = 1
            j[j >= self.n] = self.n - 1
            i = j - 1
            t = x - self.x[i]
            ans = 6. * self.d3y[i]
        return ans.reshape(xshape) if xshape != () else ans[0]            

def tri_diag_solve(a, b, c, d):
    """ Solve a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i] for x[i]

    Adapted from: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
    """
    n = len(a)
    a, b, c, d = map(numpy.array, (a, b, c, d))
    c[0] /= b[0]
    d[0] /= b[0]
    for i in range(1, n):
        m = b[i] - a[i] * c[i - 1]
        c[i] /= m
        d[i] = (d[i] - a[i] * d[i - 1]) / m
    x = d
    for i in range(n - 2, -1, -1):
        x[i] -= c[i] * x[i + 1]
    return x
