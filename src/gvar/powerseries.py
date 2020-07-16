"""
Module :mod:`gvar.powerseries` provides tools for manipulating power series
approximations of functions. A function's power series is specified by the
coefficients in its Taylor expansion with respect to an independent variable,
say ``x``::

    f(x) = f(0) + f'(0) * x + (f''(0)/2) * x**2 + (f'''(0)/6) * x**3 + ...
         = f0 + f1 * x + f2 * x**2 + f3 * x**3 + ...

In practice a power series is different from a polynomial because power
series, while infinite order in principle, are truncated at some finite
order in numerical applications. The ``order`` of a power series is the
highest power of ``x`` that is retained in the approximation; coefficients
for still higher-order terms are assumed to be unknown (as opposed to zero).

Taylor's theorem can be used to generate power series for functions of
power series::

    g(f(x)) = g(f0) + g'(f0) * (f(x)-f0) + (g''(f0)/2) * (f(x)-f0)**2 + ...
            = g0 + g1 * x + g2 * x**2 + ...

This allows us to define a full calculus for power series, where arithmetic
expressions and (sufficiently differentiable) functions of power series
return new power series.

Using power series
-----------------------
Class :class:`PowerSeries` provides a numerical implementation of the power series
calculus. ``PowerSeries([f0,f1,f2,f3...])`` is a numerical representation of
a power series with coefficients ``f0, f1, f2, f3...`` (as in ``f(x)``
above). Thus, for example, we can define a 4th-order power series
approximation ``f`` to ``exp(x)=1+x+x**2/2+...`` using

    >>> import gvar as gv 
    >>> import gvar.powerseries as ps
    >>> f = ps.PowerSeries([1., 1., 1/2., 1/6., 1/24.])
    >>> print(f)            # print the coefficients
    [ 1.          1.          0.5         0.16666667  0.04166667]

Arithmetic expressions involving instances of class :class:`PowerSeries` are
themselves :class:`PowerSeries` as in, for example,

    >>> print(1/f)              # power series for exp(-x)
    [ 1.         -1.          0.5        -0.16666667  0.04166667]
    >>> print(gv.log(f))        # power series for x
    [ 0.  1.  0. -0.  0.]
    >>> print(f / f)            # power series for 1
    [ 1.  0.  0.  0.  0.]

The standard arithmetic operators (``+,-,*,/,=,**``) are supported, as are
the usual elementary functions (``exp, log, sin, cos, tan ...``). Different
:class:`PowerSeries` can be combined arithmetically to create new
:class:`PowerSeries`; the order of the result is that of the operand with the
lowest order.

:class:`PowerSeries` can be differentiated and integrated::

    >>> print(f.deriv())    # derivative of exp(x)
    [ 1.          1.          0.5         0.16666667]
    >>> print(f.integ())    # integral of exp(x) (from x=0)
    [ 0.          1.          0.5         0.16666667  0.04166667  0.00833333]

Each :class:`PowerSeries` represents a function. The :class:`PowerSeries` for
a function of a function is easily obtained. For example, assume ``f``
represents function ``f(x)=exp(x)``, as above, and ``g``
represents ``g(x)=log(1+x)``::

    >>> g = ps.PowerSeries([0, 1, -1/2., 1/3., -1/4.])

Then ``f(g)`` gives the :class:`PowerSeries` for ``exp(log(1+x)) = 1 + x``::

    >>> print(f(g))
    [  1.0000e+00   1.0000e+00   0.0000e+00  -2.7755e-17 -7.6327e-17]

Individual coefficients from the powerseries can be accessed using
array-element notation: for example,

    >>> print(f[0], f[1], f[2], f[3])
    1.0 1.0 0.5 0.166666666667
    >>> f[0] = f[0] - 1.
    >>> print(f)            # f is now the power series for exp(x)-1
    [ 0.          1.          0.5         0.16666667  0.04166667]

Finally, a power series can be evaluated for a particular
numerical value of x:

    >>> x = 0.01
    >>> print(f(x))             # should be exp(0.01)-1 approximately
    0.0100501670833
    >>> print(gv.exp(x)-1)      # verify that it is
    0.0100501670842

The independent variable ``x`` could be of any arithmetic type (it need not
be a ``float``).

Multivariate power series
-------------------------
The coefficients in a :class:`PowerSeries` object can themselves by :class:`PowerSeries` 
objects. The is used to represent multivariate power series such as::

    f(x,y) = f00 + f10 * x + f01 * y + f20 * x**2 + f11 * x*y + f02 * y**2 + ...

One way to construct a :class:`PowerSeries` object ``f`` representing this series, through
``order=2``, is from an array ``c`` containing the coefficients::

    c = [[f00, f01, f02], [f10, f11, 0], [f20, 0, 0]]
    f = ps.multiseries(c, order=2)

Here entries for ``c[1,2]``, ``c[2,1]``,  and ``c[2,2]`` are ignored because they 
correspond to ``order=3`` or higher.
The individual coefficients ``c[i,j]`` are accessed using ``f[i,j]``, and the 
function is evaluate at point ``(x,y)`` using ``f(x,y)``. Similarly the first-order 
partial derivative with respect to ``x`` and ``y``, for example, is given by 
:class:`PowerSeries` object
``f.deriv(1,1)``, while first-order integrals with respect to ``x`` and ``y`` 
are given by ``f.integ(1,1)``.

Taylor expansions of Python functions
-------------------------------------
:class:`PowerSeries` can be used to compute Taylor series for more-or-less
arbitrary pure-Python functions provided the functions are
sufficiently differentiable. To compute the ``N``-th order
expansion of a Python function ``g(x)``, first create a ``N``-th order
:class:`PowerSeries` variable that represents the expansion parameter: say,
``x = PowerSeries([0.,1.],order=N)``. The Taylor series for function ``g``
is then given by ``g_taylor = g(x)`` which is a :class:`PowerSeries` instance.
For example, consider::

    >>> def g(x):              # an example of a Python function
    ...     return 0.5/sqrt(1+x) + 0.5/sqrt(1-x)
    ...
    >>> x = ps.PowerSeries([0.,1.],order=5)    # Taylor series for x
    >>> print(x)
    [ 0.  1.  0.  0.  0.  0.]
    >>> g_taylor = g(x)        # Taylor series for g(x) about x=0
    >>> print(g_taylor)
    [ 1.         0.         0.375      0.         0.2734375  0.       ]
    >>> exp_taylor = gv.exp(x) # Taylor series for exp(x) about x=0
    >>> print(exp_taylor)
    [ 1.          1.          0.5         0.16666667  0.04166667  0.00833333]

This generalizes easily to multivariate expansions. For example, 
one can calculate the Taylor expansion coefficients for ``exp(x+y)`` using::

    >>> x,y = ps.multivar(dim=2, order=3)
    >>> exp_taylor = gv.exp(x + y)
    >>> print(exp_taylor)
    [[1.         1.         0.5        0.16666667], [1.  1.  0.5], [0.5 0.5], [0.16666667]]
    >>> print(exp_taylor(.1,.2), gv.exp(.1 + .2))
    1.3495000000000001 1.3498588075760032

Here function :func:`multivar` creates :class:`PowerSeries` objects corresponding to the 
expansion variables through a given order.
"""
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

import numpy
import math
from numpy import exp, log, sqrt, sin, cos, tan, arcsin, arccos, arctan
from numpy import sinh, cosh, tanh, arcsinh, arccosh, arctanh
from scipy.special import eval_hermite as _scipy_eval_hermite

class PowerSeries(object):
    """
    Power series representation of a function.

    The power series created by ``PowerSeries(c)`` corresponds to::

        c[0] + c[1]*x + c[2]*x**2 + ... .

    The order of the power series is normally determined by the length of
    the input list ``c``. This can be overridden by specifying the order of
    the power series using the ``order`` parameter. The list of ``c[i]``\s
    is then padded with zeros if ``c`` is too short, or truncated if it
    is too long. Omitting ``c`` altogether results in a power series
    all of whose coefficients are zero. Individual series
    coefficients are accessed using array/list notation: for example,
    the 3rd-order coefficient of ``PowerSeries p`` is ``p[3]``. The
    order of ``p`` is ``p.order``. :class:`PowerSeries` should work
    for coefficients of any data type that supports ordinary arithmetic.

    Arithmetic expressions of :class:`PowerSeries` variables yield new
    :class:`PowerSeries` results that represent the power series
    expansion of the expression. Expressions can include the standard
    mathematical functions (``log, exp, sqrt, sin, cos, tan...``).
    :class:`PowerSeries` can also be differentiated (``p.deriv()``)
    and integrated (``p.integ()``).

    Args:
        c (array): Power series coefficients.
        order (int or None): Highest power in power series. If ``None``, 
            the order is inferred from the array of coefficients. 
            If array ``c`` is too small for the specified ``order``, 
            the array is padded with zeros at the end.
    """
    def __init__(self, c=None, order=None):
        if isinstance(c, PowerSeries):
            c = numpy.array(c.c)
        if order is None:
            if c is None:
                raise ValueError('must specify c or order')
            elif len(c) == 0:
                raise ValueError("empty coefficient array: c = "+str(c))
            order = len(c) - 1
        if order < 0:
            raise ValueError(
                "order cannot be less than zero: order = " +
                str(order)
                )
        if c is None or len(c) == 0:
            self.c = numpy.zeros(order + 1, object)
        elif order < len(c):
            self.c = numpy.array(c[:order + 1])
        elif order >= len(c):
            c = numpy.asarray(c)
            shape = (order + 1,) + c.shape[1:]
            self.c = numpy.zeros(shape, c.dtype)
            self.c[:len(c)] = c
            self.c[len(c):] *= c[0]
        # check for PowerSeries, fix orders
        for i in range(len(self.c)):
            if isinstance(self.c[i], PowerSeries):
                self.c[i] = PowerSeries(self.c[i].c, order=order - i)

    def _getorder(self):
        return len(self.c)-1
    order = property(_getorder,doc="Highest power in power series.")

    def _getcoeff(self):
        return numpy.array(self.c)
    coeff = property(_getcoeff,
                doc="Copy of power series coefficients (numpy.array).")

    def __getitem__(self, i):
        """ Return C{i}th coefficient of power series. """
        if numpy.size(i) > 1:
            return self.c[i[0]][i[1:]]
        else:
            return self.c[i if numpy.shape(i) == () else i[0]]

    def __setitem__(self, i, val):
        """ Set C{i}th coefficient of power series equal to C{val}. """
        if numpy.size(i) > 1:
            self.c[i[0]][i[1:]] = val
        else:
            self.c[i if numpy.shape(i) == () else i[0]] = val

    def __iter__(self):
        """ Iterate over coefficients of power series C{self}. """
        return iter(self.c)

    def __add__(self,x):
        try:
            order = min(self.order,x.order)
            ans = self.c[:order+1]+x.c[:order+1]
        except AttributeError:
            ans = list(self.c)
            ans[0] = ans[0] + x
        return PowerSeries(ans)

    def __radd__(self,x):
        return self+x

    def __sub__(self,x):
        try:
            order = min(self.order,x.order)
            ans = self.c[:order+1]-x.c[:order+1]
        except AttributeError:
            ans = list(self.c)
            ans[0] = ans[0] - x
        return PowerSeries(ans)

    def __neg__(self):
        return PowerSeries(-self.c)

    def __rsub__(self,x):
        return -self+x

    def __mul__(self,x):
        if isinstance(x,PowerSeries):
            order = min(self.order,x.order)
            ans = numpy.convolve(self.c,x.c,mode='full')[:order+1]
        else:
            ans = x*self.c
        return PowerSeries(ans)

    def __rmul__(self,x):
        return self*x

    # truediv and div are the same --- 1st is for python3, 2nd for python2
    def __truediv__(self,x):
        try:
            if x.c[0] == 0:     # needed for numpy scalars
                raise ZeroDivisionError
            order = min(self.order,x.order)
            ans = 0.0*self.c[:order+1]
            for n in range(order+1):
                tot = self.c[n]
                for i in range(n):
                    tot = tot - ans[i]*x.c[n-i]
                ans[n] = tot/x.c[0]
        except ZeroDivisionError:
            if self.c[0]==x.c[0] and len(self.c)>1 and len(x.c)>1:
                # strip off matching overall factor of x from denom and num
                return PowerSeries(self.c[1:])/PowerSeries(x.c[1:])
            else:
                raise ZeroDivisionError("leading coefficients in denominator equal zero")
        except AttributeError:
            ans = (1./x)*self.c
        return PowerSeries(ans)

    def __div__(self,x):
        try:
            if numpy.any(x.c[0] == 0):     # needed for numpy scalars
                raise ZeroDivisionError
            order = min(self.order,x.order)
            ans = 0.0*self.c[:order+1]
            for n in range(order+1):
                tot = self.c[n]
                for i in range(n):
                    tot = tot - ans[i]*x.c[n-i]
                ans[n] = tot/x.c[0]
        except ZeroDivisionError:
            if self.c[0]==x.c[0] and len(self.c)>1 and len(x.c)>1:
                # strip off matching overall factor of x from denom and num
                return PowerSeries(self.c[1:])/PowerSeries(x.c[1:])
            else:
                raise ZeroDivisionError("leading coefficients in denominator equal zero")
        except AttributeError:
            ans = (1./x)*self.c
        return PowerSeries(ans)

    def __rtruediv__(self,x):
        num = self.c*0
        num[0] = x
        return PowerSeries(num)/self

    def __rdiv__(self,x):
        num = self.c*0
        num[0] = x
        return PowerSeries(num)/self

    def __pow__(self,alpha):
        # use Taylor expn around x=c[0] to get answer
        # unless alpha is an integer (then multiply out)
        if numpy.issubdtype(type(alpha), numpy.int_):
            if alpha == 0:
                return 1.
            elif alpha < 0:
                alpha = - alpha
                xn = 1. / self
            else:
                xn = self
            ans = 1
            while alpha > 0:
                if alpha % 2 == 1:
                    ans *= xn
                xn *= xn
                alpha = (alpha - (alpha % 2)) / 2
            return ans
        fac = 1.
        alj = alpha
        xj = 1.
        ans = 0
        x = self-self.c[0]
        for j in range(self.order+1):
            ans = ans + fac*self.c[0]**alj*xj
            xj = xj*x
            fac = fac*alj/(j+1.)
            alj = alj-1
            # if fac==0:
            #     break
        return ans

    def __rpow__(self,x):
        return exp(self*log(x))

    def __str__(self):
        if isinstance(self.c[0], PowerSeries):
            return str([str(ci) for ci in self.c]).replace('"','').replace("'",'') # .replace(',','')
        else:
            return str(self.c)

    def __repr__(self):
        return "PowerSeries(%s)" % str(self.c.tolist())

    def __call__(self, *x):
        if len(x) > 1:
            return self(x[0])(*x[1:])
        elif len(x) == 1:
            x = x[0]
        ans = 0.0
        xn = 1.
        if isinstance(self.c[0], PowerSeries):
            # multivariate power series
            order = self.c[0].order
            for ci in self.c:
                ans += PowerSeries(ci, order=order) * xn
                xn *= x        
        else:
            for ci in self.c:
                ans += ci * xn
                xn *= x
        return ans

    def sqrt(self):
        return self**0.5

    def erf(self):
        from gvar import erf as _gvar_erf 
        jmax = self.order + 1
        jarray = numpy.arange(jmax + 1)
        deriv = (
            _scipy_eval_hermite(jarray, self.c[0])
            * (-1) ** jarray
            ) * (2 * numpy.exp(-self.c[0] ** 2) / numpy.sqrt(numpy.pi))
        x = self - self.c[0]
        fac = 1.
        ans = _gvar_erf(self.c[0])
        for j in range(1, jmax + 1):
            fac *= x / j
            ans = ans + fac * deriv[j - 1]
        return ans

    def sin(self):
        # use Taylor series about x=self.c[0]
        sc = sin(self.c[0])
        cc = cos(self.c[0])
        deriv = [sc,cc,-sc,-cc]
        fac = 1.
        xj = 1.
        ans = 0
        x = self-self.c[0]
        for j in range(self.order+1):
            ans = ans + fac*deriv[j%4]*xj
            fac = fac/(j+1.)
            xj = xj*x
        return ans

    def cos(self):
        # use Taylor series about x=self.c[0]
        sc = sin(self.c[0])
        cc = cos(self.c[0])
        deriv = [cc,-sc,-cc,sc]
        fac = 1.
        xj = 1.
        ans = 0
        x = self-self.c[0]
        for j in range(self.order+1):
            ans = ans + fac*deriv[j%4]*xj
            fac = fac/(j+1.)
            xj = xj*x
        return ans

    def tan(self):
        return sin(self)/cos(self)

    def arcsin(self):
        # use Taylors theorm to expand around x=self.c[0]
        # compute higher derivs from powerseries for 1st deriv
        x = PowerSeries([0.,1.],order=self.order-1)
        dasin = 1/sqrt(1-(self.c[0]+x)**2)
        # if True in numpy.isnan(dasin.c): # in dasin.c:
        #     raise ValueError("bad expansion point: "+str(self.c[0]))

        ans = arcsin(self.c[0])
        x = self-self.c[0]
        xn = x
        for n in range(1,self.order+1):
            ans = ans + (dasin.c[n-1]/float(n))*xn
            xn = xn*x
        return ans

    def arccos(self):
        # use Taylors theorm to expand around x=self.c[0]
        # compute higher derivs from powerseries for 1st deriv
        x = PowerSeries([0.,1.],order=self.order-1)
        dacos = -1/sqrt(1-(self.c[0]+x)**2)
        # if True in numpy.isnan(dacos.c): # in dasin.c:
        #     raise ValueError("bad expansion point: "+str(self.c[0]))

        ans = arccos(self.c[0])
        x = self-self.c[0]
        xn = x
        for n in range(1,self.order+1):
            ans = ans + (dacos.c[n-1]/float(n))*xn
            xn = xn*x
        return ans

    def arctan(self):
        # use Taylors theorm to expand around x=self.c[0]
        # compute higher derivs from powerseries for 1st deriv
        x = PowerSeries([0.,1.],order=self.order-1)
        datan = 1/(1+(self.c[0]+x)**2)

        ans = arctan(self.c[0])
        x = self-self.c[0]
        xn = x
        for n in range(1,self.order+1):
            ans = ans + (datan.c[n-1]/float(n))*xn
            xn = xn*x
        return ans

    def sinh(self):
        # use Taylor series about x=self.c[0]
        sc = sinh(self.c[0])
        cc = cosh(self.c[0])
        deriv = [sc,cc]
        fac = 1.
        xj = 1.
        ans = 0
        x = self-self.c[0]
        for j in range(self.order+1):
            ans = ans + fac*deriv[j%2]*xj
            fac = fac/(j+1.)
            xj = xj*x
        return ans

    def cosh(self):
        # use Taylor series about x=self.c[0]
        sc = sinh(self.c[0])
        cc = cosh(self.c[0])
        deriv = [cc,sc]
        fac = 1.
        xj = 1.
        ans = 0
        x = self-self.c[0]
        for j in range(self.order+1):
            ans = ans + fac*deriv[j%2]*xj
            fac = fac/(j+1.)
            xj = xj*x
        return ans

    def tanh(self):
        return sinh(self)/cosh(self)

    def arcsinh(self):
        return log(self+sqrt(self**2+1))

    def arccosh(self):
        return log(self+sqrt(self**2-1))

    def arctanh(self):
        return log((1+self)/(1-self))/2.

    def exp(self):
        f = exp(self.c[0])
        ans = 1.
        x = self - self.c[0]
        for n in range(self.order, 0, -1):
            ans = 1. + x * ans/ float(n)
        return f * ans

    def log(self):
        ans = log(self.c[0])
        x = -(self-self.c[0])/self.c[0]
        xn = 1
        for n in range(1,self.order+1):
            xn = xn*x
            ans = ans - xn/float(n)
        # ans[0] = ans[0] + log(self.c[0])
        return ans

    def deriv(self, *n):
        """ Compute *n*-th derivative (or partial derivative) of ``self``.

        Args:
            n (array): Number of derivatives in each direction.  Default is ``n=[1]``.

        Returns:
            :class:`PowerSeries` object representing the 
            *n*-th derivative or partial derivative of ``self``.
        """
        if len(n) == 0:
            n = [1]
        if len(n) > 1:
            ps = self.deriv(n[0])
            if ps == 0:
                return 0
            c = numpy.array(ps.c)
            for i in range(len(c)):
                if isinstance(c[i], PowerSeries):
                    c[i] = c[i].deriv(*n[1:])
                else:
                    c[i] = 0.0
            order = ps.order - sum(n[1:])
            if order >= 0:
                return PowerSeries(c, order=order)
            else:
                return 0.
        elif len(n) == 1:
            n = n[0]
        if n==1:
            if self.order > 0:
                return PowerSeries(self.c[1:] * range(1, len(self.c)))
            else:
                return 0.
        elif n>1:
            ans = self.deriv(1)
            return ans.deriv(n-1) if isinstance(ans, PowerSeries) else 0.
        elif n==0:
            return self

    def integ(self, *n, **kargs): # should be (self, *n, x0=1) but not with Py2
        """ Compute *n*-th indefinite integral of ``self``.

        If *x0* is specified, then the definite integral,
        integrating from point *x0*, is returned.

        Args:
            n (array): Number of integrations in each direction. Default is ``n=[1]``.
            x0 (array or float): Starting point for definite integral 
                in each direction (default is 0).
        Returns:
            :class:`PowerSeries` object representing the *n*-th integral of ``self``.
        """
        x0 = kargs.get('x0', 0.)
        if len(n) > 1:
            if numpy.shape(x0) == ():
                x0 = numpy.zeros(len(n), dtype=float) + x0 
            ps = self.integ(n[0], x0=x0[0])
            c = numpy.array(ps.c) 
            for i in range(len(c)):
                c[i] = c[i].integ(*n[1:], x0=x0[1:])
            order = ps.order + sum(n[1:])
            return PowerSeries(c, order=order)
        elif len(n) == 0:
            n = 1
        else:
            n = n[0]
        if numpy.shape(x0) != ():
            x0 = x0[0]
        if n==1:
            if self.order<0:
                ans = PowerSeries([0])
            else:
                ans = PowerSeries([0.*self.c[0]] +
                    [x/(i+1.) for i,x in enumerate(self.c)])
                return PowerSeries([ans.c[0]-ans(x0)]+list(ans.c[1:]))
        elif n>1:
            return self.integ().integ(n-1, x0=x0)
        elif n==0:
            return self


def multiseries(c, order=None):
    """ Create multivariate power series from coefficients in array ``c``.

    Args:
        c (array): :mod:`numpy`-like array containing the power series 
            coefficients. In ``d`` dimensions, ``c[i1,i2,...,id]`` is the 
            coefficient of ``x1**i1 * x2**i2 * ... * xd**id``.
        order (int or None): Highest power in power series, where the power 
            associated with term ``x1**i1 * x2**i2 * ... * xd**id`` 
            is the sum of the exponents: ``i1 + i2 + ... + id``. If ``None``,
            the order is inferred from the array of coefficients.
    
    Returns:
        :class:`PowerSeries` object representing the multivariate power series. 
    """
    c = numpy.asarray(c)
    if c.ndim <= 0:
        return None
    if c.size == 0:
        c = numpy.zeros(c.ndim * (1,), dtype=c.dtype)
    if order is None:
        order = sum([n - 1 for n in c.shape])
    ans = []
    for i, ci in enumerate(c[:order + 1]):
        if ci.shape == ():
            ans.append(ci)
        else:
            ans.append(multiseries(ci, order=order - i))
    ans = PowerSeries(ans, order=order)
    ans.ndim = c.ndim
    return ans
    
def multivar(dim, order):
    """ Create :class:`PowerSeries` objects representing the expansion variables.

    Args:
        dim (int): The dimensionality of the multivariate space.
        order (int): Highest power in the power series,  where the power 
            associated with term ``x1**i1 * x2**i2 * ... * xd**id`` 
            is the sum of the exponents: ``i1 + i2 + ... + id``.

    Returns:
        An array of ``dim`` :class:`PowerSeries` objects corresponding to 
        each of the expansion variables in a ``dim``-dimensional multivariate
        power series.
    """
    if dim <= 0 or order < 0:
        return []
    shape = dim * (order + 1,)
    ans = numpy.empty(dim, dtype=object)
    idx = numpy.zeros(dim, dtype=int)
    zeros = numpy.zeros(shape, dtype=float)
    for i in range(dim):
        ans[i] = multiseries(zeros, order=order)
        idx[i] = 1
        ans[i][idx] = 1.
        idx[i] = 0
    return ans
