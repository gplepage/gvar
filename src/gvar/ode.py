""" Differential equation integrator for GVars. """

# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2014-2020 G. Peter Lepage.
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

# sqrt needed for 0(0)/(0(0) + TINY)
TINY = sys.float_info.min ** 0.5

class Integrator(object):
    r""" Integrate ``dy/dx = deriv(x,y)``.

    An :class:`Integrator` object ``odeint`` integrates 
    ``dy/dx = f(x,y)`` to obtain ``y(x1)`` from ``y0 = y(x0)``. 
    ``y`` and ``f(x,y)`` can be scalars or :mod:`numpy` arrays. 
    Typical usage is illustrated by the following code 
    for integrating ``dy/dx = y``::

        from gvar.ode import Integrator

        def f(x, y):
            return y

        odeint = Integrator(deriv=f,  tol=1e-8)
        y0 = 1.
        y1 = odeint(y0, interval=(0, 1.))
        y2 = odeint(y1, interval=(1., 2.))
        ...

    Here the first call to ``odeint`` integrates the differential equation
    from ``x=0`` to ``x=1`` starting with ``y=y0`` at ``x=0``; the result
    is ``y1=exp(1)``, of course. Similarly the second call to ``odeint``
    continues the integration from ``x=1`` to ``x=2``, giving ``y2=exp(2)``.

    If the ``interval`` is a list with more than two entries,
    then ``odeint(y0, interval=[x0, x1, x2 ...])`` in the example above
    returns an array of solutions for points ``x1, x2 ...``. So the example
    above could have been written equivalently as ::

        ...

        odeint = Integrator(deriv=f,  tol=1e-8)
        y0 = 1.
        y1, y2 ... = odeint(y0, interval=[0, 1., 2. ...])


    An alternative interface creates a new function which is the
    solution of the differential equation for specific initial conditions.
    The code above could be rewritten::

        x0 = 0.         # initial conditions
        y0 = 1.
        y = Integrator(deriv=f, tol=1e-8).solution(x0, y0)
        y1 = y(1)
        y2 = y(2)
        ...

    Here method :meth:`Integrator.solution` returns a function ``y(x)``
    where: a) ``y(x0) = y0``; and b) ``y(x)`` uses the integator to
    integrate the differential equation to point ``x`` starting
    from  the last point at which ``y`` was evaluated
    (or from ``x0`` for the first call to ``y(x)``). The function can
    also be called with an array of ``x`` values, in which case an
    array containing the corresponding ``y`` values is returned.

    The integrator uses an adaptive Runge-Kutta algorithm that adjusts
    the integrator's step size to obtain relative accuracy ``tol`` 
    in the solution. An initial step size can be set in the 
    :class:`Integrator` by specifying parameter ``h``. A minimum 
    step size ``hmin`` can also be specified; the :class:`Integrator` 
    raises an exception if the step size becomes smaller than ``hmin``. 
    The :class:`Integrator` keeps track of the number of good steps, 
    where ``h`` is increased, and the number of bad steps, where ``h`` 
    is decreased and the step is repeated:
    ``odeint.ngood`` and ``odeint.nbad``, respectively.

    A custom criterion for step-size changes can be implemented by
    specifying a function for parameter delta. This is a function
    ``delta(yerr, y, delta_y)`` --- of the estimated error ``yerr``
    after a given step, the proposed value for ``y``, and the
    proposed change ``delta_y`` in ``y`` --- that returns a number
    to compare with tolerance ``tol``. The step size is
    decreased and the step repeated if ``delta(yerr, y, delta_y) > tol``;
    otherwise the step is accepted and the step size increased.
    The default definition of ``delta`` is equivalent to::

        import numpy as np
        import gvar as gv

        def delta(yerr, y, delta_y):
            return np.max(
                np.fabs(yerr) / 
                (np.fabs(y) + np.fabs(delta_y) + gv.ode.TINY)
                )

    An analyzer ``analyzer(x,y)`` can be specified using parameter
    ``analyzer``. This function is called after every full step of
    the integration, with the current values of ``x`` and ``y``.
    Objects of type :class:`gvar.ode.Solution` are examples of
    (simple) analyzers.

    Args:
        deriv: Function of ``x`` and ``y`` that returns ``dy/dx``.
            The return value should have the same shape as ``y`` if arrays
            are used.
        tol (float): Relative accuracy in ``y`` relative to 
            ``|y| + h|dy/dx|`` for each step in the integration. 
            Any integration step that achieves less precision is 
            repeated with a smaller step size. The step size
            is increased if precision is higher than needed. 
            Default is 1e-5.
        h (float or None): Absolute value of initial step size. 
            The default value equals the entire width of the 
            integration interval.
        hmin (float or None): Smallest step size allowed. A warning 
            is raised if a smaller step size is requested, and 
            the step size is not decreased. This prevents infinite loops 
            at singular points, but the solution may not be reliable when 
            a warning has been issued. The default value is ``None`` 
            (which does *not* prevent infinite loops).
        hmax (float or None): Largest step allowed. Ignored if 
            set to ``None``.
        maxstep (int or None): Maximum number of integration steps 
            allowed, after which a ``RuntimeError`` exception is raised. 
            Ignored if set to ``None``.
        delta: Function ``delta(yerr, y, delta_y)`` that returns
            a number to be compared  with ``tol`` at each integration step:
            if it is larger than ``tol``, the step is repeated with a smaller
            step size; if it is smaller the step is accepted and a larger
            step size used for the subsequent step. Here ``yerr`` is an
            estimate of the error in ``y`` on the last step; ``y`` is the
            proposed value; and ``delta_y`` is the change in ``y`` over
            the last step.
        analyzer: Function of ``x`` and ``y`` that is called after each
            step of the integration. This can be used to analyze intermediate
            results.
    """
    def __init__(self, deriv=None, tol=1e-5, h=None, hmin=None, hmax=None, maxstep=None, delta=None, analyzer=None):
        self.deriv = deriv
        self.tol = tol
        self.h = h
        self.hmin = hmin
        self.hmax = hmax
        self.maxstep = maxstep
        self.delta = delta
        self.analyzer = analyzer
        self.ngood = 0
        self.nbad = 0

    def __call__(self, y0, interval):
        r""" Integrate from ``x0`` to ``x1`` where ``interval=(x0,x1)`` and ``y0=y(x0)``. """
        if len(interval) > 2:
            # evaluate for sequence of intervals, one at a time
            ans = []
            xlast = interval[0]
            ylast = y0
            nbad = 0
            ngood = 0
            for x in interval[1:]:
                y = self(ylast, interval=(xlast,x))
                ylast = y 
                xlast = x
                nbad += self.nbad 
                ngood += self.ngood
                ans.append(y)
            self.nbad = nbad 
            self.ngood = ngood
            return numpy.array(ans)
        if self.deriv is None or interval[1] == interval[0]:
            return y0
        tol = numpy.fabs(self.tol)
        self.nbad = 0
        self.ngood = 0
        x0, x1 = interval
        hmax = numpy.fabs((x1 - x0) if self.hmax is None else self.hmax)
        h = self.h if (self.h is not None and self.h != 0) else (x1 - x0)
        h = numpy.fabs(h)
        if h > hmax:
            h = hmax
        xdir = 1 if x1 > x0 else -1
        hmin = 0.0 if self.hmin is None else numpy.fabs(self.hmin)
        x = x0
        y = numpy.asarray(y0)
        y_shape = y.shape
        while (xdir>0 and x<x1) or (xdir<0 and x>x1):
            if self.maxstep is not None and (self.ngood + self.nbad) >= self.maxstep:
                raise RuntimeError('maximum number of steps exceeded: ' + str(self.maxstep))
            hold = h
            xold = x
            yold = y
            if h > numpy.fabs(x - x1):
                h = numpy.fabs(x - x1)
            x, y, yerr = rk5_stepper(xold, h*xdir, yold, self.deriv, errors=True)
            if self.delta is not None:
                delta = self.delta(yerr, y, y - yold)
            else:
                delta = numpy.max(gvar.abs(yerr) / (gvar.abs(y) + gvar.abs(y - yold) + TINY))
                if isinstance(delta, gvar.GVar):
                    delta = delta.mean
            if delta >= tol:
                # need smaller step size -- adjust and redo step
                if h <= hmin:
                    warnings.warn(
                        'step size not reduced (<= hmin) --- errors may not be reliable'
                        )
                    continue
                hfac = 0.9 * (tol / delta) ** 0.25
                # limit step change
                h *= hfac if hfac > 0.1 else 0.1
                x = xold
                y = yold
                self.nbad += 1
                continue
            else:
                # want larger step size -- adjust and continue
                if delta > 0:
                    hfac = 0.9 * (tol / delta) ** 0.20
                    # limit step change
                    h *= hfac if hfac < 5. else 5.
                else:
                    h *= 5.
                if h > hmax:
                    h = hmax
                self.ngood += 1
            if self.analyzer is not None:
                self.analyzer(x, y)
        return y

    def solution(self, x0, y0):
        r""" Create a solution function ``y(x)`` such that ``y(x0) = y0``.

        A list of solution values ``[y(x0), y(x1) ...]`` is returned if the
        function is called with a list ``[x0, x1 ...]`` of ``x`` values.
        """
        def soln(x):
            if numpy.size(x) > 1:
                x = [soln.x] + list(x)
                ans = self(soln.y, interval=x)
                soln.x = x[-1]
                soln.y = ans[-1]
                return ans
            else:
                soln.y = self(soln.y, interval=(soln.x, x))
                soln.x = x
                return soln.y
        soln.x = x0
        soln.y = y0
        return soln

def rk5_stepper(x, h, y , deriv, errors=False):
    r""" Compute y(x+h) from y and dy/dx=deriv(x,y).

    Uses a one-step 5th-order Runge-Kutta algorithm.

    Returns x+h, y(x+h) if errors is False; otherwise
    returns x+h, y(x+h), yerr where yerr is an error
    estimate.

    Adapted from Numerical Recipes.
    """
    k1 = h * deriv(x,y)
    k2 = h * deriv(x + 0.2 * h, y + 0.2 * k1)
    k3 = h * deriv(x + 0.3 * h, y + 0.075 * k1 + 0.225 * k2)
    k4 = h * deriv(x + 0.6 * h, y + 0.3 * k1 - 0.9 * k2 + 1.2 * k3)
    k5 = h * deriv(
        x + h, 
        y - .2037037037037037037037037 * k1
        + 2.5 * k2 - 2.592592592592592592592593 * k3
        + 1.296296296296296296296296 * k4
        )
    k6 = h * deriv(
        x + 0.875 * h, 
        y + .2949580439814814814814815e-1 * k1
        + .341796875 * k2 + .4159432870370370370370370e-1 * k3
        + .4003454137731481481481481 * k4 + .61767578125e-1 * k5
        )
    yn = y  +  (
        .9788359788359788359788361e-1 * k1
        + .4025764895330112721417070 * k3
        + .2104377104377104377104378 * k4
        + .2891022021456804065499718 * k6
        )
    xn = x + h
    if errors:
        yerr = (
            - .429377480158730158730159e-2 * k1
            + .186685860938578329882678e-1 * k3
            - .341550268308080808080807e-1 * k4
            - .1932198660714285714285714e-1 * k5
            + .391022021456804065499718e-1 * k6
            )
        return xn,yn,yerr
    else:
        return xn,yn

class DictIntegrator(Integrator):
    r""" Integrate ``dy/dx = deriv(x,y)`` where ``y`` is a dictionary.

    An :class:`DictIntegrator` object ``odeint`` 
    integrates ``dy/dx = f(x,y)`` to obtain ``y(x1)`` from 
    ``y0 = y(x0)``. ``y`` and ``f(x,y)`` are
    dictionary types having the same keys, and containing scalars
    and/or :mod:`numpy` arrays as values. Typical usage is::

        from gvar.ode import DictIntegrator

        def f(x, y):
            ...

        odeint = DictIntegrator(deriv=f,  tol=1e-8)
        y1 = odeint(y0, interval=(x0, x1))
        y2 = odeint(y1, interval=(x1, x2))
        ...

    The first call to ``odeint`` integrates from ``x=x0`` to ``x=x1``,
    returning ``y1=y(x1)``. The second call continues the integration
    to ``x=x2``, returning ``y2=y(x2)``. Multiple integration points
    can be specified in ``interval``, in which case a list of the
    corresponding ``y`` values is returned: for example, ::

        odeint = DictIntegrator(deriv=f,  tol=1e-8)
        y1, y2 ... = odeint(y0, interval=[x0, x1, x2 ...])

    The integrator uses an adaptive Runge-Kutta algorithm that adjusts
    the integrator's step size to obtain relative accuracy ``tol`` 
    in the solution. An initial step size can be set in the 
    :class:`DictIntegrator` by specifying parameter ``h``. 
    A minimum ste psize ``hmin`` can also be specified; the 
    :class:`Integrator` raises an exception if the step size becomes
    smaller than ``hmin``. The :class:`DictIntegrator` keeps track of the
    number of good steps, where ``h`` is increased, and the number of
    bad steps, where ``h`` is decreases and the step is repeated:
    ``odeint.ngood`` and ``odeint.nbad``, respectively.

    An analyzer ``analyzer(x,y)`` can be specified using parameter
    ``analyzer``. This function is called after every full step of
    the integration with the current values of ``x`` and ``y``.
    Objects of type :class:`gvar.ode.Solution` are examples of
    (simple) analyzers.

    Args:
        deriv: Function of ``x`` and ``y`` that returns ``dy/dx``.
            The return value should be a dictionary with the same
            keys as ``y``, and values that have the same
            shape as the corresponding values in ``y``.
        tol (float): Relative accuracy in ``y`` relative to 
            ``|y| + h|dy/dx|`` for each step in the integration. 
            Any integration step that achieves less precision is 
            repeated with a smaller step size. The step size
            is increased if precision is higher than needed. 
            Default is 1e-5.
        h (float or None): Absolute value of initial step size. 
            The default value equals the entire width of the 
            integration interval.
        hmin (float or None): Smallest step size allowed. A warning 
            is raised if a smaller step size is requested, and 
            the step size is not decreased. This prevents infinite loops 
            at singular points, but the solution may not be reliable when 
            a warning has been issued. The default value is ``None`` 
            (which does *not* prevent infinite loops).
        hmax (float or None): Largest step allowed. Ignored if 
            set to ``None``.
        maxstep (int or None): Maximum number of integration steps 
            allowed, after which a ``RuntimeError`` exception is raised. 
            Ignored if set to ``None``.
        delta: Function ``delta(yerr, y, delta_y)`` that returns
            a number to be compared  with ``tol`` at each integration step:
            if it is larger than ``tol``, the step is repeated with a smaller
            step size; if it is smaller the step is accepted and a larger
            step size used for the subsequent step. Here ``yerr`` is an
            estimate of the error in ``y`` on the last step; ``y`` is the
            proposed value; and ``delta_y`` is the change in ``y`` over
            the last step.
        analyzer: Function of ``x`` and ``y`` that is called after each
            step of the integration. This can be used to analyze intermediate
            results.
    """
    def __init__(self, **args):
        super(DictIntegrator, self).__init__(**args)
    def __call__(self, y0, interval):
        if len(interval) > 2:
            ans = []
            xlast = interval[0]
            ylast = y0
            nbad = 0
            ngood = 0
            for x in interval[1:]:
                y = self(ylast, interval=(xlast,x))
                xlast = x
                ylast = y 
                nbad += self.nbad 
                ngood += self.ngood
                ans.append(y)
            self.nbad = nbad 
            self.ngood = ngood 
            return ans
        if not isinstance(y0, gvar.BufferDict):
            y0 = gvar.BufferDict(y0)
        deriv_orig = self.deriv
        def deriv(x, y):
            y = gvar.BufferDict(y0, buf=y)
            dydx = gvar.BufferDict(deriv_orig(x, y), keys=y0.keys())
            return dydx.buf
        self.deriv = deriv
        ans = super(DictIntegrator, self).__call__(y0.buf, interval)
        self.deriv = deriv_orig
        return gvar.BufferDict(y0, buf=ans)

def integral(fcn, interval, fcnshape=None, tol=1e-8, hmin=None):
    r""" Compute integral of ``fcn(x)`` on interval.

    Given a function ``fcn(x)`` the call ::

        result = integral(fcn, interval=(x0, x1))

    calculates the integral of ``fcn(x)`` from ``x0`` to ``x1``.
    For example::

        >>> def fcn(x):
        ...    return math.sin(x) ** 2 / math.pi
        >>> result = integral(fcn, (0, math.pi))
        >>> print(result)
        0.500000002834

    Function ``fcn(x)`` can return a scalar or an array (any shape):
    for example, ::

        >>> def fcn(x):
        ...    return np.array([1., x, x**3])

        >>> result = integral(fcn, (0,1))
        >>> print(result)
        [1. 0.5 0.25]

    The function can also return dictionaries whose values are
    scalars or arrays: for example, ::

        >>> def fcn(x):
        ...    return dict(x=x, x3=x**3)
        >>> result = integral(fcn, (0,1))
        >>> print(result)
        {'x': 0.5,'x3': 0.25}

    :param fcn: Function of scalar variable ``x`` that returns the integrand.
        The return value should be either a scalar or an array, or a
        dictionary whose values are scalars and/or arrays.
    :param interval: Contains the interval ``(x0,x1)`` over which the integral
        is computed.
    :param fcnshape: Contains the shape of the array returned by ``f(x)`` or
        ``()`` if the function returns a scalar. Setting ``fshape=None``
        (the default) results in an extra function evaluation to determine
        the shape.
    :param tol: Relative accuracy of result.
    :param hmin: Smallest step size allowed in adaptive integral. A warning is
        raised if a smaller step size is requested, and the step size is not
        decreased. This prevents infinite loops at singular points, but
        the integral may not be accurate when a warning has been issued. The
        default value is ``None`` (which does *not* prevent infinite loops).
    """
    if fcnshape is None:
        fx0 = fcn(interval[0])
        if hasattr(fx0, 'keys'):
            fx0 = gvar.BufferDict(fx0)
            fcnshape = None
        else:
            fcnshape = numpy.shape(fx0)
    if fcnshape is None:
        def deriv(x, y, fcn=fcn):
            return gvar.BufferDict(fcn(x)).buf
        y0 = fx0.buf * 0.0
    else:
        def deriv(x, y, fcn=fcn):
            return fcn(x)
        y0 = 0.0 if fcnshape == () else numpy.zeros(fcnshape, float)
    odeint = Integrator(deriv=deriv, tol=tol, hmin=hmin)
    ans = odeint(y0, interval=interval)
    return ans if fcnshape is not None else gvar.BufferDict(fx0, buf=ans)

class Solution:
    r""" ODE analyzer for storing intermediate values.

    Usage: eg, given ::

        odeint = Integrator(...)
        soln = Solution()
        y0 = ...
        y = odeint(y0, interval=(x0, x), analyzer=soln)

    then the ``soln.x[i]`` are the points at which the integrator
    evaluated the solution, and ``soln.y[i]`` is the solution
    of the differential equation at that point.
    """
    def __init__(self):
        self.x = []
        self.y = []

    def __call__(self,x,y):
        self.x.append(x)
        self.y.append(y)
