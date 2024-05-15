.. |GVar| replace:: :class:`gvar.GVar`

.. |~| unicode:: U+00A0
   :trim:

.. _case-study-creating-an-integrator:

Case Study: Creating an Integrator
===================================
This case study illustrates how to convert an existing numerical 
analysis routine, :func:`scipy.integrate.quad`, to work with |GVar|\s. 

The Problem
-----------------
We want a Python code that can evaluate one dimensional integrals such 
as 

.. math::

    I = \int\limits_a^b dx \, f(x)

where any of the integration limits or :math:`f(x)` are |GVar|\s and 
:math:`f(x)` is an arbitrary function coded as a Python function. 

One approach is to implement an integration function directly in 
Python, as then it is likely to work just as well for |GVar|\s as 
for floats. For example, the code ::

    >>> import gvar as gv 
    >>> import numpy as np 
    >>>
    >>> def trap_integral(f, interval, n=100):
    ...     """ Estimate integral of f(x) on interval=(a,b) using the Trapezoidal Rule. """
    ...     a, b = interval
    ...     x = a + (b - a) * np.linspace(0, 1., n+1)
    ...     fx = np.array([f(xi) for xi in x])
    ...     I =  np.sum(fx[:-1] + fx[1:]) * (b - a) / (2. * n)
    ...     return I
    ... 
    >>> A = gv.gvar(2, 0.1)
    >>> K = gv.gvar(1, 0.11)
    >>> D = gv.gvar(1., 0.4)
    >>> 
    >>> def f(x):
    ...     return A * np.cos(K * x**2 + D) ** 2
    ... 
    >>> a = gv.gvar(0, 0.1)
    >>> b = gv.gvar(4, 0.1)
    >>> Itrap = trap_integral(f, (a, b), n=100)
    >>> print(f'Itrap = {Itrap:#P}')
    Itrap = 3.45 ± 0.32

estimates the integral of ``f(x)`` over the interval between 0 ± 0.1 and 4 ± 0.1
using the Trapezoidal Rule.

This code is simple because we are using one of the simplest numerical 
estimates of the integral. A general purpose integrators needs a 
much more robust algorithm. For example, ``trap_integral`` fails badly when 
applied to a much more singular function::

    >>> def g(x):
    ...    return A * x /(K * x**2 + 1e-6)
    ...
    >>> Itrap_g = trap_integral(g, (a, b), n=100)
    >>> print(f'Itrap_g = {Itrap_g:#P}')
    Itrap_g = 10.3633 ± 4.0e+03

The correct answer is 16.6 ± 1.9. We need a much larger number of integrand 
samples ``n`` (100x larger) to get reasonable results.

Leveraging Existing Code
---------------------------
Coding a more robust integrator is complicated and time consuming. A better 
strategy is, if possible, to build on existing libraries. Here we will use 
integrators from the :mod:`scipy.integrate` module.

The integral :math:`I` is a function of its endpoints and of any parameters buried
in the definition of the function :math:`f(x)`: :math:`I = I(p)` where
:math:`p = [a, b, ...]` and :math:`p_i` for :math:`i>1`
are the parameters implicit in the integrand (e.g.,
``A`` and ``K`` in the examples above). We want an integrator 
that works when any of these parameters is replaced by a |GVar|.

We can do this using :any:`gvar.gvar_function`\ ``(p, I, dI_dp)`` where ``p``
is an array of the |GVar|-valued parameters,
``I`` is the integral evaluated with these parameters replaced 
by their mean values, and ``dI_dp`` is the array of derivatives 
of ``I`` with respect to each of these parameters 
--- :math:`[dI/dp_0, dI/dp_1, ...]` --- again 
evaluated with their mean values.

The integral ``I`` (with the parameters replaced by their mean values)
can be evaluated using standard routines as no |GVar|\s are involved.
The derivatives with respect to the endpoints are also easily evaluated:

.. math::

    \frac{dI}{da} = - f(a) \quad\quad \frac{dI}{db} = f(b)

The derivatives with respect to the function parameters involve 
different integrals, which again can be evaluated using standard 
routines:

.. math::

    \frac{dI}{dp_i} = \int\limits_a^b dx \, \frac{df(x)}{dp_i} \quad\quad \mbox{for $i>1$}

In the following code we use the integrators ``quad(...)`` and ``quad_vec(...)`` from 
:mod:`scipy.integrate` to evaluate the integrals 
needed to calculate ``I`` and elements of ``dI_dp``, respectively::

    import scipy.integrate

    def integral(f, interval, tol=1e-8):
        """ GVar-compatible integrator """
        a, b = interval

        # collect GVar-valued parameters
        p = []
        dI_dp = []
        if isinstance(a, gv.GVar):
            p += [a]
            dI_dp += [-f(a).mean]
            a = a.mean
        if isinstance(b, gv.GVar):
            p += [b]
            dI_dp += [f(b).mean]
            b = b.mean

        # evaluate integral I of f(x).mean
        sum_fx = [0]
        def fmean(x):
            fx = f(x)
            if isinstance(fx, gv.GVar):
                sum_fx[0] += fx
                return fx.mean
            else:
                return fx
        I = scipy.integrate.quad(fmean, a, b, epsrel=tol)[0]

        # parameters from the integrand
        pf = gv.dependencies(sum_fx[0], all=True)

        # evaluate dI/dpf
        if len(pf) > 0:
            # vector-valued integrand returns df(x)/dpf
            def df_dpf(x):
                fx = f(x)
                if isinstance(fx, gv.GVar):
                    return fx.deriv(pf)
                else:
                    return np.array(len(pf) * [0.0])

            # integrate df/dpf to obtain dI/dpf
            dI_dpf = scipy.integrate.quad_vec(df_dpf, a, b, epsrel=tol)[0]

            # combine with other parameters, if any
            p += list(pf)
            dI_dp += list(dI_dpf)

        return gv.gvar_function(p, I, dI_dp) if len(p) > 0 else I

A key ingredient of this code is the use of :func:`gvar.dependencies` to obtain 
an array ``pf`` of the |GVar|-valued parameters implicit in the integrand ``f(x)``. This is 
done without
knowing anything about ``f(x)`` beyond the sum ``sum_fx[0]`` of its values
at all the integration points used to calculate |~| ``I``. Given parameters |~| ``pf[i]``,
the derivatives of ``f(x)`` with respect to those parameters are obtained 
using ``f(x).deriv(pf)`` (see the documentation for :meth:`gvar.GVar.deriv`).

This new integrator works well with the first example above and gives the same result::

    >>> I = integral(f, (a, b))
    >>> print(f'I = {I:#P}')
    I = 3.45 ± 0.32

It also works well with the much more singular integrand ``g(x)``::

    >>> I_g = integral(g, (a, b))
    >>> print(f'I_g = {I_g:#P}')
    I_g = 16.6 ± 1.9

:mod:`gvar` comes with a different integrator, :func:`gvar.ode.integral`, that gives 
the same results with similar performance: for example, ::

    >>> Iode = gv.ode.integral(f, (a, b))
    >>> print(f'Iode = {Iode:#P}')
    Iode = 3.45 ± 0.32
    >>> Iode_g = gv.ode.integral(g, (a, b))
    >>> print(f'Iode_g = {Iode_g:#P}')
    Iode_g = 16.6 ± 1.9

We can generate error budgets for each of the integral estimates to see where 
the final uncertainties come from::

    >>> inputs = dict(a=a, b=b, A=A, K=K, D=D)
    >>> outputs = dict(I=I, Iode=Iode, Itrap=Itrap)
    >>> print(gv.fmt_errorbudget(inputs=inputs, outputs=outputs))
    Partial % Errors:
                       I      Iode     Itrap
    ----------------------------------------
            a:      1.69      1.69      1.69
            b:      0.44      0.44      0.52
            A:      5.00      5.00      5.00
            K:      4.53      4.53      4.35
            D:      6.29      6.29      6.25
    ----------------------------------------
        total:      9.39      9.39      9.28

As expected the different methods are in good agreement 
(the Trapezoidal Rule gives slightly different results 
because ``n`` is a bit too small).