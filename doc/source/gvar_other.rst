.. _numerical-analysis-modules-in-gvar:

Numerical Analysis Modules in :mod:`gvar`
==========================================

.. |GVar| replace:: :class:`gvar.GVar`

|GVar|\s can be used in many numerical algorithms, to propagates errors
through the algorithm. A code that is written in pure Python is likely to
work well with |GVar|\s, perhaps with minor modifications.
Here we describe some sample numerical codes, included in
:mod:`gvar`, that have been adapted
to work with |GVar|\s, as well as with ``float``\s.
More examples will follow with time.

The sub-modules included here are:

    - :mod:`gvar.cspline` --- cubic splines for 1-d data.

    - :mod:`gvar.linalg` --- basic linear algebra.

    - :mod:`gvar.ode` --- integration of systems of ordinary differential equations;
        :ref:`one-dimensional integrals <integral>`.

    - :mod:`gvar.powerseries` --- power series representation of functions.

    - :mod:`gvar.root` --- root-finding for one-dimensional functions.


.. module:: gvar.cspline
   :synopsis: Cubic splines.

Cubic Splines
-----------------

Module :mod:`gvar.cspline` implements a class for smoothing and/or
interpolating one-dimensional data using cubic splines:

.. autoclass:: gvar.cspline.CSpline


.. module:: gvar.linalg
   :synopsis: Basic linear algebra.

Linear Algebra
---------------------

Module :mod:`gvar.linalg` implements several methods for doing basic
linear algebra with matrices whose elements can be either numbers or
:class:`gvar.GVar`\s:

.. automethod:: gvar.linalg.det

.. automethod:: gvar.linalg.slogdet

.. automethod:: gvar.linalg.inv

.. automethod:: gvar.linalg.solve

.. automethod:: gvar.linalg.eigvalsh


.. module:: gvar.ode
   :synopsis: Ordinary differential equations.

Ordinary Differential Equations
------------------------------------------------

Module :mod:`gvar.ode` implements two classes for integrating systems
of first-order differential equations using an adaptive Runge-Kutta
algorithm. One integrates scalar- or array-valued equations, while the
other integrates dictionary-valued equations:

.. autoclass:: gvar.ode.Integrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

.. autoclass:: gvar.ode.DictIntegrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

A simple analyzer class is:

.. autoclass:: gvar.ode.Solution()


.. _integral:

One-Dimensional Integration
----------------------------

Module :mod:`gvar.ode` also provides a method for evaluating
one-dimensional integrals (using its adaptive Runge-Kutta algorithm):

.. automethod:: gvar.ode.integral



Power Series
--------------
.. automodule:: gvar.powerseries
    :synopsis: Power series arithmetic and evaluation.

.. autoclass:: gvar.powerseries.PowerSeries
    :members:



.. module:: gvar.root
   :synopsis: Roots (zeros) of one-dimensional functions.

Root Finding
--------------

Module :mod:`gvar.root` contains methods for finding the roots of
of one-dimensional functions: that is, finding ``x`` such that
``fcn(x)=0`` for a given function ``fcn``. Typical usage is::

    >>> import math
    >>> import gvar as gv
    >>> interval = gv.root.search(math.sin, 1.)     # bracket root
    >>> print(interval)
    (3.1384283767210035, 3.4522712143931042)
    >>> root = gv.root.refine(math.sin, interval)   # refine root
    >>> print(root)
    3.14159265359

This code finds the first root of ``sin(x)=0`` larger than 1. The first
setp is a search to find an interval containing a root. Here
:meth:`gvar.root.search` examines ``sin(x)`` for a sequence of points
``1. * 1.1 ** n`` for ``n=0,1,2...``, stopping when the function changes
sign. The last two points in the sequence then bracket a root
since ``sin(x)`` is continuous; they are returned as a tuple to ``interval``.
The final root is found by refining the interval, using ``gvar.root.refine``.
By default, the root is refined iteratively to machine precision, but this
requires only a small number (4) of iterations::

    >>> print(root.nit)                             # number of iterations
    4

The most challenging situations are ones where the function
is extremely flat in the vicinity of the root --- that is,
two or more of its leading derivatives vanish there. For
example::

    >>> import gvar as gv
    >>> def f(x):
    ...     return (x + 1) ** 3 * (x - 0.5) ** 11
    >>> root = gv.root.refine(f, (0, 2))
    >>> print(root)
    0.5
    >>> print(root.nit)                             # number of iterations
    142

This routine also works with variables of type :class:`gvar.GVar`:
for example, ::

    >>> import gvar as gv
    >>> def f(x, w=gv.gvar(1, 0.1)):
    ...     return gv.sin(w * x)
    >>> root = gv.root.refine(f, (1, 4))
    >>> print(root)
    3.14(31)

returns a root with a 10% uncertainty, reflecting the
uncertainty in parameter ``w``.

Descriptions of the two methods follow.

.. automethod:: gvar.root.search

.. automethod:: gvar.root.refine

