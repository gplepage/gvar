:mod:`gvar` - Gaussian Random Variables
==================================================

.. |GVar| replace:: :class:`gvar.GVar`

.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. the next definition is a non-breaking white space

.. |~| unicode:: U+00A0
   :trim:

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. module:: gvar
   :synopsis: Correlated Gaussian random variables.

Introduction
------------

Objects of type :class:`gvar.GVar` represent Gaussian random variables,
which are specified by means and covariances. They are created
using :func:`gvar.gvar`: for example, ::

    >>> x = gvar.gvar(10, 3)         # 10 ± 3
    >>> y = gvar.gvar(12, 4)         # 12 ± 4
    >>> z = x + y                    # 22 ± 5
    >>> print(z)
    22.0(5.0)
    >>> print(z.mean)
    22.0
    >>> print(z.sdev)
    5.0
    >>> print((z - x) / y)           # z is correlated with y
    1(0)

This module contains a variety of tools for creating and
manipulating Gaussian random variables, including:

*Creating or modifying*

    - :any:`gvar.gvar`\ ``(...)`` --- create new |GVar|

    - :any:`gvar.gvar_function`\ ``(x, f, dfdx)`` --- create |GVar| for function of |GVar|.

    - :any:`regulate`\ ``(g, svdcut|eps)`` --- regulate ``g``'s correlation matrix.

    - :any:`svd`\ ``(g, svdcut)`` --- SVD regulation of correlation matrix.

    - :any:`correlate`\ ``(g, corr)`` --- add correlations to |GVar|\s in array/dictionary ``g``.
    
*Examining and manipulating*

    - :any:`mean`\ ``(g)`` --- extract means.

    - :any:`sdev`\ ``(g)`` --- extract standard deviations.

    - :any:`var`\ ``(g)`` --- extract variances.

    - :any:`deriv`\ ``(g, x)`` --- derivatives of ``g`` with respect to ``x``.

    - :any:`is_primary`\ ``(g)`` --- test whether |GVar| is primary or derived.

    - :any:`evalcov`\ ``(g)`` --- compute covariance matrix.

    - :any:`evalcov_blocks`\ ``(g)`` --- compute diagonal blocks of covariance matrix.

    - :any:`evalcorr`\ ``(g)`` --- compute correlation matrix.

    - :any:`cov`\ ``(g1, g2)`` --- covariance between ``g1`` and ``g2``.

    - :any:`corr`\ ``(g1, g2)`` --- correlation between ``g1`` and ``g2``.

    - :any:`uncorrelated`\ ``(g1, g2)`` --- test for correlation

    - :any:`chi2`\ ``(g1, g2)`` --- ``chi**2`` of ``g1-g2``.

    - :any:`filter`\ ``(g, f, *args, **kargs)`` --- filter |GVar|\s in ``g`` through function ``f``.

*Sampling*

    - :any:`sample`\ ``(g, nbatch)``  --- random sample from collection of |GVar|\s.

    - :any:`raniter`\ ``(g, n, nbatch)`` --- iterator for random numbers.

    - :any:`gvar_from_sample`\ ``(gs)`` --- |GVar|\s from random sample.

    - :any:`bootstrap_iter`\ ``(g, n)`` --- bootstrap iterator.

    - :any:`ranseed`\ ``(seed)`` --- seed random number generator.

*Formatting*
    
    - :any:`fmt`\ ``(g)`` --- replace all |GVar|\s in array/dictionary by string representations.

    - :any:`tabulate`\ ``(g)`` --- tabulate entries in array/dictionary of |GVar|\s.

    - :any:`qqplot`\ ``(g1, g2)`` --- QQ plot of ``g1-g2``,

    - :any:`equivalent`\ ``(g1, g2)`` --- |GVar|\s the same in ``g1`` and ``g2``?

    - :any:`fmt_values`\ ``(g)`` --- list values for printing.

    - :any:`fmt_errorbudget`\ ``(g)`` --- create error-budget table for printing.

    - :any:`fmt_chi2`\ ``(f)`` --- format chi**2 information in f as string for printing.

*Saving*

    - :any:`dump`\ ``(g, outputfile)`` --- serialize data, including |GVar|\s, from ``g`` in file.

    - :any:`dumps`\ ``(g)`` --- serialize data from ``g`` in a bytes object.

    - :any:`load`\ ``(inputfile)`` ---  reconstitute data serialized in a file.

    - :any:`loads`\ ``(inputbytes)`` --- reconstitute data serialized in a bytes object.

    - :any:`gdump`\ ``(g, outputfile)`` --- serialize a collection of |GVar|\s in file.

    - :any:`gdumps`\ ``(g)`` --- serialize a collection of |GVar|\s in a string.

    - :any:`gload`\ ``(inputfile)`` --- reconstitute a collection of |GVar|\s from a file.

    - :any:`gloads`\ ``(inputstr)`` --- reconstitute a collection of |GVar|\s from a string.

    - :any:`remove_gvars`\ ``(g, gvlist)`` --- remove |GVar|\s from ``g``, appending them to ``gvlist``.

    - :any:`distribute_gvars`\ ``(g, gvlist)`` --- restore |GVar|\s to ``g`` from ``gvlist``.

    - class :class:`GVarRef` --- placeholder for |GVar| removed by :func:`gvar.remove_gvars`.

    - :any:`dependencies`\ ``(g)`` --- collect primary |GVar|\s contributing to ``g``.

    - :any:`disassemble`\ ``(g)`` --- low-level routine to disassemble a collection of |GVar|\s.

    - :any:`reassemble`\ ``(data,cov)`` --- low-level routine to reassemble a collection of |GVar|\s.

*Related classes*

    - class :class:`BufferDict` --- ordered dictionary with data buffer.

    - class :class:`SVD` --- SVD decomposition of symmetric matrix

    - class :class:`PDF` --- probability density function.

    - class :class:`PDFStatistics` --- statistical analysis of moments of a random variable.

*Analyzing Monte Carlo datasets*

    - :any:`dataset.bin_data`\ ``(data)`` --- bin random sample data.

    - :any:`dataset.avg_data`\ ``(data)`` --- estimate means of random sample data.

    - :any:`dataset.bootstrap_iter`\ ``(data,N)`` --- bootstrap random sample data.

    - class :class:`dataset.Dataset` --- class for collecting random sample data.

Functions
----------
The function used to create Gaussian variable objects is:

.. autofunction:: gvar.gvar(...)

The following function is useful for constructing new functions that
can accept |GVar|\s as arguments:

.. autofunction:: gvar.gvar_function(x, f, dfdx)

Means, standard deviations, variances, formatted strings, covariance
matrices and correlation/comparison information can be extracted from arrays
(or dictionaries) of |GVar|\s using:

.. autofunction:: gvar.mean(g)

.. autofunction:: gvar.sdev(g)

.. autofunction:: gvar.var(g)

.. autofunction:: gvar.is_primary(g)

.. autofunction:: gvar.dependencies(g, all=False)

.. autofunction:: gvar.filter(g, f, *args, **kargs)

.. autofunction:: gvar.fmt(g, ndecimal=None, sep='')

.. autofunction:: gvar.tabulate(g, ncol=1, headers=True, offset='', ndecimal=None)

.. autofunction:: gvar.correlate(g, corr, upper=False, lower=False, verify=False)

.. autofunction:: gvar.evalcov(g)

.. autofunction:: gvar.cov(g1, g2)

.. autofunction:: gvar.evalcov_blocks(g, compress=False)

.. autofunction:: gvar.evalcorr(g)

.. autofunction:: gvar.corr(g1, g2)

.. autofunction:: gvar.uncorrelated(g1, g2)

.. autofunction:: gvar.chi2(g1, g2, svdcut=1e-12, dof=None)

.. autofunction:: gvar.qqplot(g1, g2, plot=None, svdcut=1e-12, dof=None)

.. autofunction:: gvar.fmt_chi2(f)

|GVar|\s are compared by:

.. autofunction:: gvar.equivalent(g1, g2, rtol=1e-10, atol=1e-10)

|GVar|\s can be stored (serialized) and retrieved from files (or strings) using:

.. autofunction:: gvar.dump(g, outputfile=None, add_dependencies=False, **kargs)

.. autofunction:: gvar.dumps(g, add_dependencies=False, **kargs)

.. autofunction:: gvar.load(inputfile, **kargs)

.. autofunction:: gvar.loads(inputbytes, **kargs)

.. autofunction:: gvar.gdump(g, outputfile=None, method=None, add_dependencies=False, **kargs)

.. autofunction:: gvar.gdumps(g, method='json', add_dependencies=False)

.. autofunction:: gvar.gload(inputfile, method=None, **kargs)

.. autofunction:: gvar.gloads(inputstring)

.. autofunction:: gvar.remove_gvars(g, gvlist)

.. autofunction:: gvar.distribute_gvars(g, gvlist)

.. autoclass:: gvar.GVarRef

.. autofunction:: gvar.disassemble(g)

.. autofunction:: gvar.reassemble(data, cov=gvar.gvar.cov)

|GVar|\s contain information about derivatives with respect to the *primary*
|GVar|\s from which they were constructed. This information can be extracted using:

.. autofunction:: gvar.deriv(g, x)

The following functions are used to generate and analyze random arrays or dictionaries
from the distribution defined by array (or dictionary) ``g`` of |GVar|\s.
The random numbers incorporate any correlations implied by the ``g``\s.

.. autofunction:: gvar.sample(g, eps=None, svdcut=None, uniform=None, nbatch=None, mode='rbatch')

.. autofunction:: gvar.raniter(g, n=None, eps=None, svdcut=None, uniform=None, nbatch=None, mode='rbatch')

.. autofunction:: gvar.gvar_from_sample(gs, mode='rbatch', unbias=False)

.. autofunction:: gvar.bootstrap_iter(g, n=None, eps=None, svdcut=None)

:mod:`gvar`'s random number generator is ``gvar.RNG``. It is currently 
(since version 12.1) an instance of ``numpy.random.Generator``.
The following function is used to reseed the random number generator:

.. autofunction:: gvar.ranseed(seed=None, size=3, version=None)

The following two functions that are useful for tabulating results
and for analyzing where the errors in a |GVar| constructed from
other |GVar|\s come from:

.. autofunction:: gvar.fmt_errorbudget(outputs, inputs, ndecimal=2, percent=True, verify=False, colwidth=10)

.. autofunction:: gvar.fmt_values(outputs, ndecimal=None)

The following functions are used to make correlation matrices 
less singular:

.. autofunction:: gvar.regulate(g, svdcut=1e-12, eps=None, wgts=False, noise=False)

.. autofunction:: gvar.svd(g, svdcut=1e-12, wgts=False, noise=False)

This function is useful when the correlation matrix is singular
or almost singular, and its inverse is needed (as in curve fitting).

The following function can be used to rebuild collections of |GVar|\s,
ignoring all correlations with other variables. It can also be used to
introduce correlations between uncorrelated variables.

.. autofunction:: gvar.rebuild(g, corr=0.0, gvar=gvar.gvar)


The following functions creates new functions that generate |GVar|\s (to
replace :func:`gvar.gvar`):

.. autofunction:: gvar.switch_gvar()

.. autofunction:: gvar.restore_gvar()

.. autofunction:: gvar.gvar_factory(cov=None)


:class:`gvar.GVar` Objects
---------------------------
The fundamental class for representing Gaussian variables is:

.. autoclass:: gvar.GVar

   The basic attributes are:

   .. autoattribute:: mean

   .. autoattribute:: sdev

   .. autoattribute:: var

   Two methods allow one to isolate the contributions to the variance
   or standard deviation coming from other |GVar|\s:

   .. automethod:: partialvar(*args)

   .. automethod:: partialsdev(*args)

   Partial derivatives of the |GVar| with respect to the independent
   |GVar|\s from which it was constructed are given by:

   .. automethod:: deriv(x)

   There are three methods for converting ``self`` into a string for
   printing and display, and fourth for modifying the defaults used
   in formatting:

   .. automethod:: __format__

   .. automethod:: __str__

   .. automethod:: fmt(ndecimal=None, sep='', format='{}')

   .. automethod:: set

   
   Two attributes and a method make reference to the original
   variables from which ``self`` is derived:

   .. attribute:: cov

      Underlying covariance matrix (type :class:`gvar.smat`) shared by all
      |GVar|\s.

   .. autoattribute:: der

   .. automethod:: dotder(v)

:class:`gvar.BufferDict` Objects
----------------------------------
|BufferDict| objects are ordered dictionaries that are heavily used
in :mod:`gvar`'s implementation. They provide the most flexible
representation for multi-dimensional Gaussian distributions.

|BufferDict| objects differ from ordinary dictionaries in two respects. The
first difference is that the dictionary's values must be scalars or :mod:`numpy`
arrays (any shape) of scalars. The scalars can be ordinary integers or floats,
but the dictionary was designed especially for |GVar|\s.
The dictionary's  values are packed into different parts of a single
one-dimensional array or buffer. Items can be added one at a time as in other
dictionaries, ::

  >>> import gvar as gv
  >>> b = gv.BufferDict()
  >>> b['s'] = 0.0
  >>> b['v'] = [1., 2.]
  >>> b['t'] = [[3., 4.], [5., 6.]]
  >>> print(b)
  {
      's': 0.0,
      'v': array([1., 2.]),
      't': array([[3., 4.],
                  [5., 6.]]),
   }

but the values can also be accessed all at once through the buffer::

  >>> print(b.buf)
  [0. 1. 2. 3. 4. 5. 6.]

A previous entry can be overwritten, but the size, shape, and type of the 
data needs to be the same. For example, here setting ``b['s'] = [10., 20.]`` 
would generate a ``ValueError`` exception, but ``b['s'] = 22.`` is fine.

The second difference between |BufferDict|\s and other dictionaries is
illustrated by the following code::

  >>> b = gv.BufferDict()
  >>> b['log(a)'] = gv.gvar('1(1)')
  >>> print(b)
  {'log(a)': 1.0(1.0)}
  >>> print(b['a'], b['log(a)'])
  2.7(2.7) 1.0(1.0)

Even though ``'a'`` is not a key in the dictionary, ``b['a']`` is still
defined: it equals ``exp(b['log(a)'])``. This feature is used to provide
(limited) support for non-Gaussian distributions. Here |BufferDict| ``b``
specifies a distribution that is Gaussain for ``p['log(a)']``, and
therefore log-normal for ``p['a']``. Thus, for example, ::

  >>> [x['a'] for x in gv.raniter(b, n=4)]
  [2.1662650927997817, 2.3350022125310317, 8.732161128765775, 3.578188553455522]

creates a list of four random numbers drawn from a log-normal distribution.
Note that ``'a'`` in this
example is *not* a key in dictionary ``b``, even though both ``b['a']``
and ``b.get('a')`` return values::

  >>> print('a' in b)
  False
  >>> print('log(a)' in b)
  True
  >>> print(list(b))
  ['log(a)']
  >>> print(b['a'], b.get('a'))
  2.7(2.7) 2.7(2.7)

This functionality is used routinely by other modules (e.g., :mod:`lsqfit`).
The supported distributions, and methods for adding new ones, are
described in the documentation for :func:`gvar.BufferDict.add_distribution`,
below.

.. autoclass:: gvar.BufferDict

   The main attributes are:

   .. autoattribute:: size

   .. autoattribute:: flat

   .. autoattribute:: dtype

   .. attribute:: buf

      The (1d) buffer array. Allows direct access to the buffer: for example,
      ``self.buf[i] = new_val`` sets the value of the ``i-th`` element in
      the buffer to value ``new_val``.  Setting ``self.buf = nbuf``
      replaces the old buffer by new buffer ``nbuf``. This only works if
      ``nbuf`` is a one-dimensional :mod:`numpy` array having the same
      length as the old buffer, since ``nbuf`` itself is used as the new
      buffer (not a copy).

   .. attribute:: shape

      Always equal to ``None``. This attribute is included since
      |BufferDict|\s share several attributes with :mod:`numpy` arrays to
      simplify coding that might support either type. Being dictionaries
      they do not have shapes in the sense of :mod:`numpy` arrays (hence
      the shape is ``None``).

   In addition to standard dictionary methods, the main methods here are:

   .. automethod:: flatten()

   .. automethod:: slice(k)

   .. automethod:: slice_shape(k)

   .. automethod:: has_dictkey(k)

   .. automethod:: all_keys()

   .. automethod:: add_distribution(name, invfcn)

   .. automethod:: del_distribution(name)

   .. automethod:: has_distribution(name)

   .. automethod:: uniform(fname, umin, umax, shape=())

   .. automethod:: batch_iter(mode)


:class:`gvar.SVD` Objects
---------------------------
SVD analysis is handled by the following class:

.. autoclass:: gvar.SVD(mat, svdcut=None, svdnum=None, compute_delta=False, rescale=False)

   .. automethod:: decomp(n)

PDF-related Objects
-----------------------------------------------------------
Expectation values using probability density functions defined by
collections of |GVar|\s can be evaluated using the :mod:`vegas`
module (for multi-dimensional integration) and class
:class:`vegas.PDFIntegrator`. Related classes are:

.. autoclass:: gvar.PDF(g, svdcut=1e-12, mode=None)
   
   The main methods are:

   .. automethod:: chi2(p, mode='default')

   .. automethod:: chiv(p, pflat=None, mode='default')

   .. automethod:: pflat(chiv, mode='default')

   .. automethod:: __call__(p, mode='default')

   .. automethod:: logpdf(p, mode='default')

   .. automethod:: sample(nbatch=None, uniform=None, mode='default')


.. autoclass:: gvar.PDFStatistics(moments=None, histogram=None)
   :members:

Requirements
------------
:mod:`gvar` makes heavy use of :mod:`numpy` for array manipulations. It
also uses the :mod:`numpy` code for implementing elementary functions
(*e.g.*, ``sin``, ``exp`` ...) in terms of member functions.
