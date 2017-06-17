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

Objects of type :class:`gvar.GVar` represent gaussian random variables,
which are specified by a mean and standard deviation. They are created
using :func:`gvar.gvar`: for example, ::

    >>> x = gvar.gvar(10,3)          # 0 +- 3
    >>> y = gvar.gvar(12,4)          # 2 +- 4
    >>> z = x + y                    # 2 +- 5
    >>> print(z)
    22.0(5.0)
    >>> print(z.mean)
    22.0
    >>> print(z.sdev)
    5.0

This module contains a variety of tools for creating and
manipulating gaussian random variables, including:

    - :any:`mean`\ ``(g)`` --- extract means.

    - :any:`sdev`\ ``(g)`` --- extract standard deviations.

    - :any:`var`\ ``(g)`` --- extract variances.

    - :any:`fmt`\ ``(g)`` --- replace all |GVar|\s in array/dictionary by string representations.

    - :any:`tabulate`\ ``(g)`` --- tabulate entries in array/dictionary of |GVar|\s.

    - :any:`correlate`\ ``(g, corr)`` --- add correlations to |GVar|\s in array/dictionary ``g``.

    - :any:`chi2`\ ``(g1, g2)`` --- ``chi**2`` of ``g1-g2``.

    - :any:`equivalent`\ ``(g1, g2)`` --- |GVar|\s the same in ``g1`` and ``g2``?

    - :any:`evalcov`\ ``(g)`` --- compute covariance matrix.

    - :any:`evalcov_blocks`\ ``(g)`` --- compute diagonal blocks of covariance matrix.

    - :any:`evalcorr`\ ``(g)`` --- compute correlation matrix.

    - :any:`fmt_values`\ ``(g)`` --- list values for printing.

    - :any:`fmt_errorbudget`\ ``(g)`` --- create error-budget table for printing.

    - :any:`fmt_chi2`\ ``(f)`` --- format chi**2 information in f as string for printing.

    - class :class:`BufferDict` --- ordered dictionary with data buffer.

    - class :class:`PDF` --- probability density function.

    - class :class:`PDFStatistics` --- statistical analysis of moments of a random variable.

    - class :class:`PDFHistogram` --- tool for building PDF histograms.

    - :any:`dump`\ ``(g, outputfile)`` --- serialize a collection of |GVar|\s in file.

    - :any:`dumps`\ ``(g)`` --- serialize a collection of |GVar|\s in a string.

    - :any:`load`\ ``(inputfile)`` --- reconstitute a collection of |GVar|\s from a file.

    - :any:`loads`\ ``(inputstr)`` --- reconstitute a collection of |GVar|\s from a string.

    - :any:`disassemble`\ ``(g)`` --- low-level routine to disassemble a collection of |GVar|\s.

    - :any:`reassemble`\ ``(data,cov)`` --- low-level routine to reassemble a collection of |GVar|\s.

    - :any:`raniter`\ ``(g,N)`` --- iterator for random numbers.

    - :any:`bootstrap_iter`\ ``(g,N)`` --- bootstrap iterator.

    - :any:`svd`\ ``(g)`` --- SVD modification of correlation matrix.

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

.. autofunction:: gvar.fmt(g, ndecimal=None, sep='')

.. autofunction:: gvar.tabulate(g, ncol=1, headers=True, offset='', ndecimal=None)

.. autofunction:: gvar.correlate(g, corr)

.. autofunction:: gvar.evalcov(g)

.. autofunction:: gvar.cov(g1, g2)

.. autofunction:: gvar.evalcov_blocks(g)

.. autofunction:: gvar.evalcorr(g)

.. autofunction:: gvar.corr(g1, g2)

.. autofunction:: gvar.uncorrelated(g1, g2)

.. autofunction:: gvar.chi2(g1, g2, svdcut=1e-15, fmt=False)

.. autofunction:: gvar.fmt_chi2(f)

|GVar|\s are compared by:

.. autofunction:: gvar.equivalent(g1, g2, rtol=1e-10, atol=1e-10)

|GVar|\s can be stored (serialized) and retrieved from files (or strings) using:

.. autofunction:: gvar.dump(g, outputfile)

.. autofunction:: gvar.dumps(g)

.. autofunction:: gvar.load(inputfile)

.. autofunction:: gvar.loads(inputstring)

.. autofunction:: gvar.disassemble(g)

.. autofunction:: gvar.reassemble(data, cov=gvar.gvar.cov)

|GVar|\s contain information about derivatives with respect to the *independent*
|GVar|\s from which they were constructed. This information can be extracted using:

.. autofunction:: gvar.deriv(g, x)

The following function creates an iterator that generates random arrays
from the distribution defined by array (or dictionary) ``g`` of |GVar|\s.
The random numbers incorporate any correlations implied by the ``g``\s.

.. autofunction:: gvar.raniter(g, n=None, svdcut=None)

.. autofunction:: gvar.bootstrap_iter(g, n=None, svdcut=None)

.. autofunction:: gvar.ranseed(a)

The following two functions that are useful for tabulating results
and for analyzing where the errors in a |GVar| constructed from
other |GVar|\s come from:

.. autofunction:: gvar.fmt_errorbudget(outputs, inputs, ndecimal=2, percent=True, verify=False, colwidth=10)

.. autofunction:: gvar.fmt_values(outputs, ndecimal=None)

The following function applies an SVD cut to the correlation matrix
of a set of |GVar|\s:

.. autofunction:: gvar.svd

This function is useful when the correlation matrix is singular
or almost singular, and its inverse is needed (as in curve fitting).

The following function can be used to rebuild collections of |GVar|\s,
ignoring all correlations with other variables. It can also be used to
introduce correlations between uncorrelated variables.

.. autofunction:: gvar.rebuild(g, gvar=gvar, corr=0.0)


The following functions creates new functions that generate |GVar|\s (to
replace :func:`gvar.gvar`):

.. autofunction:: gvar.switch_gvar()

.. autofunction:: gvar.restore_gvar()

.. autofunction:: gvar.gvar_factory(cov=None)

|GVar|\s created by different functions cannot be combined in arithmetic
expressions (the error message "Incompatible GVars." results).


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

   There are two methods for converting ``self`` into a string, for
   printing:

   .. automethod:: __str__

   .. automethod:: fmt(ndecimal=None, sep='')

   Two attributes and a method make reference to the original
   variables from which ``self`` is derived:

   .. attribute:: cov

      Underlying covariance matrix (type :class:`gvar.smat`) shared by all
      |GVar|\s.

   .. autoattribute:: der

   .. automethod:: dotder(v)

:class:`gvar.BufferDict` Objects
----------------------------------
The following class is a specialized form of an ordered dictionary for
holding |GVar|\s (or other scalars) and arrays of |GVar|\s (or other
scalars) that supports Python pickling:

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

   The main methods are:

   .. automethod:: flatten()

   .. automethod:: slice(k)

   .. automethod:: slice_shape(k)

   .. automethod:: isscalar(k)

   .. method:: update(d)

      Add contents of dictionary ``d`` to ``self``.

   .. .. staticmethod:: BufferDict.load(fobj, use_json=False)

   ..    Load serialized |BufferDict| from file object ``fobj``.
   ..    Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
   ..    it uses :mod:`json` (obvioulsy).

   .. .. staticmethod:: loads(s, use_json=False)

   ..    Load serialized |BufferDict| from string object ``s``.
   ..    Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
   ..    it uses :mod:`json` (obvioulsy).

   .. .. automethod:: dump(fobj, use_json=False)

   .. .. automethod:: dumps(use_json=False)

:class:`gvar.SVD` Objects
---------------------------
SVD analysis is handled by the following class:

.. autoclass:: gvar.SVD(mat, svdcut=None, svdnum=None, compute_delta=False, rescale=False)

   .. automethod:: decomp(n)

:class:`vegas.PDFIntegrator` and other PDF-related Objects
-----------------------------------------------------------
Expectation values using probability density functions defined by
collections of |GVar|\s can be evaluated using the :mod:`vegas`
module (for multi-dimensional integration) and class
:class:`vegas.PDFIntegrator`. Related classes are:

.. autoclass:: gvar.PDF(g, svdcut=1e-15)

.. autoclass:: gvar.PDFStatistics(moments=None, histogram=None)

.. autoclass:: gvar.PDFHistogram(g, nbin=None, binwidth=None, bins=None)

  The main methods are:

  .. automethod:: count(data)

  .. automethod:: analyze(count)

  .. automethod:: make_plot(count, plot=None, show=False, , plottype='probability', bar=dict(alpha=0.15, color='b'), errorbar=dict(fmt='b.'), gaussian=dict(ls='--', c='r'))

Requirements
------------
:mod:`gvar` makes heavy use of :mod:`numpy` for array manipulations. It
also uses the :mod:`numpy` code for implementing elementary functions
(*e.g.*, ``sin``, ``exp`` ...) in terms of member functions.
