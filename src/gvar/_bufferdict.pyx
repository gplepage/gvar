# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-16 G. Peter Lepage.
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

import collections
import re
import numpy
import copy
import pickle
import json
try:
    # python 2
    from StringIO import StringIO as _StringIO
    _BytesIO = _StringIO
except ImportError:
    # python 3
    from io import BytesIO as _BytesIO
    from io import StringIO as _StringIO
import gvar as _gvar

try:
    from collections import MutableMapping as collections_MMapping
except ImportError:
    from collections.abc import MutableMapping as collections_MMapping

BUFFERDICTDATA = collections.namedtuple('BUFFERDICTDATA',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==() implies a scalar. """

class BufferDict(collections_MMapping):
    """ Ordered dictionary whose data are packed into a 1-d buffer.

    |BufferDict|\s can be created in the usual way dictionaries are created::

        >>> b = BufferDict()
        >>> b['a'] = 1.
        >>> b['b'] = 2
        >>> print(b)
        {'a': 1.0,'b':2.0}
        >>> b = BufferDict(a=1., b=2.)
        >>> b = BufferDict([('a',1.), ('b',2.)])

    They can also be created from other dictionaries or |BufferDict|\s::

        >>> c = BufferDict(b)
        >>> print(c)
        {'a': 1.0,'b': 2.0}
        >>> c = BufferDict(b, keys=['b'])
        >>> print(c)
        {'b': 2.0}

    The values in a |BufferDict| are scalars or arrays of a scalar type
    (|GVar|, ``float``, ``int``, etc.). The data type is normally inferred
    (dynamically) from the data itself, but can be specified when
    creating the |BufferDict| from another dictionary or list,
    using keyword ``dtype``::

        >>> b = BufferDict(dict(a=1.2), dtype=int)
        >>> print(b, b.dtype)
        {'a': 1} int64

    Some simple arithemetic is allowed between two |BufferDict|\s, say,
    ``g1`` and ``g2`` provided they have the same keys and array shapes.
    So, for example::

        >>> a = BufferDict(a=1., b=[2., 3.])
        >>> b = BufferDict(a=10., b=[20., 30.])
        >>> print(a + b)
            {'a': 11.0,'b': array([22., 33.])}

    Subtraction is also defined, as are multiplication and division
    by scalars. The corresponding ``+=``, ``-=``, ``*=``, ``/=`` operators
    are supported, as are unary ``+`` and ``-``.

    Finally a |BufferDict| can be cloned from another one but with
    a different buffer (containing different values)::

        >>> b = BufferDict(a=1., b=2.)
        >>> c = BufferDict(b, buf=[10, 20])
        >>> print(c)
        {'a': 10,'b': 20}
    """

    extension_pattern = re.compile('^([^()]+)\((.+)\)$')
    extension_fcn = {}

    def __init__(self, *args, **kargs):
        self._extension = {}
        self.shape = None
        if len(args)==0:
            # kargs are dictionary entries
            self._odict = collections.OrderedDict()
            self._buf = numpy.array([],numpy.intp)
            for k in kargs:
                self[k] = kargs[k]
        elif len(args) == 1 and 'keys' in kargs and len(kargs) == 1:
            self._odict = collections.OrderedDict()
            self._buf = numpy.array([], numpy.intp)
            try:
                for k in kargs['keys']:
                    self[k] = args[0][k]
            except KeyError:
                raise KeyError('Dictionary does not contain key in keys: ' + str(k))
        else:
            dtype = None
            if len(args)==2 and len(kargs)==0:
                bd, buf = args
            elif len(args)==1 and len(kargs)==0:
                bd = args[0]
                buf = None
            elif len(args)==1 and 'buf' in kargs and len(kargs)==1:
                bd = args[0]
                buf = kargs['buf']
            elif len(args) == 1 and 'dtype' in kargs and len(kargs)==1:
                bd = args[0]
                buf = None
                dtype = kargs['dtype']
            else:
                raise ValueError("Bad arguments for BufferDict.")
            if isinstance(bd, BufferDict):
                # make copy of BufferDict bd, possibly with new buffer
                # copy keys, slices and shapes
                self._odict = collections.OrderedDict(bd._odict)
                # copy buffer or use new one
                self._buf = (
                    numpy.array(bd._buf, dtype=dtype)
                    if buf is None else
                    numpy.asarray(buf)
                    )
                if bd.size != self.size:
                    raise ValueError("buf is wrong size --- %s not %s"
                                     % (self.size, bd.size))
                if self._buf.ndim != 1:
                    raise ValueError("buf must be 1-d, not shape = %s"
                                     % (self._buf.shape,))
            elif buf is None:
                self._odict = collections.OrderedDict()
                self._buf = numpy.array(
                    [], numpy.intp if dtype is None else dtype
                    )
                # add initial data
                if hasattr(bd,"keys"):
                    # bd a dictionary
                    for k in bd:
                        self[k] = bd[k]
                else:
                    # bd an array of tuples
                    for ki, vi in bd:
                        self[ki] = vi
                if dtype is not None and self._buf.dtype != dtype:
                    self._buf = numpy.array(self._buf, dtype=dtype)
            else:
                raise ValueError(
                    "bd must be a BufferDict in BufferDict(bd,buf), not %s"
                                    % str(type(bd)))

    def __getstate__(self):
        """ Capture state for pickling when elements are GVars. """
        state = {}
        buf = self._buf
        if len(self._buf) > 0 and isinstance(self._buf[0], _gvar.GVar):
            state['buf'] = ( _gvar.mean(buf),  _gvar.evalcov(buf))
        else:
            state['buf'] = numpy.asarray(buf)
        layout = collections.OrderedDict()
        od = self._odict
        for k in self:
            layout[k] = (od.__getitem__(k).slice, od.__getitem__(k).shape)
        state['layout'] = layout
        return state

    def __setstate__(self, state):
        """ Restore state when unpickling when elements are GVars. """
        layout = state['layout']
        buf = state['buf']
        if isinstance(buf, tuple):
            buf = _gvar.gvar(*buf)
        for k in layout:
            self._odict.__setitem__(
                k,
                BUFFERDICTDATA(
                    slice=layout[k][0],
                    shape=layout[k][1] if layout[k][1] is not None else ()
                    )
                )
        self._buf = buf
        self._extension = {}
        self.shape = None

    def __reduce_ex__(self, dummy):
        return (BufferDict, (), self.__getstate__())

    def __iadd__(self, g):
        """ self += |BufferDict| (or dictionary) """
        g = BufferDict(g, keys=self.keys())
        self.flat[:] += g.flat
        return self

    def __isub__(self, g):
        """ self += |BufferDict| (or dictionary) """
        g = BufferDict(g, keys=self.keys())
        self.flat[:] -= g.flat
        return self

    def __imul__(self, x):
        """ ``self *= x`` for scalar ``x`` """
        self.flat[:] *= x
        return self

    def __itruediv__(self, x):
        """ ``self /= x`` for scalar ``x`` """
        self.flat[:] /= x
        return self

    def __pos__(self):
        """ ``-self`` """
        return BufferDict(self, buf=+self.flat[:])

    def __neg__(self):
        """ ``-self`` """
        return BufferDict(self, buf=-self.flat[:])

    def __add__(self, g):
        """ :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        g = BufferDict(g, keys=self.keys())
        return BufferDict(self, buf=self.flat[:] + g.flat)

    def __radd__(self, g):
        """ Add ``self`` to another :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        g = BufferDict(g, keys=self.keys())
        return BufferDict(self, buf=self.flat[:] + g.flat)

    def __sub__(self, g):
        """ Subtract a :class:`BufferDict` (or a dictionary) from ``self``.

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        g = BufferDict(g, keys=self.keys())
        return BufferDict(self, buf=self.flat[:] - g.flat)

    def __rsub__(self, g):
        """ Subtract ``self`` from a :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        g = BufferDict(g, keys=self.keys())
        return BufferDict(self, buf=g.flat[:] - self.flat)

    def __mul__(self, x):
        """ Multiply ``self``` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] * x)

    def __rmul__(self, x):
        """ Multiply ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] * x)

    # truediv and div are the same --- 1st is for python3, 2nd for python2
    def __truediv__(self, x):
        """ Divide ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] / x)

    def __div__(self, x):
        """ Divide ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] / x)

    def add(self,k,v):
        """ Augment buffer with data ``v``, indexed by key ``k``.

        ``v`` is either a scalar or a :mod:`numpy` array (or a list or
        other data type that can be changed into a numpy.array).
        If ``v`` is a :mod:`numpy` array, it can have any shape.

        Same as ``self[k] = v`` except when ``k`` is already used in
        ``self``, in which case a ``ValueError`` is raised.
        """
        if k in self:
            raise ValueError("Key %s already used." % str(k))
        else:
            self[k] = v

    def __getitem__(self,k):
        """ Return piece of buffer corresponding to key ``k``. """
        try:
            d = self._odict.__getitem__(k)
            ans = self._buf[d.slice]
            return ans if d.shape is () else ans.reshape(d.shape)
        except KeyError:
            pass
        try:
            return self._extension[k]
        except KeyError:
            pass
        for f in BufferDict.extension_fcn:
            altk = f + '(' + str(k) + ')'
            try:
                d = self._odict.__getitem__(altk)
                ans = self._buf[d.slice]
                if d.shape != ():
                    ans = ans.reshape(d.shape)
                ans = BufferDict.extension_fcn[f](ans)
                self._extension[k] = ans
                return ans
            except KeyError:
                pass
        raise KeyError("undefined key: %s" % str(k))

    def extension_keys(self):
        ans = []
        for k in self:
            try:
                m = re.match(BufferDict.extension_pattern, str(k))
            except TypeError:
                continue
            if m is None:
                continue
            k_fcn, k_stripped = m.groups()
            if k_fcn in BufferDict.extension_fcn:
                ans.append(k_stripped)
        return ans

    def values(self):
        # needed for python3.5
        return [self[k] for k in self]

    def items(self):
        # needed for python3.5
        return [(k,self[k]) for k in self]

    def __setitem__(self,k,v):
        """ Set piece of buffer corresponding to ``k`` to value ``v``.

        The shape of ``v`` must equal that of ``self[k]`` if key ``k``
        is already in ``self``.
        """
        self._extension = {}
        if k not in self:
            v = numpy.asarray(v)
            if v.shape==():
                # add single piece of data
                self._odict.__setitem__(
                    k, BUFFERDICTDATA(slice=len(self._buf), shape=())
                    )
                self._buf = numpy.append(self._buf,v)
            else:
                # add array
                n = numpy.size(v)
                i = len(self._buf)
                self._odict.__setitem__(
                    k, BUFFERDICTDATA(slice=slice(i,i+n), shape=tuple(v.shape))
                    )
                self._buf = numpy.append(self._buf, v)
        else:
            d = self._odict.__getitem__(k)
            if d.shape is ():
                try:
                    self._buf[d.slice] = v
                except ValueError:
                    raise ValueError(
                        'not a scalar: shape={}'.format(str(numpy.shape(v)))
                        )
                except TypeError:
                    raise TypeError('wrong type: {} not {}'.format(
                        str(type(v)), str(self.dtype)
                        ))
            else:
                v = numpy.asarray(v)
                try:
                    self._buf[d.slice] = v.flat
                except ValueError:
                    raise ValueError('shape mismatch: {} not {}'.format(
                            str(v.shape),str(d.shape)
                            ))

    def __delitem__(self,k):
        if k not in self:
            raise ValueError('key not in BufferDict: ' + str(k))
        size = numpy.size(self[k])
        # fix buffer
        self._buf = numpy.delete(self._buf, self.slice(k), 0)
        # fix slices for keys after k
        keys = list(self.keys())
        idx = keys.index(k)
        for kk in keys[idx + 1:]:
            sl, sh = self._odict.__getitem__(kk)
            if isinstance(sl, slice):
                newsl = slice(sl.start - size, sl.stop - size)
            else:
                newsl = sl - size
            self._odict.__setitem__(
                kk, BUFFERDICTDATA(slice=newsl, shape=sh)
                )
        self._odict.__delitem__(k)

    def __len__(self):
        return len(self._odict)

    def __iter__(self):
        return iter(self._odict)

    def __contains__(self, k):
        return k in self._odict

    def __str__(self):
        ans = "{"
        for k in self:
            ans += "%s: %s," % (repr(k), repr(self[k]))
        if ans[-1] == ',':
            ans = ans[:-1]
            ans += "}"
        return ans

    def __repr__(self):
        cn = self.__class__.__name__
        return cn+"("+repr([k for k in self.items()])+")"

    def _getflat(self):
        self._extension = {}
        return self._buf.flat

    def _setflat(self,buf):
        """ Assigns buffer with buf if same size. """
        self._extension = {}
        self._buf.flat = buf

    flat = property(_getflat,_setflat,doc='Buffer array iterator.')
    def flatten(self):
        """ Copy of buffer array. """
        return numpy.array(self._buf)

    def _getdtype(self):
        return self._buf.dtype

    dtype = property(_getdtype, doc='Data type of buffer array elements.')

    def _getbuf(self):
        self._extension = {}
        return self._buf

    def _setbuf(self,buf):
        """ Replace buffer with ``buf``.

        ``buf`` must be a 1-dimensional :mod:`numpy` array of the same size
        as ``self._buf``.
        """
        self._extension = {}
        if isinstance(buf,numpy.ndarray) and buf.shape == self._buf.shape:
            self._buf = buf
        else:
            raise ValueError(
                "New buffer wrong type or shape ---\n    %s,%s   not   %s,%s"
                % (type(buf), numpy.shape(buf),
                type(self._buf), self._buf.shape))

    buf = property(_getbuf,_setbuf,doc='Buffer array (not a copy).')

    def _getsize(self):
        """ Length of buffer. """
        return len(self._buf)

    size = property(_getsize,doc='Size of buffer array.')

    def slice(self,k):
        """ Return slice/index in ``self.flat`` corresponding to key ``k``."""
        self._extension = {}
        return self._odict.__getitem__(k).slice

    def slice_shape(self,k):
        """ Return tuple ``(slice/index, shape)`` corresponding to key ``k``."""
        self._extension = {}
        return self._odict.__getitem__(k)

    def has_dictkey(self, k):
        """ Returns ``True`` if ``self[k]`` is defined; ``False`` otherwise.

        Note that ``k`` may be a key or it may be related to a
        related key associated with a non-Gaussian distribution
        (e.g., ``'log(k)'``; see :func:`gvar.BufferDict.add_distribution`
        for more information).
        """
        return _gvar.has_dictkey(self, k)

    @staticmethod
    def add_distribution(name, invfcn):
        """ Add new parameter distribution.

        |BufferDict|\s can be used to represent a  restricted
        class of  non-Gaussian distributions. For example, the code ::

            import gvar as gv
            gv.BufferDict.add_distribution('log', gv.exp)

        enables the use of log-normal distributions for parameters. So
        defining, for example, ::

            b = gv.BufferDict()
            b['log(a)'] = gv.gvar('1(1)')

        means that ``b['a']`` has a value (equal to ``exp(b['log(a)']``)
        even though ``'a'`` is not a key in the dictionary.

        The distributions available by default correspond to::

            gv.BufferDict.add_distribution('log', gv.exp)
            gv.BufferDict.add_distribution('sqrt', gv.square)
            gv.BufferDict.add_distribution('erfinv', gv.erf)

        Additional distributions can be added by specifying:

        Args:
            name (str): Distributions' function name.
            invfcn (callable): Inverse of the transformation function.
        """
        BufferDict.extension_fcn[name] = invfcn

    @staticmethod
    def del_distribution(name):
        """ Delete |BufferDict| distribution ``name``. """
        del BufferDict.extension_fcn[name]

def asbufferdict(g, dtype=None):
    """ Convert ``g`` to a BufferDict, keeping only ``g[k]`` for ``k in keylist``.

    ``asbufferdict(g)`` will return ``g`` if it is already a
    :class:`gvar.BufferDict`; otherwise it will convert the dictionary-like
    object into a :class:`gvar.BufferDict`. The data can also be
    specified: e.g., ``asbufferdict(g, dtype=int)``.
    """
    if isinstance(g, BufferDict) and dtype is None:
        return g
    kargs = {} if dtype is None else dict(dtype=dtype)
    return BufferDict(g, **kargs)

def get_dictkeys(bdict, klist):
    """ Same as ``[dictkey(bdict, k) for k in klist]``. """
    ans = []
    for k in klist:
        if k not in bdict:
            for f in BufferDict.extension_fcn:
                newk = f + '(' + str(k) + ')'
                if newk in bdict:
                    break
            else:
                raise KeyError('bad key: ' + str(k))
            ans.append(newk)
        else:
            ans.append(k)
    return ans

def dictkey(bdict, k):
    """ Find key in ``bdict`` corresponding to ``k``.

    Could be ``k`` itself or one of the standard extensions,
    such as ``log(k)`` or ``sqrt(k)``.
    """
    if k not in bdict:
        for f in BufferDict.extension_fcn:
            newk = f + '(' + str(k) + ')'
            if newk in bdict:
                return newk
        raise KeyError('key not used: ' + str(k))
    else:
        return k

def has_dictkey(b, k):
    """ Returns ``True`` if ``b[k]`` is defined; ``False`` otherwise.

    Note that ``k`` may be a key or it may be related to a
    related key associated with a non-Gaussian distribution
    (e.g., ``'log(k)'``; see :func:`gvar.MultiFitter.add_distribution`).
    """
    if k in b:
        return True
    else:
        for f in BufferDict.extension_fcn:
            newk = f + '(' + str(k) + ')'
            if newk in b:
                return True
        return False

def _stripkey(k):
    """ Return (stripped key, fcn) where fcn is exp or square or ...

    Strip off any ``"log"`` or ``"sqrt"`` or ... prefix.
    """
    if not isinstance(k, str):
        return k, None
    m = re.match(BufferDict.extension_pattern, k)
    if m is None:
        return k, None
    k_fcn, k_stripped = m.groups()
    if k_fcn not in BufferDict.extension_fcn:
        return k, None
    return k_stripped, BufferDict.extension_fcn[k_fcn]


def nonredundant_keys(keys):
    """ Return list containing only nonredundant keys in list ``keys``. """
    discards = set()
    for k in keys:
        if isinstance(k, str):
            m = re.match(BufferDict.extension_pattern, k)
            if m is not None:
                discards.add(m.groups()[1])
    ans = []
    for k in keys:
        if not isinstance(k, str) or k not in discards:
            ans.append(k)
    return ans

def add_parameter_parentheses(p):
    """ Return dictionary with proper keys for parameter distributions (legacy code).

    This utility function helps fix legacy code that uses
    parameter keys like ``logp`` or ``sqrtp`` instead of
    ``log(p)`` or ``sqrt(p)``, as now required. This method creates a
    copy of  dictionary ``p'' but with keys like ``logp`` or ``sqrtp``
    replaced by ``log(p)`` or ``sqrt(p)``. So setting ::

        p = add_parameter_parentheses(p)

    fixes the keys in ``p`` for log-normal and sqrt-normal parameters.
    """
    newp = BufferDict()
    for k in p:
        if isinstance(k, str):
            if k[:3] == 'log' and _stripkey(k)[1] is None:
                newk = 'log(' + k[3:] + ')'
            elif k[:4] == 'sqrt' and _stripkey(k)[1] is None:
                newk = 'sqrt(' + k[4:] + ')'
            else:
                newk = k
            newp[newk] = p[k]
    return newp

def trim_redundant_keys(p):
    """ Remove redundant keys from dictionary ``p``.

    A key ``'c'`` is redundant if either of ``'log(c)'``
    or ``'sqrt(c)'`` is also a key. (There are additional redundancies
    if :meth:`gvar.add_parameter_distribution` has been used to add
    extra distributions.) This function creates a copy of ``p`` but with
    the redundant keys removed.
    """
    return BufferDict(p, keys=nonredundant_keys(p.keys()))
