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

BUFFERDICTDATA = collections.namedtuple('BUFFERDICTDATA',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==() implies a scalar. """

class BufferDict(collections.OrderedDict):
    """ Ordered dictionary whose data are packed into a 1-d buffer (numpy.array).

    A |BufferDict| object is an ordered dictionary whose values must
    either be scalars or arrays (like :mod:`numpy` arrays, with arbitrary
    shapes). The scalars and arrays are assembled into different parts of a
    single one-dimensional buffer. The various scalars and arrays are
    retrieved using keys: *e.g.*,

        >>> a = BufferDict()
        >>> a['scalar'] = 0.0
        >>> a['vector'] = [1.,2.]
        >>> a['tensor'] = [[3.,4.],[5.,6.]]
        >>> print(a.flatten())              # print a's buffer
        [ 0.  1.  2.  3.  4.  5.  6.]
        >>> for k in a:                     # iterate over keys in a
        ...     print(k,a[k])
        scalar 0.0
        vector [ 1.  2.]
        tensor [[ 3.  4.]
         [ 5.  6.]]
        >>> a['vector'] = a['vector']*10    # change the 'vector' part of a
        >>> print(a.flatten())
        [  0.  10.  20.   3.   4.   5.   6.]

    The first four lines here could have been collapsed to one statement::

        a = BufferDict(scalar=0.0,vector=[1.,2.],tensor=[[3.,4.],[5.,6.]])

    or ::

        a = BufferDict([('scalar',0.0),('vector',[1.,2.]),
                        ('tensor',[[3.,4.],[5.,6.]])])

    where in the second case the order of the keys is preserved in ``a``
    (since ``BufferDict`` is an ordered dictionary).

    The keys and associated shapes in a |BufferDict| can be transferred to a
    different buffer, creating a new |BufferDict|: *e.g.*, using ``a`` from
    above,

        >>> buf = numpy.array([0.,10.,20.,30.,40.,50.,60.])
        >>> b = BufferDict(a, buf=buf)          # clone a but with new buffer
        >>> print(b['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        >>> b['scalar'] += 1
        >>> print(buf)
        [  1.  10.  20.  30.  40.  50.  60.]

    Note how ``b`` references ``buf`` and can modify it. One can also
    replace the buffer in the original |BufferDict| using, for example,
    ``a.buf = buf``:

        >>> a.buf = buf
        >>> print(a['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        >>> a['tensor'] *= 10.
        >>> print(buf)
        [  1.  10.  20.  300.  400.  500.  600.]

    ``a.buf`` is the numpy array used for ``a``'s buffer. It can be used to
    access and change the buffer directly. In ``a.buf = buf``, the new
    buffer ``buf`` must be a :mod:`numpy` array of the correct shape. The
    buffer can also be accessed through iterator ``a.flat`` (in analogy
    with :mod:`numpy` arrays), and through ``a.flatten()`` which returns a
    copy of the buffer.

    When creating a |BufferDict| from a dictionary (or another |BufferDict|),
    the keys included and their order can be specified using a list of keys:
    for example, ::

        >>> d = dict(a=0.0,b=[1.,2.],c=[[3.,4.],[5.,6.]],d=None)
        >>> print(d)
        {'a': 0.0, 'c': [[3.0, 4.0], [5.0, 6.0]], 'b': [1.0, 2.0], 'd': None}
        >>> a = BufferDict(d, keys=['d', 'b', 'a'])
        >>> for k in a:
        ...     print(k, a[k])
        d None
        b [1.0 2.0]
        a 0.0

    A |BufferDict| functions like a dictionary except: a) items cannot be
    deleted once inserted; b) all values must be either scalars or arrays
    of scalars, where the scalars can be any noniterable type that works
    with :mod:`numpy` arrays; and c) any new value assigned to an existing
    key must have the same size and shape as the original value.

    Note that |BufferDict|\s can be pickled and unpickled even when they
    store |GVar|\s (which themselves cannot be pickled separately).
    """
    def __init__(self, *args, **kargs):
        super(BufferDict, self).__init__()
        self.shape = None
        if len(args)==0:
            # kargs are dictionary entries
            self._buf = numpy.array([],numpy.intp)
            for k in kargs:
                self[k] = kargs[k]
        elif len(args) == 1 and 'keys' in kargs and len(kargs) == 1:
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
                for k in bd:
                    super(BufferDict, self).__setitem__(
                        k, super(BufferDict, bd).__getitem__(k)
                        )
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
        od = super(BufferDict, self)
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
            super(BufferDict, self).__setitem__(
                k,
                BUFFERDICTDATA(
                    slice=layout[k][0],
                    shape=layout[k][1] if layout[k][1] is not None else ()
                    )
                )
        self._buf = buf

    def __reduce_ex__(self, dummy):
        return (BufferDict, (), self.__getstate__())

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
        if not super(BufferDict, self).__contains__(k):
            raise KeyError("undefined key: %s" % str(k))
        d = super(BufferDict, self).__getitem__(k)
        ans = self._buf[d.slice]
        return ans if d.shape is () else ans.reshape(d.shape)

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
        if k not in self:
            v = numpy.asarray(v)
            if v.shape==():
                # add single piece of data
                super(BufferDict, self).__setitem__(k, BUFFERDICTDATA(slice=len(self._buf),shape=()))
                self._buf = numpy.append(self._buf,v)
            else:
                # add array
                n = numpy.size(v)
                i = len(self._buf)
                super(BufferDict, self).__setitem__(k, BUFFERDICTDATA(slice=slice(i,i+n),shape=tuple(v.shape)))
                self._buf = numpy.append(self._buf,v)
        else:
            d = super(BufferDict, self).__getitem__(k)
            if d.shape is ():
                try:
                    self._buf[d.slice] = v
                except ValueError:
                    raise ValueError("*** Not a scalar? Shape=%s"
                                     % str(numpy.shape(v)))
            else:
                v = numpy.asarray(v)
                try:
                    self._buf[d.slice] = v.flat
                except ValueError:
                    raise ValueError("*** Shape mismatch? %s not %s" %
                                     (str(v.shape),str(d.shape)))

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
            sl, sh = super(BufferDict, self).__getitem__(kk)
            if isinstance(sl, slice):
                newsl = slice(sl.start - size, sl.stop - size)
            else:
                newsl = sl - size
            super(BufferDict, self).__setitem__(
                kk, BUFFERDICTDATA(slice=newsl, shape=sh)
                )
        # delete k
        super(BufferDict, self).__delitem__(k)
        # raise NotImplementedError("Cannot delete items from BufferDict.")

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
        return self._buf.flat

    def _setflat(self,buf):
        """ Assigns buffer with buf if same size. """
        self._buf.flat = buf

    flat = property(_getflat,_setflat,doc='Buffer array iterator.')
    def flatten(self):
        """ Copy of buffer array. """
        return numpy.array(self._buf)

    def _getdtype(self):
        return self._buf.dtype

    dtype = property(_getdtype, doc='Data type of buffer array elements.')

    def _getbuf(self):
        return self._buf

    def _setbuf(self,buf):
        """ Replace buffer with ``buf``.

        ``buf`` must be a 1-dimensional :mod:`numpy` array of the same size
        as ``self._buf``.
        """
        if isinstance(buf,numpy.ndarray) and buf.shape == self._buf.shape:
            self._buf = buf
        else:
            raise ValueError(
                "New buffer wrong type or shape ---\n    %s,%s   not   %s,%s"
                % (type(buf), numpy.shape(buf),
                type(self._buf), self._buf.shape))

    buf = property(_getbuf,_setbuf,doc='The buffer array (not a copy).')
    def _getsize(self):
        """ Length of buffer. """
        return len(self._buf)

    def get(self, k, d):
        """ Return ``self[k]`` if ``k`` in ``self``, otherwise return ``d``. """
        # items(), setdefault(), etc all use __getitem__, __setitem__
        # but not get(), for some reason.
        return self[k] if k in self else d

    size = property(_getsize,doc='Size of buffer array.')
    def slice(self,k):
        """ Return slice/index in ``self.flat`` corresponding to key ``k``."""
        return super(BufferDict, self).__getitem__(k).slice

    def slice_shape(self,k):
        """ Return tuple ``(slice/index, shape)`` corresponding to key ``k``."""
        return super(BufferDict, self).__getitem__(k)

    def isscalar(self,k):
        """ Return ``True`` if ``self[k]`` is scalar else ``False``."""
        return super(BufferDict, self).__getitem__(k).shape is ()

    def dump(self, fobj, use_json=False):
        """ Serialize |BufferDict| in file object ``fobj``.

        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obviously). :mod:`json` does not handle
        non-string valued keys very well. This attempts a workaround, but
        it will only work in simpler cases. Serialization only works when
        :mod:`pickle` (or :mod:`json`) knows how to serialize the data type
        stored in the |BufferDict|'s buffer (or for |GVar|\s).
        """
        if not use_json:
            pickle.dump(self, fobj)
        else:
            if isinstance(self._buf[0], _gvar.GVar):
                tmp = _gvar.mean(self)
                cov = _gvar.evalcov(self._buf)
            else:
                tmp = self
                cov = None
            d = {}
            keys = []
            for k in tmp:
                jk = 's:' + k if str(k) == k else 'e:'+str(k)
                keys.append(jk)
                d[jk] = tmp[k] if self.isscalar(k) else tmp[k].tolist()
            d['keys'] = keys
            if cov is not None:
                d['cov'] = cov.tolist()
            json.dump(d, fobj)

    def dumps(self, use_json=False):
        """ Serialize |BufferDict| into string.

        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obviously). :mod:`json` does not handle
        non-string valued keys very well. This attempts a workaround, but
        it will only work in simpler cases (e.g., integers, tuples of
        integers, etc.). Serialization only works when :mod:`pickle` (or
        :mod:`json`) knows how to serialize the data type stored in the
        |BufferDict|'s buffer (or for |GVar|\s).
        """
        f = _StringIO() if use_json else _BytesIO()
        self.dump(f, use_json=use_json)
        return f.getvalue()

    @staticmethod
    def load(fobj, use_json=False):
        """ Load serialized |BufferDict| from file object ``fobj``.

        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obvioulsy).
        """
        if not use_json:
            return pickle.load(fobj)
        else:
            d = json.load(fobj)
            ans = BufferDict()
            for jk in d['keys']:
                k = str(jk[2:]) if jk[0] == 's' else eval(jk[2:])
                ans[k] = d[jk]
            if 'cov' in d:
                ans.buf = _gvar.gvar(ans._buf,d['cov'])
            return ans

    @staticmethod
    def loads(s, use_json=False):
        """ Load serialized |BufferDict| from file object ``fobj``.

        Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
        it uses :mod:`json` (obvioulsy).
        """
        f = _StringIO(s) if use_json else _BytesIO(s)
        return BufferDict.load(f, use_json=use_json)


def asbufferdict(g, keylist=None):
    """ Convert ``g`` to a BufferDict, keeping only ``g[k]`` for ``k in keylist``.

    ``asbufferdict(g)`` will return ``g`` if it is already a
    :class:`gvar.BufferDict`; otherwise it will convert the dictionary-like
    object into a :class:`gvar.BufferDict`. If ``keylist`` is not ``None``,
    only objects ``g[k]`` for which ``k in keylist`` are kept.
    """
    if isinstance(g, BufferDict) and keylist is None:
        return g
    if keylist is None:
        return BufferDict(g)
    ans = BufferDict()
    for k in keylist:
        ans[k] = g[k]
    return ans

class ExtendedDict(BufferDict):
    """ |BufferDict| that supports variables from extended distributions.

    Used for parameters when there may be log-normal/sqrt-normal/...  variables.
    The exponentiated/squared/... values of those variables are included in the
    BufferDict, together with  the original variables. Setting ``p.buf=buf``
    assigns a new buffer and fills in the exponentiated/squared/... values.
    (The buffer is resized if ``buf`` is sized for just the original variables.)
    Use ``p.stripped_buf`` to access the part of the buffer that has only
    the original variables.

    Use function :meth:`gvar.add_parameter_distribution` to add distributions.

    It is a bad idea to change the values of any of the entries
    separately: eg, ``p['a'] = 2.6``. I haven't redesigned setitem to
    check whether or not something else needs updating, and probably
    won't. The only "correct" way to change the values in an
    ExtendedDict is by overwriting the buffer: ``p.buf = buf``. This
    class is not really for public use; it has a very specific
    behind-the-scenes function. Might change its name to _ExtendedDict
    to emphasize this.

    N.B. ExtendedDict is *not* part of the public api yet (or maybe ever).

    Args:
        p0 : BufferDict whose keys are *not* redundant.
        buf: New buffer sized for p0 or ``None``.
    """

    extension_pattern = re.compile('^([^()]+)\((.+)\)$')
    extension_fcn = {}

    def __init__(self, p0, buf=None):
        super(ExtendedDict, self).__init__(p0)
        if isinstance(p0, ExtendedDict):
            self.extensions = list(p0.extensions)
            self._newkeys = list(p0._newkeys)
            self.stripped_buf_size = p0.stripped_buf_size
        else:
            self.stripped_buf_size = self.buf.size
            extensions = []
            newkeys = []
            for k in p0.keys():
                k_stripped, k_fcn = ExtendedDict.stripkey(k)
                if k_fcn is not None:
                    if k_stripped in p0:
                        raise ValueError('Redundant key in p0: ' + str(k_stripped))
                    self[k_stripped] = k_fcn(self[k])
                    extensions.append(
                        (self.slice(k_stripped), k_fcn, self.slice(k))
                        )
                    newkeys.append(k_stripped)
            self.extensions = extensions
            self._newkeys = newkeys
        if buf is not None:
            self.buf = buf

    def _getbuf(self):
        return self._buf

    def _setbuf(self, buf):
        """ Replace buffer with ``buf``.

        ``buf`` must be a 1-dimensional array of the same size
        as ``self._buf`` or size ``self.stripped_buf_size``
        """
        if not isinstance(buf, numpy.ndarray):
            buf = numpy.array(buf)
        if buf.ndim != 1:
            raise ValueError('New buffer not a 1-d array.')
        if len(buf) == self.stripped_buf_size:
            if self._buf.dtype == buf.dtype:
                self._buf[:self.stripped_buf_size] = buf
            else:
                self._buf = numpy.resize(buf, self._buf.size)
        elif len(buf) == self._buf.size:
            self._buf = buf
        else:
            raise ValueError(
                'New buffer wrong size: {} != {} or {}'.format(
                    buf.size,self.stripped_buf_size, self._buf.size
                    )
                )
        # restore derived values
        for s1, f, s2 in self.extensions:
            self.buf[s1] = f(self.buf[s2])

    buf = property(_getbuf, _setbuf, doc='The buffer array (not a copy).')

    def _getstrippedbuf(self):
        return self._buf[:self.stripped_buf_size]

    stripped_buf = property(_getstrippedbuf, doc='Part of buffer array for nonredundant keys.')

    def newkeys(self):
        " Iterator containing new keys generated by :class:`ExtendedDict`. "
        return iter(self._newkeys)

    @staticmethod
    def basekey(prior, k):
        """ Find base key in ``prior`` corresponding to ``k``. """
        if not isinstance(k, str):
            return k
        for f in ExtendedDict.extension_fcn:
            newk = f + '(' + k + ')'
            if newk in prior:
                return newk
        return k

    @staticmethod
    def stripkey(k):
        """ Return (stripped key, fcn) where fcn is exp or square or ...

        Strip off any ``"log"`` or ``"sqrt"`` or ... prefix.
        """
        if not isinstance(k, str):
            return k, None
        m = re.match(ExtendedDict.extension_pattern, k)
        if m is None:
            return k, None
        k_fcn, k_stripped = m.groups()
        if k_fcn not in ExtendedDict.extension_fcn:
            return k, None
        return k_stripped, ExtendedDict.extension_fcn[k_fcn]

def nonredundant_keys(keys):
    """ Return list containing only nonredundant keys in list ``keys``. """
    discards = set()
    for k in keys:
        if isinstance(k, str):
            m = re.match(ExtendedDict.extension_pattern, k)
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
            if k[:3] == 'log' and ExtendedDict.stripkey(k)[1] is None:
                newk = 'log(' + k[3:] + ')'
            elif k[:4] == 'sqrt' and ExtendedDict.stripkey(k)[1] is None:
                newk = 'sqrt(' + k[4:] + ')'
            else:
                newk = k
            newp[newk] = p[k]
    return newp

def add_parameter_distribution(name, invfcn):
    """ Add new parameter distribution for use in fits.

    This function adds new distributions for the parameters used in
    :class:`lsqfit.nonlinear_fit`. For example, the code ::

        import gvar as gv
        gv.add_parameter_distribution('log', gv.exp)

    enables the use of log-normal distributions for parameters. The log-normal
    distribution is invoked for a parameter ``p`` by including ``log(p)``
    rather than ``p`` itself in the fit prior. log-normal, sqrt-normal,  and
    erfinv-normal distributions are included by default. (Setting  a prior
    ``prior[erfinv(w)]`` equal to ``gv.gvar('0(1)') / gv.sqrt(2)``  means that
    the prior probability for ``w`` is distributed uniformly between -1 and 1,
    and is zero elsewhere.)

    These distributions are implemented by replacing a fit parameter ``p``
    by a new fit parameter ``fcn(p)`` where ``fcn`` is some function. ``fcn(p)``
    is assumed to have a Gaussian distribution, and parameter ``p`` is
    recovered using the inverse function ``invfcn`` where ``p=invfcn(fcn(p))``.

    :param name: Distribution's name.
    :type name: str
    :param invfcn: Inverse of the transformation function.
    """
    ExtendedDict.extension_fcn[name] = invfcn

def del_parameter_distribution(name):
    """ Delete parameter distribution ``name``. """
    del ExtendedDict.extension_fcn[name]

def trim_redundant_keys(p):
    """ Remove redundant keys from dictionary ``p``.

    A key ``'c'`` is redundant if either of ``'log(c)'``
    or ``'sqrt(c)'`` is also a key. (There are additional redundancies
    if :meth:`gvar.add_parameter_distribution` has been used to add
    extra distributions.) This function creates a copy of ``p`` but with
    the redundant keys removed.
    """
    return BufferDict(p, keys=nonredundant_keys(p.keys()))
