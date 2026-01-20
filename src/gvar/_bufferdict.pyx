# cython: language_level=3str, binding=True
# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-22 G. Peter Lepage.
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
import gvar as _gvar
import sys 

try:
    from collections.abc import MutableMapping as collections_MMapping
except ImportError:
    from collections import MutableMapping as collections_MMapping

if sys.version_info.major == 2:
    ORDEREDDICT = collections.OrderedDict
else:
    if sys.version_info.major == 3 and sys.version_info.minor < 6:
        ORDEREDDICT = collections.OrderedDict
    else:
        ORDEREDDICT = dict

BUFFERDICTDATA = collections.namedtuple('BUFFERDICTDATA',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==() implies a scalar. """

class BufferDict(collections_MMapping):
    r""" Ordered dictionary whose values are packed into a 1-d buffer (:mod:`numpy` array).

    |BufferDict|\s can be created in the usual way dictionaries are created::

        >>> b = BufferDict()
        >>> b['a'] = 1
        >>> b['b'] = [2.7, 3.2]
        >>> print(b)
        {'a': 1.0, 'b': array([2.7, 3.2])}
        >>> print(b.buf)
        [1. 2.7 3.2]
        >>> b = BufferDict(a=1, b=[2.7, 3.2])
        >>> b = BufferDict([('a',1.), ('b',[2.7, 3.2])])

    They can also be created from other dictionaries or |BufferDict|\s::

        >>> c = BufferDict(b)
        >>> print(c)
        {'a': 1.0, 'b': 2.7}
        >>> c = BufferDict(b, keys=['b'])
        >>> print(c)
        {'b': 2.7}

    The ``keys`` keyword restricts the copy of a dictionary to entries whose
    keys are in ``keys`` and in the same order as in ``keys``.

    The values in a |BufferDict| are scalars or arrays of a scalar type
    (|GVar|, ``float``, ``int``, etc.). The data type of the buffer is 
    normally inferred (dynamically) from the data itself::

        >>> b = BufferDict(a=1)
        >>> print(b, b.dtype)
        {'a': 1} int64
        >>> b['b'] = 2.7
        >>> print(b, b.dtype)
        {'a': 1.0, 'b': 2.7} float64     
    
    The data type of the |BufferDict|'s buffer can be specified
    when creating the |BufferDict| from another dictionary or list
    by using keyword ``dtype``::

        >>> b = BufferDict(dict(a=1.2), dtype=int)  
        >>> print(b, b.dtype)
        {'a': 1} int64        
        >>> b['b'] = 2.7
        >>> print(b, b.dtype)
        {'a': 1, 'b': 2} int64  
  
    Note in this example that the data type is *not* changed by subsequent
    additions to the |BufferDict| when the ``dtype`` keyword is specified.
    To create an empty |BufferDict| with a specified type use, for example,
    ``BufferDict({}, dtype=int)``. Any data type supported by :mod:`numpy` 
    arrays can be specified.

    Some simple arithemetic is allowed between two |BufferDict|\s, say,
    ``g1`` and ``g2`` provided they have the same keys and array shapes.
    So, for example::

        >>> a = BufferDict(a=1, b=[2, 3])
        >>> b = BufferDict(b=[20., 30.], a=10.)
        >>> print(a + b)
            {'a': 11.0, 'b': array([22., 33.])}

    Subtraction is also defined, as are multiplication and division
    by scalars. The corresponding ``+=``, ``-=``, ``*=``, ``/=`` operators
    are supported, as are unary ``+`` and ``-``.

    Finally a |BufferDict| can be cloned from another one using
    a different buffer (containing different values)::

        >>> b = BufferDict(a=1., b=2.)
        >>> c = BufferDict(b, buf=[10, 20])
        >>> print(c)
        {'a': 10, 'b': 20}

    The  new buffer itself (i.e., not a copy) is used in the 
    new |BufferDict| if the buffer is a :mod:`numpy` array::

        >>> import numpy as np
        >>> b = BufferDict(a=1., b=2.)
        >>> nbuf = np.array([10, 20])
        >>> c = BufferDict(b, buf=nbuf)
        >>> c['a'] += 1
        >>> print(c)
        {'a': 11, 'b': 20}
        >>> print(nbuf)
        [11 20]

    If many clones of a |BufferDict| are needed, it is usually substantially
    more efficient to create them together in a single batch (or a 
    small number of large batches). For example,
    the following code 

        >>> a = BufferDict(s=1., v=[2., 3., 4.])
        >>> print(a, '   ', a.buf)
        {'s': 1.0, 'v': array([2., 3.])}    [1. 2. 3. 4.]
        >>> lbuf = np.array(
        ... [[10., 20., 30., 40.], [100., 200., 300., 400.], [1000., 2000., 3000., 4000.]]
        ... )
        >>> lb = BufferDict(a, lbatch_buf=lbuf)
        >>> print(lb)    
        {
            's': array([  10.,  100., 1000.]),
            'v':
            array([[  20.,   30.,   40.],
                [ 200.,  300.,  400.],
                [2000., 3000., 4000.]]),
        }

    merges three clones of the original |BufferDict| |~| ``a`` into 
    |BufferDict| ``lb``. Each clone has its own buffer ``lbuf[i]``
    where ``i=0...2`` labels the clone. Each element of ``lb`` has 
    an extra (leftmost) index labeling the clone: ``lb['s'][i]`` 
    and ``lb['v'][i, :]``. If needed, the individual clones 
    can be pulled out into separate dictionaries using
    :meth:`gvar.BufferDict.batch_iter`::

        >>> for clone in b.batch_iter('lbatch'):
        ...     print(clone)
        ...
        {'s': 10.0, 'v': array([20., 30., 40.])}
        {'s': 100.0, 'v': array([200., 300., 400.])}
        {'s': 1000.0, 'v': array([2000., 3000., 4000.])}

    These clones all provide views into ``b.buf``.

    The batch index is always the leftmost index in the example above. It is 
    frequently more convenient for the batch index to be the rightmost 
    index. This is done by setting  ``rbatch_buf`` rather than 
    ``lbatch_buf`` when making the batch |BufferDict|: for example,  ::

        >>> rbuf = np.array(
        ... [[10., 100., 1000.], [20., 200., 2000.], [30., 300., 3000.], [40., 400., 4000.]]
        ... )
        >>> rb = BufferDict(a, rbatch_buf=rbuf)
        >>> print(rb)
        {
            's': array([  10.,  100., 1000.]),
            'v':
            array([[  20.,  200., 2000.],
                   [  30.,  300., 3000.]]),
                   [  40.,  400., 4000.]]),
        }

    creates a batch |BufferDict| equivalent to the previous one but now
    where the batch index is the last index for each element:
    ``rb[k][...,i]`` for each key ``k`` and batch index ``i=0..2``.
    The batch buffer ``rbuf`` used here is the transpose of that 
    used for the ``lbatch`` case; ``rb.buf`` is a view into ``rbuf`` 
    (not a copy as is the case for ``lbatch``). Again we can iterate over 
    the individual clones (and these are the same as above)::

        >>> for clone in rb.batch_iter('rbatch'):
        ...     print(clone)
        ...
        {'s': 10.0, 'v': array([20., 30., 40.])}
        {'s': 100.0, 'v': array([200., 300., 400.])}
        {'s': 1000.0, 'v': array([2000., 3000., 4000.])}


    """

    extension_pattern = re.compile('^([^()]+)\((.+)\)$')
    invfcn = {}

    def __init__(self, *args, **kargs):
        self._extension = {}
        self._ver_l = BufferDict._ver_g
        self.shape = None
        self._dtype = None     # enforced dtype (default is None)
        if len(args)==0:
            # kargs are dictionary entries
            self._odict = ORDEREDDICT()
            self._buf = numpy.array([])
            for k in kargs:
                self[k] = kargs[k]
            return
        if len(args) > 2:
            raise RuntimeError('wrong number of arguments')
        self._dtype = kargs.get('dtype', None)
        keys = list(kargs['keys']) if 'keys' in kargs else None
        if 'buf' in kargs:
            buf = kargs['buf']
            if len(args) > 1 or 'rbatch_buf' in kargs or 'lbatch_buf' in kargs:
                raise RuntimeError('too many arguments: buffer specified more than once')
        elif len(args) == 2:
            buf = args[1]
            if 'rbatch_buf' in kargs or 'lbatch_buf' in kargs:
                raise RuntimeError('too many arguments: buffer specified more than once')
        else:
            buf = None
        if isinstance(args[0], BufferDict) and keys is None:
            # make copy of BufferDict args[0], possibly with new buffer
            # copy keys, slices and shapes (fast build)
            bd = args[0]
            self._odict = ORDEREDDICT(bd._odict)
            # copy buffer or use new one
            if buf is None:
                if 'rbatch_buf' in kargs:
                    buf = kargs['rbatch_buf']
                elif 'lbatch_buf' in kargs:
                    # transpose to convert to rbatch_buf
                    buf = numpy.transpose(kargs['lbatch_buf'])
                else:
                    buf = numpy.array(bd._buf, dtype=self._dtype)
            if numpy.shape(buf) == bd._buf.shape:
                self._buf = numpy.asarray(buf, dtype=self._dtype)
            elif numpy.ndim(buf) == 2 and len(buf) == len(bd._buf):
                # buf has extra indices => rbatch mode
                extra_sh = numpy.shape(buf)[1:]
                rescale = numpy.prod(extra_sh)
                for k in self._odict:
                    sh = self._odict[k].shape + extra_sh 
                    sl = self._odict[k].slice
                    if isinstance(sl, slice):
                        sl = slice(sl.start * rescale, sl.stop * rescale)
                    else:
                        sl = slice(sl * rescale, (sl + 1) * rescale)
                    self._odict[k] = BUFFERDICTDATA(slice=sl, shape=sh)
                self._buf = numpy.asarray(buf, dtype=self._dtype).reshape(-1)
            else:
                raise ValueError("buf is wrong shape --- %s not %s"
                                    % (numpy.shape(buf), bd._buf.shape))
            if 'lbatch_buf' in kargs:
                # move batch index to convert back from rbatch format
                tmp = BufferDict()
                tmp._odict = self._odict 
                tmp._buf = self._buf
                self._odict = ORDEREDDICT()
                self._buf = numpy.array([], dtype=self._dtype) 
                for k in tmp:
                    self[k] = numpy.moveaxis(tmp[k], -1, 0)
            return
        elif buf is not None:
            raise RuntimeError("can't specify buffer unless first argument is a BufferDict")
        data = args[0] if hasattr(args[0], 'keys') else dict(args[0])
        self._odict = ORDEREDDICT()
        self._buf = numpy.array([], dtype=self._dtype) 
        if keys is None:
            for k in data:
                self[k] = data[k]
        else:
            for k in keys:
                if k in data:
                    self[k] = data[k]
            # for k, v in data:
            #     if k in keys:
            #         self[k] = v 

    def _r2lbatch(self):
        r""" Returns ``self, nbatch`` converted from ``'rbatch'`` to ``'lbatch'`` mode."""
        ans = _gvar.BufferDict() 
        nbatch = None
        for k in self:
            ans[k] = numpy.moveaxis(self[k], -1, 0)
            if nbatch is None:
                nbatch = ans[k].shape[0]
            elif nbatch != ans[k].shape[0]:
                raise ValueError('Not an rbatch BufferDict')
        return ans, nbatch 

    def _l2rbatch(self):
        r""" Returns ``self, nbatch`` converted from ``'lbatch'`` to ``'rbatch'`` mode."""
        ans = _gvar.BufferDict()
        nbatch = None
        for k in self:
            ans[k] = numpy.moveaxis(self[k], 0, -1)
            if nbatch is None:
                nbatch = ans[k].shape[-1]
            elif nbatch != ans[k].shape[-1]:
                raise ValueError('Not an lbatch BufferDict')
        return ans, nbatch 

    def __copy__(self):
        return BufferDict(self)
    
    def __deepcopy__(self, memo):
        return BufferDict(self, buf=copy.deepcopy(self.buf, memo))

    def __getstate__(self):
        r""" Capture state for pickling. """
        return (_gvar.dumps(collections.OrderedDict(self)), BufferDict.invfcn)

    def __setstate__(self, state):
        r""" Restore state when unpickling. """
        if not hasattr(state, 'keys'):
            if isinstance(state, tuple):
                contents, invfcn = state
            else:
                # legacy code
                contents = state 
                invfcn = {}
            tmp = _gvar.BufferDict(_gvar.loads(contents))
            self._odict = tmp._odict 
            self._buf = tmp._buf 
            self._extension = {}
            self.shape = None
            for k in invfcn:
                if k not in self.invfcn:
                    self.invfcn[k] = invfcn[k]
            return
        # even older legacy code
        layout = state['layout']
        buf = state['buf']
        if isinstance(buf, tuple):
            if len(buf) == 2:
                # old format (for legacy)
                buf = _gvar.gvar(*buf)  
            else:
                means, bcovs = buf[:2]
                buf = numpy.array(means, dtype=object)
                for idx, bcov in bcovs:
                    buf[idx] = _gvar.gvar(buf[idx], bcov)
        else:
            buf = numpy.array(buf)
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

    def _remove_gvars(self, gvlist):
        tmp = BufferDict(self, buf=numpy.array(self._buf))
        tmp._buf = _gvar.remove_gvars(tmp._buf, gvlist)
        return tmp

    def _distribute_gvars(self, gvlist):
        tmp = BufferDict(self, buf=numpy.array(self._buf))
        tmp._buf = _gvar.distribute_gvars(tmp._buf, gvlist)
        return tmp

    def __reduce_ex__(self, dummy):
        return (BufferDict, (), self.__getstate__())

    def __iadd__(self, g):
        r""" self += |BufferDict| (or dictionary) """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        self.flat[:] += g.flat
        return self

    def __isub__(self, g):
        r""" self -= |BufferDict| (or dictionary) """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        self.flat[:] -= g.flat
        return self

    def __imul__(self, x):
        r""" ``self *= x`` for scalar ``x`` """
        self.flat[:] *= x
        return self

    def __itruediv__(self, x):
        r""" ``self /= x`` for scalar ``x`` """
        self.flat[:] /= x
        return self

    def __pos__(self):
        r""" ``+self`` """
        return BufferDict(self, buf=+self.flat[:])

    def __neg__(self):
        r""" ``-self`` """
        return BufferDict(self, buf=-self.flat[:])

    def __add__(self, g):
        r""" :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        return BufferDict(self, buf=self.flat[:] + g.flat)

    def __radd__(self, g):
        r""" Add ``self`` to another :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        return BufferDict(self, buf=self.flat[:] + g.flat)

    def __sub__(self, g):
        r""" Subtract a :class:`BufferDict` (or a dictionary) from ``self``.

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        return BufferDict(self, buf=self.flat[:] - g.flat)

    def __rsub__(self, g):
        r""" Subtract ``self`` from a :class:`BufferDict` (or a dictionary).

        The two dictionaries need to have compatible layouts: i.e., the
        same keys and array shapes.
        """
        try:
            g = BufferDict({k:g[k] for k in self})
        except KeyError:
            raise KeyError('g missing a key')
        return BufferDict(self, buf=g.flat[:] - self.flat)

    def __mul__(self, x):
        r""" Multiply ``self``` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] * x)

    def __rmul__(self, x):
        r""" Multiply ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] * x)

    # truediv and div are the same --- 1st is for python3, 2nd for python2
    def __truediv__(self, x):
        r""" Divide ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] / x)

    def __div__(self, x):
        r""" Divide ``self`` by scalar ``x``. """
        return BufferDict(self, buf=self.flat[:] / x)

    def add(self,k,v):
        r""" Augment buffer with data ``v``, indexed by key ``k``.

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

    def __getitem__(self, k):
        r""" Return piece of buffer corresponding to key ``k``. """
        try:
            d = self._odict.__getitem__(k)
            ans = self._buf[d.slice]
            return ans if d.shape is () else ans.reshape(d.shape)
        except KeyError:
            pass
        if self._ver_l == BufferDict._ver_g:
            try:
                return self._extension[k]
            except KeyError:
                pass
        else:
            self._extension = {}
            self._ver_l = BufferDict._ver_g 
        for f in BufferDict.invfcn:
            altk = f + '(' + str(k) + ')'
            try:
                d = self._odict.__getitem__(altk)
                ans = self._buf[d.slice]
                if d.shape != ():
                    ans = ans.reshape(d.shape)
                ans = BufferDict.invfcn[f](ans)
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
            if k_fcn in BufferDict.invfcn and k_stripped not in self:
                ans.append(k_stripped)
        return ans

    def all_keys(self):
        r""" Iterator over all keys and implicit keys.
        
        For example, the following code ::

            b = BufferDict()
            b['log(x)'] = 1.
            for k in b.keys():
                print(k, b[k])
            print()
            for k in b.all_keys():
                print(k, b[k])

        gives the following output::

            log(x) 1

            log(x) 1
            x 2.718281828459045

        Here ``'x'`` is not a key in dictionary ``b`` 
        but ``b['x']`` is defined implicitly from ``b['log(x)']``.
        See :meth:`gvar.BufferDict.add_distribution` for more 
        information.
        """
        # Code from Giacomo Petrillo.
        for k in self:
            yield k
            if isinstance(k, str):
                m = self.extension_pattern.match(k)
                if m and m.group(1) in BufferDict.invfcn and m.group(2) not in self:
                    yield m.group(2)

    def values(self):
        # needed for python3.5
        return [self[k] for k in self]

    def items(self):
        # needed for python3.5
        return [(k,self[k]) for k in self]

    def __setitem__(self, k, v):
        r""" Set piece of buffer corresponding to ``k`` to value ``v``.

        The shape of ``v`` must equal that of ``self[k]`` if key ``k``
        is already in ``self``.
        """
        self._extension = {}
        if k not in self:
            v = numpy.asarray(v, dtype=self._dtype)
            if v.shape==():
                # add single piece of data
                self._odict.__setitem__(
                    k, BUFFERDICTDATA(slice=len(self._buf), shape=())
                    )
                # self._buf = numpy.append(self._buf,v) ####
            else:
                # add array
                n = v.size  # numpy.size(v)    #########
                i = len(self._buf)
                self._odict.__setitem__(
                    k, BUFFERDICTDATA(slice=slice(i,i+n), shape=tuple(v.shape))
                    )
                # self._buf = numpy.append(self._buf, v) #####
            if len(self._buf) == 0:
                self._buf = v.flatten()
            else:
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

    def __delitem__(self, k):
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

    def __repr__(self):
        if self._dtype is None:
            return self.__class__.__name__ + '(' + str(self) + ')'
        else:
            return self.__class__.__name__ + '(' + str(self) + ', dtype=' + str(self._dtype) + ')'

    def __str__(self):
        # Code from Giacomo Petrillo.
        out = '{'

        listrepr = [(repr(k), repr(v)) for k, v in self.items()]
        newlinemode = any('\n' in rv for _, rv in listrepr)
        
        for rk, rv in listrepr:
            if not newlinemode:
                out += '{}: {}, '.format(rk, rv)
            elif '\n' in rv:
                rv = rv.replace('\n', '\n    ')
                out += '\n    {}:\n    {},'.format(rk, rv)
            else:
                out += '\n    {}: {},'.format(rk, rv)
                
        if out.endswith(', '):
            out = out[:-2]
        elif newlinemode:
            out += '\n'
        out += '}'
        
        return out

    def _getflat(self):
        self._extension = {}
        return self._buf.flat

    def _setflat(self, buf):
        r""" Assigns buffer with buf if same size. """
        self._extension = {}
        self._buf.flat = buf

    flat = property(_getflat, _setflat, doc='Buffer array iterator.')
    def flatten(self, mode=None):
        r""" Copy of buffer array. """
        if mode is None:
            return numpy.array(self._buf)
        else:
            return self.rbatch_buf if mode=='rbatch' else self.lbatch_buf

    def _getdtype(self):
        return self._buf.dtype

    dtype = property(_getdtype, doc='Data type of buffer array elements.')

    def _getbuf(self):
        self._extension = {}
        return self._buf

    def _setbuf(self, buf):
        r""" Replace buffer with ``buf``.

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
        r""" Length of buffer. """
        return len(self._buf)

    size = property(_getsize,doc='Size of buffer array.')

    def slice(self,k):
        r""" Return slice/index in ``self.flat`` corresponding to key ``k``."""
        self._extension = {}
        return self._odict.__getitem__(k).slice

    def slice_shape(self,k):
        r""" Return tuple ``(slice/index, shape)`` corresponding to key ``k``."""
        self._extension = {}
        return self._odict.__getitem__(k)

    def has_dictkey(self, k):
        r""" Returns ``True`` if ``self[k]`` is defined; ``False`` otherwise.

        Note that ``k`` may be a key or it may be related to a
        another key associated with a non-Gaussian distribution
        (e.g., ``'log(k)'``; see :func:`gvar.BufferDict.add_distribution`
        for more information).
        """
        return _gvar.has_dictkey(self, k)

    def _getlbatch_buf(self):
        if self.size == 0:
            return numpy.array([], dtype=self.dtype)
        rdict, nbatch = self._l2rbatch()
        return rdict.flat[:].reshape(-1, nbatch).T

    lbatch_buf = property(_getlbatch_buf, doc='lbatch_buf')

    def _getrbatch_buf(self):
        if self.size == 0:
            return numpy.array([], dtype=self.dtype)
        nbatch = None
        for k in self:
            if nbatch is None:
                nbatch = self[k].shape[-1]
            elif nbatch != self[k].shape[-1]:
                raise ValueError('Not an rbatch BufferDict')
        return numpy.array(self._buf).reshape(-1, nbatch)

    rbatch_buf = property(_getrbatch_buf, doc='rbatch_buf')

    def batch_iter(self, mode):
        r""" Iterate over dictionaries in a batch |BufferDict|.

        Args:
            mode (str): Type of batch |BufferDict|: ``'rbatch'`` if the 
                batch index on each entry is on the right; or ``'lbatch'``
                if the batch index is on the left. 

        Returns:
            An iterator that yields the individual dictionaries merged into
            the batch |BufferDict|. The returned dictionaries are type ``dict``
            (not type |BufferDict|). The values are views into the original
            |BufferDict|, not copies.
        """
        if not self:
            return []
        if mode not in ['lbatch', 'rbatch']:
            raise ValueError('unknown mode: ' + str(mode))
        lbatch = True if mode == 'lbatch' else False
        nbatch = self[list(self.keys())[0]].shape[0 if lbatch else -1]

        # main loop
        for i in range(nbatch):
            ans = {}
            if lbatch:
                for k in self:
                    ans[k] = self[k][i,...]
            else:
                for k in self:
                    ans[k] = self[k][...,i]
            yield ans

    @staticmethod
    def add_distribution(name, invfcn):
        r""" Add new parameter distribution.

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

        See also :meth:`gvar.BufferDict.uniform` for uniform distributions.

        Args:
            name (str): Distributions' function name. A ``ValueError`` is 
                raised if ``name`` is already being used for 
                a distribution; the error can be avoided by deleting
                the old definition first using 
                :meth:`BufferDict.del_distribution`.
            invfcn (callable): Inverse of the transformation function.
        """
        if '(' in name or ')' in name:
            raise ValueError('distribution name contains parenthesis (not allowed): {}'. format(name))
        if name in BufferDict.invfcn:
            raise ValueError('distribution {} already defined'.format(name))
        BufferDict.invfcn[name] = invfcn

    @staticmethod
    def del_distribution(name):
        r""" Delete |BufferDict| distribution ``name``. 
        
        Raises a ``ValueError`` if ``name`` is not the name of 
        an existing distribution.
        """
        if name in BufferDict.invfcn:
            del BufferDict.invfcn[name]
        else:
            raise ValueError('{} is not a distribution'.format(name))
        BufferDict._ver_g += 1

    @staticmethod
    def has_distribution(name):
        r""" ``True`` if ``name`` has been defined as a distribution; ``False`` otherwise. """
        return name in BufferDict.invfcn
    
    @staticmethod
    def uniform(fname, umin, umax, shape=()):
        r""" Create uniform distribution on interval ``[umin, umax]``.

        The code ::

            import gvar as gv
            b = gv.BufferDict()
            b['f(w)'] = gv.BufferDict.uniform('f', 2., 3.)
        
        adds a distribution function ``f(w)`` designed so that ``b['w']``
        corresponds to a uniform distribution on the interval ``[2., 3.]``
        (see :meth:`gvar.BufferDict.add_distribution` for more about 
        distributions).

        Args:
            fname (str): Name of function used in the :class:`BufferDict` key. 
                Note that names can be reused provided they correspond to the 
                same interval as in previous calls.
            umin (float): Minimum value of the uniform distribution.
            umax (float): Maximum value of the uniform distribution.
            shape (tuple): Shape of array of uniform variables. Default is ``()``.
        
        Returns:
            :class:`gvar.GVar` object corresponding to a uniform distribution.
        """
        if not isinstance(fname, str):
            raise ValueError('fname must be a string')
        if fname in BufferDict.invfcn:
            invfcn = BufferDict.invfcn[fname]
            if not isinstance(invfcn, _BDict_UDistribution):
                raise ValueError("distribution {} already defined".format(fname))
            elif sorted((invfcn.umin, invfcn.umax)) != sorted((umin, umax)):
                raise ValueError("distribution {} already defined".format(fname))
        else:
            BufferDict.add_distribution(fname, _BDict_UDistribution(umin, umax))
        if shape == ():
            return _gvar.gvar(0, 1)
        else:
            ans = _gvar.gvar(int(numpy.prod(shape)) * [(0, 1.)]) 
            ans.shape = shape 
            return ans

class _BDict_UDistribution(object):
    def __init__(self, umin, umax):
        self.umin = umin 
        self.umax = umax
        self.umax_umin_2 = (umax - umin) / 2.
        self.root2 = numpy.sqrt(2)
    def __call__(self, x):
        return self.umin + (_gvar.erf(x / self.root2) + 1) * self.umax_umin_2

BufferDict._ver_g = 0    # version of distribution collection (for cache synch)

def asbufferdict(g, dtype=None):
    r""" Convert ``g`` to a BufferDict, keeping only ``g[k]`` for ``k in keylist``.

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
    r""" Same as ``[dictkey(bdict, k) for k in klist]``. """
    ans = []
    for k in klist:
        if k not in bdict:
            for f in BufferDict.invfcn:
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
    r""" Find key in ``bdict`` corresponding to ``k``.

    Could be ``k`` itself or one of the standard extensions,
    such as ``log(k)`` or ``sqrt(k)``.
    """
    if k not in bdict:
        for f in BufferDict.invfcn:
            newk = f + '(' + str(k) + ')'
            if newk in bdict:
                return newk
        raise KeyError('key not used: ' + str(k))
    else:
        return k

def has_dictkey(b, k):
    r""" Returns ``True`` if ``b[k]`` is defined; ``False`` otherwise.

    Note that ``k`` may be a key or it may be related to a
    another key associated with a non-Gaussian distribution
    (e.g., ``'log(k)'``; see :func:`gvar.BufferDict.add_distribution`).
    """
    if k in b:
        return True
    else:
        for f in BufferDict.invfcn:
            newk = f + '(' + str(k) + ')'
            if newk in b:
                return True
        return False

def _stripkey(k):
    r""" Return (stripped key, fcn) where fcn is exp or square or ...

    Strip off any ``"log"`` or ``"sqrt"`` or ... prefix.
    """
    if not isinstance(k, str):
        return k, None
    m = re.match(BufferDict.extension_pattern, k)
    if m is None:
        return k, None
    k_fcn, k_stripped = m.groups()
    if k_fcn not in BufferDict.invfcn:
        return k, None
    return k_stripped, BufferDict.invfcn[k_fcn]


def nonredundant_keys(keys):
    r""" Return list containing only nonredundant keys in list ``keys``. """
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
    r""" Return dictionary with proper keys for parameter distributions (legacy code).

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
    r""" Remove redundant keys from dictionary ``p``.

    A key ``'c'`` is redundant if either of ``'log(c)'``
    or ``'sqrt(c)'`` is also a key. (There are additional redundancies
    if :meth:`gvar.add_parameter_distribution` has been used to add
    extra distributions.) This function creates a copy of ``p`` but with
    the redundant keys removed.
    """
    return BufferDict(p, keys=nonredundant_keys(p.keys()))
