# Created by Peter Lepage (Cornell University) in 2012.
# Copyright (c) 2012-14 G. Peter Lepage.
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
import fileinput
import re
import warnings

import numpy

import gvar as _gvar

cimport numpy, cython

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

# tools for random data: Dataset, avg_data, bin_data

def _vec_median(v, spread=False, noerror=False):
    """ estimate the median, with errors, of data in 1-d vector ``v``.

    If ``spread==True``, the error on the median is replaced by the spread
    of the data (which is larger by ``sqrt(len(v))``).

    If ``noerror==True``, error estimates on the median are omitted.
    """
    nv = len(v)
    v = sorted(v)
    if nv%2==0:
        im = int(nv/2)
        di = int(0.341344746*nv)
        median = 0.5*(v[im-1]+v[im])
        if noerror:
            return median
        sdev = max(v[im+di]-median,median-v[im-di-1])
    else:
        im = int((nv-1)/2)
        di = int(0.341344746*nv+0.5)
        median = v[im]
        if noerror:
            return median
        sdev = max(v[im+di]-median,median-v[im-di])
    if not spread:
        sdev = sdev/nv**0.5
    return _gvar.gvar(median,sdev)


def bin_data(random_data, binsize=2):
    """ Bin random data.

    ``random_data`` is a list of random numbers or random arrays, or a
    dictionary of lists of random numbers/arrays.
    ``bin_data(random_data, binsize)`` replaces consecutive groups of
    ``binsize`` numbers/arrays by the average of those numbers/arrays. The
    result is new data list (or dictionary) with ``1/binsize`` times as much
    random data: for example, ::

        >>> print(bin_data([1,2,3,4,5,6,7],binsize=2))
        [1.5, 3.5, 5.5]
        >>> print(bin_data(dict(s=[1,2,3,4,5],v=[[1,2],[3,4],[5,6],[7,8]]),binsize=2))
        {'s': [1.5, 3.5], 'v': [array([ 2.,  3.]), array([ 6.,  7.])]}

    Data is dropped at the end if there is insufficient data to from complete
    bins. Binning is used to make calculations
    faster and to reduce measurement-to-measurement correlations, if they
    exist. Over-binning erases useful information.
    """
    if hasattr(random_data,'keys'):
        # random_data is a dictionary
        if not random_data:
            return Dataset()
        newdata = Dataset()
        for k in random_data:
            newdata[k] = bin_data(random_data[k],binsize=binsize)
        return newdata

    # random_data is a list
    if len(random_data) == 0:
        return []
    # force random_data into a numpy array of floats
    try:
        random_data = numpy.asarray(random_data, numpy.float_)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in random_data.")

    nd = random_data.shape[0] - random_data.shape[0] % binsize
    accum = 0.0
    for i in range(binsize):
        accum += random_data[i:nd:binsize]
    return list(accum/float(binsize))



def avg_data(random_data, median=False, spread=False, bstrap=False, noerror=False, warn=True):
    """ Average ``random_data`` to estimate means and covariance.

    ``random_data`` is a list of random numbers, a list of random arrays, or a dictionary
    of lists of random numbers and/or arrays: for example, ::

        >>> random_numbers = [1.60, 0.99, 1.28, 1.30, 0.54, 2.15]
        >>> random_arrays = [[12.2,121.3],[13.4,149.2],[11.7,135.3],
        ...                  [7.2,64.6],[15.2,69.0],[8.3,108.3]]
        >>> random_dict = dict(n=random_numbers,a=random_arrays)

    where in each case there are six random numbers/arrays. ``avg_data``
    estimates the means of the distributions from which the random
    numbers/arrays are drawn, together with the uncertainties in those
    estimates. The results are returned as a |GVar| or an array of
    |GVar|\s, or a dictionary of |GVar|\s and/or arrays of |GVar|\s::

        >>> print(avg_data(random_numbers))
        1.31(20)
        >>> print(avg_data(random_arrays))
        [11.3(1.1) 108(13)]
        >>> print(avg_data(random_dict))
        {'a': array([11.3(1.1), 108(13)], dtype=object),'n': 1.31(20)}

    The arrays in ``random_arrays`` are one dimensional; in general, they
    can have any shape.

    ``avg_data(random_data)`` also estimates any correlations between different
    quantities in ``random_data``. When ``random_data`` is a dictionary, it does this by
    assuming that the lists of random numbers/arrays for the different
    ``random_data[k]``\s are synchronized, with the first element in one list
    corresponding to the first elements in all other lists, and so on. If
    some lists are shorter than others, the longer lists are truncated to
    the same length as the shortest list (discarding data samples).

    There are four optional arguments. If argument ``spread=True`` each
    standard deviation in the results refers to the spread in the data, not
    the uncertainty in the estimate of the mean. The former is ``sqrt(N)``
    larger where ``N`` is the number of random numbers (or arrays) being
    averaged::

        >>> print(avg_data(random_numbers,spread=True))
        1.31(50)
        >>> print(avg_data(random_numbers))
        1.31(20)
        >>> print((0.50 / 0.20) ** 2)   # should be (about) 6
        6.25

    This is useful, for example, when averaging bootstrap data. The default
    value is ``spread=False``.

    The second option is triggered by setting ``median=True``. This
    replaces the means in the results by medians, while the standard
    deviations are approximated by the half-width of the interval, centered
    around the median, that contains 68% of the data. These estimates are
    more robust than the mean and standard deviation when averaging over
    small amounts of data; in particular, they are unaffected by extreme
    outliers in the data. The default is ``median=False``.

    The third option is triggered by setting ``bstrap=True``. This is
    shorthand for setting ``median=True`` and ``spread=True``, and
    overrides any explicit setting for these keyword arguments. This is the
    typical choice for analyzing bootstrap data --- hence its name. The
    default value is ``bstrap=False``.

    The fourth option is to omit the error estimates on the averages, which
    is triggered by setting ``noerror=True``. Just the mean values are
    returned. The default value is ``noerror=False``.

    The final option ``warn`` determines whether or not a warning is issued when
    different components of a dictionary data set have different sample sizes.
    """
    if bstrap:
        median = True
        spread = True
    if hasattr(random_data,'keys'):
        # random_data is a dictionary
        if not random_data:
            return _gvar.BufferDict()
        newdata = []                    # random_data repacked as a list of arrays
        samplesize_list = [len(random_data[k]) for k in random_data]
        samplesize = min(samplesize_list)
        if warn and samplesize != max(samplesize_list):
            warnings.warn(
                'sample sizes differ for different entries: %d %d'
                % (samplesize, max(samplesize_list))
                )
        if samplesize<=0:
            raise ValueError(
                "Largest consistent sample size is zero --- no data."
                )
        bd = _gvar.BufferDict()
        for k in random_data:
            data_k = numpy.asarray(random_data[k][:samplesize])
            bd[k] = data_k[0]
            newdata.append(data_k.reshape(samplesize,-1))
        newdata = numpy.concatenate(tuple(newdata),axis=1)
        return _gvar.BufferDict(
            bd,
            buf=avg_data(newdata, median=median, spread=spread, noerror=noerror)
            )

    # random_data is list
    if len(random_data) == 0:
        return None
    # force random_data into a numpy array of floats
    try:
        random_data = numpy.asarray(random_data,numpy.float_)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in random_data.")

    # avg_data.nmeas = len(random_data)
    if median:
        # use median and spread
        if len(random_data.shape)==1:
            return _vec_median(random_data,spread=spread, noerror=noerror)
        else:
            tdata = random_data.transpose()
            tans = numpy.empty(random_data.shape[1:],object).transpose()
            for ij in numpy.ndindex(tans.shape):
                tans[ij] = _vec_median(tdata[ij],spread=spread, noerror=noerror)
            ans = tans.transpose()
            if noerror:
                return ans
            cov = numpy.cov(random_data.reshape(random_data.shape[0],ans.size),
                            rowvar=False,bias=True)
            if ans.size==1:                 # rescale std devs
                D = _gvar.sdev(ans)/cov**0.5
            else:
                D = _gvar.sdev(ans).reshape(ans.size)/numpy.diag(cov)**0.5
            cov = ((cov*D).transpose()*D).transpose()
            return _gvar.gvar(_gvar.mean(ans),cov.reshape(ans.shape+ans.shape))

    else:
        # use mean and standard deviation
        means = random_data.mean(axis=0)
        if noerror:
            return means
        norm = 1.0 if spread else float(len(random_data))
        if len(random_data)>=2:
            cov = numpy.cov(
                random_data.reshape(random_data.shape[0], means.size),
                rowvar=False, bias=True
                ) / norm
        else:
            cov = numpy.zeros(means.shape + means.shape, numpy.float_)
        if cov.shape==() and means.shape==():
            cov = cov**0.5
        return _gvar.gvar(means, cov.reshape(means.shape+means.shape))


def autocorr(random_data):
    """ Compute autocorrelation in ``random_data``.

    ``random_data`` is a list of random numbers or random arrays, or a dictionary
    of lists of random numbers/arrays.

    When ``random_data`` is a list of random numbers, ``autocorr(random_data)`` returns
    an array where ``autocorr(random_data)[i]`` is the correlation between
    elements in ``random_data`` that are separated by distance ``i`` in the list:
    for example, ::

        >>> print(autocorr([2,-2,2,-2,2,-2]))
        [ 1. -1.  1. -1.  1. -1.]

    shows perfect correlation between elements separated by an even
    interval in the list, and perfect anticorrelation between elements by
    an odd interval.

    ``autocorr(random_data)`` returns a list of arrays of autocorrelation
    coefficients when ``random_data`` is a list of random arrays. Again
    ``autocorr(random_data)[i]`` gives the autocorrelations for ``random_data`` elements
    separated by distance ``i`` in the list. Similarly ``autocorr(random_data)``
    returns a dictionary when ``random_data`` is a dictionary.

    ``autocorr(random_data)`` uses FFTs to compute the autocorrelations; the cost
    of computing the autocorrelations should grow roughly linearly with the
    number of random samples in ``random_data`` (up to logarithms).
    """
    if hasattr(random_data,'keys'):
        # random_data is a dictionary
        ans = dict()
        for k in random_data:
            ans[k] = autocorr(random_data[k])

        return ans
    # random_data is an array
    if numpy.ndim(random_data) < 1 or len(random_data) < 2:
        raise ValueError("Need at least two samples to compute autocorr.")
    # force random_data into a numpy array of floats
    try:
        random_data = numpy.asarray(random_data,numpy.float_)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in random_data.")

    datat = random_data.transpose()
    ans = numpy.zeros(datat.shape,numpy.float_)
    idxlist = numpy.ndindex(datat.shape[:-1])
    for idx in numpy.ndindex(datat.shape[:-1]):
        f = datat[idx]
        dft = numpy.fft.fft(f-f.mean())
        ans[idx] = numpy.fft.ifft(dft*dft.conjugate()).real/f.var()/len(f)
    return ans.transpose()



def bootstrap_iter(random_data, n=None):
    """ Create iterator that returns bootstrap copies of ``random_data``.

    ``random_data`` is a list of random numbers or random arrays, or a dictionary
    of lists of random numbers/arrays. ``bootstrap_iter(random_data,n)`` is an
    iterator that returns ``n`` bootstrap copies of ``random_data``. The random
    numbers/arrays in a bootstrap copy are drawn at random (with repetition
    allowed) from among the samples in ``random_data``: for example, ::

        >>> random_data = [1.1, 2.3, 0.5, 1.9]
        >>> data_iter = bootstrap_iter(random_data)
        >>> print(next(data_iter))
        [ 1.1  1.1  0.5  1.9]
        >>> print(next(data_iter))
        [ 0.5  2.3  1.9  0.5]

        >>> random_data = dict(a=[1,2,3,4],b=[1,2,3,4])
        >>> data_iter = bootstrap_iter(random_data)
        >>> print(next(data_iter))
        {'a': array([3, 3, 1, 2]), 'b': array([3, 3, 1, 2])}
        >>> print(next(data_iter))
        {'a': array([1, 3, 3, 2]), 'b': array([1, 3, 3, 2])}

        >>> random_data = [[1,2],[3,4],[5,6],[7,8]]
        >>> data_iter = bootstrap_iter(random_data)
        >>> print(next(data_iter))
        [[ 7.  8.]
         [ 1.  2.]
         [ 1.  2.]
         [ 7.  8.]]
        >>> print(next(data_iter))
        [[ 3.  4.]
         [ 7.  8.]
         [ 3.  4.]
         [ 1.  2.]]

    The distribution of bootstrap copies is an approximation to the
    distribution from which ``random_data`` was drawn. Consequently means,
    variances and correlations for bootstrap copies should be similar to
    those in ``random_data``. Analyzing variations from bootstrap copy to copy is
    often useful when dealing with non-gaussian behavior or complicated
    correlations between different quantities.

    Parameter ``n`` specifies the maximum number of copies; there is no
    maximum if ``n is None``.
    """
    if hasattr(random_data,'keys'):
        # random_data is a dictionary
        if not random_data:
            return
        ns = min(len(random_data[k]) for k in random_data)  # number of samples
        datadict = {}
        for k in random_data:
            datadict[k] = numpy.asarray(random_data[k],numpy.float_)
        ct = 0
        while (n is None) or (ct<n):
            ct += 1
            idx = numpy.random.randint(0,ns,ns)
            ans = Dataset()
            for k in datadict:
                ans[k] = datadict[k][idx]
            yield ans
    else:
        # random_data is an array
        if len(random_data) == 0:
            return
        # force random_data into a numpy array of floats
        try:
            random_data = numpy.asarray(random_data,numpy.float_)
        except ValueError:
            raise ValueError( #
                "Inconsistent array shapes or data types in random_data.")
        ns = len(random_data)
        ct = 0
        while (n is None) or (ct<n):
            ct += 1
            idx = numpy.random.randint(0,ns,ns)
            yield random_data[idx]


class Dataset(collections.OrderedDict):
    """ Dictionary for collecting random data.

    A :class:`gvar.dataset.Dataset` is an ordered dictionary whose values
    represent collections of random samples. Each value is  a :mod:`numpy`
    array whose first index labels the random  sample. Random samples can be
    numbers or arrays of numbers. The keys identify the quantity being
    sampled.

    A ``Dataset`` can be assembled piece by piece, as random data is
    accumulated, or it can be read from a file. Consider a situation
    where there are four random values for a scalar ``s`` and
    four random values for vector ``v``. These can be collected as
    follows::

        >>> dset = Dataset()
        >>> dset.append(s=1.1, v=[12.2, 20.6])
        >>> dset.append(s=0.8, v=[14.1, 19.2])
        >>> dset.append(s=0.95, v=[10.3, 19.7])
        >>> dset.append(s=0.91, v=[8.2, 21.0])
        >>> print(dset['s'])       # 4 random values of s
        [ 1.1, 0.8, 0.95, 0.91]
        >>> print(dset['v'])       # 4 random vector-values of v
        [array([ 12.2,  20.6]), array([ 14.1,  19.2]), array([ 10.3,  19.7]), array([  8.2,  21. ])]

    The argument to ``dset.append()`` can also be a dictionary: for example,
    ``dd = dict(s=1.1,v=[12.2,20.6]); dset.append(dd)`` is equivalent to the
    first ``append`` statement above. One can also append data key-by-key: for
    example, ``dset.append('s',1.1); dset.append('v',[12.2,20.6])`` is
    equivalent to the first ``append`` in the example above.

    Use ``extend`` in place of ``append`` to add data in batches: for
    example, ::

        >>> dset = Dataset()
        >>> dset.extend(s=[1.1, 0.8], v=[[12.2, 20.6], [14.1, 19.2]])
        >>> dset.extend(s=[0.95, 0.91], v=[[10.3, 19.7],[8.2, 21.0]])
        >>> print(dset['s'])       # 4 random values of s
        [ 1.1, 0.8, 0.95, 0.91]

    gives the same dataset as the first example above.

    The same ``Dataset`` can also be created from a text file named
    ``'datafile'`` with the following contents::

        # file: datafile
        s 1.1
        v [12.2, 20.6]
        s 0.8
        v [14.1, 19.2]
        s 0.95
        v [10.3, 19.7]
        s 0.91
        v [8.2, 21.0]

    Here each line consists of a key  followed by a new random sample for that
    key. Lines that begin with ``#`` are ignored. The file is  read using::

        >>> dset = Dataset('datafile')
        >>> print(dset['s'])
        [ 1.1, 0.8, 0.95, 0.91]

    Data can be binned while reading it in, which might be useful if
    the data set is huge or if correlations are a concern.
    To bin the data contained in file ``datafile`` in
    bins of bin size 2 we use::

        >>> dset = Dataset('datafile', binsize=2)
        >>> print(dset['s'])
        [0.95, 0.93]

    The keys read from a data file are restricted to those listed in keyword
    ``keys`` and those that are matched (or partially matched) by regular
    expression ``grep`` if one or other of these is specified: for
    example, ::

        >>> dset = Dataset('datafile')
        >>> print([k for k in dset])
        ['s', 'v']
        >>> dset = Dataset('datafile', keys=['v'])
        >>> print([k for k in dset])
        ['v']
        >>> dset = Dataset('datafile', grep='[^v]')
        >>> print([k for k in dset])
        ['s']
        >>> dset = Dataset('datafile', keys=['v'], grep='[^v]')
        >>> print([k for k in dset])
        []

    In addition to text files, hdf5 files can also be read (provided
    module :mod:`h5py` is available): for example, ::

        >>> dset = Dataset('datafile.h5', h5group='/mcdata')

    reads the hdf5 datasets in hdf5 group ``'/mcdata'``. An hdf5
    equivalent to the text file above would contain two groups,
    one with key ``'s'`` that is a one-dimensional array with shape (4,),
    and another with key ``'v'`` that is a two-dimensional array
    with shape (4, 2):

        >>> import h5py
        >>> for v in h5py.File('datafile.h5')['/mcdata'].values():
        ...     print(v)
        <HDF5 dataset "s": shape (4,), type "<f8">
        <HDF5 dataset "v": shape (4, 2), type "<f8">

    Finally, :class:`Dataset`\s can also be constructed from other
    dictionaries (including other :class:`Dataset`\s), or lists of key-data
    tuples. For example, ::

        >>> dset = Dataset('datafile')
        >>> dset_binned = Dataset(dset, binsize=2)
        >>> dset_v = Dataset(dset, keys=['v'])

    reads data from file ``'datafile'`` into :class:`Dataset` ``dset``, and
    then creates a new :class:`Dataset` with the data binned
    (``dset_binned``), and another that only contains the data with key
    ``'v'`` (``dset_v``).

    Args:
        inputdata (str or list or dictionary): If ``inputdata`` is a string,
            it is the name of a file containing datasets. Two formats are
            supported. If the filename ends in '.h5', the file is in hdf5
            format,  with datasets that are :mod:`numpy` arrays whose first
            index labels the random sample.

            The other file format is a text file where each line consists of
            a key followed by a number or array of numbers representing a new
            random sample associated  with that key. Lines beginning with
            ``#`` are comments. A list of text file names can also be
            supplied, and text files can be compressed (with names ending in
            ``.gz`` or ``.bz2``).

            If ``inputdata`` is a dictionary or a list of (key,value) tuples,
            its keys and values are copied into the dataset. Its values should
            be arrays whose first index labels the random sample.
        binsize (int): Bin the random samples in bins of size ``binsize``.
            Default value is ``binsize=1`` (*i.e.*, no binning).
        grep (str or ``None``): If not ``None``, only keys that match or
            partially match regular expression ``grep`` are retained in
            the data set. Keys that don't match are ignored. Default is
            ``grep=None``.
        keys (list): List of keys to retain in data set. Keys that are not
            in the list are ignored. Default is ``keys=None`` which implies
            that all keys are kept.
        h5group (str or list): Address within the hdf5 file identified by
            ``inputdata`` that contains the relevant datasets. Every
            hdf5 dataset in group ``h5group`` is
            read into the dataset, with the same key as in ``h5group``.
            Default is the top group in the file: ``h5group='/'``.
            ``h5group`` can also be a list of groups, in which case
            datasets from all of the groups are read.
    """
    def __init__(
        self, inputdata=None, int binsize=1, grep=None, keys=None, h5group='/',
        nbin=None,
        ):
        if inputdata is None:
            super(Dataset, self).__init__()
            return
        if nbin is not None and binsize is 1:
            binsize = nbin
        if grep is not None:
            grep = re.compile(grep)
        if isinstance(inputdata, str) and inputdata[-3:] == '.h5':
            try:
                import h5py
            except ImportError:
                raise ImportError('need module h5py to read hpf5 .h5 file')
            if isinstance(h5group, str):
                h5group = [h5group]
            if h5group == [] or h5group is None:
                h5group = '/'
            with h5py.File(inputdata, 'r') as h5file:
                inputdata = collections.OrderedDict()
                for h5g in h5group:
                    h5g = h5file[h5g]
                    for k in h5g:
                        if isinstance(h5g[k], h5py.Dataset):
                            inputdata[k] = list(numpy.array(h5g[k]))
        try:
            # inputdata = Dataset or dictionary
            super(Dataset, self).__init__(inputdata)
            if grep is not None:
                for k in list(self.keys()):
                    if grep.search(k) is None:
                        del self[k]
            if keys:
                for k in list(self.keys()):
                    if k not in keys:
                        del self[k]
            if binsize > 1:
                for k in self:
                    self[k] = bin_data(self[k], binsize=binsize)
            return
        except ValueError:
            pass
        # inputdata = files
        super(Dataset, self).__init__()
        if binsize>1:
            acc = {}
        if isinstance(inputdata, fileinput.FileInput):
            finput = inputdata
        else:
            finput = fileinput.input(inputdata, openhook=fileinput.hook_compressed)
        for line in finput:
            f = line.split()
            if len(f)<2 or f[0][0]=='#':
                continue
            k = f[0]
            if keys and k not in keys:
                continue
            if grep is not None and grep.search(k) is None:
                continue
            if len(f)==2:
                d = eval(f[1])
            elif f[1][0] in "[(":
                d = eval(" ".join(f[1:]), {}, {})
            else:
                try:
                    d = [float(x) for x in f[1:]]
                except ValueError:
                    raise ValueError('Bad input line: "%s"'%line[:-1])
            if binsize<=1:
                self.append(k, d)
            else:
                acc.setdefault(k, []).append(d)
                if len(acc[k])==binsize:
                    d = numpy.sum(acc[k], axis=0)/float(binsize)
                    del acc[k]
                    self.append(k, d)

    def toarray(self):
        """ Create new dictionary ``d`` where ``d[k]=numpy.array(self[k])`` for all ``k``. """
        ans = collections.OrderedDict()
        for k in self:
            ans[k] = numpy.array(self[k],numpy.float_)
        return ans

    def append(self, *args, **kargs):
        """ Append data to dataset.

        There are three equivalent ways of adding data to a dataset
        ``data``: for example, each of ::

            data.append(n=1.739,a=[0.494,2.734])        # method 1

            data.append(n,1.739)                        # method 2
            data.append(a,[0.494,2.734])

            dd = dict(n=1.739,a=[0.494,2.734])          # method 3
            data.append(dd)

        adds one new random number to ``data['n']``, and a new
        vector to ``data['a']``.
        """
        if len(args)>2 or (args and kargs):
            raise ValueError("Too many arguments.")
        if len(args)==2:
            # append(k, m)
            k = args[0]
            try:
                d = numpy.asarray(args[1],numpy.float_)
            except ValueError:
                raise ValueError("Unreadable data: " + str(args[1]))
            if d.shape==():
                d = d.flat[0]
            if k not in self:
                self[k] = [d]
            elif d.shape!=self[k][0].shape:
                raise ValueError(
                    "Shape mismatch between samples %s: %s,%s"%
                    (k, d.shape, self[k][0].shape)
                    )
            else:
                self[k].append(d)
            return
        if len(args)==1:
            # append(kmdict)
            kargs = args[0]
            if not hasattr(kargs, 'keys'):
                raise ValueError("Argument not a dictionary.")
        for k in kargs:
            self.append(k, kargs[k])

    def extend(self, *args, **kargs):
        """ Add batched data to dataset.

        There are three equivalent ways of adding batched data, containing
        multiple samples for each quantity, to a dataset ``data``: for
        example, each of ::

            data.extend(n=[1.739,2.682],
                        a=[[0.494,2.734],[ 0.172, 1.400]])  # method 1

            data.extend(n,[1.739,2.682])                    # method 2
            data.extend(a,[[0.494,2.734],[ 0.172, 1.400]])

            dd = dict(n=[1.739,2.682],
                        a=[[0.494,2.734],[ 0.172, 1.400]])  # method 3
            data.extend(dd)

        adds two new random numbers to ``data['n']``, and two new
        random vectors to ``data['a']``.

        This method can be used to merge two datasets, whether or not they
        share keys: for example, ::

            data = Dataset("file1")
            data_extra = Dataset("file2")
            data.extend(data_extra)   # data now contains all of data_extra
        """
        if len(args) > 2 or (args and kargs):
            raise ValueError("Too many arguments.")
        if len(args)==2:
            # extend(k,m)
            k = args[0]
            try:
                d = [numpy.asarray(di,numpy.float_) for di in args[1]]
            except TypeError:
                raise TypeError('Bad argument.')
            if not d:
                return
            if any(d[0].shape!=di.shape for di in d):
                raise ValueError("Inconsistent shapes.")
            if d[0].shape==():
                d = [di.flat[0] for di in d]
            if k not in self:
                self[k] = d
            elif self[k][0].shape!=d[0].shape:
                raise ValueError( #
                    "Shape mismatch between samples %s: %s,%s"%
                    (k,d[0].shape,self[k][0].shape))
            else:
                self[k].extend(d)
            return
        if len(args)==1:
            # extend(kmdict)
            kargs = args[0]
            if not hasattr(kargs,'keys'):
                raise ValueError("Argument not a dictionary.")
        for k in kargs:
            self.extend(k,kargs[k])

    def slice(self, sl):
        """ Create new dataset with ``self[k] -> self[k][sl].``

        Parameter ``sl`` is a slice object that is applied to every
        item in the dataset to produce a new :class:`gvar.Dataset`.
        Setting ``sl = slice(0,None,2)``, for example, discards every
        other sample for each quantity in the dataset. Setting
        ``sl = slice(100,None)`` discards the first 100 samples for
        each quantity.

        If parameter ``sl`` is a tuple of slice objects, these
        are applied to successive indices of ``self[k]``. An exception
        is called if the number of slice objects exceeds the number
        of dimensions for any ``self[k]``.
        """
        if isinstance(sl, tuple) and len(sl) > 1:
            ans = Dataset()
            s0 = sl[0]
            s1 = sl[1:]
            for k  in self:
                ans[k] = [d[s1] for d in self[k][s0]]
            return ans
        ans = Dataset()
        for k in self:
            ans[k] = self[k][sl]
        return ans

    def grep(self, rexp):
        """ Create new dataset containing items whose keys match ``rexp``.

        Returns a new :class:`gvar.dataset.Dataset`` containing only the
        items ``self[k]`` whose keys ``k`` match regular expression
        ``rexp`` (a string) according to Python module :mod:`re`::

            >>> a = Dataset()
            >>> a.append(xx=1.,xy=[10.,100.])
            >>> a.append(xx=2.,xy=[20.,200.])
            >>> print(a.grep('y'))
            {'yy': [array([  10.,  100.]), array([  20.,  200.])]}
            >>> print(a.grep('x'))
            {'xx': [1.0, 2.0], 'xy': [array([  10.,  100.]), array([  20.,  200.])]}
            >>> print(a.grep('x|y'))
            {'xx': [1.0, 2.0], 'xy': [array([  10.,  100.]), array([  20.,  200.])]}
            >>> print a.grep('[^y][^x]')
            {'xy': [array([  10.,  100.]), array([  20.,  200.])]}

        Items are retained even if ``rexp`` matches only part of the item's
        key.
        """
        prog = re.compile(rexp)
        ans = Dataset()
        for k in self:
            if prog.search(k) is not None:
                ans[k] = self[k]
        return ans

    def trim(self):
        """ Create new dataset where all entries have same sample size. """
        ns = self.samplesize
        ans = Dataset()
        for k in self:
            ans[k] = self[k][:ns]
        return ans

    def _get_samplesize(self):
        return min([len(self[k]) for k in self])

    samplesize = property(_get_samplesize,
                          doc="Smallest number of samples for any key.")

    def arrayzip(self, template):
        """ Merge lists of random data according to ``template``.

        ``template`` is an array of keys in the dataset, where the shapes
        of ``self[k]`` are the same for all keys ``k`` in ``template``.
        ``self.arrayzip(template)`` merges the lists of random
        numbers/arrays associated with these keys to create a new list of
        (merged) random arrays whose layout is specified by ``template``:
        for example, ::

            >>> d = Dataset()
            >>> d.append(a=1,b=10)
            >>> d.append(a=2,b=20)
            >>> d.append(a=3,b=30)
            >>> print(d)            # three random samples each for a and b
            {'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 30.0]}
            >>> # merge into list of 2-vectors:
            >>> print(d.arrayzip(['a','b']))
            [[  1.  10.]
             [  2.  20.]
             [  3.  30.]]
            >>> # merge into list of (symmetric) 2x2 matrices:
            >>> print(d.arrayzip([['b','a'],['a','b']]))
            [[[ 10.   1.]
              [  1.  10.]]

             [[ 20.   2.]
              [  2.  20.]]

             [[ 30.   3.]
              [  3.  30.]]]

        The number of samples in each merged result is the same as the
        number samples for each key (here 3). The keys used in this example
        represent scalar quantities; in general, they could be either
        scalars or arrays (of any shape, so long as all have the same
        shape).
        """
        # regularize and test the template
        template = numpy.array(template, dtype=numpy.object)
        template_shape = template.shape
        template_flat = template.flat
        if not template_flat:
            return Dataset()
        try:
            assert all((k in self) for k in template_flat), \
                "Some keys in template not in Dataset."
        except TypeError:
            raise ValueError("Poorly formed template.")
        shape = numpy.shape(self[template_flat[0]])
        if not all(numpy.shape(self[k]) == shape for k in template_flat[1:]):
            raise ValueError(           #
                "Different shapes for different elements in template.")
        n_sample = shape[0]
        ans_shape = shape[:1] + template_shape + shape[1:]
        ans = numpy.zeros(ans_shape, numpy.float_)
        ans = ans.reshape(n_sample, template.size, -1)
        for i,k in enumerate(template_flat):
            ans[:, i, :] = numpy.reshape(self[k], (n_sample,-1))
        return ans.reshape(ans_shape)


class svd_diagnosis(object):
    """ Diagnose the need for an SVD cut.

    :class:`gvar.dataset.svd_diagnosis` bootstraps the spectrum of
    the correlation matrix for the data in ``data`` to determine
    how much of that spectrum is reliably determined by this data.

    Here ``data`` is a list of random arrays or a dictionary
    (e.g., :class:`gvar.dataset.Dataset`) whose values are lists
    of random numbers or random arrays. The random numbers or
    arrays are averaged (using :func:`gvar.dataset.avg_data`)
    to produce a set |GVar|\s and their correlation matrix.
    The smallest eigenvalues of the correlation matrix are poorly
    estimated when the number of random samples is insufficiently
    large --- the number of samples should typically be significantly
    larger than the number of random variables being analyzed in
    order to get good estimates of the correlations between these
    variables.

    Typical usage is ::

        import gvar as gv

        s = gv.dataset.svd_diagnosis(data)
        avgdata = gv.svd(s.avgdata, svdcut=s.svdcut)
        s.plot_ratio(show=True)

    where the defective part of the correlation matrix is corrected by
    applying an SVD cut to the averaged data. A plot showing the ratio
    of bootstrapped eigenvalues to the actual eigenvalues is displayed
    by the ``s.plot_ratio`` command.

    Args:
        random_data: List of random arrays or a dictionary
            (e.g., :class:`gvar.dataset.Dataset`) whose values are lists
            of random numbers or random arrays.

        nbstrap: Number of bootstrap copies used (default is 50).

        models: For use in conjunction with :class:`lsqfit.MultiFitter`;
            ignored when not specified. When specified, it is a list of multi-
            fitter models used to specify which parts of the data are being
            analyzed. The correlation matrix is restricted to the data
            specified by the models and the data returned are "processed data"
            for use with a multi-fitter using keyword ``pdata`` rather than
            ``data``.

        mincut: Minimum SVD cut (default 1e-14).

    The main attributes are:

    Attributes:
        svdcut: SVD cut for bad eigenvalues in correlation matrix.
        avgdata: Averaged data (``gvar.dataset.avg_data(random_data)``).
        val: Eigenvalues of the correlation matrix.
        bsval: Bootstrap average of correlation matrix eigenvalues.
        nmod: Number of eigenmodes modified by SVD cut ``svdcut``.
    """
    def __init__(self, random_data, nbstrap=50, mincut=1e-14, models=None):
        if models is None or models == []:
            avg_data = _gvar.dataset.avg_data
        else:
            try:
                import lsqfit
            except:
                raise ValueError('Need lsqfit module to use models.')
            def avg_data(dset, models=models):
                return lsqfit.MultiFitter.process_dataset(dset, models)
        avgdata = avg_data(random_data)
        self.avgdata = avgdata
        self.mincut = mincut
        if hasattr(avgdata, 'keys'):
            isdict = True
            avgdata = avgdata.buf
        elif numpy.shape(avgdata) == ():
            # scalar --- no correlation matrix
            self.val = numpy.array([avgdata.var])
            self.bsval = self.val
            self.svdcut = 1.
            self.nmod = 0
            return
        else:
            isdict = False
        self.val = _gvar.SVD(_gvar.evalcorr(avgdata)).val
        bsval_list = []
        for bsdata in _gvar.dataset.bootstrap_iter(random_data, n=nbstrap):
            bsavgdata = avg_data(bsdata)
            if isdict:
                bsavgdata = bsavgdata.buf
            bsval_list.append(_gvar.SVD(_gvar.evalcorr(bsavgdata)).val)
        self.bsval = _gvar.dataset.avg_data(bsval_list, bstrap=True)
        ratio = self.bsval / self.val
        cuts = self.val / self.val[-1]
        idx = cuts > self.mincut
        ratio = ratio[idx]
        cuts = cuts[idx]
        idx = numpy.where(
            _gvar.fabs(_gvar.mean(ratio) - 1.) / _gvar.sdev(ratio) < 1.
            )[0]
        if len(idx) == 0:
            idx = [len(self.val) - 1]
        if idx[0] > 0:
            self.svdcut = (cuts[idx[0]] + cuts[idx[0] - 1]) / 2.
        else:
            self.svdcut = 0.9 * cuts[idx[0]]
        self.nmod = idx[0]

    def plot_ratio(self, plot=None, show=False):
        """ Plot ratio of bootstrapped eigenvalues divided by actual eigenvalues.

        Ratios are plotted versus the value of the actual eigenvalues
        divided by the maximum eigenvalue. A dashed line shows the
        position of the proposed SVD cut. The plot object is returned.

        Args:
            plot: :class:`matplotlib` plotter used to make plot.
                Uses ``plot = matplotlib.pyplot`` if ``plot=None`` (default).
            show: Displays the plot if ``show=True`` (default ``False``).
            minx: Minimum ``x`` value in plot (default is 1e-14).
        """
        if plot is None:
            try:
                import matplotlib.pyplot as plot
            except:
                warnings.warn('Need matplotlib library to make plots')
                return None
        x = self.val / self.val[-1]
        ratio = self.bsval / self.val
        idx = x > self.mincut
        ratio = ratio[idx]
        x = x[idx]
        y = _gvar.mean(ratio)
        yerr = _gvar.sdev(ratio)
        plot.errorbar(x=x, y=y, yerr=yerr, fmt='+', color='b')
        plot.plot([x[0], x[-1]], [1., 1.], 'k--')
        plot.ylabel('(bootstrap / exact) for eigenvalues')
        plot.xlabel('eigenvalue / largest eigenvalue')
        plot.xscale('log')
        plot.plot([self.svdcut, self.svdcut], [0.8, 1.2], 'r:')
        if show == True:
            plot.show()
        return plot






