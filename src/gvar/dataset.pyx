# cython: language_level=3str, binding=True
# Created by Peter Lepage (Cornell University) in 2012.
# Copyright (c) 2012-20 G. Peter Lepage.
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
import os
import re
import sys
import warnings

import numpy

# try:
#     import matplotlib.pyplot as _PLOT
# except ImportError:
#     _PLOT = None

import gvar as _gvar

if sys.version_info > (3,):
    import gzip
    import bz2
    import io
    # Python 3 needs to decode the compressed output
    # modified (fixed) from stackoverflow suggestion
    def hook_compressed_alt(encoding):
        def hook_compressed(filename, mode):
            ext = os.path.splitext(filename)[1]
            if ext == '.gz':
                return io.TextIOWrapper(gzip.open(filename, mode=mode), encoding=encoding)
            elif ext == '.bz2':
                return io.TextIOWrapper(bz2.open(filename, mode=mode), encoding=encoding)
            else:
                return open(filename, mode, encoding=encoding)
        return hook_compressed
else:
    # Python 2 can use normal routine
    def hook_compressed_alt(encoding):
        return fileinput.hook_compressed


cimport numpy, cython

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

# tools for random data: Dataset, avg_data, bin_data

def bin_data(dataset, binsize=2):
    """ Bin random data.

    ``dataset`` is a list of random numbers or random arrays, or a
    dictionary of lists of random numbers/arrays.
    ``bin_data(dataset, binsize)`` replaces consecutive groups of
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
    if hasattr(dataset,'keys'):
        # dataset is a dictionary
        if not dataset:
            return Dataset()
        newdata = Dataset()
        for k in dataset:
            newdata[k] = bin_data(dataset[k],binsize=binsize)
        return newdata

    # dataset is a list
    if len(dataset) == 0:
        return []
    # force dataset into a numpy array of floats
    try:
        dataset = numpy.asarray(dataset, numpy.float_)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in dataset.")

    nd = dataset.shape[0] - dataset.shape[0] % binsize
    accum = 0.0
    for i in range(binsize):
        accum += dataset[i:nd:binsize]
    return list(accum/float(binsize))

def avg_data(
    dataset, median=False, spread=False, bstrap=False, noerror=False, 
    mismatch='truncate', warn=False
    ):
    """ Average ``dataset`` to estimate means and covariance.

    ``dataset`` is a list of random numbers, a list of random arrays, or a dictionary
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

    ``avg_data(dataset)`` also estimates any correlations between different
    quantities in ``dataset``. When ``dataset`` is a dictionary, it does this by
    assuming that the lists of random numbers/arrays for the different
    ``dataset[k]``\s are synchronized, with the first element in one list
    corresponding to the first elements in all other lists, and so on.

    Note that estimates of the correlations are robust only if the 
    number of samples being averaged is substantially larger (eg, 10x) than
    the number of quantities being averaged. The correlation matrix is 
    poorly conditioned or singular if the number of samples is too small.
    Function :func:`gvar.dataset.svd_diagnosis` can be used to determine 
    whether there is a problem, and, if so, the problem can be ameliorated by 
    applying an SVD cut to the data after averaging (:func:`gvar.regulate`).

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
    deviations are approximated by the width of the larger interval 
    above or below the median that contains 34% of the data. These 
    estimates are more robust than the mean and standard deviation when 
    averaging over small amounts of data; in particular, they are 
    unaffected by extreme outliers in the data. The default 
    is ``median=False``.

    The third option is triggered by setting ``bstrap=True``. This is
    shorthand for setting ``median=True`` and ``spread=True``, and
    overrides any explicit setting for these keyword arguments. This is the
    typical choice for analyzing bootstrap data --- hence its name. The
    default value is ``bstrap=False``.

    The fourth option is to omit the error estimates on the averages, which
    is triggered by setting ``noerror=True``. Just the mean values are
    returned. The default value is ``noerror=False``.

    The fifth option, ``mismatch``, is only relevant when ``dataset`` is a
    dictionary whose entries ``dataset[k]`` have different sample sizes. This
    complicates the calculation of correlations between the different entries.
    There are three choices for ``mismatch``:

        ``mismatch='truncate'``: Samples are discarded from the ends of
        entries so that all entries have the same sample size (equal to the 
        smallest sample size). This is the default. 
        
        ``mismatch='wavg'``: The data set is decomposed into a collection of data sets 
        so that the entries in any given data set all have the same sample size;
        no samples are discarded. Each of the resulting data sets is averaged 
        separately, and the final result is the weighted average of these averages.
        This choice is the most accurate but also the slowest, especially for 
        large problems. It should only be used when the smallest sample size is 
        much larger (eg, 10x) than the number of quantities being averaged.
        It also requires the :mod:`lsqfit` Python module.

        ``mismatch='decorrelate'``: Ignores correlations between different
        entries ``dataset[k]``. All samples are used and correlations within
        an entry ``dataset[k]`` are retained. This is the fastest choice.
    
    The final option ``warn`` determines whether or not a warning is issued
    when different entries in a dictionary data set have different sample
    sizes. The default is ``warn=False``.
    """
    if bstrap:
        median = True
        spread = True
    kargs = dict(median=median, spread=spread, noerror=noerror, warn=False)
    if not hasattr(dataset,'keys'):
        # dataset is a list of arrays (all same shape) or numbers
        if len(dataset) == 0:
            return None
        
        try:
            dataset = numpy.asarray(dataset, float)
        except ValueError:
            raise ValueError("Inconsistent array shapes or data types in dataset.")
        if noerror:
            return numpy.median(dataset, axis=0) if median else dataset.mean(axis=0) 

        # calculate covariance
        dataset_shape = dataset.shape 
        data_shape = dataset.shape[1:]
        dataset.shape = (dataset.shape[0], -1)
        if dataset.shape[0] >= 2:
            cov = numpy.cov(dataset, rowvar=False, bias=True)
        else:
            cov = numpy.zeros(dataset.shape[1:] + dataset.shape[1:], float)
        if not spread:
            cov /= dataset.shape[0]
        if median:
            meanm, mean, meanp = numpy.percentile(dataset, q=[50 - 34.1344746, 50, 50 + 34.1344746], axis=0)
            sdev = numpy.maximum(meanp - mean, mean - meanm)
            if not spread:
                sdev /= dataset.shape[0] ** 0.5 
            # replace std dev by sdev from percentiles in cov
            if data_shape == ():
                cov = sdev ** 2
            elif mean.size == 1:
                cov.flat[:] = sdev ** 2
            else:
                D = sdev / numpy.diagonal(cov) ** 0.5 
                cov = D[None, :] * cov * D[:, None]
        else:
            mean = dataset.mean(axis=0)
        mean.shape = data_shape 
        cov.shape = data_shape + data_shape
        dataset.shape = dataset_shape
        if mean.shape == ():
            cov = cov ** 0.5
        return _gvar.gvar(mean, cov, fast=True)    

    else:
        # dataset is a dictionary filled with lists of arrays or numbers
        if len(dataset) == 0:
            return _gvar.BufferDict()
        samplesize_list = [len(dk) for dk in dataset.values()]
        samplesize = min(samplesize_list)
        if noerror:
            ans = _gvar.BufferDict()
            for k in dataset:
                ans[k] = avg_data(dataset[k], **kargs)
            return ans

        if samplesize<=0:
            raise ValueError(
                "Empty entries in dataset; can't compute correlations."
                )

        elif samplesize == 1:
            # can't compute correlations between different entries
            ans = _gvar.BufferDict()
            for k in dataset:
                ans[k] = avg_data(dataset[k], **kargs)
            return ans

        if samplesize == max(samplesize_list):
            # all samples same size
            newdataset = []                     # dataset repacked as a list of flat arrays
            ans = _gvar.BufferDict()
            for k in dataset:
                data_k = numpy.asarray(dataset[k][:samplesize])
                ans[k] = data_k[0]              # dummy place holder
                newdataset.append(data_k.reshape(samplesize, -1))
            newdataset = numpy.concatenate(tuple(newdataset),axis=1)
            return _gvar.BufferDict(
                ans,
                buf=avg_data(newdataset, **kargs)
                )

        # samplesize differs in different dataset[k]
        if warn:
            warnings.warn(
            'sample sizes differ for different entries: %d -> %d'
            % (samplesize, max(samplesize_list))
            )
        if mismatch == 'truncate':
            truncated_dataset = _gvar.BufferDict()
            for k in dataset:
                truncated_dataset[k] = dataset[k][:samplesize]
            return avg_data(truncated_dataset) 
        
        if mismatch == 'decorrelate':
            ans = _gvar.BufferDict()
            for k in dataset:
                ans[k] = avg_data(dataset[k])
            return ans 

        elif mismatch == 'wavg':
            try:
                import lsqfit
            except ImportError:
                raise ImportError('lsqfit module required for wavg')
            extra_dataset = _gvar.BufferDict()
            for k in dataset:
                if len(dataset[k]) > samplesize + 1: # want at least 2 samples
                    extra_dataset[k] = dataset[k][samplesize:]
            ans1 = avg_data(dataset, mismatch='truncate', **kargs)
            ans2 = avg_data(extra_dataset, mismatch='wavg', **kargs)
            return lsqfit.wavg([ans1, ans2])


def autocorr(dataset):
    """ Compute autocorrelation in ``dataset``.

    ``dataset`` is a list of random numbers or random arrays, or a dictionary
    of lists of random numbers/arrays.

    When ``dataset`` is a list of random numbers, ``autocorr(dataset)`` returns
    an array where ``autocorr(dataset)[i]`` is the correlation between
    elements in ``dataset`` that are separated by distance ``i`` in the list:
    for example, ::

        >>> print(autocorr([2,-2,2,-2,2,-2]))
        [ 1. -1.  1. -1.  1. -1.]

    shows perfect correlation between elements separated by an even
    interval in the list, and perfect anticorrelation between elements by
    an odd interval.

    ``autocorr(dataset)`` returns a list of arrays of autocorrelation
    coefficients when ``dataset`` is a list of random arrays. Again
    ``autocorr(dataset)[i]`` gives the autocorrelations for ``dataset`` elements
    separated by distance ``i`` in the list. Similarly ``autocorr(dataset)``
    returns a dictionary when ``dataset`` is a dictionary.

    ``autocorr(dataset)`` uses FFTs to compute the autocorrelations; the cost
    of computing the autocorrelations should grow roughly linearly with the
    number of random samples in ``dataset`` (up to logarithms).
    """
    if hasattr(dataset,'keys'):
        # dataset is a dictionary
        ans = dict()
        for k in dataset:
            ans[k] = autocorr(dataset[k])

        return ans
    # dataset is an array
    if numpy.ndim(dataset) < 1 or len(dataset) < 2:
        raise ValueError("Need at least two samples to compute autocorr.")
    # force dataset into a numpy array of floats
    try:
        dataset = numpy.asarray(dataset,numpy.float_)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in dataset.")

    datat = dataset.transpose()
    ans = numpy.zeros(datat.shape,numpy.float_)
    idxlist = numpy.ndindex(datat.shape[:-1])
    for idx in numpy.ndindex(datat.shape[:-1]):
        f = datat[idx]
        dft = numpy.fft.fft(f-f.mean())
        ans[idx] = numpy.fft.ifft(dft*dft.conjugate()).real/f.var()/len(f)
    return ans.transpose()



def bootstrap_iter(dataset, n=None):
    """ Create iterator that returns bootstrap copies of ``dataset``.

    ``dataset`` is a list of random numbers or random arrays, or a dictionary
    of lists of random numbers/arrays. ``bootstrap_iter(dataset,n)`` is an
    iterator that returns ``n`` bootstrap copies of ``dataset``. The random
    numbers/arrays in a bootstrap copy are drawn at random (with repetition
    allowed) from among the samples in ``dataset``: for example, ::

        >>> dataset = [1.1, 2.3, 0.5, 1.9]
        >>> data_iter = bootstrap_iter(dataset)
        >>> print(next(data_iter))
        [ 1.1  1.1  0.5  1.9]
        >>> print(next(data_iter))
        [ 0.5  2.3  1.9  0.5]

        >>> dataset = dict(a=[1,2,3,4],b=[1,2,3,4])
        >>> data_iter = bootstrap_iter(dataset)
        >>> print(next(data_iter))
        {'a': array([3, 3, 1, 2]), 'b': array([3, 3, 1, 2])}
        >>> print(next(data_iter))
        {'a': array([1, 3, 3, 2]), 'b': array([1, 3, 3, 2])}

        >>> dataset = [[1,2],[3,4],[5,6],[7,8]]
        >>> data_iter = bootstrap_iter(dataset)
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
    distribution from which ``dataset`` was drawn. Consequently means,
    variances and correlations for bootstrap copies should be similar to
    those in ``dataset``. Analyzing variations from bootstrap copy to copy is
    often useful when dealing with non-gaussian behavior or complicated
    correlations between different quantities.

    Parameter ``n`` specifies the maximum number of copies; there is no
    maximum if ``n is None``.
    """
    if hasattr(dataset,'keys'):
        # dataset is a dictionary
        if not dataset:
            return
        ns = min(len(dataset[k]) for k in dataset)  # number of samples
        datadict = collections.OrderedDict()
        for k in dataset:
            datadict[k] = numpy.asarray(dataset[k],numpy.float_)
        ct = 0
        while (n is None) or (ct<n):
            ct += 1
            idx = numpy.random.randint(0,ns,ns)
            ans = Dataset()
            for k in datadict:
                ans[k] = datadict[k][idx]
            yield ans
    else:
        # dataset is an array
        if len(dataset) == 0:
            return
        # force dataset into a numpy array of floats
        try:
            dataset = numpy.asarray(dataset,numpy.float_)
        except ValueError:
            raise ValueError( #
                "Inconsistent array shapes or data types in dataset.")
        ns = len(dataset)
        ct = 0
        while (n is None) or (ct<n):
            ct += 1
            idx = numpy.random.randint(0,ns,ns)
            yield dataset[idx]


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
            acc = collections.OrderedDict()
        if isinstance(inputdata, fileinput.FileInput):
            finput = inputdata
        else:
            finput = fileinput.input(
                inputdata, openhook=hook_compressed_alt('utf-8')
                )
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
                d = eval(" ".join(f[1:]), collections.OrderedDict(), collections.OrderedDict())
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
        template = numpy.array(template, dtype=object)
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
    the correlation matrix for the data in ``dataset`` to determine
    how much of that spectrum is reliably determined by this data.

    Here ``dataset`` is a list of random arrays or a dictionary
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

        s = gv.dataset.svd_diagnosis(dataset)
        avgdata = gv.svd(s.avgdata, svdcut=s.svdcut)
        s.plot_ratio(show=True)

    where the defective part of the correlation matrix is corrected by
    applying an SVD cut to the averaged data. A plot showing the ratio
    of bootstrapped eigenvalues to the actual eigenvalues is displayed
    by the ``s.plot_ratio`` command.

    Args:
        dataset: List of random arrays or a dictionary
            (e.g., :class:`gvar.dataset.Dataset`) whose values are lists
            of random numbers or random arrays. Alternatively it can
            be a tuple ``(g, Ns)`` where: ``g`` is an array of
            |GVar|\s or a dictionary whose values are |GVar|\s or
            arrays of |GVar|\s; and ``Ns`` is the number of random
            samples. Then the list of random data that is analyzed is
            created is created using ``gvar.raniter(g, n=Ns)``.

        nbstrap: Number of bootstrap copies used (default is 50).

        models: For use in conjunction with :class:`lsqfit.MultiFitter`;
            ignored when not specified. When specified, it is a list of multi-
            fitter models used to specify which parts of the data are being
            analyzed. The correlation matrix is restricted to the data
            specified by the models and the data returned are "processed data"
            for use with a multi-fitter using keyword ``pdata`` rather than
            ``data``. Ignored if keyword ``process_datasets`` is specified.

        process_dataset: Function that converts datasets into averaged
            data. Function :func:`gvar.dataset.avg_data` is used if
            set equal to ``None`` (default).

        mincut: Minimum SVD cut (default 1e-12).

    The main attributes are:

    Attributes:
        svdcut: SVD cut for bad eigenvalues in correlation matrix.
        eps: ``eps`` corresponding to ``svdcut``, for use in :func:`gvar.regulate`.
        avgdata: Averaged data (``gvar.dataset.avg_data(dataset)``).
        val: Eigenvalues of the correlation matrix.
        bsval: Bootstrap average of correlation matrix eigenvalues.
        nmod: Number of eigenmodes modified by SVD cut ``svdcut``.
    """
    def __init__(self, dataset, nbstrap=50, mincut=1e-12, models=None, process_dataset=None):
        if isinstance(dataset, tuple):
            data, ns = dataset
            tset = _gvar.dataset.Dataset()
            for d in _gvar.raniter(data, n=ns):
                tset.append(d)
            dataset = tset
        if process_dataset is not None:
            avg_data = process_dataset
        elif models is None or models == []:
            avg_data = _gvar.dataset.avg_data
        else:
            try:
                import lsqfit
            except:
                raise ValueError('Need lsqfit module to use models.')
            def avg_data(dset, models=models):
                return lsqfit.MultiFitter.process_dataset(dset, models)
        avgdata = avg_data(dataset)
        self.avgdata = avgdata
        self.mincut = mincut
        if hasattr(avgdata, 'keys'):
            isdict = True
            avgdata = avgdata.buf
        elif numpy.shape(avgdata) == ():
            # scalar --- no correlation matrix
            self.val = numpy.array([avgdata.var])
            self.bsval = self.val
            self.svdcut = 0.
            self.nmod = 0
            self.eps = 0.
            return
        else:
            isdict = False
        avgdata_corr = _gvar.evalcorr(avgdata)
        self.val = _gvar.SVD(avgdata_corr).val
        bsval_list = []
        for bsdata in _gvar.dataset.bootstrap_iter(dataset, n=nbstrap):
            bsavgdata = avg_data(bsdata)
            if isdict:
                bsavgdata = bsavgdata.buf
            bsval_list.append(_gvar.SVD(_gvar.evalcorr(bsavgdata)).val)
        self.bsval = _gvar.dataset.avg_data(bsval_list, bstrap=True)
        # use heuristic to find recommended svdcut
        ratio = _gvar.mean(self.bsval) / self.val
        cuts = self.val / self.val[-1]
        chi_sig = (2. / self.avgdata.size) ** 0.5
        # 0) impose mincut
        idx = numpy.where(
            cuts > self.mincut
            )[0]
        if len(idx) > 0:
            cuts = cuts[idx[0]:]
            ratio = ratio[idx[0]:]
        # 1) find last place that is 2 sig down
        idx = numpy.where(
            ratio < 1. - 2 * chi_sig
            )[0]
        if len(idx) > 0:
            ratio = ratio[idx[-1]:]
            cuts = cuts[idx[-1]:]
        # 2) find first place (after position 1) where 1 sig down
        idx = numpy.where(
            ratio >= 1. - chi_sig
            )[0]
        if len(idx) == 0:
            self.svdcut = cuts[0]
        elif idx[0] > 0:
            self.svdcut = (cuts[idx[0]] + cuts[idx[0] - 1]) / 2.
        else:
            self.svdcut = cuts[idx[0]]
        # determine nmod
        idx = numpy.where(
            self.val < self.svdcut * self.val[-1]
            )[0]
        if len(idx) == 0:
            self.nmod = 0
        else:
            self.nmod = idx[-1] + 1
        self.eps = (
            self.svdcut * self.val[-1] / 
            numpy.linalg.norm(avgdata_corr, numpy.inf)
            )

    def plot_ratio(self, plot=None, show=False):
        """ Plot ratio of bootstrapped eigenvalues divided by actual eigenvalues.

        Ratios (blue points) are plotted versus the value of the actual
        eigenvalues divided by the maximum eigenvalue. Error bars on
        the ratios show the range of variation across bootstrapped copies.
        A dotted line is drawn at ``1 - sqrt(2/N)``, where ``N`` is the
        number of data points. The proposed SVD cut is where the
        ratio curve intersects this line; that point is indicated
        by a vertical dashed red line. The plot object is returned.

        Args:
            plot: :class:`matplotlib` plotter used to make plot.
                Uses ``plot = matplotlib.pyplot`` if ``plot=None`` (default).
            show: Displays the plot if ``show=True`` (default ``False``).
        """
        if plot is None:
            try:
                import matplotlib.pyplot as plot 
            except ImportError:
            # if _PLOT is not None:
            #     plot = _PLOT
            # else:
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
        sig = (2. / len(self.val)) ** 0.5
        plot.plot([x[0], x[-1]], [1., 1.], 'k--')
        plot.plot([x[0], x[-1]], [1. - sig, 1. - sig], 'k:')
        plot.ylabel('bootstrap eigenvalue / exact eigenvalue')
        plot.xlabel('eigenvalue / largest eigenvalue')
        plot.xscale('log')
        plot.plot([self.svdcut, self.svdcut], [0.8, 1.2], 'r:')
        if show == True:
            plot.show()
        return plot






