import numpy as np
cimport numpy as np
from numpy cimport npy_intp as INTP_TYPE

from ._svec_smat import svec, smat
from ._svec_smat cimport svec, smat

from ._bufferdict import BufferDict

from ._scipy_connected_components cimport _connected_components_directed

cdef double _cov_raw(smat cov, svec x, svec y):
    """Covariance between two gvars. x and y are the d members, cov is the
    global covariance matrix."""
    
    cdef INTP_TYPE i = 0
    cdef INTP_TYPE j = 0
    cdef INTP_TYPE k = 0
    cdef double out = 0.0
    cdef INTP_TYPE xind
    cdef INTP_TYPE yind
    
    while i < x.size and j < y.size:
        xind = x[i].i
        yind = y[j].i
        row = cov.row[xind]
        
        if xind == yind:
            while k < row.size and row[k].i < yind:
                k += 1
            if k < row.size and row[k].i == yind:
                out += x[i].v * y[j].v * row[k].v
        
        if xind <= yind:
            i += 1
            k = 0
        if yind <= xind:
            j += 1
            
    return out

cdef double _cov(g1, g2):
    """Covariance between two gvars."""
    assert g1.cov is g2.cov
    return _cov_raw(g1.cov, g1.d, g2.d)

cdef tuple _evalcov_sparse(np.ndarray[object, ndim=1] g):
    """
    Return the covariance matrix of g as a sparse CSR matrix.
    Returned values: data, indices, indptr, like scipy.sparse.csr_matrix.
    """
    
    cdef np.ndarray[object, ndim=1] rows_data = np.empty(len(g), object)
    cdef np.ndarray[object, ndim=1] rows_indices = np.empty(len(g), object)
    cdef np.ndarray[INTP_TYPE, ndim=1] indptr = np.empty(len(g) + 1, np.intp)
    
    cdef np.ndarray[np.float_t, ndim=1] row_buffer = np.empty(len(g))
    cdef np.ndarray[INTP_TYPE, ndim=1] indices_buffer = np.empty(len(g), np.intp)
    
    indptr[0] = 0
    cdef INTP_TYPE buflen
    for i in range(g.size):
        buflen = 0
        for j in range(g.size):
            cov = _cov(g[i], g[j])
            if cov > 0.0:
                row_buffer[buflen] = cov
                indices_buffer[buflen] = j
                buflen += 1
        indptr[i + 1] = indptr[i] + buflen
        rows_data[i] = np.copy(row_buffer[:buflen])
        rows_indices[i] = np.copy(indices_buffer[:buflen])
    
    data = np.concatenate(rows_data)
    indices = np.concatenate(rows_indices)
    return data, indices, indptr

cdef _compress_labels(np.ndarray[INTP_TYPE, ndim=1] labels):
    """Convert the labels output of _connected_components_directed to a list
    of arrays of indices, where each array contains all the indices for a
    single label. The first array contains indices for all the labels with only
    one index."""

    nlabels = 1 + np.max(labels)

    cdef np.ndarray[INTP_TYPE, ndim=1] length = np.zeros(nlabels, np.intp)
    cdef np.ndarray[INTP_TYPE, ndim=1] end = np.zeros(1 + nlabels, np.intp)
    cdef np.ndarray[object, ndim=1] indices = np.empty(1 + nlabels, object)
    
    # count number of indices for each label
    for l in labels:
        length[l] += 1
    assert np.all(length)
    
    # array for indices of single-index labels
    nones = 0
    for l in length:
        if l == 1:
            nones += 1
    indices[0] = np.empty(nones, np.intp)

    # fill arrays of indices
    for i, l in enumerate(labels):
        if length[l] == 1:
            indices[0][end[0]] = l
            end[0] += 1
        else:
            idx = 1 + l
            if end[idx] == 0:
                indices[idx] = np.empty(length[l], np.intp)
            indices[idx][end[idx]] = i
            end[idx] += 1
    
    # remove empty arrays
    cdef np.ndarray[object, ndim=1] output = np.empty(nlabels + 1 - nones, object)
    output[0] = indices[0]
    end = 1
    for idxs in indices[1:]:
        if idxs:
            output[end] = idxs
            end += 1
    assert end == len(output)
    
    return output

cdef np.ndarray[np.float_t, ndim=1] _sub_sdev(
    np.ndarray[INTP_TYPE, ndim=1] outindices,
    np.ndarray[np.float_t, ndim=1] data,
    np.ndarray[INTP_TYPE, ndim=1] indices,
    np.ndarray[INTP_TYPE, ndim=1] indptr,
):
    """Extract the square root of the diagonal from a CSR matrix at indices
    `outindices`."""
    
    cdef np.ndarray[np.float_t, ndim=1] out = np.empty(len(outindices))
    for iout, i in enumerate(outindices):
        rowslice = slice(indptr[i], indptr[i + 1])
        rowdata = data[rowslice]
        rowindices = indices[rowslice]
        j = np.searchsorted(rowindices, i)
        if j < len(rowindices) and rowindices[j] == i:
            out[iout] = np.sqrt(rowdata[j])
        else:
            out[iout] = 0
    return out

cdef np.ndarray[np.float_t, ndim=2] _sub_cov(
    np.ndarray[INTP_TYPE, ndim=1] outindices,
    np.ndarray[np.float_t, ndim=1] data,
    np.ndarray[INTP_TYPE, ndim=1] indices,
    np.ndarray[INTP_TYPE, ndim=1] indptr,
):
    """Extract the submatrix from a CSR matrix for indices `outindices`."""

    cdef np.ndarray[np.float_t, ndim=2] out = np.empty(2 * (len(outindices),))
    for iout, i in enumerate(outindices):
        rowslice = slice(indptr[i], indptr[i + 1])
        rowdata = data[rowslice]
        rowindices = indices[rowslice]
        rowj = np.searchsorted(rowindices, outindices)
        for jout, j in enumerate(rowj):
            if j < len(rowindices) and rowindices[j] == outindices[jout]:
                out[iout, jout] = rowdata[j]
            else:
                out[iout, jout] = 0
    return out

def _evalcov_blocks(np.ndarray[object, ndim=1] g):
    """
    Like evalcov_blocks with compress=True for 1D array input.
    """
    data, indices, indptr = _evalcov_sparse(g)
    labels = np.zeros(len(g), np.intp)
    n = _connected_components_directed(indices, indptr, labels)
    indices_list = _compress_labels(labels)
    covs = [_sub_sdev(indices_list[0], data, indices, indptr)]
    for idxs in indices_list[1:]:
        covs.append(_sub_cov(idxs, data, indices, indptr))
    return list(zip(indices_list, covs))

def _sanitize(g):
    """convert g to ndarray or BufferDict"""
    if hasattr(g, 'keys'): # a dictionary
        return BufferDict(g)
    else:
        return np.array(g)

def _flat(g):
    """convert g to a flat array"""
    if hasattr(g, 'keys'): # a bufferdict
        return g.buf
    else: # an array
        return g.reshape(-1)

def _decompress(index, sdev):
    """Expand index, sdev into a list of pairs (idx, [[sdev ** 2]])"""
    return [
        (idx, np.array([[s ** 2]]))
        for idx, s in zip(index, sdev)
    ]

def evalcov_blocks(g, compress=False):
    """Faster version of gvar.evalcov_blocks"""
    g = _sanitize(g)
    g = _flat(g)
    idxs_covs = _evalcov_blocks(g)
    if not compress:
        idxs_covs = _decompress(*idxs_covs[0]) + idxs_covs[1:]
    return idxs_covs
