import numpy as np
cimport numpy as np
from numpy cimport npy_intp as INTP_TYPE

from ._svec_smat cimport svec, smat
from ._gvarcore cimport GVar

from ._bufferdict import BufferDict

from ._scipy_connected_components cimport _connected_components_undirected

cpdef tuple _evalcov_sparse(np.ndarray[object, ndim=1] g):
    """
    Return the covariance matrix of g as a sparse CSR matrix.
    Returned values: data, indices, indptr, like scipy.sparse.csr_matrix.
    Only the upper triangular part is returned.
    """
    # get the global covariance matrix
    cdef smat cov
    if hasattr(g[0], 'cov'):
        cov = g[0].cov
    else:
        raise ValueError("g does not contain GVar's")
    
    # make a mask for the indices we need from cov
    cdef INTP_TYPE ng = len(g)
    cdef INTP_TYPE nc = cov.nrow
    cdef np.ndarray[np.int8_t, ndim=1] imask = np.zeros(nc, np.int8)
    cdef GVar ga
    cdef svec da
    for a in range(ng):
        ga = g[a]
        da = ga.d
        for i in range(da.size):
            imask[da.v[i].i] = True
    
    # compute the covariance matrix
    cdef np.ndarray[object, ndim=1] covd = np.zeros(ng, object)
    cdef INTP_TYPE col, row
    cdef np.ndarray[object, ndim=1] rows_data = np.empty(ng, object)
    cdef np.ndarray[object, ndim=1] rows_indices = np.empty(ng, object)
    cdef np.ndarray[np.float_t, ndim=1] data
    cdef np.ndarray[INTP_TYPE, ndim=1] indices
    cdef np.ndarray[INTP_TYPE, ndim=1] buflen = np.zeros(ng, np.intp)
    cdef np.ndarray[INTP_TYPE, ndim=1] length = np.zeros(ng, np.intp)
    cdef np.float_t c
    for col in range(ng):
        ga = g[col]
        covd[col] = cov.masked_dot(ga.d, imask)
        for row in range(col + 1):
            c = ga.d.dot(covd[row])
            if c != 0:
            
                # create the row buffers if needed
                if buflen[row] == 0:
                    rows_data[row] = np.empty(1)
                    rows_indices[row] = np.empty(1, np.intp)
                    buflen[row] = 1
                    
                # expand the row buffers if needed
                if buflen[row] <= length[row]:
                    rows_data[row] = np.append(rows_data[row], np.empty_like(rows_data[row]))
                    rows_indices[row] = np.append(rows_indices[row], np.empty_like(rows_indices[row]))
                    buflen[row] *= 2
                
                data = rows_data[row]
                indices = rows_indices[row]
                data[length[row]] = c
                indices[length[row]] = col
                length[row] += 1
    
    # short buffers to their used length and remove empty buffers
    cdef INTP_TYPE outrow = 0
    for row in range(ng):
        if length[row] > 0:
            rows_data[outrow] = rows_data[row][:length[row]]
            rows_indices[outrow] = rows_indices[row][:length[row]]
            outrow += 1
    rows_data = rows_data[:outrow]
    rows_indices = rows_indices[:outrow]
    
    # concatenate buffers
    data = np.concatenate(rows_data)
    indices = np.concatenate(rows_indices)
    
    # compute rows span
    cdef np.ndarray[INTP_TYPE, ndim=1] indptr = np.empty(ng + 1, np.intp)
    indptr[0] = 0
    indptr[1:] = np.cumsum(length)
    
    return data, indices, indptr

cpdef _compress_labels(np.ndarray[INTP_TYPE, ndim=1] labels):
    """Convert the labels output of _connected_components_* to a list
    of arrays of indices, where each array contains all the indices for a
    single label. The first array contains indices for all the labels with only
    one index."""

    nlabels = 1 + np.max(labels)

    cdef np.ndarray[INTP_TYPE, ndim=1] length = np.zeros(nlabels, np.intp)
    cdef np.ndarray[INTP_TYPE, ndim=1] end = np.zeros(1 + nlabels, np.intp)
    cdef np.ndarray[object, ndim=1] indices = np.empty(1 + nlabels, object)
    cdef np.ndarray[INTP_TYPE, ndim=1] idxs
    
    # count number of indices for each label
    cdef INTP_TYPE l
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
    cdef INTP_TYPE i
    for i, l in enumerate(labels):
        if length[l] == 1:
            indices[0][end[0]] = i
            end[0] += 1
        else:
            idx = 1 + l
            if end[idx] == 0:
                indices[idx] = np.empty(length[l], np.intp)
            idxs = indices[idx]
            idxs[end[idx]] = i
            end[idx] += 1
    
    # remove empty arrays
    cdef np.ndarray[object, ndim=1] output = np.empty(nlabels + 1 - nones, object)
    output[0] = indices[0]
    cdef INTP_TYPE outlen = 1
    for idxs in indices[1:]:
        if idxs is not None:
            output[outlen] = idxs
            outlen += 1
    assert outlen == len(output)
    
    return output

cpdef np.ndarray[np.float_t, ndim=1] _sub_sdev(
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

cpdef np.ndarray[np.float_t, ndim=2] _sub_cov(
    np.ndarray[INTP_TYPE, ndim=1] outindices,
    np.ndarray[np.float_t, ndim=1] data,
    np.ndarray[INTP_TYPE, ndim=1] indices,
    np.ndarray[INTP_TYPE, ndim=1] indptr,
):
    """Extract the submatrix from a CSR matrix for indices `outindices`."""

    cdef INTP_TYPE size = len(outindices)
    cdef np.ndarray[np.float_t, ndim=2] out = np.empty((size, size))
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
            out[jout, iout] = out[iout, jout]
    return out

def _evalcov_blocks(np.ndarray[object, ndim=1] g):
    """
    Like evalcov_blocks with compress=True for 1D array input.
    """
    data, indices, indptr = _evalcov_sparse(g)
    labels = np.zeros(len(g), np.intp)
    n = _connected_components_undirected(indices, indptr, indices, indptr, labels)
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
