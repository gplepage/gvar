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
    Matrix format: data, indices, indptr, like scipy.sparse.csr_matrix.
    Two matrices are returned: the upper triangular part and the lower
    triangular part. Thus the function output is:
        (udata, uindices, uindptr, ldata, lindices, lindptr)
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
    
    # variables for computing the covariance matrix
    cdef np.ndarray[object, ndim=1] covd = np.zeros(ng, object)
    cdef INTP_TYPE col, row
    cdef np.float_t c
    cdef np.ndarray[np.float_t, ndim=1] data
    cdef np.ndarray[INTP_TYPE, ndim=1] indices
    
    # variables only for upper output
    cdef np.ndarray[object, ndim=1] upper_rows_data = np.empty(ng, object)
    cdef np.ndarray[object, ndim=1] upper_rows_indices = np.empty(ng, object)
    cdef np.ndarray[INTP_TYPE, ndim=1] upper_buflen = np.zeros(ng, np.intp)
    cdef np.ndarray[INTP_TYPE, ndim=1] upper_length = np.zeros(ng, np.intp)
    
    # variables only for lower output
    cdef np.ndarray[object, ndim=1] lower_rows_data = np.empty(ng, object)
    cdef np.ndarray[object, ndim=1] lower_rows_indices = np.empty(ng, object)
    cdef np.ndarray[INTP_TYPE, ndim=1] lindptr = np.empty(1 + ng, np.intp)
    lindptr[0] = 0
    cdef np.ndarray[np.float_t, ndim=1] lower_row_data_buffer = np.empty(ng)
    cdef np.ndarray[INTP_TYPE, ndim=1] lower_row_indices_buffer = np.empty(ng, np.intp)
    cdef INTP_TYPE lower_buflen
    cdef INTP_TYPE lower_nbuf = 0
    
    # compute the covariance matrix
    for col in range(ng):
        
        # compute (global covariance) @ transform[:,col]
        ga = g[col]
        covd[col] = cov.masked_dot(ga.d, imask)
        
        lower_buflen = 0
        for row in range(col + 1):
        
            # compute transform[row,:] @ (global covariance) @ transform[:,col]
            c = ga.d.dot(covd[row])
            if c != 0:
            
                ### UPPER OUTPUT ###
                
                # create the row buffers if needed
                if upper_buflen[row] == 0:
                    upper_rows_data[row] = np.empty(1)
                    upper_rows_indices[row] = np.empty(1, np.intp)
                    upper_buflen[row] = 1
                    
                # expand the row buffers if needed
                if upper_buflen[row] <= upper_length[row]:
                    data = upper_rows_data[row]
                    indices = upper_rows_indices[row]
                    upper_rows_data[row] = np.append(data, np.empty_like(data))
                    upper_rows_indices[row] = np.append(indices, np.empty_like(indices))
                    upper_buflen[row] *= 2
                
                # append the value to the row buffers
                data = upper_rows_data[row]
                indices = upper_rows_indices[row]
                data[upper_length[row]] = c
                indices[upper_length[row]] = col
                upper_length[row] += 1
                
                ### LOWER OUTPUT ###
                # here row, col is transposed
                
                lower_row_data_buffer[lower_buflen] = c
                lower_row_indices_buffer[lower_buflen] = row
                lower_buflen += 1
            
        if lower_buflen > 0:
            lower_rows_data[lower_nbuf] = np.copy(lower_row_data_buffer[:lower_buflen])
            lower_rows_indices[lower_nbuf] = np.copy(lower_row_indices_buffer[:lower_buflen])
            lower_nbuf += 1
        lindptr[col + 1] = lindptr[col] + lower_buflen
    
    ### UPPER OUTPUT ###
    
    # short buffers to their used length and remove empty buffers
    cdef INTP_TYPE outrow = 0
    for row in range(ng):
        if upper_length[row] > 0:
            upper_rows_data[outrow] = upper_rows_data[row][:upper_length[row]]
            upper_rows_indices[outrow] = upper_rows_indices[row][:upper_length[row]]
            outrow += 1
    upper_rows_data = upper_rows_data[:outrow]
    upper_rows_indices = upper_rows_indices[:outrow]
    
    # concatenate buffers
    udata = np.concatenate(upper_rows_data)
    uindices = np.concatenate(upper_rows_indices)
    
    # compute rows span
    cdef np.ndarray[INTP_TYPE, ndim=1] uindptr = np.empty(ng + 1, np.intp)
    uindptr[0] = 0
    uindptr[1:] = np.cumsum(upper_length)
    
    ### LOWER OUTPUT ###
    
    # concatenate buffers truncating to used buffers
    ldata = np.concatenate(lower_rows_data[:lower_nbuf])
    lindices = np.concatenate(lower_rows_indices[:lower_nbuf])
    
    return udata, uindices, uindptr, ldata, lindices, lindptr

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
    """Extract the submatrix from a CSR matrix for indices `outindices`. The
    matrix must be the upper triangular part of a symmetric matrix."""

    cdef INTP_TYPE size = len(outindices)
    cdef np.ndarray[np.float_t, ndim=2] out = np.zeros((size, size))
    cdef INTP_TYPE iout, i, jout, j
    cdef np.ndarray[INTP_TYPE, ndim=1] rowindices, rowj
    cdef np.ndarray[np.float_t, ndim=1] rowdata
    for iout, i in enumerate(outindices):
        rowslice = slice(indptr[i], indptr[i + 1])
        rowdata = data[rowslice]
        rowindices = indices[rowslice]
        rowj = np.searchsorted(rowindices, outindices)
        for jout, j in enumerate(rowj):
            if j < len(rowindices) and rowindices[j] == outindices[jout]:
                out[iout, jout] = rowdata[j]
                out[jout, iout] = out[iout, jout]
    return out

def _evalcov_blocks(np.ndarray[object, ndim=1] g):
    """
    Like evalcov_blocks with compress=True for 1D array input.
    """
    udata, uindices, uindptr, ldata, lindices, lindptr = _evalcov_sparse(g)
    labels = np.zeros(len(g), np.intp)
    n = _connected_components_undirected(uindices, uindptr, lindices, lindptr, labels)
    indices_list = _compress_labels(labels)
    covs = [_sub_sdev(indices_list[0], udata, uindices, uindptr)]
    for idxs in indices_list[1:]:
        covs.append(_sub_cov(idxs, udata, uindices, uindptr))
    return list(zip(indices_list, covs))

def _sanitize(g):
    """convert g to ndarray or BufferDict"""
    if hasattr(g, 'keys'): # a dictionary
        return BufferDict(g)
    else: # array or scalar
        return np.array(g)

def _flat(g):
    """convert g to a flat array"""
    if hasattr(g, 'keys'): # a bufferdict
        return g.buf
    else: # an array
        return g.reshape(-1)

def _decompress(index, sdev):
    """Expand index, sdev into a list of pairs ([idx], [[sdev ** 2]])"""
    return [
        (np.array([idx]), np.array([[s ** 2]]))
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
