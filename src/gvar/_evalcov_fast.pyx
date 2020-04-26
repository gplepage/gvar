import numpy as np
cimport numpy as np
from numpy cimport npy_intp as INTP_TYPE

from ._svec_smat cimport svec, smat
from ._gvarcore cimport GVar

from ._bufferdict import BufferDict

from ._scipy_connected_components cimport _connected_components_undirected

cpdef tuple _transpose_csmatrix_indices(tuple shape,
                           # np.ndarray[np.float_t, ndim=1] data,
                           np.ndarray[INTP_TYPE, ndim=1] indices,
                           np.ndarray[INTP_TYPE, ndim=1] indptr):
    """transpose a CSR/CSC matrix
    copied and adapted from scipy/sparse/sparsetools/csr.h
    the original code is in comments
    returns indices, indptr of the transposed matrix"""
# void csr_tocsc(const I n_row,
#                const I n_col,
#                const I Ap[],
#                const I Aj[],
#                const T Ax[],
#                      I Bp[],
#                      I Bi[],
#                      T Bx[])
# {
    cdef INTP_TYPE n_row = shape[0]
    cdef INTP_TYPE n_col = shape[1]
    cdef np.ndarray[INTP_TYPE, ndim=1] Ap = indptr
    cdef np.ndarray[INTP_TYPE, ndim=1] Aj = indices
    # cdef np.ndarray[np.float_t, ndim=1] Ax = data
    cdef np.ndarray[INTP_TYPE, ndim=1] Bp = np.zeros(1 + n_col, np.intp)
    cdef np.ndarray[INTP_TYPE, ndim=1] Bi = np.empty_like(indices)
    # cdef np.ndarray[np.float_t, ndim=1] Bx = np.empty_like(data)

    # const I nnz = Ap[n_row];
    cdef INTP_TYPE nnz = Ap[n_row]

    # //compute number of non-zero entries per column of A
    # std::fill(Bp, Bp + n_col, 0);

    # for (I n = 0; n < nnz; n++){
    #     Bp[Aj[n]]++;
    # }
    cdef INTP_TYPE n
    for n in range(nnz):
        Bp[Aj[n]] += 1

    # //cumsum the nnz per column to get Bp[]
    # for(I col = 0, cumsum = 0; col < n_col; col++){
    #     I temp  = Bp[col];
    #     Bp[col] = cumsum;
    #     cumsum += temp;
    # }
    # Bp[n_col] = nnz;
    cdef INTP_TYPE col = 0
    cdef INTP_TYPE cumsum = 0
    cdef INTP_TYPE temp
    for col in range(n_col):
        temp = Bp[col]
        Bp[col] = cumsum
        cumsum += temp
    Bp[n_col] = nnz

    # for(I row = 0; row < n_row; row++){
    #     for(I jj = Ap[row]; jj < Ap[row+1]; jj++){
    #         I col  = Aj[jj];
    #         I dest = Bp[col];
    #
    #         Bi[dest] = row;
    #         Bx[dest] = Ax[jj];
    #
    #         Bp[col]++;
    #     }
    # }
    cdef INTP_TYPE row, jj, dest
    for row in range(n_row):
        for jj in range(Ap[row], Ap[row + 1]):
            col = Aj[jj]
            dest = Bp[col]
            Bi[dest] = row
            # Bx[dest] = Ax[jj]
            Bp[col] += 1

    # for(I col = 0, last = 0; col <= n_col; col++){
    #     I temp  = Bp[col];
    #     Bp[col] = last;
    #     last    = temp;
    # }
    cdef INTP_TYPE last = 0
    for col in range(n_col + 1):
        temp = Bp[col]
        Bp[col] = last
        last = temp
    
    return Bi, Bp
# }

cpdef tuple _evalcov_sparse(np.ndarray[object, ndim=1] g):
    """
    Compute the covariance matrix of g as a sparse CSR matrix.
    Returns data, indices, indptr, like scipy.sparse.csr_matrix.
    Only the lower triangular part is computed.
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
    cdef INTP_TYPE row, col
    cdef np.float_t c
    
    cdef np.ndarray[np.float_t, ndim=1] data
    cdef np.ndarray[INTP_TYPE, ndim=1] indices
    cdef np.ndarray[INTP_TYPE, ndim=1] indptr = np.empty(1 + ng, np.intp)
    indptr[0] = 0
    
    cdef np.ndarray[object, ndim=1] rows_data = np.empty(ng, object)
    cdef np.ndarray[object, ndim=1] rows_indices = np.empty(ng, object)

    cdef np.ndarray[np.float_t, ndim=1] row_data_buffer = np.empty(ng)
    cdef np.ndarray[INTP_TYPE, ndim=1] row_indices_buffer = np.empty(ng, np.intp)
    cdef INTP_TYPE buflen
    cdef INTP_TYPE nbuf = 0
    
    # compute the covariance matrix
    for row in range(ng):
        
        # compute (global covariance) @ transform[row, :].T
        ga = g[row]
        covd[row] = cov.masked_dot(ga.d, imask)
        
        buflen = 0
        for col in range(row + 1):
        
            # compute transform[row,:] @ (global covariance) @ transform[:,col]
            c = ga.d.dot(covd[col])
            if c != 0:
                row_data_buffer[buflen] = c
                row_indices_buffer[buflen] = col
                buflen += 1
            
        if buflen > 0:
            rows_data[nbuf] = np.copy(row_data_buffer[:buflen])
            rows_indices[nbuf] = np.copy(row_indices_buffer[:buflen])
            nbuf += 1
        indptr[col + 1] = indptr[col] + buflen
    
    # concatenate buffers truncating to used buffers
    data = np.concatenate(rows_data[:nbuf])
    indices = np.concatenate(rows_indices[:nbuf])
    
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
    """Extract the submatrix from a CSR matrix for indices `outindices`. The
    matrix must be the triangular part of a symmetric matrix."""

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
    ldata, lindices, lindptr = _evalcov_sparse(g)
    uindices, uindptr = _transpose_csmatrix_indices((len(g), len(g)), lindices, lindptr)
    labels = np.zeros(len(g), np.intp)
    n = _connected_components_undirected(uindices, uindptr, lindices, lindptr, labels)
    indices_list = _compress_labels(labels)
    covs = [_sub_sdev(indices_list[0], ldata, lindices, lindptr)]
    for idxs in indices_list[1:]:
        covs.append(_sub_cov(idxs, ldata, lindices, lindptr))
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
