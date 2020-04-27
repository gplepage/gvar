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
    
    # Should I use lists? Can I type a list like list[np.ndarray[etc]]?
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
            # TODO should I avoid using np.copy? Will it call python code?
            rows_data[nbuf] = np.copy(row_data_buffer[:buflen])
            rows_indices[nbuf] = np.copy(row_indices_buffer[:buflen])
            nbuf += 1
        indptr[col + 1] = indptr[col] + buflen
    
    # concatenate buffers truncating to used buffers
    data = np.concatenate(rows_data[:nbuf]) if nbuf else np.empty(0)
    indices = np.concatenate(rows_indices[:nbuf]) if nbuf else np.empty(0, np.intp)
    
    return data, indices, indptr

cpdef list _extract_blocks(
    np.ndarray[INTP_TYPE, ndim=1] labels,
    np.ndarray[np.float_t, ndim=1] data,
    np.ndarray[INTP_TYPE, ndim=1] indices,
    np.ndarray[INTP_TYPE, ndim=1] indptr
):
    """Extract the connected blocks from the lower part of the covariance
    matrix, output format is like evalcov_blocks(*, compress=True)"""

    # count number of indices for each label
    cdef INTP_TYPE nlabels = 1 + np.max(labels)
    cdef np.ndarray[INTP_TYPE, ndim=1] length = np.zeros(nlabels, np.intp)
    cdef INTP_TYPE label
    for label in labels:
        length[label] += 1
    # assert np.all(length), 'np.all(length)'
    
    # count single-appearance labels
    cdef INTP_TYPE nones = 0
    cdef INTP_TYPE l
    for l in length:
        if l == 1:
            nones += 1
    
    # arrays for saving indices and sdev of solitary labels
    cdef np.ndarray[INTP_TYPE, ndim=1] ones_indices = np.empty(nones, np.intp)
    cdef np.ndarray[np.float_t, ndim=1] ones_sdev = np.empty(nones)
    cdef INTP_TYPE ones_end = 0
    
    # arrays of arrays for saving indices and covariance blocks
    # TODO use a single buffer with indptrs for covs and covs_indices
    cdef np.ndarray[object, ndim=1] covs
    cdef np.ndarray[object, ndim=1] covs_indices
    cdef np.ndarray[INTP_TYPE, ndim=1] covs_end, subindices
    if nlabels - nones:
        covs = np.empty(nlabels, object)
        covs_indices = np.empty(nlabels, object)
        covs_end = np.zeros(nlabels, np.intp)
        subindices = np.empty(len(labels), np.intp)

    # variables for cycle
    cdef INTP_TYPE row, start, end, cov_end, col, i, j
    cdef np.ndarray[np.float_t, ndim=2] cov
    cdef np.ndarray[np.float_t, ndim=1] rowdata
    cdef np.ndarray[INTP_TYPE, ndim=1] rowindices, cov_indices
    
    # fill arrays
    for row, label in enumerate(labels):
        
        # extract row from sparse matrix
        start = indptr[row]
        end = indptr[row + 1]
        rowdata = data[start:end]
        rowindices = indices[start:end]

        # extract sdev if label is solitary
        l = length[label]
        if l == 1:
            # assert end - start <= 1, 'end - start <= 1'
            # assert end - start == 0 or rowindices[0] == row, 'rowindices[0] == row'
            ones_indices[ones_end] = row
            ones_sdev[ones_end] = np.sqrt(rowdata[0]) if end - start else 0
            ones_end += 1
            
        # extract square block
        else:
            
            # get arrays for the label
            cov_end = covs_end[label]
            if cov_end == 0:
                covs[label] = np.zeros((l, l))
                covs_indices[label] = np.empty(l, np.intp)
            cov = covs[label]
            cov_indices = covs_indices[label]
            
            # save indices
            subindices[row] = cov_end
            cov_indices[cov_end] = row
            
            # # some checks
            # assert end - start <= cov_end + 1, 'end - start <= cov_end + 1'
            # for i in rowindices:
            #     assert i in cov_indices, 'i in cov_indices'

            # fill cov
            for i, col in enumerate(rowindices):
                # assert col <= row, 'col <= row'
                j = subindices[col]
                # assert j <= cov_end, 'j <= cov_end'
                cov[cov_end, j] = rowdata[i]
                cov[j, cov_end] = rowdata[i]
            covs_end[label] += 1
    
    # put everything in a list, removing empty arrays
    output = [(ones_indices, ones_sdev)]
    for label in range(nlabels):
        if length[label] > 1:
            # assert covs_end[label] == length[label], 'covs_end[label] == length[label]'
            output.append((covs_indices[label], covs[label]))
    
    return output

def _evalcov_blocks(np.ndarray[object, ndim=1] g):
    """
    Like evalcov_blocks with compress=True for 1D array input.
    """
    ldata, lindices, lindptr = _evalcov_sparse(g)
    uindices, uindptr = _transpose_csmatrix_indices((len(g), len(g)), lindices, lindptr)
    labels = np.zeros(len(g), np.intp)
    _connected_components_undirected(uindices, uindptr, lindices, lindptr, labels)
    return _extract_blocks(labels, ldata, lindices, lindptr)

def _flat(g):
    """convert g to a flat array"""
    if hasattr(g, 'keys'):
        # a dictionary
        if not isinstance(g, BufferDict):
            g = BufferDict(g)
        return g.buf
    else:
        # array or scalar
        return np.array(g, copy=False).reshape(-1)

def _decompress(index, sdev):
    """Expand index, sdev into a list of pairs ([idx], [[sdev ** 2]])"""
    return list(zip(index.reshape(-1, 1), sdev.reshape(-1, 1, 1)))

def evalcov_blocks(g, compress=False):
    """Faster version of gvar.evalcov_blocks"""
    g = _flat(g)
    idxs_covs = _evalcov_blocks(g)
    if not compress:
        idxs_covs = _decompress(*idxs_covs[0]) + idxs_covs[1:]
    return idxs_covs
