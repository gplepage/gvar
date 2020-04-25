import numpy as np
cimport numpy as np
from np cimport npy_intp as INTP_TYPE

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

cdef (
    np.ndarray[np.float_t, ndim=1],
    np.ndarray[np.intp, ndim=1],
    np.ndarray[np.intp, ndim=1]
)
_evalcov_sparse(np.ndarray[object, ndim=1] g):
    """
    Return the covariance matrix of g as a sparse CSR matrix.
    Returned values: data, indices, indptr, like scipy.sparse.csr_matrix.
    """
    
    cdef np.ndarray[np.ndarray[np.float_t, ndim=1], ndim=1] rows_data = np.empty(len(g), object)
    cdef np.ndarray[np.ndarray[np.intp, ndim=1], ndim=1] rows_indices = np.empty(len(g), object)
    cdef np.ndarray[np.intp, ndim=1] indptr = np.empty(len(g) + 1, np.intp)
    
    cdef np.ndarray[np.float_t, ndim=1] row_buffer = np.empty(len(g))
    cdef np.ndarray[np.intp, ndim=1] indices_buffer = np.empty(len(g), np.intp)
    
    indptr[0] = 0
    for i in range(g.size):
        cdef INTP_TYPE buflen = 0
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
    indices = np.concatenate(indices_data)
    return data, indices, indptr

def _evalcov_blocks(np.ndarray[object, ndim=1] g):
    """
    Like evalcov_blocks, for 1D array input.
    """
    data, indices, indptr = _evalcov_sparse(g)
    labels = np.empty(len(g), dtype=ITYPE)
    labels.fill(NULL_IDX)
    n = _connected_components_directed(indices, indptr, labels)
    
    # TODO cdef _compress_labels
    