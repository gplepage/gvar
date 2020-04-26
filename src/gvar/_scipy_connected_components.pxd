cimport numpy as np
cimport cython
from numpy cimport npy_intp as ITYPE_t

cdef int _connected_components_directed(
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] labels)
