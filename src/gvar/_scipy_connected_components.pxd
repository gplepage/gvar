cimport numpy as np
from numpy cimport npy_intp as ITYPE_t

cdef int _connected_components_directed(
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indices,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr,
                                 np.ndarray[ITYPE_t, ndim=1, mode='c'] labels)

cdef int _connected_components_undirected(
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] indices1,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr1,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] indices2,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr2,
                        np.ndarray[ITYPE_t, ndim=1, mode='c'] labels)
