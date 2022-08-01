# cython: language_level=3str, binding=True
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

cimport numpy

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

cdef packed struct svec_element:
    double v
    INTP_TYPE i

cdef class svec:
    cdef svec_element * v
    cdef readonly int size
    cpdef numpy.ndarray[numpy.float_t, ndim=1] toarray(svec,INTP_TYPE msize=?)
    cpdef numpy.ndarray[INTP_TYPE, ndim=1] indices(svec)
    cpdef numpy.ndarray[numpy.float_t, ndim=1] values(svec)
    cpdef _assign(self, const numpy.float_t[:], const INTP_TYPE[:])
    cpdef double dot(svec, svec)
    cpdef svec clone(svec)
    cpdef svec add(svec,svec,double a=*,double b=*)
    cpdef svec mul(svec self,double a)
    cpdef numpy.ndarray[numpy.float_t, ndim=1] masked_vec(svec, smask, out=*)

cdef class smask:
    cdef readonly numpy.int8_t[::1] mask
    cdef readonly INTP_TYPE[::1] map 
    cdef readonly INTP_TYPE starti, stopi, len
 
cdef class smat:
    cdef object[::1] row
    # cdef svec[::1] row
    cdef INTP_TYPE[::1] block
    cdef INTP_TYPE nrow, nrow_max, next_block
    cpdef _add_memory(smat self)
    cpdef INTP_TYPE blockid(smat self, INTP_TYPE i)
    cpdef numpy.ndarray[INTP_TYPE, ndim=1] append_diag(self, const numpy.float_t[:])
    cpdef numpy.ndarray[INTP_TYPE, ndim=1] append_diag_m(self, const numpy.float_t[:,:])
    cpdef add_offdiag_m(self, const numpy.npy_intp[:], const numpy.npy_intp[:], const numpy.float_t[:, :])
    cpdef svec dot(self,svec)
    cpdef svec masked_dot(self, svec vv, const numpy.int8_t[:] imask)
    cpdef double expval(self,svec)
    cpdef numpy.ndarray[numpy.float_t, ndim=2] toarray(self)
    cpdef numpy.ndarray[numpy.float_t, ndim=2] masked_mat(smat self, smask mask, out=*)
