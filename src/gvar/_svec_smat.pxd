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

cdef packed struct svec_element:
    double v
    Py_ssize_t i

cdef class svec:
    cdef svec_element * v
    cdef readonly int size
    cpdef object toarray(svec,Py_ssize_t msize=?)
    cpdef object indices(svec)
    cpdef object values(svec)
    cpdef _assign(self, const double[:], const Py_ssize_t[:])
    cpdef double dot(svec, svec)
    cpdef svec clone(svec)
    cpdef svec add(svec, svec, double a=*, double b=*)
    cpdef svec mul(svec self, double a)
    cpdef object masked_vec(svec, smask, out=*)

cdef class smask:
    cdef readonly char[::1] mask
    cdef readonly Py_ssize_t[::1] map 
    cdef readonly Py_ssize_t starti, stopi, len
 
cdef class smat:
    cdef object[::1] row
    # cdef svec[::1] row
    cdef Py_ssize_t[::1] block
    cdef Py_ssize_t nrow, nrow_max, next_block
    cpdef _add_memory(smat self)
    cpdef Py_ssize_t blockid(smat self, Py_ssize_t i)
    cpdef object append_diag(self, const double[:])
    cpdef object append_diag_m(self, const double[:,:])
    cpdef add_offdiag_m(self, const Py_ssize_t[:], const Py_ssize_t[:], const double[:, :])
    cpdef svec dot(self,svec)
    cpdef svec masked_dot(self, svec vv, const char[:] imask)
    cpdef double expval(self,svec)
    cpdef object toarray(self)
    cpdef object masked_mat(smat self, smask mask, out=*)
