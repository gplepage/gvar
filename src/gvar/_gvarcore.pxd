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

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

from ._svec_smat cimport svec, smat

cdef class GVar:
    cdef double v
    cdef svec d
    cdef readonly smat cov
    cpdef GVar clone(self)
    cpdef bint is_primary(self)
