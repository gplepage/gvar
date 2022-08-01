# cython: boundscheck=False, intializedcheck=False, wraparound=False, language_level=3str, binding=True 
# Created by Peter Lepage (Cornell University) on 2012-05-31.
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

import numpy

cimport numpy
cimport cython

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

# from libc.stdlib cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free #, sizeof
from libc.string cimport memset

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

cdef class svec:
    """ sparse vector --- for GVar derivatives (only)"""
    # cdef svec_element * v
    # cdef readonly usigned int size ## number of elements in v

    def __cinit__(svec self, INTP_TYPE size, *arg, **karg):
        self.v = <svec_element *> PyMem_Malloc(size * sizeof(self.v[0]))
        memset(self.v, 0, size * sizeof(self.v[0]))
        self.size = size

    def __dealloc__(self):
        PyMem_Free(<void *> self.v)

    def __getstate__(self):
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx = numpy.empty(self.size, numpy.intp)
        cdef numpy.ndarray[numpy.float_t, ndim=1] val = numpy.empty(self.size, numpy.float_)
        cdef INTP_TYPE i
        for i in range(self.size):
            idx[i] = self.v[i].i
            val[i] = self.v[i].v
        return (val, idx)

    def __setstate__(self, data):
        cdef INTP_TYPE i
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx
        cdef numpy.ndarray[numpy.float_t, ndim=1] val
        val, idx = data
        for i in range(self.size):
            self.v[i].v = val[i]
            self.v[i].i = idx[i]

    def __reduce_ex__(self, dummy):
        return (svec, (self.size,), self.__getstate__())

    def __len__(self):
        """ """
        cdef INTP_TYPE i
        if self.size==0:
            return 0
        else:
            return max([self.v[i].i for i in range(self.size)])

    cpdef svec clone(self):
        cdef svec ans
        cdef INTP_TYPE i
        ans = svec(self.size)
        for i in range(self.size):
            ans.v[i].v = self.v[i].v
            ans.v[i].i = self.v[i].i
        return ans

    cpdef numpy.ndarray[INTP_TYPE, ndim=1] indices(self):
        cdef INTP_TYPE i
        cdef numpy.ndarray [INTP_TYPE, ndim=1] ans
        ans = numpy.zeros(self.size, numpy.intp)
        for i in range(self.size):
            ans[i] = self.v[i].i
        return ans

    cpdef numpy.ndarray[numpy.float_t, ndim=1] values(self):
        cdef INTP_TYPE i
        cdef numpy.ndarray [numpy.float_t, ndim=1] ans
        ans = numpy.zeros(self.size, float)
        for i in range(self.size):
            ans[i] = self.v[i].v
        return ans    

    cpdef numpy.ndarray[numpy.float_t,ndim=1] toarray(self, INTP_TYPE msize=0):
        """ Create numpy.array version of self, padded with zeros to length
        msize if msize is not None and larger than the actual size.
        """
        cdef INTP_TYPE i,nsize
        cdef numpy.ndarray[numpy.float_t, ndim=1] ans
        if self.size==0:
            return numpy.zeros(msize, numpy.float_)
        nsize = max(self.v[self.size-1].i + 1, msize)
        ans = numpy.zeros(nsize, numpy.float_)
        for i in range(self.size):
            ans[self.v[i].i] = self.v[i].v
        return ans

    cpdef _assign(self, const numpy.float_t[:] v, const INTP_TYPE[:] idx):
        """ Assign v and idx to self.v[i].v and self.v[i].i.

        Assumes that len(v)==len(idx)==self.size and idx sorted
        """
        cdef INTP_TYPE i, j
        j = 0
        for i in range(self.size):
            # only keep non-zero items
            if v[i] != 0.0:
                self.v[j].v = v[i]
                self.v[j].i = idx[i]
                j = j + 1
        if j < self.size:
            self.size = j
            self.v = <svec_element *> PyMem_Realloc(
                <void*> self.v, self.size * sizeof(self.v[0])
                )

    def assign(self, v, idx):
        """ assign v and idx to self.v[i].v and self.v[i].i """
        cdef INTP_TYPE nv, i, j
        nv = len(v)
        assert nv==len(idx) and nv==self.size,"v,idx length mismatch"
        if nv>0:
            idx,v = zip(*sorted(zip(idx,v)))
            j = 0
            for i in range(self.size):
                if v[i] != 0.0:
                    self.v[i].v = v[i]
                    self.v[i].i = idx[i]
                    j = j + 1
            if j < self.size:
                self.size = j
                self.v = <svec_element *> PyMem_Realloc(
                    <void*> self.v, self.size * sizeof(self.v[0])
                    )

    cpdef double dot(svec self, svec v):
        """ Compute dot product of self and v: <self|v> """
        cdef svec va,vb
        cdef INTP_TYPE ia,ib
        cdef double ans
        va = self
        vb = v
        ia = 0
        ib = 0
        ans = 0.0
        if va.size==0 or vb.size==0:
            return 0.0
        if va.v[va.size-1].i<vb.v[0].i or vb.v[vb.size-1].i<va.v[0].i:
            return ans
        while ia<va.size and ib<vb.size:
            if va.v[ia].i==vb.v[ib].i:
                ans += va.v[ia].v*vb.v[ib].v
                ia += 1
                ib += 1
            elif va.v[ia].i<vb.v[ib].i:
                ia += 1
            else:
                ib += 1
        return ans

    cpdef svec add(svec self, svec v, double a=1., double b=1.):
        """ Compute a*self + b*v. """
        cdef svec va, vb
        cdef INTP_TYPE ia, ib, i, ians
        cdef svec ans
        va = self
        vb = v
        if va.size == 0 or a == 0:
            return vb.mul(b)
        elif vb.size == 0 or b == 0:
            return va.mul(a)
        ans = svec.__new__(svec, va.size + vb.size) # svec(va.size+vb.size)     # could be too big
        ia = 0
        ib = 0
        ians = 0
        while ia<va.size or ib<vb.size:
            if va.v[ia].i==vb.v[ib].i:
                ans.v[ians].i = va.v[ia].i
                ans.v[ians].v = a * va.v[ia].v + b * vb.v[ib].v
                ians += 1
                ia += 1
                ib += 1
                if ia>=va.size:
                    while ib<vb.size:
                        ans.v[ians].i = vb.v[ib].i
                        ans.v[ians].v = b*vb.v[ib].v
                        ib += 1
                        ians += 1
                    break
                elif ib>=vb.size:
                    while ia<va.size:
                        ans.v[ians].i = va.v[ia].i
                        ans.v[ians].v = a*va.v[ia].v
                        ia += 1
                        ians += 1
                    break
            elif va.v[ia].i<vb.v[ib].i:
                ans.v[ians].i = va.v[ia].i
                ans.v[ians].v = a*va.v[ia].v
                ians += 1
                ia += 1
                if ia>=va.size:
                    while ib<vb.size:
                        ans.v[ians].i = vb.v[ib].i
                        ans.v[ians].v = b*vb.v[ib].v
                        ib += 1
                        ians += 1
                    break
            else:
                ans.v[ians].i = vb.v[ib].i
                ans.v[ians].v = b*vb.v[ib].v
                ians += 1
                ib += 1
                if ib>=vb.size:
                    while ia<va.size:
                        ans.v[ians].i = va.v[ia].i
                        ans.v[ians].v = a*va.v[ia].v
                        ia += 1
                        ians += 1
                    break
        ans.size = ians
        ans.v = <svec_element *> PyMem_Realloc(<void*> ans.v,
                                         ans.size*sizeof(self.v[0]))
        return ans

    cpdef svec mul(svec self, double a):
        """ Compute a*self. """
        cdef INTP_TYPE i
        if a == 0:
            return svec.__new__(svec, 0)
        cdef svec ans = svec.__new__(svec, self.size) # svec(self.size)
        for i in range(self.size):
            ans.v[i].i = self.v[i].i
            ans.v[i].v = a * self.v[i].v
        return ans
    
    @cython.initializedcheck(False)
    cpdef numpy.ndarray[numpy.float_t, ndim=1] masked_vec(svec self, smask mask, out=None):
        """ Returns compact vector containing the unmasked components of the svec. 
        
        N.B. If use ``out`` make sure it is zeroed first.
        """
        cdef INTP_TYPE i
        cdef numpy.ndarray[numpy.float_t, ndim=1] ans
        if out is None:
            ans = numpy.zeros(mask.len, dtype=float)
        else:
            ans = out
        for i in range(self.size):
            if mask.mask[self.v[i].i]:
                ans[mask.map[self.v[i].i]] = self.v[i].v
        return ans

cdef class smask:
    " mask for smat, svec "

    def __cinit__(smask self, numpy.int8_t[::1] mask):
        cdef INTP_TYPE i 
        cdef numpy.int8_t ib 
        self.mask = mask
        self.map = numpy.zeros(len(self.mask), dtype=numpy.intp)
        self.starti = -1
        self.stopi = -1
        i = 0
        self.len = 0
        for ib in mask:
            if ib:
                self.map[i] = self.len
                self.len += 1
                if self.starti == -1:
                    self.starti = i
                self.stopi = i
            i += 1
        self.stopi += 1
    
    def __len__(smask self):
        return self.len

cdef class smat:
    """ sym. sparse matrix --- for GVar covariance matrices (only) """
    # cdef object rowlist

    def __cinit__(smat self):
        self.row = numpy.empty(2000, object)
        self.block = numpy.empty(2000, numpy.intp)
        self.nrow = 0
        self.next_block = 0
        self.nrow_max = 2000

    def __reduce_ex__(self, dummy):
        return (smat, (), self.__getstate__())

    def __getstate__(self):
        return (self.nrow, self.nrow_max, numpy.asarray(self.row))

    def __setstate__(self, data):
        self.nrow, self.nrow_max, self.row = data

    def __len__(self):
        """ Dimension of matrix. """
        return self.nrow # len(self.rowlist)

    cpdef INTP_TYPE blockid(smat self, INTP_TYPE i):
        return self.block[i]
    
    cpdef _add_memory(smat self):
        cdef object[:] oldrow = self.row
        cdef INTP_TYPE[::1] oldblock = self.block 
        cdef INTP_TYPE i
        self.row = numpy.empty(2 * self.nrow_max, object)
        self.block = numpy.empty(2 * self.nrow_max, numpy.intp)
        for i in range(self.nrow_max):
            self.row[i] = oldrow[i]
            self.block[i] = oldblock[i]
        self.nrow_max = 2 * self.nrow_max
        # print('**** added memory')

    cpdef numpy.ndarray[INTP_TYPE,ndim=1] append_diag(self, const numpy.float_t[:] d):
        """ Add d[i] along diagonal. """
        cdef INTP_TYPE i, nr
        cdef numpy.ndarray[numpy.float_t, ndim=1] v
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx, vrange
        cdef svec new_svec
        idx = numpy.zeros(1, numpy.intp)
        nr = self.nrow # len(self.rowlist)
        v = numpy.zeros(1, numpy.float_)
        vrange = numpy.arange(nr, nr+d.shape[0], dtype=numpy.intp)
        for i in range(d.shape[0]):
            v[0] = d[i]
            idx[0] = self.nrow # len(self.rowlist)
            # self.rowlist.append(svec(1))
            # self.rowlist[-1]._assign(v,idx)
            if self.nrow >= self.nrow_max:
                self._add_memory()
            new_svec = svec(1)
            new_svec._assign(v, idx)
            self.row[self.nrow] = new_svec
            self.block[self.nrow] = self.next_block
            self.next_block += 1
            self.nrow += 1
        return vrange
        
    cpdef numpy.ndarray[INTP_TYPE,ndim=1] append_diag_m(self, const numpy.float_t[:, :] m):
        cdef INTP_TYPE i, j, nr, nm, n_nonzero
        cdef numpy.ndarray[numpy.float_t, ndim=1] v
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx,vrange
        cdef svec new_svec
        assert m.shape[0]==m.shape[1], "m must be square matrix"
        nm = m.shape[0]
        idx = numpy.zeros(nm, numpy.intp)
        v = numpy.zeros(nm, numpy.float_)
        nr = self.nrow # len(self.rowlist)
        vrange = numpy.arange(nr, nr + nm, dtype=numpy.intp)
        for i in range(nm):
            n_nonzero = 0
            for j in range(nm):
                # only keep non-zero elements + diagonal elements
                if m[i, j] != 0.0 or i == j:
                    v[n_nonzero] = m[i, j]
                    idx[n_nonzero] = j + nr
                    n_nonzero += 1
            if n_nonzero == 0:
                continue
            if self.nrow >= self.nrow_max:
                self._add_memory()
            new_svec = svec(n_nonzero) # nm)
            new_svec._assign(v[:n_nonzero], idx[:n_nonzero])
            self.row[self.nrow] = new_svec
            self.block[self.nrow] = self.next_block
            self.nrow += 1
        self.next_block += 1
        return vrange

    cpdef add_offdiag_m(self, const numpy.npy_intp[:] xrow, const numpy.npy_intp[:] yrow, const numpy.float_t[:, :] xym):
        cdef INTP_TYPE i, j, k
        cdef svec x, y, newx, newy
        try:
            assert xym.shape[0] == xrow.shape[0] and xym.shape[1] == yrow.shape[0]
        except:
            raise ValueError('m.shape != ', str((xrow.shape[0], yrow.shape[0])))
        # fix block ids
        blockid = self.nrow
        idset = set()
        for i in xrow:
            if self.block[i] < blockid:
                blockid = self.block[i]
            idset.add(self.block[i])
        for i in range(self.nrow):
            if self.block[i] in idset:
                self.block[i] = blockid 
        for i in yrow:
            self.block[i] = blockid
        for i in range(len(xrow)):
            x = self.row[xrow[i]]
            newx = svec(x.size + len(yrow))
            k = 0
            for j in range(x.size):
                newx.v[k].i = x.v[j].i
                newx.v[k].v = x.v[j].v
                k += 1
            for j in range(len(yrow)):
                newx.v[k].i = yrow[j]
                newx.v[k].v = xym[i, j]
                k += 1
            self.row[xrow[i]] = newx
        for i in range(len(yrow)):
            y = self.row[yrow[i]]
            newy = svec(y.size + len(xrow))
            k = 0
            for j in range(len(xrow)):
                newy.v[k].i = xrow[j]
                newy.v[k].v = xym[j, i]
                k += 1
            for j in range(y.size):
                newy.v[k].i = y.v[j].i
                newy.v[k].v = y.v[j].v
                k += 1
            self.row[yrow[i]] = newy
            




    cpdef double expval(self, svec vv):
        """ Compute expectation value <vv|self|vv>. """
        cdef INTP_TYPE i
        cdef svec row
        cdef double ans
        ans = 0.0
        for i in range(vv.size):
            row = self.row[vv.v[i].i] # self.rowlist[vv.v[i].i]
            ans += row.dot(vv) * vv.v[i].v
        return ans

    cpdef svec dot(self,svec vv):
        """ Compute dot product self|vv>. """
        cdef numpy.ndarray[numpy.float_t,ndim=1] v
        cdef numpy.ndarray[INTP_TYPE,ndim=1] idx
        cdef double rowv
        cdef INTP_TYPE nr, size, i
        cdef svec row
        nr = self.nrow # len(self.rowlist)
        v = numpy.zeros(nr, numpy.float_)
        idx = numpy.zeros(nr, numpy.intp)
        size = 0
        for i in range(nr):
            row = self.row[i] # self.rowlist[i]
            rowv = row.dot(vv)
            if rowv!=0.0:
                idx[size] = i
                v[size] = rowv
                size += 1
        ans = svec(size)
        for i in range(size):
            ans.v[i].v = v[i]
            ans.v[i].i = idx[i]
        return ans

    cpdef svec masked_dot(self, svec vv, const numpy.int8_t[:] imask):
        """ Compute masked dot product self|vv>.

        imask indicates which components to compute and keep in final result;
        disregard components i where imask[i]==False.
        """
        cdef numpy.ndarray[numpy.float_t,ndim=1] v
        cdef numpy.ndarray[INTP_TYPE,ndim=1] idx
        cdef INTP_TYPE nr, size, i
        cdef double rowv
        cdef svec row
        cdef svec ans
        nr = self.nrow # len(self.rowlist)
        v = numpy.zeros(nr,numpy.float_)
        idx = numpy.zeros(nr,numpy.intp)
        size = 0
        for i in range(nr):
            if not imask[i]:
                continue
            row = self.row[i] # self.rowlist[i]
            rowv = row.dot(vv)
            if rowv!=0.0:
                idx[size] = i
                v[size] = rowv
                size += 1
        ans = svec(size)
        for i in range(size):
            ans.v[i].v = v[i]
            ans.v[i].i = idx[i]
        return ans

    cpdef numpy.ndarray[numpy.float_t,ndim=2] toarray(self):
        """ Create numpy ndim=2 array version of self. """
        cdef numpy.ndarray[numpy.float_t,ndim=2] ans
        cdef INTP_TYPE nr = self.nrow # len(self.rowlist)
        cdef INTP_TYPE i
        ans = numpy.zeros((nr,nr),numpy.float_)
        for i in range(nr):
            row = self.row[i].toarray() # self.rowlist[i].toarray()
            ans[i][:len(row)] = row
        return ans

    @cython.initializedcheck(False)
    cpdef numpy.ndarray[numpy.float_t, ndim=2] masked_mat(smat self, smask mask, out=None):
        """ Returns compact matrix containing the unmasked components of the smat. 
        
        N.B. If use ``out`` make sure it is zeroed first.
        """
        cdef INTP_TYPE i
        cdef numpy.ndarray[numpy.float_t, ndim=2] ans 
        cdef svec rowi
        if out is None:
            ans = numpy.zeros((mask.len, mask.len), dtype=float)
        else:
            ans = out
        for i in range(mask.starti, mask.stopi):
            if mask.mask[i]:
                rowi = self.row[i]
                # ans[mask.map[i], :] = rowi.masked_vec(mask)
                rowi.masked_vec(mask, out=ans[mask.map[i],:])
        return ans
