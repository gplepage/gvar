# cython: boundscheck=False, language_level=3str, binding=True
# cython: c_api_binop_methods=True
# c#ython: profile=True
# remove extra # above for profiling

# Created by Peter Lepage (Cornell University) on 2011-08-17.
# Copyright (c) 2011-2021 G. Peter Lepage.
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

import re
from scipy.sparse.csgraph import connected_components as _connected_components
from gvar._svec_smat import svec, smat
from gvar._bufferdict import BufferDict

cimport numpy
cimport cython
from ._svec_smat cimport svec, smat


cdef extern from "math.h":
    double c_pow "pow" (double x,double y)
    double c_sin "sin" (double x)
    double c_cos "cos" (double x)
    double c_tan "tan" (double x)
    double c_sinh "sinh" (double x)
    double c_cosh "cosh" (double x)
    double c_tanh "tanh" (double x)
    double c_log "log" (double x)
    double c_exp "exp" (double x)
    double c_sqrt "sqrt" (double x)
    double c_asin "asin" (double x)
    double c_acos "acos" (double x)
    double c_atan "atan" (double x)


import numpy
from numpy import sin, cos, tan, exp, log, sqrt, fabs
from numpy import sinh, cosh, tanh, arcsin, arccos, arctan, arctan2
from numpy import arcsinh, arccosh, arctanh, square
import copy
import warnings

import gvar.powerseries
_ARRAY_TYPES = [numpy.ndarray, gvar.powerseries.PowerSeries]

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

# GVar
cdef class GVar:
    # cdef double v     -- value or mean
    # cdef svec d       -- vector of derivatives
    # cdef readonly smat cov    -- covariance matrix

    def __init__(self,double v,svec d,smat cov):
        self.v = v
        self.d = d
        self.cov = cov

    cpdef GVar clone(self):
        return GVar(self.v,self.d,self.cov)

    def __getstate__(self):
        warnings.warn('Pickling GVars with pickle.dump/load loses correlations; use gvar.dump/load to preserve them.')
        return (self.mean, self.sdev)

    def __setstate__(self, state):
        from gvar import gvar
        if len(state) == 3:
            # legacy code (bad idea to use it, however)
            self.cov, self.d, self.v = state
        else:
            self.v, self.d, self.cov = gvar(*state).internaldata

    def __deepcopy__(self, *args):
        return self

    def __copy__(self):
        return self

    def __format__(self, fmt=None):
        """ Convert to string and format the string """
        return format(str(self), fmt)

    def __str__(self):
        """ Return string representation of ``self``.

        The representation is designed to show at least
        one digit of the mean and two digits of the standard deviation.
        For cases where mean and standard deviation are not
        too different in magnitude, the representation is of the
        form ``'mean(sdev)'``. When this is not possible, the string
        has the form ``'mean +- sdev'``.
        """
        def ndec(x, offset=2):
            ans = offset - numpy.log10(x)
            ans = int(ans)
            if ans > 0 and x * 10. ** ans >= [0.5, 9.5, 99.5][offset]:
                ans -= 1
            return 0 if ans < 0 else ans
        dv = abs(self.sdev)
        v = self.mean

        # special cases
        if numpy.isnan(v) or numpy.isnan(dv):
            return '%g +- %g'%(v, dv)
        elif dv == float('inf'):
            return '%g +- inf' % v
        elif v == 0 and (dv >= 1e5 or dv < 1e-4):
            if dv == 0:
                return '0(0)'
            else:
                ans = ("%.1e" % dv).split('e')
                return "0.0(" + ans[0] + ")e" + ans[1]
        elif v == 0:
            if dv >= 9.95:
                return '0(%.0f)' % dv
            elif dv >= 0.995:
                return '0.0(%.1f)' % dv
            else:
                ndecimal = ndec(dv)
                return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)
        elif dv == 0:
            ans = ('%g' % v).split('e')
            if len(ans) == 2:
                return ans[0] + "(0)e" + ans[1]
            else:
                return ans[0] + "(0)"
        # elif dv < 1e-6 * abs(v):
        #     return '%g +- %.2g' % (v, dv)
        elif dv > 1e4 * abs(v):
            return '%.1g +- %.2g' % (v, dv)
        elif abs(v) >= 1e6 or abs(v) < 1e-5:
            # exponential notation for large |self.mean|
            exponent = numpy.floor(numpy.log10(abs(v)))
            fac = 10.**exponent
            mantissa = str(self/fac)
            exponent = "e" + ("%.0e" % fac).split("e")[-1]
            return mantissa + exponent

        # normal cases
        if dv >= 9.95:
            if abs(v) >= 9.5:
                return '%.0f(%.0f)' % (v, dv)
            else:
                ndecimal = ndec(abs(v), offset=1)
                return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv)
        if dv >= 0.995:
            if abs(v) >= 0.95:
                return '%.1f(%.1f)' % (v, dv)
            else:
                ndecimal = ndec(abs(v), offset=1)
                return '%.*f(%.*f)' % (ndecimal, v, ndecimal, dv)
        else:
            ndecimal = max(ndec(abs(v), offset=1), ndec(dv))
            return '%.*f(%.0f)' % (ndecimal, v, dv * 10. ** ndecimal)

    def __repr__(self):
        """ Same as ``str(self)``. """
        return self.__str__()

    # def __hash__(self):  # conflicts with equality (unless make it hash(self.mean) -- dumb)
    #     return id(self)

    def __richcmp__(x, y, op):
        """ Compare mean values. """
        xx = x.mean if isinstance(x, GVar) else x
        yy = y.mean if isinstance(y, GVar) else y
        if op == 0:
            return xx < yy
        elif op == 2:
            return xx == yy
        elif op == 3:
            return xx != yy
        elif op == 4:
            return xx > yy
        elif op == 1:
            return xx <= yy
        elif op == 5:
            return xx >= yy
        else:
            raise TypeError("undefined comparison for GVars")

    def __call__(self):
        """ Generate random number from ``self``'s distribution."""
        return numpy.random.normal(self.mean,self.sdev)

    def __neg__(self):
        return GVar(-self.v,self.d.mul(-1.),self.cov)

    def __pos__(self):
        return self

    def __add__(xx,yy):
        cdef GVar ans = GVar.__new__(GVar)
        cdef GVar x,y
        # cdef INTP_TYPE i,nx,di,ny
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                ans.v = x.v + y.v
                ans.d = x.d.add(y.d)
                ans.cov = x.cov
                # return GVar(x.v+y.v,x.d.add(y.d),x.cov)
            else:
                x = xx
                ans.v = x.v + yy
                ans.d = x.d
                ans.cov = x.cov
                # return GVar(x.v+yy,x.d,x.cov)
        elif isinstance(yy,GVar):
            y = yy
            ans.v = y.v + xx
            ans.d = y.d
            ans.cov = y.cov
            # return GVar(y.v+xx,y.d,y.cov)
        else:
            return NotImplemented
        return ans

    def __sub__(xx,yy):
        cdef GVar x,y
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                return GVar(x.v-y.v,x.d.add(y.d,1.,-1.),x.cov)
            else:
                x = xx
                return GVar(x.v-yy,x.d,x.cov)
        elif isinstance(yy,GVar):
            y = yy
            return GVar(xx-y.v,y.d.mul(-1.),y.cov)
        else:
            return NotImplemented

    def __mul__(xx,yy):
        cdef GVar x,y

        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                return GVar(x.v * y.v, x.d.add(y.d, y.v, x.v), x.cov)
            else:
                x = xx
                return GVar(x.v*yy, x.d.mul(yy), x.cov)
        elif isinstance(yy,GVar):
            y = yy
            return GVar(xx*y.v, y.d.mul(xx), y.cov)
        else:
            return NotImplemented

    # truediv and div are the same --- 1st is for python3, 2nd for python2
    def __truediv__(xx,yy):
        cdef GVar x,y
        cdef double xd,yd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                return GVar(x.v/y.v,x.d.add(y.d,1./y.v,-x.v/y.v**2),x.cov)
            else:
                x = xx
                yd=yy
                return GVar(x.v/yd,x.d.mul(1./yd),x.cov)
        elif isinstance(yy,GVar):
            y = yy
            xd=xx
            return GVar(xd/y.v,y.d.mul(-xd/y.v**2),y.cov)
        else:
            return NotImplemented

    def __div__(xx, yy):
        cdef GVar x, y
        cdef double xd, yd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx, GVar):
            if isinstance(yy, GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                return GVar(
                    x.v / y.v,
                    x.d.add(y.d, 1. / y.v, -x.v / y.v**2),
                    x.cov,
                    )
            else:
                x = xx
                yd=yy
                return GVar(x.v / yd, x.d.mul(1. / yd), x.cov)
        elif isinstance(yy, GVar):
            y = yy
            xd=xx
            return GVar(xd / y.v, y.d.mul(-xd / y.v **2), y.cov)
        else:
            return NotImplemented

    def __pow__(xx, yy, zz):
        cdef GVar x, y
        cdef double ans, f1, f2, yd, xd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx, GVar):
            if isinstance(yy, GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"incompatible GVars"
                ans = c_pow(x.v, y.v)
                f1 = c_pow(x.v,y.v-1)*y.v
                f2 = ans*c_log(x.v)
                return GVar(ans, x.d.add(y.d, f1, f2), x.cov)
            else:
                x = xx
                yd= yy
                ans = c_pow(x.v,yd)
                f1 = c_pow(x.v,yd-1)*yy
                return GVar(ans, x.d.mul(f1), x.cov)
        elif isinstance(yy, GVar):
            y = yy
            xd= xx
            ans = c_pow(xd, y.v)
            f1 = ans*c_log(xd)
            return GVar(ans, y.d.mul(f1), y.cov)
        else:
            return NotImplemented

    def sin(self):
        return GVar(c_sin(self.v), self.d.mul(c_cos(self.v)), self.cov)

    # def sin(self):
    #     cdef GVar ans = GVar.__new__(GVar)
    #     ans.v = c_sin(self.v)
    #     ans.d = self.d.mul(c_cos(self.v))
    #     ans.cov = self.cov
    #     return ans

    def cos(self):
        return GVar(c_cos(self.v), self.d.mul(-c_sin(self.v)), self.cov)

    def tan(self):
        cdef double ans = c_tan(self.v)
        return GVar(ans, self.d.mul(1 + ans * ans), self.cov)

    def arcsin(self):
        return GVar(
            c_asin(self.v),
            self.d.mul(1. / (1. - self.v ** 2) ** 0.5),
            self.cov
            )

    def asin(self):
        return self.arcsin()

    def arccos(self):
        return GVar(
            c_acos(self.v),
            self.d.mul(-1. / (1. - self.v**2) ** 0.5),
            self.cov,
            )

    def acos(self):
        return self.arccos()

    def arctan(self):
        return GVar(c_atan(self.v), self.d.mul(1. / (1. + self.v ** 2)), self.cov)

    def atan(self):
        return self.arctan()

    def arctan2(yy, xx):
        # following code by Matt Wingate 10/2014
        # returns angle in (-pi, pi]
        if type(xx) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        if xx > 0.0:
            return numpy.arctan(yy / xx)
        elif xx < 0.0:
            return (
                numpy.arctan(yy / xx) + numpy.pi
                if yy >= 0 else
                numpy.arctan(yy / xx) - numpy.pi
                )
        else:
            return (
                numpy.pi / 2 - numpy.arctan(xx / yy)
                if yy >= 0 else
                - numpy.pi / 2 - numpy.arctan(xx / yy)
                )

    def atan2(yy, xx):
        return yy.arctan2(xx)

    def sinh(self):
        return GVar(c_sinh(self.v), self.d.mul(c_cosh(self.v)), self.cov)

    def cosh(self):
        return GVar(c_cosh(self.v), self.d.mul(c_sinh(self.v)), self.cov)

    def tanh(self):
        return GVar(
            c_tanh(self.v),
            self.d.mul(1. / (c_cosh(self.v) ** 2)),
            self.cov,
            )

    def arcsinh(self):
        return log(self + sqrt(self * self + 1.))

    def asinh(self):
        return self.arcsinh()

    def arccosh(self):
        return log(self + sqrt(self * self - 1.))

    def acosh(self):
        return self.arccosh()

    def arctanh(self):
        return log((1. + self) / (1. - self)) / 2.

    def atanh(self):
        return self.arctanh()

    def exp(self):
        cdef double ans
        ans = c_exp(self.v)
        return GVar(ans, self.d.mul(ans), self.cov)

    def log(self):
        return GVar(c_log(self.v), self.d.mul(1./self.v), self.cov)

    def sqrt(self):
        cdef double ans = c_sqrt(self.v)
        return GVar(ans, self.d.mul(0.5/ans), self.cov)

    def fabs(self):
        if self.v >= 0:
            return self
        else:
            return -self

    def deriv(GVar self, x):
        """ Derivative of ``self`` with respest to primary |GVar|\s in ``x``.

        All |GVar|\s are constructed from primary |GVar|\s (see 
        :func:`gvar.is_primary`).  ``self.deriv(x)`` returns the 
        partial derivative of ``self`` with respect to 
        primary |GVar| ``x``, holding all of the other
        primary |GVar|\s constant.

        Args:
            x: A primary |GVar| or an array of primary |GVar|\s.

        Returns:
            Derivatives of ``self`` with respect to the 
            |GVar|\s in ``x``.  The result has the same 
            shape as ``x``.
        """
        cdef INTP_TYPE i, ider
        cdef double xder
        cdef numpy.ndarray[numpy.float_t, ndim=1] ans
        cdef GVar xi
        x = numpy.asarray(x)
        ans = numpy.zeros(x.size, dtype=float)
        self_deriv = dict([(self.d.v[i].i, self.d.v[i].v) for i in range(self.d.size)])
        for i in range(x.size):
            xi = x.flat[i]
            if xi.d.size != 1:
                raise ValueError("derivative ambiguous -- x is not primary")
            xder = xi.d.v[0].v 
            ider = xi.d.v[0].i 
            if xder == 0:
                continue
            ans[i] = self_deriv.get(ider, 0.0) / xder
        return ans.flat[0] if x.shape == () else ans.reshape(x.shape)

    def fmt(self, ndecimal=None, sep='', d=None):
        """ Convert to string with format: ``mean(sdev)``.

        Leading zeros in the standard deviation are omitted: for example,
        ``25.67 +- 0.02`` becomes ``25.67(2)``. Parameter ``ndecimal``
        specifies how many digits follow the decimal point in the mean.
        Parameter ``sep`` is a string that is inserted between the ``mean``
        and the ``(sdev)``. If ``ndecimal`` is ``None`` (default), it is set
        automatically to the larger of ``int(2-log10(self.sdev))`` or
        ``0``; this will display at least two digits of error. Very large
        or very small numbers are written with exponential notation when
        ``ndecimal`` is ``None``.

        Setting ``ndecimal < 0`` returns ``mean +- sdev``.
        """
        if d is not None:
            ndecimal = d            # legacy name
        if ndecimal is None:
            ans = str(self)
            if sep != '':
                if 'e' not in ans:
                    ans = ans.split('(')
                    if len(ans) > 1:
                        ans = ans[0] + sep + '(' + ans[1]
                    else:
                        ans = ans[0]
            return ans

        dv = abs(self.sdev)
        v = self.mean

        if dv == float('inf'):
            # infinite sdev
            if ndecimal > 0:
                ft = "%%.%df" % int(ndecimal)
                return (ft % v) + ' +- inf'
            else:
                return str(v) + ' +- inf'

        if ndecimal<0 or ndecimal != int(ndecimal):
            # do not use compact notation
            return "%g +- %g" % (v,dv)

        dv = round(dv, ndecimal)
        if dv<1.0:
            ft =  '%.' + str(ndecimal) + 'f%s(%.0f)'
            return ft % (v, sep, dv * 10. ** ndecimal)
        else:
            ft = '%.' + str(ndecimal) + 'f%s(%.' + str(ndecimal) + 'f)'
            return ft % (v, sep, dv)

    def partialvar(self,*args):
        """ Compute partial variance due to |GVar|\s in ``args``.

        This method computes the part of ``self.var`` due to the |GVar|\s
        in ``args``. If ``args[i]`` is correlated with other |GVar|\s, the
        variance coming from these is included in the result as well. (This
        last convention is necessary because variances associated with
        correlated |GVar|\s cannot be disentangled into contributions
        corresponding to each variable separately.)

        Args:
            args[i]: A |GVar| or array/dictionary of |GVar|\s
                 contributing to the partial variance.

        Returns:
            Partial variance due to all of ``args``.
        """
        cdef GVar ai
        cdef svec md
        cdef smat cov
        cdef numpy.ndarray[INTP_TYPE,ndim=1] dmask
        cdef numpy.ndarray[INTP_TYPE,ndim=1] md_idx
        cdef numpy.ndarray[numpy.float_t,ndim=1] md_v
        cdef INTP_TYPE i,j,md_size
        cdef INTP_TYPE dstart,dstop
        if self.d.size<=0:
            return 0.0
        dstart = self.d.v[0].i
        dstop = self.d.v[self.d.size-1].i+1
        # create a mask = 1 if (cov * args[i].der) component!=0; 0 otherwise
        # a) collect all indices referenced in args[i].der
        iset = set()
        for a in args:
            if a is None:
                continue
            if hasattr(a,'keys'):
                if not hasattr(a,'flat'):
                    a = BufferDict(a)
            else:
                a = numpy.asarray(a)
            for ai in a.flat:
                if ai is None:
                    continue
                else:
                    assert ai.cov is self.cov,"Incompatible |GVar|\s."
                iset.update(ai.d.indices())

        # b) collect indices connected to args[i].der indices by self.cov
        cov = self.cov
        jset = set()
        for i in iset:
            # NB: iset is contained in jset since cov always has diagonal elements
            jset.update(cov.row[i].indices())

        # c) build the mask to restrict to indices in or connected to args
        dmask = numpy.zeros(dstop-dstart, numpy.intp)
        for j in sorted(jset):
            if j<dstart:
                continue
            elif j>=dstop:
                break
            else:
                dmask[j-dstart] |= 1

        # create masked derivative vector for self
        md_size = 0
        md_idx = numpy.zeros(dstop-dstart, numpy.intp)
        md_v = numpy.zeros(dstop-dstart,numpy.float_)
        for i in range(self.d.size):
            if dmask[self.d.v[i].i-dstart]==0:
                continue
            else:
                md_idx[md_size] = self.d.v[i].i
                md_v[md_size] = self.d.v[i].v
                md_size += 1
        md = svec(md_size)
        md._assign(md_v[:md_size],md_idx[:md_size])

        return md.dot(self.cov.dot(md))

    def partialsdev(self,*args):
        """ Compute partial standard deviation due to |GVar|\s in ``args``.

        This method computes the part of ``self.sdev`` due to the |GVar|\s
        in ``args``. If ``args[i]`` is correlated with other |GVar|\s, the
        standard deviation coming from these is included in the result as
        well. (This last convention is necessary because variances
        associated with correlated |GVar|\s cannot be disentangled into
        contributions corresponding to each variable separately.)

        :param args[i]: Variables contributing to the partial standard
            deviation.
        :type args[i]: |GVar| or array/dictionary of |GVar|\s
        :returns: Partial standard deviation due to ``args``.
        """
        ans = self.partialvar(*args)
        return ans**0.5 if ans>0 else -(-ans)**0.5

    cpdef bint is_primary(self):
        """ ``True`` if a primary |GVar| ; ``False`` otherwise. 
        
        A *primary* |GVar| is one created using :func:`gvar.gvar` (or a 
        function of such a variable). A *derived* |GVar| is one that 
        is constructed from arithmetic expressions and functions that 
        combine multiple primary |GVar|\s. The standard deviations for 
        all |GVar|\s originate with the primary |GVar|\s. 
        In particular, :: 

            z = z.mean + sum_p (p - p.mean) * dz/dp

        is true for any |GVar| ``z``, where the sum is over all primary 
        |GVar|\s ``p``.
        """
        return self.d.size == 1
    
    property shape:
        """ Shape = () """
        def __get__(self):
            return ()

    property val:
        """ Mean value. """
        def __get__(self):
            return self.v

    property der:
        """ Array of derivatives with respect to  underlying (original)
        |GVar|\s.
        """
        def __get__(self):
            return self.d.toarray(len(self.cov))

    property mean:
        """ Mean value. """
        def __get__(self):
            return self.v

    property sdev:
        """ Standard deviation. """
        def __get__(self):
            cdef double var = self.cov.expval(self.d)
            if var>=0:
                return c_sqrt(var)
            else:
                return -c_sqrt(-var)

    property var:
        """ Variance. """
        # @cython.boundscheck(False)
        def __get__(self):
            return self.cov.expval(self.d)

    property internaldata:
        """ Data contained in |GVar|.

        This attribute is useful when creating a class that
        inherits from a |GVar|: for example, ::

            import gvar as gv
            class newGVar(gv.GVar):
                def __init__(self, g, a):
                    super(newGVar, self).__init__(*g.internaldata)
                    g.a = a

        creates a variation on |GVar| that, in effect, adds a new attribute
        ``a``  to an existing |GVar| ``g`` (being careful not to avoid
        names that collide with an existing |GVar| attribute).
        """
        def __get__(self):
            return self.v, self.d, self.cov

    def dotder(self, numpy.float_t[:] v not None):
        """ Return the dot product of ``self.der`` and ``v``. """
        cdef double ans = 0
        cdef INTP_TYPE i
        for i in range(self.d.size):
            ans += v[self.d.v[i].i] * self.d.v[i].v
        return ans

    def mdotder(self, numpy.float_t[:, :]  m not None):
        """ Return the dot product of m and ``self.der``. """
        cdef numpy.ndarray[numpy.float_t, ndim=1] ans 
        cdef INTP_TYPE i, j 
        ans = numpy.zeros(m.shape[0], dtype=float)
        for j in range(m.shape[0]):
            for i in range(self.d.size):
                ans[j] += m[j, self.d.v[i].i] * self.d.v[i].v
        return ans



# GVar factory functions

_RE1 = re.compile(r"(.*)\s*[+][-]\s*(.*)")
_RE2 = re.compile(r"(.*)[e](.*)")
_RE3 = re.compile(r"([-+]?)([0-9]*)[.]?([0-9]*)\s*\(([0-9]+)\)")
_RE3a = re.compile(r"([-+]?[0-9]*[.]?[0-9]*)\s*\(([.0-9]+)\)")

class GVarFactory:
    """ Creates one or more new |GVar|\s.

    ``gvar.gvar`` is an object of type :class:`gvar.GVarFactory`.
    Each of the following creates new |GVar|\s:

    .. function:: gvar.gvar(x, xsdev)
        :noindex:

        Returns a |GVar| with mean ``x`` and standard deviation ``xsdev``.
        Returns an array of |GVar|\s if ``x`` and ``xsdev`` are arrays
        with the same shape; the shape of the result is the same as the
        shape of ``x``. Returns a |BufferDict| if ``x`` and ``xsdev``
        are dictionaries with the same keys and layout; the result has
        the same keys and layout as ``x``.

    .. function:: gvar.gvar(x, xcov)
        :noindex:

        Returns an array of |GVar|\s with means given by array ``x`` and a
        covariance matrix given by array ``xcov``, where ``xcov.shape =
        2*x.shape``; the result has the same shape as ``x``. Returns a
        |BufferDict| if ``x`` and ``xcov`` are dictionaries, where the
        keys in ``xcov`` are ``(k1,k2)`` for any keys ``k1`` and ``k2``
        in ``x``. Returns a single |GVar| if ``x`` is a number and
        ``xcov`` is a one-by-one matrix. The layout for ``xcov`` is
        compatible with that produced by :func:`gvar.evalcov` for
        a single |GVar|, an array of |GVar|\s, or a dictionary whose
        values are |GVar|\s and/or arrays of |GVar|\s. Therefore
        ``gvar.gvar(gvar.mean(g), gvar.evalcov(g))`` creates |GVar|\s
        with the same means and covariance matrix as the |GVar|\s
        in ``g`` provided ``g`` is a single |GVar|, or an array or
        dictionary of |GVar|\s.

    .. function:: gvar.gvar(x, xcov, verify=True)
        :noindex:

        Same as ``gvar.gvar(x, xcov)`` above but checks that the covariance 
        matrix is symmetric and positive definite (which covariance matrices 
        should be). This check is expensive for large matrices and so is 
        *not* done by default. Note, however, that unpredictable outcomes 
        will result from specifying an improper covariance matrix.
        
    .. function:: gvar.gvar(x, xcov, fast=True)
        :noindex:

        Normally ``gvar.gvar(x, xcov)`` tries to break the covariance matrix
        into disjoint diagonal blocks, if there are any. For example, ::
        
            xcov = [[1,1,0], [1,2,0], [0,0,3]]
        
        can be decomposed into two blocks. This decomposition saves memory, 
        and can make later manipulations of the resulting |GVar|\s 
        faster. This is at the expense of extra processing to 
        create the |GVar|\s. Setting keyword ``fast=True`` prevents 
        ``gvar.gvar`` from doing this, which would make sense, for example, 
        if it was known ahead of time that ``xcov`` has no sub-blocks. The 
        default is ``fast=False``. Either choice gives correct answers; 
        the difference is about efficiency.
        
    .. function:: gvar.gvar((x, xsdev))
        :noindex:

        Returns a |GVar| with mean ``x`` and standard deviation ``xsdev``.

    .. function:: gvar.gvar(xstr)
        :noindex:

        Returns a |GVar| corresponding to string ``xstr`` which is
        either of the form ``"xmean +- xsdev"`` or ``"x(xerr)"`` (see
        :meth:`GVar.fmt`).

    .. function:: gvar.gvar(xgvar)
        :noindex:

        Returns |GVar| ``xgvar`` unchanged.

    .. function:: gvar.gvar(xdict)
        :noindex:

        Returns a dictionary (:class:`BufferDict`) ``b`` where
        ``b[k] = gvar.gvar(xdict[k])`` for every key in dictionary ``xdict``.
        The values in ``xdict``, therefore, can be strings, tuples or
        |GVar|\s (see above), or arrays of these.

    .. function:: gvar.gvar(xarray)
        :noindex:

        Returns an array ``a`` having the same shape as ``xarray`` where
        every element ``a[i...] = gvar.gvar(xarray[i...])``. The values in
        ``xarray``, therefore, can be strings, tuples or |GVar|\s (see
        above).

    .. function:: gvar.gvar(ymean, ycov, x, xycov)
        :noindex:

        Returns a 1-d array of |GVar|\s ``y[i]`` constructed from the 1-d array 
        of mean values ``ymean`` and the 2-d covariance matrix ``ycov``. The 
        ``y[i]`` are correlated with the primary |GVar|\s in 1-d array ``x``.
        The ``x-y`` covariance matrix is ``xycov`` whose shape 
        is ``x.shape + y.shape``. Note that this changes the |GVar|\s
        in ``x`` (because they are correlated with the ``y[i]``); it 
        has no effect on the variance or on correlations between 
        different ``x[i]``\s.
    """
    def __init__(self,cov=None):
        if cov is None:
            self.cov = smat()
        else:
            assert isinstance(cov,smat),"cov not type gvar.smat"
            self.cov = cov

    def __call__(self, *args, verify=False, fast=False):
        cdef INTP_TYPE nx, i, nd, ib, nb, n1, n2
        cdef svec der
        cdef smat cov
        cdef GVar gv, xg, yg
        cdef numpy.ndarray[numpy.float_t, ndim=1] d
        cdef numpy.ndarray[numpy.float_t, ndim=1] d_v
        cdef numpy.ndarray[numpy.float_t, ndim=2] xcov
        cdef numpy.ndarray[INTP_TYPE, ndim=1] d_idx
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx
        cdef numpy.ndarray[INTP_TYPE, ndim=1] xrow, yrow

        if len(args)==2:
            if hasattr(args[0], 'keys'):
                # args are dictionaries -- convert to arrays
                if not hasattr(args[1], 'keys'):
                    raise ValueError(
                        'Argument mismatch: %s, %s'
                        % (str(type(args[0])), str(type(args[1])))
                        )
                if set(args[0].keys()) == set(args[1].keys()):
                    # means and stdevs
                    x = BufferDict(args[0])
                    xsdev = BufferDict(x, buf=numpy.empty(x.size, numpy.float_))
                    for k in x:
                        xsdev[k] = args[1][k]
                    xflat = self(x.flat, xsdev.flat)
                    return BufferDict(x, buf=xflat)
                else:
                    # means and covariance matrix
                    x = BufferDict(args[0])
                    xcov = numpy.empty((x.size, x.size), numpy.float_)
                    for k1 in x:
                        k1_sl, k1_sh = x.slice_shape(k1)
                        if k1_sh == ():
                            k1_sl = slice(k1_sl, k1_sl + 1)
                            n1 = 1
                        else:
                            n1 = numpy.prod(k1_sh)
                        for k2 in x:
                            k2_sl, k2_sh = x.slice_shape(k2)
                            if k2_sh == ():
                                k2_sl = slice(k2_sl, k2_sl + 1)
                                n2 = 1
                            else:
                                n2 = numpy.prod(k2_sh)
                            xcov[k1_sl, k2_sl] = (
                                numpy.asarray(args[1][k1, k2]).reshape(n1, n2)
                                )
                    xflat = self(x.flat, xcov, verify=verify, fast=fast)
                    return BufferDict(x, buf=xflat)
            # else:
            # (x,xsdev) or (xarray,sdev-array) or (xarray,cov)
            # unpack arguments and verify types
            try:
                x = numpy.asarray(args[0],numpy.float_)
                xsdev = numpy.asarray(args[1],numpy.float_)
            except (ValueError,TypeError):
                raise TypeError(
                    "arguments must be numbers or arrays of numbers"
                    )

            if len(x.shape)==0:
                # single gvar from x and xsdev
                if xsdev.shape == (1, 1):
                    # xsdev is actually a variance (1x1 matrix)
                    xsdev = c_sqrt(abs(xsdev[0, 0]))
                elif len(xsdev.shape) != 0:
                    raise ValueError("x and xsdev different shapes.")
                if verify and xsdev < 0:
                    raise ValueError('negative standard deviation: ' + str(xsdev))
                idx = self.cov.append_diag(numpy.array([xsdev**2]))
                der = svec(1)
                der.v[0].i = idx[0]
                der.v[0].v = 1.0
                # gv = GVar.__new__(GVar)
                # gv.v = x
                # gv.d = der
                # gv.cov = self.cov
                # return gv
                return GVar(x, der, self.cov)
            else:
                # array of gvars from x and sdev/cov arrays
                nx = len(x.flat)
                if x.shape==xsdev.shape:  # x,sdev
                    if verify and numpy.any(xsdev < 0):
                        raise ValueError('negative standard deviation: ' + str(xsdev))
                    idx = self.cov.append_diag(xsdev.reshape(nx) ** 2)
                elif xsdev.shape==2 * x.shape: # x,cov
                    xcov = xsdev.reshape(nx, nx)
                    with numpy.errstate(under='ignore'):
                        if not numpy.allclose(xcov, xcov.T, equal_nan=True):
                            raise ValueError('non-symmetric covariance matrix:\n' + str(xcov))
                    if verify:
                        try:
                            numpy.linalg.cholesky(xcov)
                        except numpy.linalg.LinAlgError:
                            raise ValueError('covariance matrix not positive definite')
                    if fast:
                        idx = self.cov.append_diag_m(xcov)
                    else:
                        allxcov = numpy.arange(nx)
                        ans = numpy.empty(nx, dtype=object)
                        nb, key = _connected_components(xcov != 0, directed=False)
                        for ib in range(nb):
                            bidx = allxcov[key == ib]
                            ans[bidx] = self(x.flat[bidx], xcov[bidx[:, None], bidx], fast=True)
                        return ans.reshape(x.shape)
                else:
                    raise ValueError("Argument shapes mismatched: " +
                        str(x.shape) + ' ' + str(xsdev.shape))
                d = numpy.ones(nx, dtype=numpy.float_)
                ans = numpy.empty(nx, dtype=object)
                for i in range(nx):
                    der = svec(1)
                    der.v[0].i = idx[i]
                    der.v[0].v = 1.0
                    ans[i] = GVar(x.flat[i], der, self.cov)
                return ans.reshape(x.shape)
        elif len(args)==1:
            # ('1(1)') etc
            return self._call1(*args)
        elif len(args)==3:
            # (x,der,cov)
            return self._call3(*args)
        elif len(args) == 4:
            # (ymean, ycov, x, xycov)
            return self._call4(*args, verify=verify, fast=fast)
        else:
            raise ValueError("Wrong number of arguments: "+str(len(args)))
    
    def _call1(self, *args):
        # gvar('1(1)') etc
        x = args[0]
        if isinstance(x, str):
            # case 1: x is a string like "3.72(41)" or "3.2 +- 4"
            x = x.strip()
            try:
                # eg: 3.4 +- 0.7e-4
                a,c = _RE1.match(x).groups()
                return self(float(a), float(c))
            except AttributeError:
                pass
            try:
                # eg: 3.4(1)e+10
                a, c = _RE2.match(x).groups()
                return self(a)*float("1e"+c)
            except AttributeError:
                pass
            try:
                # eg: +3.456(33)
                s, a, b, c = _RE3.match(x).groups()
                s = -1. if s == '-' else 1.
                if not a and not b:
                    raise ValueError("Poorly formatted string: "+x)
                elif not b:
                    return s*self(float(a),float(c))
                else:
                    if not a:
                        a = '0'
                    fac = 1./10.**len(b)
                    a,b,c = [float(xi) for xi in [a,b,c]]
                    return s*self(a + b*fac, c*fac)
            except AttributeError:
                pass
            try:
                # eg: 3.456(1.234)
                a, c = _RE3a.match(x).groups()
                return self(float(a), float(c))
            except AttributeError:
                raise ValueError("Poorly formatted string: "+x)

        elif isinstance(x,GVar):
            # case 2: x is a GVar
            return x

        elif isinstance(x,tuple) and len(x)==2:
            # case 3: x = (x,sdev) tuple
            return self(x[0], x[1])

        elif hasattr(x,'keys'):
            # case 4: x is a dictionary
            ans = BufferDict()
            for k in x:
                ans[k] = self(x[k])
            return ans

        elif hasattr(x, '__iter__'):
            # case 5: x is an array
            if isinstance(x, numpy.ndarray) and x.shape == ():
                return self(x.flat[0])
            return numpy.array(
                [xi if isinstance(xi, GVar) else self(xi) for xi in x], 
                object,
                )

        else:   # case 6: a number
            return self(x, 0.0)
    
    def _call3(self, *args):
        # gvar(x,der,cov)
        cdef INTP_TYPE nx, i, nd, ib, nb
        cdef svec der
        cdef smat cov
        cdef numpy.ndarray[numpy.float_t, ndim=1] d
        cdef numpy.ndarray[numpy.float_t, ndim=1] d_v
        cdef numpy.ndarray[INTP_TYPE, ndim=1] d_idx
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx
        try:
            x = float(args[0])
        except (ValueError, TypeError):
            raise TypeError('Value not a number.')
        cov = args[2]
        assert isinstance(cov, smat), "cov not type gvar.smat."
        if isinstance(args[1], svec):
            return GVar(x, args[1], cov)
        elif isinstance(args[1], tuple):
            try:
                d_idx = numpy.asarray(args[1][1], numpy.intp)
                d_v = numpy.asarray(args[1][0], numpy.float_)
                assert d_idx.ndim == 1 and d_v.ndim == 1 and d_idx.shape[0] == d_v.shape[0]
            except (ValueError, TypeError, AssertionError):
                raise TypeError('Badly formed derivative.')
        else:
            try:
                d = numpy.asarray(args[1], numpy.float_)
                assert d.ndim == 1
            except (ValueError, TypeError, AssertionError):
                raise TypeError('Badly formed derivative.')
            d_idx = d.nonzero()[0]
            d_v = d[d_idx]
        assert len(cov) > d_idx[-1], "length mismatch between der and cov"
        der = svec(len(d_idx))
        der.assign(d_v, d_idx)
        return GVar(x, der, cov)
                
    def _call4(self, *args, verify=False, fast=False):
        # gvar(ymean, ycov, x, xycov)
        # y,x 1-d arrays, ycov, xycov 2-d arrays
        cdef INTP_TYPE i, j, k, ni, nj
        cdef smat cov
        cdef GVar xg, yg
        cdef numpy.ndarray[numpy.float_t, ndim=1] ymean
        cdef numpy.ndarray[numpy.float_t, ndim=2] ycov, xycov
        cdef numpy.ndarray[INTP_TYPE, ndim=1] idx
        cdef numpy.ndarray[INTP_TYPE, ndim=1] xrow, yrow
        try:
            ymean = numpy.asarray(args[0], float)
            assert ymean.ndim == 1 and ymean.shape[0] > 0
        except:
            raise ValueError('y must be a 1-d array of numbers')
        try:
            ycov = numpy.asarray(args[1], float)
            assert ycov.shape[0] == ymean.shape[0]
            assert ycov.shape[1] == ymean.shape[0]
        except:
            raise ValueError(
                'ycov must be a matrix of numbers '
                + str((ymean.shape[0], ymean.shape[0]))
                )
        try:
            x = numpy.asarray(args[2])
            assert len(x.shape) == 1
        except:
            raise ValueError('x must be a 1-d array of numbers')
        try:
            xycov = numpy.asarray(args[3])
            assert xycov.shape[0] == x.shape[0]
            assert xycov.shape[1] == ymean.shape[0]
        except:
            raise ValueError(
                'xycov must be a matrix of numbers with shape ' 
                + str((x.shape[0], ymean.shape[0]))
                )
        if x.shape[0] == 0:
            return self(ymean, ycov, verify=verify, fast=fast)
        else:
            y = self(ymean, ycov, verify=verify, fast=False)
        xrow = numpy.zeros(len(x), numpy.intp)
        yrow = numpy.zeros(len(y), numpy.intp)
        for j, xg in enumerate(x):
            if xg.d.size != 1 or xg.d.v[0].v != 1.:
                raise ValueError('x[i] must be primary GVars')
            xrow[j] = xg.d.v[0].i
        for j, yg in enumerate(y):
            yrow[j] = yg.d.v[0].i
        idx = numpy.argsort(xrow)
        y[0].cov.add_offdiag_m(xrow[idx], yrow, xycov[idx])
        if verify:
            allrow = list(xrow) + list(yrow)
            allcov = numpy.zeros(2*(len(allrow),), float)
            cov = self.cov
            for ni, i in enumerate(allrow):
                indices = cov.row[i].indices()
                values = cov.row[i].values()
                for nj, j in enumerate(allrow):
                    idx = (indices == j).nonzero()[0]
                    if len(idx) == 1:
                        allcov[ni, nj] = values[idx[0]]
                    if nj < ni and allcov[nj, ni] != allcov[ni, nj]:
                        raise ValueError('covariance matrix not symmetric')
            try:
                numpy.linalg.cholesky(allcov)
            except numpy.linalg.LinAlgError:
                raise ValueError('covariance matrix not positive definite')
        return y 


def gvar_function(x, double f, dfdx):
    """ Create a |GVar| for function f(x) given f and df/dx at x.

    This function creates a |GVar| corresponding to a function of |GVar|\s ``x``
    whose value is ``f`` and whose derivatives with respect to each
    ``x`` are given by ``dfdx``. Here ``x`` can be a single |GVar|,
    an array of |GVar|\s (for a multidimensional function), or
    a dictionary whose values are |GVar|\s or arrays of |GVar|\s, while
    ``dfdx`` must be a float, an array of floats, or a dictionary
    whose values are floats or arrays of floats, respectively.

    This function is useful for creating functions that can accept
    |GVar|\s as arguments. For example, ::

        import math
        import gvar as gv

        def sin(x):
            if isinstance(x, gv.GVar):
                f = math.sin(x.mean)
                dfdx = math.cos(x.mean)
                return gv.gvar_function(x, f, dfdx)
            else:
                return math.sin(x)

    creates a version of ``sin(x)`` that works with either floats or
    |GVar|\s as its argument. This particular function is unnecessary since
    it is already provided by :mod:`gvar`.

    :param x: Point at which the function is evaluated.
    :type x: |GVar|, array of |GVar|\s, or a dictionary of |GVar|\s

    :param f: Value of function at point ``gvar.mean(x)``.
    :type f: float

    :param dfdx: Derivatives of function with respect to x at
        point ``gvar.mean(x)``.
    :type dfdx: float, array of floats, or a dictionary of floats

    :returns: A |GVar| representing the function's value at ``x``.
    """
    cdef svec f_d
    cdef GVar x_i
    cdef double dfdx_i
    if hasattr(x, 'keys'):
        if not isinstance(x, BufferDict):
            x = BufferDict(x)
        if x.size == 0 or not isinstance(x.buf[0], GVar):
            raise ValueError('x has no GVars')
        if not hasattr(dfdx, 'keys'):
            raise ValueError('x is a dictionary, dfdx is not')
        tmp = BufferDict()
        try:
            for k in x:
                tmp[k] = dfdx[k]
                assert numpy.shape(tmp[k]) == numpy.shape(x[k])
        except KeyError:
            raise ValueError("dfdx[k] doesn't exist for k = " + str(k))
        except AssertionError:
            raise ValueError('shape(dfdx[k]) != shape(x[k]) for k = ' + str(k))
        dfdx = tmp
    else:
        x = numpy.asarray(x)
        if x.size == 0 or not isinstance(x.flat[0], GVar):
            raise ValueError('x has no GVars')
        dfdx = numpy.asarray(dfdx)
        if x.shape != dfdx.shape:
            raise ValueError('shape(dfdx) != shape(x)')
    f_d = None
    for x_i, dfdx_i in zip(x.flat, dfdx.flat):
        if f_d is None:
            f_d = x_i.d.mul(dfdx_i)
        else:
            f_d = f_d.add(x_i.d, 1., dfdx_i)
    return GVar(f, f_d, x_i.cov)


def abs(g):
    try:
        return fabs(g)
    except:
        return numpy.absolute(g)





