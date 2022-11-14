"""
test-gvar.py

"""
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

import os
import unittest
import collections
import math
import pickle
import numpy as np
import random
import gvar as gv
from gvar import *
try:
    import vegas
    have_vegas = True
except:
    have_vegas = False

FAST = False

class ArrayTests(object):
    def __init__(self):
        pass

    def assert_gvclose(self,x,y,rtol=1e-5,atol=1e-8,prt=False):
        """ asserts that the means and sdevs of all x and y are close """
        if hasattr(x,'keys') and hasattr(y,'keys'):
            if sorted(x.keys())==sorted(y.keys()):
                for k in x:
                    self.assert_gvclose(x[k],y[k],rtol=rtol,atol=atol)
                return
            else:
                raise ValueError("x and y have mismatched keys")
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.asarray(x).flat
        y = np.asarray(y).flat
        if prt:
            print(np.array(x))
            print(np.array(y))
        for xi,yi in zip(x,y):
            self.assertGreater(atol+rtol*abs(yi.mean),abs(xi.mean-yi.mean))
            self.assertGreater(10*(atol+rtol*abs(yi.sdev)),abs(xi.sdev-yi.sdev))

    def assert_arraysclose(self,x,y,rtol=1e-5,prt=False):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        max_val = max(np.abs(list(x)+list(y)))
        max_rdiff = max(np.abs(x-y))/max_val
        if prt:
            print(x)
            print(y)
            print(max_val,max_rdiff,rtol)
        self.assertAlmostEqual(max_rdiff,0.0,delta=rtol)

    def assert_arraysequal(self,x,y):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = [float(xi) for xi in np.array(x).flatten()]
        y = [float(yi) for yi in np.array(y).flatten()]
        self.assertSequenceEqual(x,y)


class test_svec(unittest.TestCase,ArrayTests):
    def test_v(self):
        """ svec svec.assign svec.toarray """
        v = svec(3)   # [1,2,0,0,3]
        v.assign([1.,3.,2.],[0,4,1])
        self.assert_arraysequal(v.toarray(),[1.,2.,0.,0.,3.])

    def test_null_v(self):
        """ svec(0) """
        v = svec(0)
        self.assertEqual(len(v.toarray()),0)
        self.assertEqual(len(v.clone().toarray()),0)
        self.assertEqual(len(v.mul(10.).toarray()),0)
        u = svec(1)
        u.assign([1],[0])
        self.assertEqual(v.dot(u),0.0)
        self.assertEqual(u.dot(v),0.0)
        self.assert_arraysequal(u.add(v).toarray(),v.add(u).toarray())

    def test_v_clone(self):
        """ svec.clone """
        v1 = svec(3)   # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = v1.clone() # [0,10,0,0,20]
        self.assert_arraysequal(v1.toarray(),v2.toarray())
        v2.assign([10.,20.,30.],[0,1,2])
        self.assert_arraysequal(v2.toarray(),[10.,20.,30.])

    def test_v_dot(self):
        """ svec.dot """
        v1 = svec(3)   # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = svec(2)
        v2.assign([10.,20.],[1,4])
        self.assertEqual(v1.dot(v2),v2.dot(v1))
        self.assertEqual(v1.dot(v2),80.)
        v1 = svec(3)
        v1.assign([1,2,3],[0,1,2])
        v2 = svec(2)
        v2.assign([4,5],[3,4])
        self.assertEqual(v1.dot(v2),v2.dot(v1))
        self.assertEqual(v1.dot(v2),0.0)

    def test_v_add(self):
        """ svec.add """
        v1 = svec(3)    # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = svec(2)    # [0,10,0,0,20]
        v2.assign([10.,20.],[1,4])
        self.assert_arraysequal(v1.add(v2).toarray(),v2.add(v1).toarray())
        self.assert_arraysequal(v1.add(v2).toarray(),[1,12,0,0,23])
        self.assert_arraysequal(v1.add(v2,10,100).toarray(),[10.,1020.,0,0,2030.])
        self.assert_arraysequal(v2.add(v1,100,10).toarray(),[10.,1020.,0,0,2030.])
        v1 = svec(2)            # overlapping
        v1.assign([1,2],[0,1])
        v2.assign([3,4],[1,2])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,31.,28.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,31.,28.])
        v1 = svec(3)
        v2 = svec(3)
        v1.assign([1,2,3],[0,1,2])
        v2.assign([10,20,30],[1,2,3])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,80.,155.,210.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,80.,155.,210.])
        v1 = svec(2)
        v2 = svec(2)
        v1.assign([1,2],[0,1])  # non-overlapping
        v2.assign([3,4],[2,3])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,10.,21.,28.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,10.,21.,28.])
        v1 = svec(4)            # one encompasses the other
        v1.assign([1,2,3,4],[0,1,2,3])
        v2.assign([10,20],[1,2])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,80.,155.,20.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,80.,155.,20.])

    def test_v_mul(self):
        """ svec.mul """
        v1 = svec(3)    # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        self.assert_arraysequal(v1.mul(10).toarray(),[10,20,0,0,30])

    def test_pickle(self):
        v = svec(4)
        v.assign([1.,2.,5.,22], [3,5,1,0])
        with open('outputfile.p', 'wb') as ofile:
            pickle.dump(v, ofile)
        with open('outputfile.p', 'rb') as ifile:
            newv = pickle.load(ifile)
        self.assertEqual(type(v), type(newv))
        self.assertTrue(np.all(v.toarray() == newv.toarray()))
        os.remove('outputfile.p')

class test_smat(unittest.TestCase,ArrayTests):
    def setUp(self):
        """ make mats for tests """
        global smat_m,np_m
        smat_m = smat()
        smat_m.append_diag(np.array([0.,10.,200.]))
        smat_m.append_diag_m(np.array([[1.,2.],[2.,1.]]))
        smat_m.append_diag(np.array([4.,5.]))
        smat_m.append_diag_m(np.array([[3.]]))
        np_m = np.array([[ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,   10.,    0.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,    0.,  200.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    1.,    2.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    2.,    1.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    4.,    0.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    0.,    5.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    3.]])

    def tearDown(self):
        global smat_m,np_m
        smat_m = None
        np_m = None

    def test_m_append(self):
        """ smat.append_diag smat.append_diag_m smat.append_row smat.toarray"""
        self.assert_arraysequal(smat_m.toarray(),np_m)

    def test_m_dot(self):
        """ smat.dot """
        global smat_m,np_m
        v = svec(2)
        v.assign([10,100],[1,4])
        np_v = v.toarray()
        nv = len(np_v)
        self.assert_arraysequal(smat_m.dot(v).toarray(),np.dot(np_m[:nv,:nv],np_v))
        self.assert_arraysequal(smat_m.dot(v).toarray(),[0.,100.,0.,200.,100.])
        self.assertEqual(smat_m.dot(v).dot(v),np.dot(np.dot(np_m[:nv,:nv],np_v),np_v))
        self.assertEqual(smat_m.dot(v).size,3)

    def test_m_expval(self):
        """ smat.expval """
        global smat_m,np_m
        v = svec(2)
        v.assign([10.,100.],[1,4])
        np_v = v.toarray()
        nv = len(np_v)
        self.assertEqual(smat_m.expval(v),np.dot(np.dot(np_m[:nv,:nv],np_v),np_v))

    def test_pickle(self):
        """ pickle.dump(smat, outfile) """
        global smat_m
        with open('outputfile.p', 'wb') as ofile:
            pickle.dump(smat_m, ofile)
        with open('outputfile.p', 'rb') as ifile:
            m = pickle.load(ifile)
        self.assertEqual(type(smat_m), type(m))
        self.assertTrue(np.all(smat_m.toarray() == m.toarray()))
        os.remove('outputfile.p')

class test_smask(unittest.TestCase):
    def test_smask(self):
        def _test(imask):
            mask = smask(imask)
            np.testing.assert_array_equal(sum(imask[mask.starti:mask.stopi]), mask.len)
            np.testing.assert_array_equal(imask, np.asarray(mask.mask))
            np.testing.assert_array_equal(np.asarray(mask.map)[imask != 0], np.arange(mask.len))
            np.testing.assert_array_equal(np.cumsum(imask[imask != 0]) - 1, np.asarray(mask.map)[imask != 0])
        g = gvar([1, 2, 3], [4, 5, 6])
        gvar(1,0)
        imask = np.array(g[0].der + g[2].der, dtype=np.int8)
        _test(imask)
    
    def test_masked_ved(self):
        def _test(imask, g):
            mask = smask(imask)
            vec = g.internaldata[1].masked_vec(mask)
            np.testing.assert_array_equal(vec, g.der[imask!=0])
        g = gvar([1, 2, 3], [4, 5, 6])
        gvar(1,0)
        imask = np.array(g[0].der + g[1].der, dtype=np.int8)
        g[1:] += g[:-1]
        g2 = g**2
        _test(imask, g2[0])
        _test(imask, g2[1])
        _test(imask, g2[2])
    
    def test_masked_mat(self):
        a = np.random.rand(3,3)
        g = gvar([1, 2, 3], a.dot(a.T))
        imask = np.array((g[0].der + g[2].der) != 0, dtype=np.int8)
        cov = evalcov([g[0], g[2]])
        gvar(1,0)
        mask = smask(imask)
        np.testing.assert_allclose(cov, g[1].cov.masked_mat(mask))


class test_gvar1(unittest.TestCase,ArrayTests):
    """ gvar1 - part 1 """
    def setUp(self):
        """ setup for tests """
        global x,xmean,xsdev,gvar
        # NB - powers of two important
        xmean = 0.5
        xsdev = 0.25
        gvar = switch_gvar()
        x = gvar(xmean,xsdev)
        # ranseed((1968,1972,1972,1978,1980))
        # random.seed(1952)

    def tearDown(self):
        """ cleanup after tests """
        global x,gvar
        gvar = restore_gvar()
        x = None

    def test_str(self):
        """ str(x) """
        global x,xmean,xsdev,gvar
        self.assertEqual(str(x), x.fmt())

    def test_call(self):
        """ x() """
        global x,xmean,xsdev,gvar
        n = 10000
        fac = 5.   # 4 sigma
        xlist = [x() for i in range(n)]
        avg = np.average(xlist)
        std = np.std(xlist)
        self.assertAlmostEqual(avg,x.mean,delta=fac*x.sdev/n**0.5)
        self.assertAlmostEqual(std,(1-1./n)**0.5*xsdev,delta=fac*x.sdev/(2*n)**0.5)

    def test_cmp(self):
        """ x==y x!=y x>y x<y"""
        global x,xmean,xsdev,gvar
        x = gvar(1, 10)
        y = gvar(2, 20)
        self.assertTrue(y!=x and 2*x==y and x==1 and y!=1 and 1==x and 1!=y)
        self.assertTrue(
            not y==x and not 2*x!=y and not x!=1 and not y==1
            and not 1!=x and not 1==y
            )
        self.assertTrue(y>x and x<y and y>=x and x<=y and y>=2*x and 2*x<=y)
        self.assertTrue(not y<x and not x>y and not y<=x and not x>=y)
        self.assertTrue(y>1 and x<2 and y>=1 and x<=2 and y>=2 and 2*x<=2)
        self.assertTrue(not y<1 and not x>2 and not y<=1 and not x>=2)
        self.assertTrue(1<y and 2>x and 1<=y and 2>=x and 2<=y and 2>=2*x)
        self.assertTrue(not 1>y and not 2<x and not 1>=y and not 2<=x)

    def test_neg(self):
        """ -x """
        global x,xmean,xsdev,gvar
        z = -x
        self.assertEqual(x.mean,-z.mean)
        self.assertEqual(x.var,z.var)

    def test_pos(self):
        """ +x """
        global x,xmean,xsdev,gvar
        z = +x
        self.assertEqual(x.mean, z.mean)
        self.assertEqual(x.var, x.var)


class test_gvar2(unittest.TestCase,ArrayTests):
    """ gvar - part 2 """
    def setUp(self):
        global x,y,gvar
        # NB x.mean < 1 and x.var < 1 and y.var > 1 (assumed below)
        # and powers of 2 important
        gvar = switch_gvar()
        x,y = gvar([0.125,4.],[[0.25,0.0625],[0.0625,1.]])
        # ranseed((1968,1972,1972,1978,1980))
        # random.seed(1952)
        self.label = None

    def tearDown(self):
        """ cleanup after tests """
        global x,y,gvar
        x = None
        y = None
        gvar = restore_gvar()
        # if self.label is not None:
        #     print self.label

    def test_add(self):
        """ x+y """
        global x,y,gvar
        z = x+y
        cov = evalcov([x,y])
        self.assertEqual(z.mean,x.mean+y.mean)
        self.assertEqual(z.var,cov[0,0]+cov[1,1]+2*cov[0,1])
        z = x + y.mean
        self.assertEqual(z.mean,x.mean+y.mean)
        self.assertEqual(z.var,x.var)
        z = y.mean + x
        self.assertEqual(z.mean,x.mean+y.mean)
        self.assertEqual(z.var,x.var)

    def test_sub(self):
        """ x-y """
        global x,y,gvar
        z = x-y
        cov = evalcov([x,y])
        self.assertEqual(z.mean,x.mean-y.mean)
        self.assertEqual(z.var,cov[0,0]+cov[1,1]-2*cov[0,1])
        z = x - y.mean
        self.assertEqual(z.mean,x.mean-y.mean)
        self.assertEqual(z.var,x.var)
        z = y.mean - x
        self.assertEqual(z.mean,y.mean-x.mean)
        self.assertEqual(z.var,x.var)

    def test_mul(self):
        """ x*y """
        z = x*y
        dz = [y.mean,x.mean]
        cov = evalcov([x,y])
        self.assertEqual(z.mean,x.mean*y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = x * y.mean
        dz = [y.mean,0.]
        self.assertEqual(z.mean,x.mean*y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = y.mean * x
        self.assertEqual(z.mean,x.mean*y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))

    def test_div(self):
        """ x/y """
        z = x/y
        dz = [1./y.mean,-x.mean/y.mean**2]
        cov = evalcov([x,y])
        self.assertEqual(z.mean,x.mean/y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = x / y.mean
        dz = [1./y.mean,0.]
        self.assertEqual(z.mean,x.mean/y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = y.mean / x
        dz = [-y.mean/x.mean**2,0.]
        self.assertEqual(z.mean,y.mean/x.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))

    def test_pow(self):
        """ x**y """
        z = x**y
        dz = [y.mean*x.mean**(y.mean-1),x.mean**y.mean*log(x.mean)]
        cov = evalcov([x,y])
        self.assertEqual(z.mean,x.mean**y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = x ** y.mean
        dz = [y.mean*x.mean**(y.mean-1),0.]
        self.assertEqual(z.mean,x.mean**y.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))
        z = y.mean ** x
        dz = [y.mean**x.mean*log(y.mean),0.]
        self.assertEqual(z.mean,y.mean**x.mean)
        self.assertEqual(z.var,np.dot(dz,np.dot(cov,dz)))

    def t_fcn(self,f,df):
        """ tester for test_fcn """
        gdict = dict(globals())
        gdict['x'] = x          # with GVar
        fx = eval(f,gdict)
        gdict['x'] = x.mean     # with float
        fxm = eval(f,gdict)
        dfxm = eval(df,gdict)
        self.assertAlmostEqual(fx.mean,fxm)
        self.assertAlmostEqual(fx.var,x.var*dfxm**2)

    def test_fcn(self):
        """ f(x) """
        flist = [
        ("sin(x)","cos(x)"), ("cos(x)","-sin(x)"), ("tan(x)","1 + tan(x)**2"),
        ("arcsin(x)","(1 - x**2)**(-1./2.)"), ("arccos(x)","-1/(1 - x**2)**(1./2.)"),
        ("arctan(x)","1/(1 + x**2)"),
        ("sinh(x)","cosh(x)"), ("cosh(x)","sinh(x)"), ("tanh(x)","1 - tanh(x)**2"),
        ("arcsinh(x)","1./sqrt(x**2+1.)"),("arccosh(1+x)","1./sqrt(x**2+2*x)"),
        ("arctanh(x)","1./(1-x**2)"),
        ("exp(x)","exp(x)"), ("log(x)","1/x"), ("sqrt(x)","1./(2*x**(1./2.))")
        ]
        for f,df in flist:
            self.label = f
            self.t_fcn(f,df)
        # arctan2 tests
        x = gvar('0.5(0.5)')
        y = gvar('2(2)')
        f = arctan2(y, x)
        fc = arctan(y / x)
        self.assertAlmostEqual(f.mean, fc.mean)
        self.assertAlmostEqual(f.sdev, fc.sdev)
        self.assertAlmostEqual(arctan2(y, x).mean, numpy.arctan2(y.mean, x.mean))
        self.assertAlmostEqual(arctan2(y, -x).mean, numpy.arctan2(y.mean, -x.mean))
        self.assertAlmostEqual(arctan2(-y, -x).mean, numpy.arctan2(-y.mean, -x.mean))
        self.assertAlmostEqual(arctan2(-y, x).mean, numpy.arctan2(-y.mean, x.mean))
        self.assertAlmostEqual(arctan2(y, x*0).mean, numpy.arctan2(y.mean, 0))
        self.assertAlmostEqual(arctan2(-y, x*0).mean, numpy.arctan2(-y.mean, 0))

    def test_gvar_function(self):
        """ gvar_function(x, f, dfdx) """
        x = sqrt(gvar(0.1, 0.1) + gvar(0.2, 0.5))
        def fcn(x):
            return sin(x + x**2)
        def dfcn_dx(x):
            return cos(x + x**2) * (1 + 2*x)
        f = fcn(x).mean
        dfdx = dfcn_dx(x).mean
        diff = gvar_function(x, f, dfdx) - fcn(x)
        self.assertAlmostEqual(diff.mean, 0.0)
        self.assertAlmostEqual(diff.sdev, 0.0)
        diff = gvar_function([x, x + gvar(2,2)], f, [dfdx, 0]) - fcn(x)
        self.assertAlmostEqual(diff.mean, 0.0)
        self.assertAlmostEqual(diff.sdev, 0.0)
        x = gvar(dict(a='1(1)', b=['2(2)', '3(3)']))
        z = gvar(1,1)
        def fcn(x):
            return sin(x['a'] * x['b'][0]) * x['b'][1]
        f = fcn(x)
        dfdx = dict(a=f.deriv(x['a']), b=[f.deriv(x['b'][0]), f.deriv(x['b'][1])])
        f = f.mean
        diff = gvar_function(x, f, dfdx) - fcn(x)
        self.assertAlmostEqual(diff.mean, 0.0)
        self.assertAlmostEqual(diff.sdev, 0.0)
        x = gvar(['1(1)', '2(2)', '3(3)'])
        def fcn(x):
            return sin(x[0] + x[1]) * x[2]
        f = fcn(x)
        dfdx = np.array([f.deriv(x[0]), f.deriv(x[1]), f.deriv(x[2])])
        f = f.mean
        diff = gvar_function(x, f, dfdx) - fcn(x)
        self.assertAlmostEqual(diff.mean, 0.0)
        self.assertAlmostEqual(diff.sdev, 0.0)

    def test_wsum_der(self):
        """ wsum_der """
        gv = GVarFactory()
        x = gv([1,2],[[3,4],[4,5]])
        self.assert_arraysequal(wsum_der(np.array([10.,100]),x),[10.,100.])

    def test_wsum_gvar(self):
        """ wsum_gvar """
        gv = GVarFactory()
        x = gv([1,2],[[3,4],[4,5]])
        v = np.array([10.,100.])
        ws = wsum_gvar(v,x)
        self.assertAlmostEqual(ws.val,np.dot(v,mean(x)))
        self.assert_arraysclose(ws.der,wsum_der(v,x))

    def test_dotder(self):
        """ GVar.dotder """
        gv = GVarFactory()
        x = gv([1,2],[[3,4],[4,5]])*2
        v = np.array([10.,100.])
        self.assertAlmostEqual(x[0].dotder(v),20.)
        self.assertAlmostEqual(x[1].dotder(v),200.)

    def test_fmt(self):
        """ x.fmt """
        self.assertEqual(x.fmt(None), x.fmt(2))
        self.assertEqual(x.fmt(3),"%.3f(%d)"%(x.mean,round(x.sdev*1000)))
        self.assertEqual(y.fmt(3),"%.3f(%.3f)"%(y.mean,round(y.sdev,3)))
        self.assertEqual(gvar(".1234(341)").fmt(), "0.123(34)")
        self.assertEqual(gvar(" .1234(341)").fmt(), "0.123(34)")
        self.assertEqual(gvar(".1234(341) ").fmt(), "0.123(34)")
        self.assertEqual(gvar(".1234(341)").fmt(1), "0.1(0)")
        self.assertEqual(gvar(".1234(341)").fmt(5), "0.12340(3410)")
        self.assertEqual(gvar(".1234(0)").fmt(), "0.1234(0)")
        self.assertEqual(gvar("-.1234(341)").fmt(), "-0.123(34)")
        self.assertEqual(gvar("+.1234(341)").fmt(), "0.123(34)")
        self.assertEqual(gvar("-0.1234(341)").fmt(), "-0.123(34)")
        self.assertEqual(gvar("10(1.3)").fmt(), "10.0(1.3)")
        self.assertEqual(gvar("10.2(1.3)").fmt(), "10.2(1.3)")
        self.assertEqual(gvar("-10.2(1.3)").fmt(), "-10.2(1.3)")
        self.assertEqual(gvar("10(1.3)").fmt(0),"10(1)")
        self.assertEqual(gvar("1e-9 +- 1.23e-12").fmt(), "1.0000(12)e-09")
        self.assertEqual(gvar("1e-9 +- 1.23e-6").fmt(), '1(1230)e-09')
        self.assertEqual(gvar("1e+9 +- 1.23e+6").fmt(), "1.0000(12)e+09")
        self.assertEqual(gvar("1e-9 +- 0").fmt(), "1(0)e-09")
        self.assertEqual(gvar("0(0)").fmt(), "0(0)")
        self.assertEqual(gvar("1.234e-9 +- 0.129").fmt(), '1e-09 +- 0.13')
        self.assertEqual(gvar("1.23(4)e-9").fmt(), "1.230(40)e-09")
        self.assertEqual(gvar("1.23 +- 1.23e-12").fmt(), "1.2300000000000(12)")
        self.assertEqual(gvar("1.23 +- 1.23e-6").fmt(), "1.2300000(12)")
        self.assertEqual(gvar("1.23456 +- inf").fmt(3), "1.235 +- inf")
        self.assertEqual(gvar("1.23456 +- inf").fmt(), str(1.23456) + " +- inf")
        self.assertEqual(gvar("10.23 +- 1e-10").fmt(), "10.23000000000(10)")
        self.assertEqual(gvar("10.23(5.1)").fmt(), "10.2(5.1)")
        self.assertEqual(gvar("10.23(5.1)").fmt(-1),"10.23 +- 5.1")
        self.assertEqual(gvar(0.021, 0.18).fmt(), '0.02(18)')
        self.assertEqual(gvar(0.18, 0.021).fmt(), '0.180(21)')
        # boundary cases
        self.assertEqual(gvar(0.096, 9).fmt(), '0.1(9.0)')
        self.assertEqual(gvar(0.094, 9).fmt(), '0.09(9.00)')
        self.assertEqual(gvar(0.96, 9).fmt(), '1.0(9.0)')
        self.assertEqual(gvar(0.94, 9).fmt(), '0.9(9.0)')
        self.assertEqual(gvar(-0.96, 9).fmt(), '-1.0(9.0)')
        self.assertEqual(gvar(-0.94, 9).fmt(), '-0.9(9.0)')
        self.assertEqual(gvar(9.6, 90).fmt(), '10(90)')
        self.assertEqual(gvar(9.4, 90).fmt(), '9(90)')
        self.assertEqual(gvar(99.6, 91).fmt(), '100(91)')
        self.assertEqual(gvar(99.4, 91).fmt(), '99(91)')
        self.assertEqual(gvar(0.1, 0.0996).fmt(), '0.10(10)')
        self.assertEqual(gvar(0.1, 0.0994).fmt(), '0.100(99)')
        self.assertEqual(gvar(0.1, 0.994).fmt(), '0.10(99)')
        self.assertEqual(gvar(0.1, 0.996).fmt(), '0.1(1.0)')
        self.assertEqual(gvar(12.3, 9.96).fmt(), '12(10)')
        self.assertEqual(gvar(12.3, 9.94).fmt(), '12.3(9.9)')
        # 0 +- stuff
        self.assertEqual(gvar(0, 0).fmt(), '0(0)')
        self.assertEqual(gvar(0, 99.6).fmt(), '0(100)')
        self.assertEqual(gvar(0, 99.4).fmt(), '0(99)')
        self.assertEqual(gvar(0, 9.96).fmt(), '0(10)')
        self.assertEqual(gvar(0, 9.94).fmt(), '0.0(9.9)')
        self.assertEqual(gvar(0, 0.996).fmt(), '0.0(1.0)')
        self.assertEqual(gvar(0, 0.994).fmt(), '0.00(99)')
        self.assertEqual(gvar(0, 1e5).fmt(), '0.0(1.0)e+05')
        self.assertEqual(gvar(0, 1e4).fmt(), '0(10000)')
        self.assertEqual(gvar(0, 1e-5).fmt(), '0.0(1.0)e-05')
        self.assertEqual(gvar(0, 1e-4).fmt(), '0.00000(10)')


    def test_fmt2(self):
        """ fmt(x) """
        g1 = gvar(1.5,0.5)
        self.assertEqual(fmt(g1),g1.fmt())
        g2 = [g1,2*g1]
        fmtg2 = fmt(g2)
        self.assertEqual(fmtg2[0],g2[0].fmt())
        self.assertEqual(fmtg2[1],g2[1].fmt())
        g3 = dict(g1=g1,g2=g2)
        fmtg3 = fmt(g3)
        self.assertEqual(fmtg3['g1'],g1.fmt())
        self.assertEqual(fmtg3['g2'][0],g2[0].fmt())
        self.assertEqual(fmtg3['g2'][1],g2[1].fmt())

    def test_tabulate(self):
        """ tabulate(g) """
        g = BufferDict()
        g['scalar'] = gv.gvar('10.3(1.2)')
        g['vector'] = gv.gvar(['0.52(3)', '0.09(10)', '1.2(1)'])
        g['tensor'] = gv.gvar([
            ['0.01(50)', '0.001(20)', '0.033(15)'],
            ['0.001(20)', '2.00(5)', '0.12(52)'],
            ['0.007(45)', '0.237(4)', '10.23(75)'],
            ])
        table = gv.tabulate(g, ncol=2)
        correct = '\n'. join([
            '   key/index          value     key/index          value',
            '---------------------------  ---------------------------',
            '      scalar     10.3 (1.2)           1,0     0.001 (20)',
            '    vector 0     0.520 (30)           1,1     2.000 (50)',
            '           1      0.09 (10)           1,2      0.12 (52)',
            '           2      1.20 (10)           2,0     0.007 (45)',
            '  tensor 0,0      0.01 (50)           2,1    0.2370 (40)',
            '         0,1     0.001 (20)           2,2     10.23 (75)',
            '         0,2     0.033 (15)',
            ])
        self.assertEqual(table, correct, 'tabulate wrong')

    def test_partialvar(self):
        """ x.partialvar x.partialsdev fmt_errorbudget """
        gvar = gvar_factory()
        ## test basic functionality ##
        x = gvar(1,2)
        y = gvar(3,4)
        a,b = gvar([1,2],[[4,5],[5,16]])
        z = x+y+2*a+3*b
        self.assertEqual(z.var,x.var+y.var
                         +np.dot([2.,3.],np.dot(evalcov([a,b]),[2.,3.])))
        self.assertEqual(z.partialvar(x,y),x.var+y.var)
        self.assertEqual(z.partialvar(x,a),
                         x.var+np.dot([2.,3.],np.dot(evalcov([a,b]),[2.,3.])))
        self.assertEqual(z.partialvar(a),z.partialvar(a))
        ##
        ## test different arg types, fmt_errorbudget, fmt_values ##
        s = gvar(1,2)
        a = np.array([[gvar(3,4),gvar(5,6)]])
        d = BufferDict(s=gvar(7,8),v=[gvar(9,10),gvar(10,11)])
        z = s + sum(a.flat) + sum(d.flat)
        self.assertEqual(z.partialvar(s,a,d),z.var)
        self.assertEqual(z.partialvar(s),s.var)
        self.assertEqual(z.partialvar(a),sum(var(a).flat))
        self.assertEqual(z.partialvar(d),sum(var(d).flat))
        self.assertAlmostEqual(z.partialsdev(s,a,d),z.sdev)
        tmp = fmt_errorbudget(
            outputs=dict(z=z),
            inputs=collections.OrderedDict([
                ('a', a), ('s', s), ('d', d),
                ('ad', [a,d]), ('sa', [s,a]), ('sd', [s,d]), ('sad', [s,a,d])
                ]),
            ndecimal=1
            )
        out = "\n".join([
            "Partial % Errors:",
            "                   z",
            "--------------------",
            "        a:      20.6",
            "        s:       5.7",
            "        d:      48.2",
            "       ad:      52.5",
            "       sa:      21.4",
            "       sd:      48.6",
            "      sad:      52.8",
            "--------------------",
            "    total:      52.8",
            ""
            ])
        self.assertEqual(tmp,out,"fmt_errorbudget output wrong")
        tmp = fmt_errorbudget(
            outputs=dict(z=z),
            inputs=collections.OrderedDict([
                ('a', a), ('s', s), ('d', d),
                ('ad', [a,d]), ('sa', [s,a]), ('sd', [s,d]), ('sad', [s,a,d])
                ]),
            ndecimal=1, colwidth=25
            )
        out = "\n".join([
            "Partial % Errors:",
            "                                                 z",
            "--------------------------------------------------",
            "                       a:                     20.6",
            "                       s:                      5.7",
            "                       d:                     48.2",
            "                      ad:                     52.5",
            "                      sa:                     21.4",
            "                      sd:                     48.6",
            "                     sad:                     52.8",
            "--------------------------------------------------",
            "                   total:                     52.8",
            ""
            ])
        self.assertEqual(tmp,out,"fmt_errorbudget output wrong (with colwidth)")
        tmp = fmt_values(outputs=collections.OrderedDict([('s',s),('z',z)]),ndecimal=1)
        out = "\n".join([
        "Values:",
        "                  s: 1.0(2.0)            ",
        "                  z: 35.0(18.5)          ",
        ""
        ])
        self.assertEqual(tmp,out,"fmt_value output wrong")

    def test_errorbudget_warnings(self):
        """ fmt_errorbudget(...verify=True) """
        a, b, c = gvar(3 * ['1.0(1)'])
        b , c = (b+c) / 2., (b-c) /2.
        outputs = dict(sum=a+b+c)
        warnings.simplefilter('error')
        fmt_errorbudget(outputs=outputs, inputs=dict(a=a, b=b), verify=True)
        with self.assertRaises(UserWarning):
            fmt_errorbudget(outputs=outputs, inputs=dict(a=a, b=b, c=c), verify=True)
        with self.assertRaises(UserWarning):
            fmt_errorbudget(outputs=outputs, inputs=dict(a=a), verify=True)

    def test_der(self):
        """ x.der """
        global x,y
        self.assert_arraysequal(x.der,[1.,0.])
        self.assert_arraysequal(y.der,[0.,1.])
        z = x*y**2
        self.assert_arraysequal(z.der,[y.mean**2,2*x.mean*y.mean])

    def test_construct_gvar(self):
        """ construct_gvar """
        v = 2.0
        dv = np.array([0.,1.])
        cov = smat()
        cov.append_diag_m(np.array([[2.,4.],[4.,16.]]))
        y = gvar(v,np.array([1,0.]),cov)
        z = gvar(v,dv,cov)
        cov = evalcov([y,z])
        self.assertEqual(z.mean,v)
        self.assert_arraysequal(z.der,dv)
        self.assertEqual(z.var,np.dot(dv,np.dot(cov,dv)))
        self.assertEqual(z.sdev,sqrt(z.var))

        cov = smat()
        cov.append_diag_m(np.array([[2.,4.],[4.,16.]]))
        y = gvar(v,([1.], [0]), cov)
        z = gvar(v, ([1.], [1]), cov)
        cov = evalcov([y,z])
        self.assertEqual(z.mean,v)
        self.assert_arraysequal(z.der,dv)
        self.assertEqual(z.var,np.dot(dv,np.dot(cov,dv)))
        self.assertEqual(z.sdev,sqrt(z.var))

        # zero covariance
        x = gvar([1.], [[0.]])
        self.assertEqual(str(x), '[1(0)]')
        x = gvar(1, 0.)
        self.assertEqual(str(x), '1(0)')

    def t_gvar(self,args,xm,dx,xcov,xder):
        """ worker for test_gvar """
        gvar = gvar_factory()
        x = gvar(*args)
        self.assertEqual(x.mean,xm)
        self.assertEqual(x.sdev,dx)
        self.assert_arraysequal(evalcov([x]),xcov)
        self.assert_arraysequal(x.der,xder)

    def test_gvar(self):
        """ gvar """
        ## tests for arguments corresponding to a single gvar ##
        cov = smat()
        cov.append_diag_m(np.array([[1.,2.],[2.,16.]]))
        x = gvar(2.,np.array([0.,1.]),cov)
        arglist = [(["4.125(250)"],4.125,0.25,[[.25**2]],[1.0],'"x(dx)"'), #]
                (["-4.125(2.0)"],-4.125,2.0,[[2.**2]],[1.0],'"x(dx)"'),
                (["4.125 +- 0.5"],4.125,0.5,[[0.5**2]],[1.0],'"x +- dx"'),
                ([x],x.mean,x.sdev,evalcov([x]),x.der,"x"),
                ([2.0],2.0,0.0,[[0.0]],[1.0],"x"),
                ([(2.0,4.0)],2.,4.,[[4.**2]],[1.0],"(x,dx)"),
                ([2.0,4.0],2.,4.,[[4.**2]],[1.0],"x,dx"),
                ([x.mean,x.der,x.cov],x.mean,x.sdev,evalcov([x]),x.der,"x,der,cov")
                 ]
        for a in arglist:
            self.label = "gvar(%s)" % a[0]
            self.t_gvar(a[0],a[1],a[2],a[3],a[4])

        # tests involving a single argument that is sequence
        x = gvar([[(0,1),(1,2)],[(3,4),(5,6)],[(7,8),(9,10)]])
        y = np.array([[gvar(0,1),gvar(1,2)],[gvar(3,4),gvar(5,6)],
                    [gvar(7,8),gvar(9,10)]])
        self.assert_gvclose(x,y)
        x = gvar([[["0(1)"],["2(3)"]]])
        y = np.array([[[gvar(0,1)],[gvar(2,3)]]])
        self.assert_gvclose(x,y)
        x = gvar([[1.,2.],[3.,4.]])
        y = np.array([[gvar(1.,0),gvar(2.,0)],[gvar(3.,0),gvar(4.,0)]])
        self.assert_gvclose(x,y)
        x = gvar([gvar(0,1),gvar(2,3)])
        y = np.array([gvar(0,1),gvar(2,3)])
        self.assert_gvclose(x,y)

        # tests involving dictionary arguments
        x = gvar(dict(a=1,b=[2,3]), dict(a=10, b=[20,30]))
        y = dict(a=gvar(1,10), b=[gvar(2,20), gvar(3,30)])
        self.assert_gvclose(x,y)
        a, b = gvar([1,2],[10,20])
        a, b = a+b, a-b
        x = gvar([a, a+b, b, b-a])
        y = BufferDict()
        y['a'] = [a, a+b]
        y['b'] = [b, b-a]
        self.assert_gvclose(y.flat, x)
        z = gvar(mean(y), evalcov(y))
        self.assert_gvclose(z.flat, y.flat)
        self.assert_arraysclose(evalcov(z.flat), evalcov(x))

    def test_phased_construction(self):
        " gvar(ymean, ycov, x, xycov)"
        assert_almost_equal = np.testing.assert_almost_equal
        acov = [[4, .1], [.1, 9]]
        a = gvar([2, 3], acov)
        b = gvar('1(1)')

        # test 1
        accov = [[12.], [13.]]
        c = gvar([4], [[16]], a, accov)
        abc = BufferDict(a=a, b=b, c=c)
        assert_almost_equal(
            evalcov(abc.buf), 
            [[ 4,   0.1, 0., 12.],
            [ 0.1, 9.,  0., 13.],
            [ 0.,  0.,  1.,  0.],
            [12., 13.,  0., 16.]]
            )
        abcblk = evalcov_blocks(abc.buf, compress=True)
        assert_almost_equal(abcblk[0][0], [2])  # b
        assert_almost_equal(abcblk[0][1], [1.])
        assert_almost_equal(abcblk[1][0], [0, 1, 3]) # a and c
        assert_almost_equal(
            abcblk[1][1], 
            [[ 4.,   0.1, 12.],
            [ 0.1,  9.,  13.],
            [12.,  13.,  16.]]
            )
        
        abccov = evalcov(abc)
        assert_almost_equal(abccov['a', 'a'], acov)
        assert_almost_equal(abccov['c', 'a'], np.transpose(accov))
        assert_almost_equal(abccov['a', 'c'], accov)
        assert_almost_equal(abccov['c', 'c'], [[16]])
        assert_almost_equal(abccov['b', 'b'], [[1]])
        assert_almost_equal(abccov['b', 'c'], [[0]])
        assert_almost_equal(abccov['b', 'a'], [[0, 0]])
        assert_almost_equal(abccov['c', 'b'], [[0]])
        assert_almost_equal(abccov['a', 'b'], [[0], [0]])

        # test 2 (out of order)
        d = gvar([5.], [[25.]])
        adcov = [[14.], [15.]]
        c = gvar([4], [[16]], list(reversed(a)), list(reversed(adcov)))
        adc = BufferDict(a=a, d=d, c=c)
        assert_almost_equal(
            evalcov(adc.buf), 
            [[ 4,   0.1, 0., 14.],
            [ 0.1, 9.,  0., 15.],
            [ 0.,  0., 25.,  0.],
            [14., 15.,  0., 16.]]
            )
        adcblk = evalcov_blocks(adc.buf, compress=True)
        assert_almost_equal(adcblk[0][0], [2])  # b
        assert_almost_equal(adcblk[0][1], [5.])
        assert_almost_equal(adcblk[1][0], [0, 1, 3]) # a and d
        assert_almost_equal(
            adcblk[1][1], 
            [[ 4.,   0.1, 14.],
            [ 0.1,  9.,  15.],
            [14.,  15.,  16.]]    
            )
        adccov = evalcov(adc)
        assert_almost_equal(adccov['a', 'a'], acov)
        assert_almost_equal(adccov['c', 'a'], np.transpose(adcov))
        assert_almost_equal(adccov['a', 'c'], adcov)
        assert_almost_equal(adccov['c', 'c'], [[16]])
        assert_almost_equal(adccov['d', 'd'], [[25]])
        assert_almost_equal(adccov['d', 'c'], [[0]])
        assert_almost_equal(adccov['d', 'a'], [[0, 0]])
        assert_almost_equal(adccov['c', 'd'], [[0]])
        assert_almost_equal(adccov['a', 'd'], [[0], [0]])

        # test 3 (partial overlap)
        ecov = [[36, .2], [.2, 49]]
        aecov = [[16., 17.]]
        e = gvar([6, 7], ecov, [a[1]], aecov)
        ade = list(a) + [d[0]] + list(e)
        adeblk = evalcov_blocks(ade, compress=True)
        assert_almost_equal(adeblk[0][0], [2])
        assert_almost_equal(adeblk[0][1], [5])
        assert_almost_equal(adeblk[1][0], [0, 1, 3, 4])
        assert_almost_equal(
            adeblk[1][1], 
            [[4.,  0.1, 0.,   0.],
            [0.1, 9., 16.,  17.],
            [0., 16., 36.,   0.2],
            [0., 17.,  0.2, 49.]]
            )
        assert_almost_equal(
            evalcov(ade),
            [[ 4. ,  0.1,  0. ,  0. ,  0. ],
             [ 0.1,  9. ,  0. , 16. , 17. ],
             [ 0. ,  0. , 25. ,  0. ,  0. ],
             [ 0. , 16. ,  0. , 36. ,  0.2],
             [ 0. , 17. ,  0. ,  0.2, 49. ]]
            )

        # test 4 (big matrices)
        cov = numpy.random.uniform(size=(100,100))
        cov = cov.T @ cov
        mean = numpy.linspace(0, 1., 100)
        g = gvar(mean, cov)
        h = gvar('1(8)')
        z = g[0] + g[1]  # function of g evaluated before m defined but will be correlated
        gmcov = np.linspace(0.,1., 100).reshape(1,-1)
        m = gvar(mean, cov, [g[0]], gmcov)
        assert_almost_equal(evalcov([z, m[-1]])[0, 1],gmcov[0, -1])
        ghm = BufferDict(g=g, h=h, m=m)
        ghmcov = evalcov(ghm)
        assert_almost_equal(ghmcov['g', 'g'], cov)
        assert_almost_equal(ghmcov['m', 'm'], cov)
        assert_almost_equal(ghmcov['g', 'm'][:1, :], gmcov)
        assert_almost_equal(ghmcov['m', 'g'][:, :1], gmcov.T)

        blks = iter(evalcov_blocks(ghm, compress=True))
        blk = next(blks)
        assert_almost_equal(blk[0][0], 100)
        assert_almost_equal(blk[1][0], 8.)
        blk = next(blks)
        assert_almost_equal(blk[0][:100], numpy.arange(100))
        assert_almost_equal(blk[0][100:], 101 + numpy.arange(100))
        assert_almost_equal(blk[1][0, 100:], gmcov[0])
        assert_almost_equal(blk[1][100:, 0], gmcov[0])
        with self.assertRaises(StopIteration):
            next(blks)

        # test misc. exceptions
        with self.assertRaises(ValueError):
            f = gvar([8], [[64.]], d, [[200.]], verify=True)
        with self.assertRaises(ValueError):
            f = gvar([8], [[64.]], d, [[.2, .3]])
        with self.assertRaises(ValueError):
            f = gvar([8], [[64.]], d[0], [[.2, .3]])
        with self.assertRaises(ValueError):
            f = gvar(8, [[64.]], d, [[.2, .3]])
        with self.assertRaises(ValueError):
            f = gvar([8], [[64.]], d, [[.2], [.3]])

    def _tst_compare_evalcovs(self):
        " evalcov evalcov_blocks evalcov_blocks_dense agree "
        def reconstruct(x, blocks, compress):
            ans = np.zeros((len(x), len(x)), float)
            if compress:
                idx, sdev = blocks[0]
                ans[idx, idx] = sdev ** 2
                n = (len(idx), len(blocks))
                blocks = blocks[1:]
            else:
                n = len(blocks)
            for idx, bcov in blocks:
                ans[idx[:,None], idx[:]] = bcov 
            return ans, n
        for setup, compress in [
            ("N=10; a=np.random.rand(N,N); x=gv.gvar(N*[1.],a.dot(a.T)); x=a.dot(x);", True),
            ("N=10; a=np.random.rand(N,N); x=gv.gvar(N*[1.],a.dot(a.T)); x=a.dot(x);", False),
            ("N=10; x=gv.gvar(N*[1.],N*[1.]);", True),
            ("N=10; x=gv.gvar(N*[1.],N*[1.]);", False),
            ("N=10;  x=gv.gvar(N*[1.],N*[1.]); x[1:] += x[:-1];", True),
            ("N=10;  x=gv.gvar(N*[1.],N*[1.]); x[1:] += x[:-1];", False),
            ("N=10; a=np.random.rand(N,N); x=gv.gvar(N*[1.],a.dot(a.T));", True),
            ("N=10; a=np.random.rand(N,N); x=gv.gvar(N*[1.],a.dot(a.T));", False),
            ]:
            tmp = locals()
            exec(setup, globals(), tmp)
            x = tmp['x']
            ec = gv.evalcov(x)
            ecb, necb = reconstruct(x, gv.evalcov_blocks(x, compress=compress), compress)
            ecbd, necbd = reconstruct(x, gv.evalcov_blocks_dense(x, compress=compress), compress)
            np.testing.assert_allclose(ec, ecbd)
            np.testing.assert_allclose(ec, ecb)
            self.assertEqual(necb, necbd)
            # print(necb)

    def test_compare_evalcovs(self):
        " evalcov evalcov_blocks evalcov_blocks_dense agree "
        self._tst_compare_evalcovs()
        tmp, gv._CONFIG['evalcov_blocks'] = gv._CONFIG['evalcov_blocks'], 1
        self._tst_compare_evalcovs()
        gv._CONFIG['evalcov_blocks'] = tmp

    def test_gvar_blocks(self):
        " block structure created by gvar.gvar "
        def blockid(g):
            return g.cov.blockid(g.internaldata[1].indices()[0])
        x = gvar([1., 2., 3.], [1., 1., 1.])
        id = [blockid(xi) for xi in x]
        self.assertNotEqual(id[0], id[1])
        self.assertNotEqual(id[0], id[2])
        self.assertNotEqual(id[1], id[2])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertNotEqual(id[0], id[1])
        self.assertNotEqual(id[0], id[2])
        self.assertNotEqual(id[1], id[2])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0.1, 0.], [0.1, 1., 0.], [0., 0., 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertEqual(id[0], id[1])
        self.assertNotEqual(id[0], id[2])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0., 0.1], [0.0, 1., 0.0], [0.1, 0., 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertEqual(id[0], id[2])
        self.assertNotEqual(id[0], id[1])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0., 0.0], [0.0, 1., 0.1], [0.0, 0.1, 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertEqual(id[1], id[2])
        self.assertNotEqual(id[0], id[1])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0.1, 0.0], [0.1, 1., 0.1], [0.0, 0.1, 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertEqual(id[1], id[2])
        self.assertEqual(id[0], id[1])
        idlast = max(id)
        x = gvar([1., 2., 3.], [[1., 0.1, 0.1], [0.1, 1., 0.1], [0.1, 0.1, 1.]], fast=False)
        id = [blockid(xi) for xi in x]
        self.assertEqual(min(id), idlast + 1)
        self.assertEqual(id[1], id[2])
        self.assertEqual(id[0], id[1])
        

    def test_gvar_verify(self):
        " gvar(x, xx, verify=True) "
        # case that does not generate an error
        gvar([1., 2.], [[1., 2./10], [2./10., 1.]])
        with self.assertRaises(ValueError):
            gvar([1., 2.], [[1., .5], [.6, 1.]])
        # cases that do generate errors
        for a,b in [
            (1., -1.), ([1., 2.], [2., -2.]), 
            ([1., 2.], [[1., 2.], [2., 1.]]),
            ]:
            with self.assertRaises(ValueError):
                gvar(a, b, verify=True)

    def test_asgvar(self):
        """ gvar functions as asgvar """
        z = gvar(x)
        self.assertTrue(z is x)
        z = gvar("2.00(25)")
        self.assertEqual(z.mean,2.0)
        self.assertEqual(z.sdev,0.25)

    def test_basis5(self):
        """ gvar(x,dx) """
        xa = np.array([[2.,4.]])
        dxa = np.array([[16.,64.]])
        x = gvar(xa,dxa)
        xcov = evalcov(x)
        self.assertEqual(xcov.shape,2*x.shape)
        for xai,dxai,xi in zip(xa.flat,dxa.flat,x.flat):
            self.assertEqual(xai,xi.mean)
            self.assertEqual(dxai,xi.sdev)
        self.assertEqual(np.shape(xa),np.shape(x))
        xcov = xcov.reshape((2,2))
        self.assert_arraysequal(xcov.diagonal(),[dxa[0,0]**2,dxa[0,1]**2])

    def test_basis6(self):
        """ gvar(x,cov) """
        xa = np.array([2.,4.])
        cov = np.array([[16.,64.],[64.,4.]])
        x = gvar(xa,cov)
        xcov = evalcov(x)
        for xai,dxai2,xi in zip(xa.flat,cov.diagonal().flat,x.flat):
            self.assertEqual(xai,xi.mean)
            self.assertEqual(dxai2,xi.sdev**2)
        self.assertEqual(np.shape(xa),np.shape(x))
        self.assert_arraysequal(xcov,cov.reshape((2,2)))

    def test_mean_sdev_var(self):
        """ mean(g) sdev(g) var(g) """
        def compare(x,y):
            self.assertEqual(set(x.keys()),set(y.keys()))
            for k in x:
                self.assertEqual(np.shape(x[k]),np.shape(y[k]))
                if np.shape(x[k])==():
                    self.assertEqual(x[k],y[k])
                else:
                    self.assertTrue(all(x[k]==y[k]))
        # dictionaries of GVars
        a = dict(x=gvar(1,2),y=np.array([gvar(3,4),gvar(5,6)]))
        a_mean = dict(x=1.,y=np.array([3.,5.]))
        a_sdev = dict(x=2.,y=np.array([4.,6.]))
        a_var = dict(x=2.**2,y=np.array([4.**2,6.**2]))
        compare(a_mean,mean(a))
        compare(a_sdev,sdev(a))
        compare(a_var,var(a))
        # arrays of GVars
        b = np.array([gvar(1,2),gvar(3,4),gvar(5,6)])
        b_mean = np.array([1.,3.,5.])
        b_sdev = np.array([2.,4.,6.])
        self.assertTrue(all(b_mean==mean(b)))
        self.assertTrue(all(b_sdev==sdev(b)))
        self.assertTrue(all(b_sdev**2==var(b)))
        # single GVar
        self.assertEqual(mean(gvar(1,2)),1.)
        self.assertEqual(sdev(gvar(1,2)),2.)
        self.assertEqual(var(gvar(1,2)),4.)
        # single non-GVar
        self.assertEqual(mean(1.25), 1.25)
        self.assertEqual(sdev(1.25), 0.0)
        self.assertEqual(var(1.25), 0.0)
        b = np.array([gvar(1,2), 3.0, gvar(5,6)])
        self.assertTrue(all(mean(b)==[1., 3., 5.]))
        self.assertTrue(all(sdev(b)==[2., 0., 6.]))
        self.assertTrue(all(var(b)==[4., 0., 36.]))

    def test_sdev_var(self):
        " sdev var from covariance matrices "
        a = np.random.rand(10, 10)
        cov = a.dot(a.T)
        x = gvar(cov.shape[0] * [1], cov)
        xd = gvar(cov.shape[0] * [1], cov.diagonal() ** 0.5)
        xt = a.dot(x)
        covt = a.dot(cov.dot(a.T))
        for nthreshold in [1, 1000]:
            tmp, gv._CONFIG['var'] = gv._CONFIG['var'], nthreshold
            numpy.testing.assert_allclose(var(x), cov.diagonal())
            numpy.testing.assert_allclose(sdev(x), cov.diagonal() ** 0.5)
            numpy.testing.assert_allclose(var(xd), cov.diagonal())
            numpy.testing.assert_allclose(sdev(xd), cov.diagonal() ** 0.5)
            numpy.testing.assert_allclose(var(xt), covt.diagonal())
            numpy.testing.assert_allclose(sdev(xt), covt.diagonal() ** 0.5)
            gv._CONFIG['var'] = tmp

    def test_empty(self):
        self.assertEqual(mean([]).size, 0)
        self.assertEqual(sdev([]).size, 0)
        self.assertEqual(var([]).size, 0)
        self.assertEqual(evalcov([]).size, 0)
        self.assertEqual(evalcorr([]).size, 0)
        for i, b in evalcov_blocks([]):
            self.assertEqual(i.size, 0)
            self.assertEqual(b.size, 0)

    def test_uncorrelated(self):
        """ uncorrelated(g1, g2) """
        a = dict(x=gvar(1,2),y=np.array([gvar(3,4),gvar(5,6)]))
        b = dict(x=gvar(1,2),y=np.array([gvar(3,4),gvar(5,6)]))
        c = np.array([gvar(1,2),gvar(3,4),gvar(5,6)])
        d = np.array([gvar(1,2),gvar(3,4),gvar(5,6)])
        self.assertTrue(uncorrelated(a,b))
        self.assertTrue(not uncorrelated(a,a))
        self.assertTrue(uncorrelated(a['x'],a['y']))
        self.assertTrue(not uncorrelated(a['x'],a))
        self.assertTrue(uncorrelated(a,c))
        self.assertTrue(uncorrelated(c,a))
        self.assertTrue(uncorrelated(c,d))
        self.assertTrue(not uncorrelated(c,c))
        a['x'] += b['x']
        self.assertTrue(not uncorrelated(a,b))
        d += c[0]
        self.assertTrue(not uncorrelated(c,d))
        self.assertTrue(not uncorrelated(a,b['x']))
        a, b = gvar([1,2],[[1,.1],[.1,4]])
        c = 2*a
        self.assertTrue(not uncorrelated(a,c))
        self.assertTrue(not uncorrelated(b,c))
        self.assertTrue(not uncorrelated(a,b))

    def test_deriv(self):
        global x, y, gvar
        f = 2 * x ** 2. + 3 * y
        self.assertEqual(deriv(f, x), 4. * x.mean)
        self.assertEqual(deriv(f, y), 3.)
        with self.assertRaises(ValueError):
            deriv(f, x+y)
        self.assertEqual(f.deriv(x), 4. * x.mean)
        self.assertEqual(f.deriv(y), 3.)
        with self.assertRaises(ValueError):
            f.deriv(x+y)
        self.assertEqual(deriv(f, [x, y]).tolist(), [4. * x.mean, 3.])
        self.assertEqual(deriv(f, [[x], [y]]).tolist(), [[4. * x.mean], [3.]])
        self.assertEqual(deriv([f], [x, y]).tolist(), [[4. * x.mean, 3.]])
        f = [2 * x + 3 * y, 4 * x]
        self.assertEqual(deriv(f, x).tolist(), [2., 4.])
        self.assertEqual(deriv(f, y).tolist(), [3., 0.])
        with self.assertRaises(ValueError):
            deriv(f, x+y)
        df = deriv(f, [[x, y]])
        self.assertEqual(df.tolist(), [[[2., 3.]], [[4., 0.]]])
        f = BufferDict([('a', 2 * x + 3 * y), ('b', 4 * x)])
        self.assertEqual(deriv(f, x), BufferDict([('a',2.), ('b',4.)]))
        self.assertEqual(deriv(f, y), BufferDict([('a',3.), ('b',0.)]))
        df = deriv(f, [x, y])
        self.assertEqual(df['a'].tolist(), [2., 3.])
        self.assertEqual(df['b'].tolist(), [4., 0.])        
        with self.assertRaises(ValueError):
            deriv(f, x+y)

    def test_correlate(self):
        " correlate(g, corr) "
        x = gvar([1., 2.], [[64., 4.], [4., 16.]])
        xmean = mean(x)
        xsdev = sdev(x)
        xx = correlate(gvar(xmean, xsdev), evalcorr(x))
        self.assert_arraysequal(xmean, mean(xx))
        self.assert_arraysequal(evalcov(x), evalcov(xx))
        # with upper, verify
        corr = evalcorr(x)
        corr[1, 0] = 0.
        corr[1, 1] = 10.
        with self.assertRaises(ValueError):
            xx = correlate(gvar(xmean, xsdev), corr, upper=False, verify=True)
        xx = correlate(gvar(xmean, xsdev), corr, upper=True, verify=True)
        self.assert_arraysequal(xmean, mean(xx))
        self.assert_arraysequal(evalcov(x), evalcov(xx))
        # with lower, verify
        corr = evalcorr(x)
        corr[0, 1] = 0.
        corr[0, 0] = 0.
        with self.assertRaises(ValueError):
            xx = correlate(gvar(xmean, xsdev), corr, lower=False, verify=True)
        xx = correlate(gvar(xmean, xsdev), corr, lower=True, verify=True)
        self.assert_arraysequal(xmean, mean(xx))
        self.assert_arraysequal(evalcov(x), evalcov(xx))
        # matrix
        x.shape = (2, 1)
        xmean = mean(x)
        xsdev = sdev(x)
        xx = correlate(gvar(xmean, xsdev), evalcorr(x))
        self.assert_arraysequal(xmean, mean(xx))
        self.assert_arraysequal(evalcov(x), evalcov(xx))
        # dict
        y = BufferDict()
        y['a'] = x[0, 0]
        y['b'] = x
        ymean = mean(y)
        ysdev = sdev(y)
        yy = correlate(gvar(ymean, ysdev), evalcorr(y))
        for k in y:
            self.assert_arraysequal(mean(y[k]), mean(yy[k]))
        ycov = evalcov(y)
        yycov = evalcov(yy)
        for k in ycov:
            self.assert_arraysequal(ycov[k], yycov[k])

    def test_evalcorr(self):
        " evalcorr(array) "
        x = gvar([1., 2.], [[64., 4.], [4., 16.]])
        a, b = x
        c = evalcorr([a, b])
        self.assertEqual(corr(a,b), 1/8.)
        self.assert_arraysequal(c, [[1., 1/8.], [1/8., 1.]])
        c = evalcorr(x.reshape(2, 1))
        self.assertEqual(c.shape, 2 * (2, 1))
        self.assert_arraysequal(c.reshape(2,2), [[1., 1/8.], [1/8., 1.]])
        y = dict(a=x[0], b=x)
        c = evalcorr(y)
        self.assertEqual(c['a', 'a'], [[1]])
        self.assert_arraysequal(c['a', 'b'], [[1., 1/8.]])
        self.assert_arraysequal(c['b', 'a'], [[1.], [1./8.]])
        self.assert_arraysequal(c['b', 'b'], [[1., 1/8.], [1/8., 1.]])

    def _tst_evalcov1(self):
        """ evalcov(array) """
        a,b = gvar([1.,2.],[[64.,4.],[4.,36.]])
        c = evalcov([a,b/2])
        self.assert_arraysequal(c,[[ 64.,2.],[ 2.,9.]])
        self.assertEqual(cov(a, b/2), 2.)
        c = evalcov([a/2,b])
        self.assert_arraysequal(c,[[ 16.,2.],[ 2.,36.]])
        z = gvar(8.,32.)
        c = evalcov([x,y,z])
        self.assert_arraysequal(c[:2,:2],evalcov([x,y]))
        self.assertEqual(c[2,2],z.var)
        self.assert_arraysequal(c[:2,2],np.zeros(np.shape(c[:2,2])))
        self.assert_arraysequal(c[2,:2],np.zeros(np.shape(c[2,:2])))
        rc = evalcov([x+y/2,2*x-y])
        rotn = np.array([[1.,1/2.],[2.,-1.]])
        self.assert_arraysequal(rc,np.dot(rotn,np.dot(c[:2,:2],rotn.transpose())))

    def test_evalcov1(self):
        """ evalcov(array) """
        self._tst_evalcov1()
        tmp, gv._CONFIG['evalcov'] = gv._CONFIG['evalcov'], 1
        self._tst_evalcov1()
        gv._CONFIG['evalcov'] = tmp

    def _tst_evalcov2(self):
        """ evalcov(dict) """
        c = evalcov({0:x + y / 2, 1:2 * x - y})
        rotn = np.array([[1., 1/2.], [2., -1.]])
        cz = np.dot(rotn, np.dot(evalcov([x, y]), rotn.transpose()))
        c = [[c[0,0][0,0], c[0,1][0,0]], [c[1,0][0,0], c[1,1][0,0]]]
        self.assert_arraysequal(c, cz)
        c = evalcov(dict(x=x, y=[x, y]))
        self.assert_arraysequal(c['y','y'], evalcov([x, y]))
        self.assertEqual(c['x','x'], [[x.var]])
        self.assert_arraysequal(c['x','y'], [[x.var, evalcov([x,y])[0,1]]])
        self.assert_arraysequal(c['y','x'], c['x','y'].T)

    def test_evalcov2(self):
        """ evalcov(dict) """
        self._tst_evalcov2()
        tmp, gv._CONFIG['evalcov'] = gv._CONFIG['evalcov'], 1
        self._tst_evalcov2()
        gv._CONFIG['evalcov'] = tmp

    def test_sample(self):
        " sample(g) "
        glist = [
            gvar('1(2)'), gv.gvar(['10(2)', '20(2)']) * gv.gvar('1(1)'),
            gv.gvar(dict(a='100(2)', b=['200(2)', '300(2)'])),
            ]
        for g in glist:
            ranseed(12)
            svdcut = 0.9
            s1 = sample(g, svdcut=svdcut)
            ranseed(12)
            s2 = next(raniter(g, svdcut=svdcut))
            self.assertEqual(str(s1), str(s2))
            ranseed(12)
            eps = 0.9
            s1 = sample(g, eps=eps)
            ranseed(12)
            s2 = next(raniter(g, eps=eps))
            self.assertEqual(str(s1), str(s2))

    @unittest.skipIf(FAST,"skipping test_raniter for speed")
    def test_raniter(self):
        """ raniter """
        global x,y,gvar
        n = 1000
        rtol = 5./n**0.5
        x = gvar(x.mean, x.sdev)
        y = gvar(y.mean, y.sdev)
        f = raniter([x,y],n)
        ans = [fi for fi in f]
        # print(x, y, evalcov([x,y]))
        # print (ans)
        ans = np.array(ans).transpose()
        self.assertAlmostEqual(ans[0].mean(),x.mean,delta=x.sdev*rtol)
        self.assertAlmostEqual(ans[1].mean(),y.mean,delta=y.sdev*rtol)
        self.assert_arraysclose(np.cov(ans[0],ans[1]),evalcov([x,y]),rtol=rtol)

    @unittest.skipIf(FAST,"skipping test_raniter2 for speed")
    def test_raniter2(self):
        """ raniter & svd """
        for svdcut in [1e-20,1e-2]:
            pr = BufferDict()
            pr[0] = gvar(1,1)
            pr[1] = pr[0]+gvar(0.1,1e-4)
            a0 = []
            da = []
            n = 10000
            rtol = 5./n**0.5            # 5 sigma
            for p in raniter(pr,n,svdcut=svdcut):
                a0.append(p[0])
                da.append(p[1]-p[0])
            a0 = np.array(a0)
            da = np.array(da)
            dda = max(2*svdcut**0.5,1e-4) # largest eig is 2 -- hence 2*sqrt(svdcut)
            self.assertAlmostEqual(da.std(),dda,delta=rtol*dda)
            self.assertAlmostEqual(a0.mean(),1.,delta=rtol)
            self.assertAlmostEqual(da.mean(),0.1,delta=rtol*da.std())

    def test_raniter4(self):
        """ raniter with svdcut<0 """
        svdcut = -0.9e-1     # only one eigenmode survives
        cov = ( 
            np.array([[1,1,1], [1,1,1], [1,1,1]]) 
            + np.array([[1,0,0],[0,1,0],[0,0,1]]) * 1e-1
            )
        g = gvar(np.zeros(len(cov)), cov)
        y = next(raniter(g, svdcut=svdcut))
        np.testing.assert_allclose(y, y[0] * np.array([1, 1, 1]))

    def test_bootstrap_iter(self):
        """ bootstrap_iter """
        p = BufferDict()
        p = gvar(1,1)*np.array([1,1])+gvar(0.1,1e-4)*np.array([1,-1])
        p_sw = np.array([p[0]+p[1],p[0]-p[1]])/2.
        p_cov = evalcov(p_sw.flat)
        p_mean = mean(p_sw.flat)
        p_sdev = mean(p_sw.flat)
        for pb in bootstrap_iter(p,3,svdcut=1e-20):
            pb_sw = np.array([pb[0]+pb[1],pb[0]-pb[1]])/2.
            self.assert_arraysclose(p_cov,evalcov(pb_sw.flat))
            dp = np.abs(mean(pb_sw.flat)-p_mean)
            self.assertGreater(p_sdev[0]*5,dp[0])
            self.assertGreater(p_sdev[1]*5,dp[1])
        for pb in bootstrap_iter(p,3,svdcut=1e-2):
            pb_sw = np.array([pb[0]+pb[1],pb[0]-pb[1]])/2.
            pb_mean = mean(pb_sw.flat)
            pb_sdev = sdev(pb_sw.flat)
            self.assertAlmostEqual(pb_sdev[0],p_sdev[0])
            self.assertAlmostEqual(pb_sdev[1],p_sdev[0]/10.)
            dp = abs(pb_mean-p_mean)
            self.assertGreater(p_sdev[0]*5,dp[0])
            self.assertGreater(p_sdev[0]*5./10.,dp[1])

    def test_raniter3(self):
        """ raniter & BufferDict """
        pr = BufferDict()
        pr['s'] = gvar(2.,4.)
        pr['v'] = [gvar(4.,8.),gvar(8.,16.)]
        pr['t'] = [[gvar(16.,32.),gvar(32.,64.)],[gvar(64.,128.),gvar(128.,256.)]]
        pr['ps'] = gvar(256.,512.)
        nran = 49
        delta = 5./nran**0.5     # 5 sigma
        prmean = mean(pr)
        prsdev = sdev(pr)
        ans = dict((k,[]) for k in pr)
        for p in raniter(pr,nran):
            for k in p:
                ans[k].append(p[k])
        for k in p:
            ansmean = np.mean(ans[k],axis=0)
            anssdev = np.std(ans[k],axis=0)
            pkmean = prmean[k]
            pksdev = prsdev[k]
            self.assertAlmostEqual(np.max(np.abs((pkmean-ansmean)/pksdev)),0.0,delta=delta)
            self.assertAlmostEqual(np.max(np.abs((pksdev-anssdev)/pksdev)),0.0,delta=delta)

    def test_SVD(self):
        """ SVD """
        # error system
        with self.assertRaises(ValueError):
            SVD([1,2])
        # non-singular
        x,y = gvar([1,1],[1,4])
        cov = evalcov([(x+y)/2**0.5,(x-y)/2**0.5])
        s = SVD(cov)
        e = s.val
        v = s.vec
        k = s.kappa
        self.assert_arraysclose(e,[1.,16.],rtol=1e-6)
        self.assert_arraysclose(e[0]/e[1],1./16.,rtol=1e-6)
        self.assert_arraysclose(np.dot(cov,v[0]),e[0]*v[0],rtol=1e-6)
        self.assert_arraysclose(np.dot(cov,v[1]),e[1]*v[1],rtol=1e-6)
        self.assertTrue(np.allclose([np.dot(v[0],v[0]),np.dot(v[1],v[1]),np.dot(v[0],v[1])],
                            [1.,1.,0],rtol=1e-6))
        self.assert_arraysclose(sum(np.outer(vi,vi)*ei for ei,vi in zip(e,v)),
                             cov,rtol=1e-6)
        self.assertAlmostEqual(s.kappa,1/16.)

        # on-axis 0
        cov = np.array([[4.,0.0], [0.0, 0.0]])
        s = SVD(cov, rescale=False, svdcut=None)
        self.assertTrue(np.all(s.val == [0., 4.]))

        # singular case
        cov = evalcov([(x+y)/2**0.5,(x-y)/2**0.5,x,y])
        s = SVD(cov)
        e,v,k = s.val,s.vec,s.kappa
        self.assert_arraysclose(e,[0,0,2.,32.],rtol=1e-6)
        self.assert_arraysclose(sum(np.outer(vi,vi)*ei for ei,vi in zip(e,v)),
                            cov,rtol=1e-6)
        s = SVD(cov,svdcut=1e-4,compute_delta=True)
        e,v,k,d = s.val,s.vec,s.kappa,s.delta
        self.assert_arraysclose(e,[32*1e-4,32*1e-4,2.,32.],rtol=1e-6)
        ncov = sum(np.outer(vi,vi)*ei for ei,vi in zip(e,v))-evalcov(d)
        self.assert_arraysclose(cov,ncov,rtol=1e-6)
        s = SVD(cov,svdnum=2,compute_delta=True)
        e,v,k,d = s.val,s.vec,s.kappa,s.delta
        self.assert_arraysclose(e,[2.,32.],rtol=1e-6)
        self.assertTrue(d is None)
        s = SVD(cov,svdnum=3,svdcut=1e-4,compute_delta=True)
        e,v,k,d = s.val,s.vec,s.kappa,s.delta
        self.assert_arraysclose(e,[32*1e-4,2.,32.],rtol=1e-6)

        # s.delta s.decomp
        for rescale in [False,True]:
            mat = [[1.,.25],[.25,2.]]
            s = SVD(mat,rescale=rescale)
            if rescale==False:
                self.assertTrue(s.D is None)
            else:
                diag = np.diag(s.D)
                self.assert_arraysclose(np.diag(np.dot(diag,np.dot(mat,diag))),
                                    [1.,1.])
            self.assert_arraysclose(mat, sum(np.outer(wj,wj) for wj in s.decomp(1)))
            s = SVD(mat,svdcut=0.9,compute_delta=True,rescale=rescale)
            mout = sum(np.outer(wj,wj) for wj in s.decomp(1))
            self.assert_arraysclose(mat+evalcov(s.delta),mout)
            self.assertTrue(not np.allclose(mat,mout))
            s = SVD(mat,rescale=rescale)
            minv = sum(np.outer(wj,wj) for wj in s.decomp(-1))
            self.assert_arraysclose([[1.,0.],[0.,1.]],np.dot(mat,minv))
            if rescale==False:
                m2 = sum(np.outer(wj,wj) for wj in s.decomp(2))
                self.assert_arraysclose(mat,np.dot(m2,minv))

    def test_diagonal_blocks(self):
        """ find_diagonal_blocks """
        def make_blocks(*m_list):
            m_list = [np.asarray(m, float) for m in m_list]
            n = sum([m.shape[0] for m in m_list])
            ans = np.zeros((n,n), float)
            i = 0
            for m in m_list:
                j = i + m.shape[0]
                ans[i:j, i:j] = m
                i = j
            # mean is irrelevant
            return gvar(ans[0], ans)
        def compare_blocks(b1, b2):
            s1 = set([tuple(list(b1i)) for b1i in b1])
            s2 = set([tuple(list(b2i)) for b2i in b2])
            self.assertEqual(s1, s2)
        m = make_blocks(
            [[1]],
            [[1, 1], [1, 1]],
            [[1]]
            )
        idx = [idx.tolist() for idx,bcov in evalcov_blocks(m)]
        compare_blocks(idx, [[0], [3], [1, 2]])
        m = make_blocks(
            [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            [[1, 1], [1, 1]],
            [[1]],
            [[1]]
            )
        idx = [idx.tolist() for idx,bcov in evalcov_blocks(m)]
        compare_blocks(idx, [[1], [5], [6], [0, 2], [3, 4]])
        m = make_blocks(
            [[1, 0, 1, 1],
             [0, 1, 0, 1],
             [1, 0, 1, 1],
             [1, 1, 1, 1]],
            [[1, 1], [1, 1]],
            [[1]],
            [[1]]
            )
        idx = [idx.tolist() for idx,bcov in evalcov_blocks(m)]
        compare_blocks(idx, [[6], [7], [0, 1, 2, 3] , [4, 5]])

    def test_evalcov_blocks(self):
        def test_cov(g):
            if hasattr(g, 'keys'):
                g = BufferDict(g)
            g = g.flat[:]
            cov = np.zeros((len(g), len(g)), dtype=float)
            for idx, bcov in evalcov_blocks(g):
                cov[idx[:,None], idx] = bcov
            self.assertEqual(str(evalcov(g)), str(cov))
        g = gv.gvar(5 * ['1(1)'])
        test_cov(g)
        g[-1] = g[0] + g[1]
        test_cov(g)
        test_cov(g * gv.gvar('2(1)'))
        g = gv.gvar(5 * ['1(1)'])
        g[0] = g[-1] + g[-2]
        test_cov(g)

    def test_evalcov_blocks_compress(self):
        def test_cov(g):
            if hasattr(g, 'keys'):
                g = BufferDict(g)
            blocks = evalcov_blocks(g, compress=True)
            g = g.flat[:]
            cov = np.zeros((len(g), len(g)), dtype=float)
            idx, bsdev = blocks[0]
            if len(idx) > 0:
                cov[idx, idx] = bsdev ** 2
            for idx, bcov in blocks[1:]:
                cov[idx[:,None], idx] = bcov
            self.assertEqual(str(evalcov(g)), str(cov))
        g = gv.gvar(5 * ['1(1)'])
        test_cov(g)
        test_cov(dict(g=g))
        g[-1] = g[0] + g[1]
        test_cov(g)
        test_cov(dict(g=g))
        test_cov(g * gv.gvar('2(1)'))
        g = gv.gvar(5 * ['1(1)'])
        g[0] = g[-1] + g[-2]
        test_cov(g)
        test_cov(dict(g=g))
        g[1:] += g[:-1]
        test_cov(g)
        test_cov(dict(g=g))

    def test_svd(self):
        """ svd """
        def make_mat(wlist, n):
            ans = np.zeros((n,n), float)
            i, wgts = wlist[0]
            if len(i) > 0:
                ans[i, i] = np.array(wgts) ** 2
            for i, wgts in wlist[1:]:
                for w in wgts:
                    ans[i, i[:, None]] += np.outer(w, w)
            return ans
        def test_gvar(a, b):
            self.assertEqual(a.fmt(4), b.fmt(4))
        def test_cov(wgts, cov, atol=1e-7):
            invcov = make_mat(wgts, cov.shape[0])
            np.testing.assert_allclose(
                invcov.dot(cov), np.eye(*cov.shape), atol=atol
                )
            np.testing.assert_allclose(svd.logdet, np.log(np.linalg.det(cov)), atol=atol)
        # diagonal
        f = gvar(['1(2)', '3(4)'])
        g, wgts = svd(f, svdcut=0.9, wgts=-1)
        test_gvar(g[0], f[0])
        test_gvar(g[1], f[1])
        test_cov(wgts, evalcov(g))
        self.assertEqual(svd.nmod, 0)
        self.assertEqual(svd.eigen_range, 1.)

        # degenerate
        g, wgts = svd(3 * [gvar('1(1)')], svdcut=1e-10, wgts=-1)
        test_cov(wgts, evalcov(g), atol=1e-4)
        self.assertEqual(svd.nmod, 2)
        self.assertAlmostEqual(svd.eigen_range, 0.0)

        # blocks
        x = gvar(10 * ['1(1)'])
        x[:5] += gvar('1(1)')       # half are correlated
        g = svd(x, svdcut=0.5)
        self.assertEqual(svd.nmod, 4)
        p = np.random.permutation(10)
        gp = svd(x[p], svdcut=0.5)
        self.assertEqual(svd.nmod, 4)
        invp = np.argsort(p)
        np.testing.assert_allclose(evalcov(g), evalcov(gp[invp]), atol=1e-7)
        np.testing.assert_allclose(mean(g), mean(gp[invp]), atol=1e-7)


        # cov[i,i] independent of i, cov[i,j] != 0
        x, dx = gvar(['1(1)', '0.01(1)'])
        g, wgts = svd([(x+dx)/2, (x-dx)/2.], svdcut=0.2 ** 2, wgts=-1)
        y = g[0] + g[1]
        dy = g[0] - g[1]
        test_gvar(y, x)
        test_gvar(dy, gvar('0.01(20)'))
        test_cov(wgts, evalcov(g))
        self.assertEqual(svd.nmod, 1)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)

        # negative svdcut
        x, dx = gvar(['1(1)', '0.01(1)'])
        g, wgts = svd([(x+dx)/2, (x-dx)/20.], svdcut=-0.2 ** 2, wgts=-1)
        y = g[0] + g[1] * 10
        dy = g[0] - g[1] * 10
        np.testing.assert_allclose(evalcov([y, dy]), [[1, 0], [0, 0]], atol=1e-7)
        test_gvar(y, x)
        test_gvar(dy, gvar('0(0)'))
        self.assertEqual(svd.dof, 1)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)

        # cov[i,i] independent of i, cov[i,j] != 0 --- cut too small
        x, dx = gvar(['1(1)', '0.01(1)'])
        g, wgts = svd([(x+dx)/2, (x-dx)/2.], svdcut=0.0099999** 2, wgts=-1)
        y = g[0] + g[1]
        dy = g[0] - g[1]
        test_gvar(y, x)
        test_gvar(dy, dx)
        test_cov(wgts, evalcov(g))
        self.assertEqual(svd.nmod, 0)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)


        # cov[i,i] independent of i after rescaling, cov[i,j] != 0
        # rescaling turns this into the previous case
        g, wgts = svd([(x+dx)/2., (x-dx)/20.], svdcut=0.2 ** 2, wgts=-1)
        y = g[0] + g[1] * 10.
        dy = g[0] - g[1] * 10.
        test_gvar(y, x)
        test_gvar(dy, gvar('0.01(20)'))
        test_cov(wgts, evalcov(g))
        self.assertEqual(svd.nmod, 1)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)

        # dispersed correlations
        g2, g4 = gvar(['2(2)', '4(4)'])
        orig_g = np.array([g2, (x+dx)/2., g4, (x-dx)/20.])
        g, wgts = svd(orig_g, svdcut=0.2 ** 2, wgts=-1)
        y = g[1] + g[3] * 10.
        dy = g[1] - g[3] * 10.
        test_gvar(y, x)
        test_gvar(dy, gvar('0.01(20)'))
        test_gvar(g[0], g2)
        test_gvar(g[2], g4)
        test_cov(wgts, evalcov(g))
        self.assertEqual(svd.nmod, 1)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)
        self.assertEqual(svd.nblocks[1], 2)
        self.assertEqual(svd.nblocks[2], 1)

        # remove svd correction
        g -= g.correction
        y = g[1] + g[3] * 10.
        dy = g[1] - g[3] * 10.
        test_gvar(y, x)
        test_gvar(dy, dx)
        test_gvar(g[0], g2)
        test_gvar(g[2], g4)
        np.testing.assert_allclose(evalcov(g.flat), evalcov(orig_g), atol=1e-7)

        # noise=True
        x, dx = gvar(['1(1)', '0.01(1)'])
        g, wgts = svd([(x+dx)/2, (x-dx)/2.], svdcut=0.2 ** 2, wgts=-1, noise=True)
        y = g[0] + g[1]
        dy = g[0] - g[1]
        offsets = mean(g.correction)
        self.assertEqual(g.nmod, 1)
        self.assertAlmostEqual(offsets[0], -offsets[1])
        self.assertGreater(chi2(g.correction[0]).Q, 0.01)
        self.assertLess(chi2(g.correction[0]).Q, 0.99)
        with self.assertRaises(AssertionError):
            test_gvar(y, x)
            test_gvar(dy, gvar('0.01(20)'))
        self.assertTrue(equivalent(
            g - g.correction, [(x+dx)/2, (x-dx)/2.]
            ))
        self.assertTrue(not equivalent(
            g, [(x+dx)/2, (x-dx)/2.]
            ))

        # bufferdict
        g = {}
        g[0] = (x+dx)/2.
        g[1] = (x-dx)/20.
        g, wgts = svd({0:(x+dx)/2., 1:(x-dx)/20.}, svdcut=0.2 ** 2, wgts=-1)
        assert isinstance(g, BufferDict)
        y = g[0] + g[1] * 10.
        dy = g[0] - g[1] * 10.
        test_gvar(y, x)
        test_gvar(dy, gvar('0.01(20)'))
        test_cov(wgts, evalcov(g.flat))
        self.assertEqual(svd.nmod, 1)
        self.assertAlmostEqual(svd.eigen_range, 0.01**2)
        self.assertTrue(equivalent(
            g - g.correction, {0:(x+dx)/2, 1:(x-dx)/20.}
            ))
        self.assertTrue(not equivalent(
            g, {0:(x+dx)/2, 1:(x-dx)/20.}
            ))

    def test_valder(self):
        """ valder_var """
        alist = [[1.,2.,3.]]
        a = valder([[1.,2.,3.]])
        alist = np.array(alist)
        self.assertEqual(np.shape(a),np.shape(alist))
        na = len(alist.flat)
        for i,(ai,ali) in enumerate(zip(a.flat,alist.flat)):
            der = np.zeros(na,float)
            der[i] = 1.
            self.assert_arraysequal(ai.der,der)
            self.assertEqual(ai.val,ali)

    def test_ranseed(self):
        """ ranseed """
        f = raniter([x,y])
        ranseed((1,2))
        f1 = next(f)
        x1 = x()
        y1 = y()
        ranseed((1,2))
        self.assert_arraysequal(f1,next(f))
        self.assertEqual(x1,x())
        self.assertEqual(y1,y())
        # default initialization
        ranseed()
        f1 = next(f)
        ranseed(ranseed.seed)
        self.assert_arraysequal(f1, next(f))

    def test_rebuild(self):
        """ rebuild """
        gvar = gvar_factory()
        a = gvar([1.,2.],[[4.,2.],[2.,16.]])
        b = a*gvar(1.,10.)
        c = rebuild(b)
        self.assert_arraysequal(c[0].der[-2:],b[0].der[:-1])
        self.assert_arraysclose(evalcov(c),evalcov(b))
        gvar = gvar_factory()
        c = rebuild({0:b[0],1:b[1]},gvar=gvar)
        c = np.array([c[0],c[1]])
        self.assert_arraysequal(c[0].der,b[0].der[:-1])
        self.assert_arraysclose(evalcov(c),evalcov(b)    )

    def test_chi2(self):
        """ chi2(g1, g2) """
        # uncorrelated
        g = gvar([1., 2.], [1., 2.])
        x = [2., 4.]
        ans = chi2(x, g)
        self.assertAlmostEqual(ans, 2., places=5)
        self.assertEqual(ans.dof, 2)
        self.assertAlmostEqual(ans.Q, 0.36787944, places=2)

        # correlated
        g = np.array([g[0]+g[1], g[0]-g[1]])
        x = np.array([x[0]+x[1], x[0]-x[1]])
        ans = chi2(x, g)
        self.assertAlmostEqual(ans, 2., places=5)
        self.assertEqual(ans.dof, 2)
        self.assertAlmostEqual(ans.Q, 0.36787944, places=2)

        # correlated with 0 mode and svdcut < 0
        g = np.array([g[0], g[1], g[0]+g[1]])
        x = np.array([x[0], x[1], x[0]+x[1]])
        ans = chi2(x, g, svdcut=-1e-10)
        self.assertAlmostEqual(ans, 2., places=5)
        self.assertEqual(ans.dof, 2)
        self.assertAlmostEqual(ans.Q, 0.36787944, places=2)

        # dictionaries with different keys
        g = dict(a=gvar(1,1), b=[[gvar(2,2)], [gvar(3,3)], [gvar(4,4)]], c=gvar(5,5))
        x = dict(a=2., b=[[4.], [6.]])
        ans = chi2(x,g)
        self.assertAlmostEqual(ans, 3.)
        self.assertEqual(ans.dof, 3)
        self.assertAlmostEqual(ans.Q, 0.3916252, places=2)
        ans = chi2(g,x)
        self.assertAlmostEqual(ans, 3.)
        self.assertEqual(ans.dof, 3)
        self.assertAlmostEqual(ans.Q, 0.3916252, places=2)
        ans = chi2(2., gvar(1,1))
        self.assertAlmostEqual(ans, 1.)
        self.assertEqual(ans.dof, 1)
        self.assertAlmostEqual(ans.Q, 0.31731051, places=2)

        # two dictionaries
        g1 = dict(a=gvar(1, 1), b=[gvar(2, 2)])
        g2 = dict(a=gvar(2, 2), b=[gvar(4, 4)])
        ans = chi2(g1, g2)
        self.assertAlmostEqual(ans, 0.2 + 0.2)
        self.assertEqual(ans.dof, 2)
        self.assertAlmostEqual(ans.Q, 0.81873075, places=2)

    def test_corr(self):
        """ rebuild (corr!=0) """
        a = gvar([1., 2.], [3., 4.])
        corr = 1.
        b = rebuild(a, corr=corr)
        self.assert_arraysclose(evalcov(a).diagonal(),evalcov(b).diagonal())
        bcov = evalcov(b)
        self.assert_arraysclose(bcov[0,1],corr*(bcov[0,0]*bcov[1,1])**0.5)
        self.assert_arraysclose(bcov[1,0],bcov[0,1])
        self.assert_arraysclose((b[1]-b[0]).sdev,1.0)
        self.assert_arraysclose((a[1]-a[0]).sdev,5.0)

    def test_filter(self):
        g = collections.OrderedDict([('a', 2.3), ('b', [gv.gvar('12(2)'), 3.]), ('c', 'string')])
        gm = collections.OrderedDict([('a', 2.3), ('b', [2., 3.]), ('c', 'string')])
        self.assertEqual(str(gv.filter(g, gv.sdev)), str(gm))

    def test_pickle(self):
        """ pickle strategies """
        for g in [
            '1(5)',
            [['2(1)'], ['3(2)']],
            dict(a='4(2)', b=[['5(5)', '6(9)']]),
            ]:
            g1 = gvar(g)
            gtuple = (mean(g1), evalcov(g1))
            gpickle = pickle.dumps(gtuple)
            g2 = gvar(pickle.loads(gpickle))
            self.assertEqual(str(g1), str(g2))
            self.assertEqual(str(evalcov(g1)), str(evalcov(g2)))

    def test_dump_load(self):
        dict = collections.OrderedDict
        gs = gv.gvar('1(2)') * gv.gvar('4(2)')
        ga = gv.gvar([2, 3], [[5., 1.], [1., 10.]]) 
        gd = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
        gd['v'] += gv.gvar('0(1)')
        gd[(1,3)] = gv.gvar('13(13)')
        gd['v'] = 1 / gd['v']
        def _test(g, outputfile=None, test_cov=True):
            s = dump(g, outputfile=outputfile)
            d = load(s if outputfile is None else outputfile)
            self.assertEqual( str(g), str(d))
            if test_cov and getattr(g,'size',1) > 1:
                self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
            # cleanup
            if isinstance(outputfile, str):
                os.remove(outputfile) 
            return d
        for g in [gs, ga, gd]:
            _test(g)
            _test(g, outputfile='xxx.pickle')
            _test(g, outputfile='xxx')
        gd['x'] = 5.0
        _test(gd, test_cov=False)
        _test(gd, outputfile='xxx', test_cov=False)
        for g in [gs, ga, gd]:
            g = gv.mean(g)
            _test(g, test_cov=False)
        # misc types
        g = dict(
            s=set([1,2,12.2]),
            a=1, 
            b=[1,[gv.gvar('3(1)') * gv.gvar('2(1)'), 4]], 
            c=dict(a=gv.gvar(5 * ['1(2)']), b=np.array([[4]])),
            d=collections.deque([1., 2, gv.gvar('4(1)')]),
            e='a string',
            g=(3, 'hi', gv.gvar('-1(2)')),
            )
        g['f'] = ['str', g['b'][1][0] * gv.gvar('5(2)')]
        d = _test(g, outputfile='xxx', test_cov=False)

        # dumping classes, without and with special methods and/or __slots__
        fac = gvar('10(1)')
        g['C'] = C(gv.gvar(2 * ['3(4)']) * fac, 'str', (1,2, fac * gv.gvar('2(1)')))
        d = _test(g, test_cov=False)
        self.assertEqual(str(gv.evalcov(d['C'].x)), str(gv.evalcov(g['C'].x)))
        self.assertAlmostEqual(corr(d['C'].x[0], d['C'].z[2]), corr(g['C'].x[0], g['C'].z[2]))        
        g['CS'] = CS(gv.gvar(2 * ['3(4)']) * fac, 'str', (1,2,fac * gv.gvar('2(1)')))
        d = _test(g, test_cov=False)
        self.assertEqual(str(gv.evalcov(d['CS'].x)), str(gv.evalcov(g['CS'].x)))
        self.assertAlmostEqual(corr(d['CS'].x[0], d['C'].z[2]), corr(g['CS'].x[0], g['C'].z[2]))        
        g['C'] = CC(gv.gvar(2 * ['3(4)']) * gv.gvar('10(1)'), 'str', 12.)
        d = gv.loads(gv.dumps(g))
        self.assertEqual(d['C'].z, None)
        self.assertEqual(g['C'].z, 12.)
        self.assertEqual(str(gv.evalcov(d['C'].x)), str(gv.evalcov(g['C'].x)))

    def test_dump_load_errbudget(self):
        dict = collections.OrderedDict
        def _test(d, add_dependencies=False):
            d = gv.BufferDict(d)
            newd = loads(dumps(d, add_dependencies=add_dependencies))
            str1 = str(d) + fmt_errorbudget(
                outputs=dict(a=d['a'], b=d['b']), 
                inputs=dict(x=d['x'], y=d['y'], z=d['z']),
                )
            d = newd
            str2 = str(d) + fmt_errorbudget(
                outputs=dict(a=d['a'], b=d['b']), 
                inputs=dict(x=d['x'], y=d['y'], z=d['z']),
                )
            self.assertEqual(str1, str2)    
        # all primaries included
        x = gv.gvar('1(2)')
        y = gv.gvar('2(3)') ** 2
        z = gv.gvar('3(4)') ** 0.5 
        u = gv.gvar([2, 3], [[5., 1.], [1., 10.]]) 
        a  = x*y
        b = x*y - z
        d = dict(a=a, b=b, x=x, y=y, z=z, u=u, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d)
        del d['xx']
        _test(d)
        # a,b are primaries
        a, b = gvar(mean([d['a'], d['b']]), evalcov([d['a'], d['b']]))
        d['a'] = a
        d['b'] = b 
        _test(d)
        # no primaries included explicitly
        x = gv.gvar('1(2)') + gv.gvar('1(2)')
        y = gv.gvar('2(3)') ** 2 + gv.gvar('3(1)')
        z = gv.gvar('3(4)') ** 0.5 + gv.gvar('4(1)')
        a  = x*y
        b = x*y - z + gv.gvar('10(1)')
        d = dict(a=a, b=b, x=x, y=y, z=z, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d, add_dependencies=True)
        # mixture
        x = gv.gvar('1(2)') 
        y = gv.gvar('2(3)') ** 2  + gv.gvar('3(1)')
        z = gv.gvar('3(4)') ** 0.5 + gv.gvar('4(1)')
        a  = x*y
        b = x*y - z + gv.gvar('10(1)')
        d = dict(a=a, b=b, x=x, y=y, z=z, u=u, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d, add_dependencies=True)

    def test_more_dump(self):
        " check on particular issue "
        x = gv.gvar(4 * ['1(2)']) 
        x[0] -= x[1] * gv.gvar('1(10)')
        x[2] += x[1]
        str1 = str(x) +  str(evalcov(x))
        x = loads(dumps(x))
        str2 = str(x) +  str(evalcov(x))
        self.assertEqual(str1, str2)

    def test_dumps_loads(self):
        dict = collections.OrderedDict
        gs = gv.gvar('1(2)')
        ga = (gv.gvar([['2(2)', '3(3)']]) + gv.gvar('0(1)') )
        gd = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
        gd['v'] += gv.gvar('0(1)')
        gd[(1,3)] = gv.gvar('13(13)')
        gd['v'] = 1 / gd['v']
        def _test(g):
            s = dumps(g)
            d = loads(s)
            self.assertEqual( str(g), str(d))
            self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
        for g in [gs, ga, gd]:
            _test(g)

###############
    def test_gdump_gload(self):
        gs = gv.gvar('1(2)') * gv.gvar('3(2)')
        ga = gv.gvar([2, 3], [[5., 1.], [1., 10.]]) 
        gd = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
        gd['v'] += gv.gvar('0(1)')
        gd[(1,3)] = gv.gvar('13(13)')
        gd['v'] = 1 / gd['v']
        def _test(g, outputfile=None, method=None):
            s = gdump(g, outputfile=outputfile, method=method)
            d = gload(s if outputfile is None else outputfile, method=method)
            self.assertEqual( str(g), str(d))
            if getattr(g, 'size', 1) > 1:
                self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
            # cleanup
            if isinstance(outputfile, str):
                os.remove(outputfile) 
        for g in [gs, ga, gd]:
            _test(g)
            _test(g, outputfile='xxx.json')
            _test(g, outputfile='xxx.pickle')
            _test(g, outputfile='xxx')
            _test(g, outputfile='xxx', method='pickle')
            _test(g, method='json')
            _test(g, method='pickle')    
            _test(g, method='dict')

    def test_gdump_gload_errbudget(self):
        def _test(d, add_dependencies=False):
            d = gv.BufferDict(d)
            newd = gloads(gdumps(d, add_dependencies=add_dependencies))
            str1 = str(d) + fmt_errorbudget(
                outputs=dict(a=d['a'], b=d['b']), 
                inputs=dict(x=d['x'], y=d['y'], z=d['z']),
                )
            d = newd
            str2 = str(d) + fmt_errorbudget(
                outputs=dict(a=d['a'], b=d['b']), 
                inputs=dict(x=d['x'], y=d['y'], z=d['z']),
                )
            self.assertEqual(str1, str2)    
        # all primaries included
        x = gv.gvar('1(2)')
        y = gv.gvar('2(3)') ** 2
        z = gv.gvar('3(4)') ** 0.5 
        u = gv.gvar([2, 3], [[5., 1.], [1., 10.]]) 
        a  = x*y
        b = x*y - z
        d = dict(a=a, b=b, x=x, y=y, z=z, u=u, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d)
        del d['xx']
        _test(d)
        # a,b are primaries
        a, b = gvar(mean([d['a'], d['b']]), evalcov([d['a'], d['b']]))
        d['a'] = a
        d['b'] = b 
        _test(d)
        # no primaries included explicitly
        x = gv.gvar('1(2)') + gv.gvar('1(2)')
        y = gv.gvar('2(3)') ** 2 + gv.gvar('3(1)')
        z = gv.gvar('3(4)') ** 0.5 + gv.gvar('4(1)')
        a  = x*y
        b = x*y - z + gv.gvar('10(1)')
        d = dict(a=a, b=b, x=x, y=y, z=z, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d, add_dependencies=True)
        # mixture
        x = gv.gvar('1(2)') 
        y = gv.gvar('2(3)') ** 2  + gv.gvar('3(1)')
        z = gv.gvar('3(4)') ** 0.5 + gv.gvar('4(1)')
        a  = x*y
        b = x*y - z + gv.gvar('10(1)')
        d = dict(a=a, b=b, x=x, y=y, z=z, u=u, uu=u*gv.gvar('1(1)'), xx=x)
        _test(d, add_dependencies=True)

    def test_more_gdump(self):
        " check on particular issue "
        x = gv.gvar(4 * ['1(2)']) 
        x[0] -= x[1] * gv.gvar('1(10)')
        x[2] += x[1]
        str1 = str(x) +  str(evalcov(x))
        x = gloads(gdumps(x))
        str2 = str(x) +  str(evalcov(x))
        self.assertEqual(str1, str2)

    def test_gdumps_gloads(self):
        gs = gv.gvar('1(2)')
        ga = (gv.gvar(['2(2)', '3(3)']) + gv.gvar('0(1)') )
        gd = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
        gd['v'] += gv.gvar('0(1)')
        gd[(1,3)] = gv.gvar('13(13)')
        gd['v'] = 1 / gd['v']
        # json (implicit)
        def _test(g):
            s = gdumps(g)
            d = gloads(s)
            self.assertEqual( str(g), str(d))
            self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
        for g in [gs, ga, gd]:
            _test(g)
        # json
        def _test(g):
            s = gdumps(g, method='json')
            d = gloads(s)
            self.assertEqual( str(g), str(d))
            self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
        for g in [gs, ga, gd]:
            _test(g)
        # pickle
        def _test(g):
            s = gdumps(g, method='pickle')
            d = gloads(s)
            self.assertEqual( str(g), str(d))
            self.assertEqual( str(gv.evalcov(g)), str(gv.evalcov(d)))
        for g in [gs, ga, gd]:
            _test(g)

################
    def test_oldload(self):
        gd = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
        for g in [gd, gd['s'], gd['v']]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = gv.gvar(dict(s='1(2)', v=['2(2)', '3(3)'], g='4(4)'))
                olddump(g, 'xxx.p')
                d = load('xxx.p')
                assert str(g) == str(d)
                assert str(gv.evalcov(g)) == str(gv.evalcov(d))
                olddump(g, 'xxx.json', method='json')
                d = load('xxx.json', method='json')
                assert str(g) == str(d)
                assert str(gv.evalcov(g)) == str(gv.evalcov(d))

    def test_dependencies(self):
        def _test(g):
            dep = dependencies(g)
            new_g = g.mean + sum(dep * g.deriv(dep))
            self.assertEqual(str(new_g - g), str(gvar('0(0)')))
            self.assertTrue(equivalent(g, new_g))
        x = gv.gvar('1(2)')
        y = gv.gvar('2(3)') ** 2
        z = gv.gvar('3(4)') ** 0.5 * y
        _test(x * y)
        _test(x * y - z)
        self.assertEqual(len(dependencies([y, x])), 0)
        self.assertEqual(len(dependencies([y, 'string', x])), 0)
        self.assertEqual(len(dependencies([y, x, x**2, 2*y])), 0)
        self.assertEqual(len(dependencies([x*y, x])), 1)
        self.assertEqual(len(dependencies([x*y, x, x, x])), 1)
        self.assertEqual(len(dependencies([x*y, x], all=True)), 2)
        self.assertEqual(len(dependencies([x*y, x, 'string'], all=True)), 2)
        self.assertEqual(len(dependencies([x*y, x, x, x], all=True)), 2)
        self.assertTrue(missing_dependencies([x*y, x]))
        self.assertTrue(missing_dependencies([x*y, x+y, x, x]))
        self.assertTrue(not missing_dependencies([y, x]))
        self.assertTrue(not missing_dependencies([x*y, x, y]))            

    def test_gammaQ(self):
        " gammaQ(a, x) "
        cases = [
            (2.371, 5.243, 0.05371580082389009, 0.9266599665892222),
            (20.12, 20.3, 0.4544782602230986, 0.4864172139106905),
            (100.1, 105.2, 0.29649013488390663, 0.6818457585776236),
            (1004., 1006., 0.4706659307021259, 0.5209695379094582),
            ]
        for a, x, gax, gxa in cases:
            np.testing.assert_allclose(gax, gv._utilities.gammaQ(a, x), rtol=0.01)
            np.testing.assert_allclose(gxa, gv._utilities.gammaQ(x, a), rtol=0.01)

    def test_erf(self):
        " erf(x) "
        for x in [-1.1, 0.2]:
            self.assertAlmostEqual(erf(x), math.erf(x))
        x = [[-1.1], [0.2]]
        np.testing.assert_allclose(erf(x), [[math.erf(-1.1)], [math.erf(0.2)]])
        x = gv.gvar('0(2)')
        erfx = erf(x)
        self.assertAlmostEqual(erfx.mean, math.erf(0))
        self.assertAlmostEqual(
            erfx.sdev,
            2 * (math.erf(1e-10) - math.erf(-1e-10)) / 2e-10
            )
        x = gv.gvar('1.5(2)')
        self.assertAlmostEqual(erf(x).mean, math.erf(x.mean))
        x = gv.gvar(['0(2)', '1.5(2)'])
        erfx = erf(x)
        self.assertAlmostEqual(erfx[0].mean, math.erf(x[0].mean))
        self.assertAlmostEqual(erfx[1].mean, math.erf(x[1].mean))

    def test_equivalent(self):
        " equivalent(g1, g2) "
        x = gvar(['1(1)', '2(2)'])
        y = gvar(['1(1)', '2(2)'])
        u = 2 ** 0.5 * np.array([[0.5, 0.5],[-0.5, 0.5]])
        ux = u.dot(x)
        uTy = u.T.dot(y)
        ux_y = ux + y
        xnew = u.T.dot(ux_y) - uTy
        self.assertTrue(equivalent(x, xnew))
        self.assertTrue(not equivalent(x, y))
        self.assertTrue(equivalent(x[0], xnew[0]))
        d = dict(x=x, y0=y[0])
        dnew = dict(x=xnew, y0=y[0])
        self.assertTrue(equivalent(d, dnew))
        dnew = dict(x=y, y0=y[0])
        self.assertTrue(not equivalent(d, dnew))
        dnew = dict(x=xnew, y0=x[0])
        self.assertTrue(not equivalent(d, dnew))

    def test_is_primary(self):
        " is_primary(g) "
        self.assertTrue(gvar('1(1)').is_primary())
        self.assertTrue((2 * gvar('1(1)')).is_primary())
        self.assertFalse((gvar('2(1)') * gvar('1(1)')).is_primary())
        gs = gvar('1(1)')
        ga = gvar(2 * [3 * ['1(1)']])
        gd = dict(s=gs, a=ga)
        self.assertEqual(is_primary(gs), True)
        self.assertEqual(is_primary(ga).tolist(), 2 * [3 * [True]])
        self.assertEqual(is_primary(gd).buf.tolist(), 7 * [True])
        self.assertEqual(is_primary([gs, gs]).tolist(), [True, False])
        gs = gs + gvar('1(1)')
        ga[0, 0] += gvar('2(1)')
        ga[1, 0] *= 5.
        gd = BufferDict()
        gd['s'] = gs 
        gd['a'] = ga
        self.assertEqual(is_primary(gs), False)
        self.assertEqual(is_primary(ga).tolist(), [[False, True, True], [True, True, True]])
        self.assertEqual(is_primary(gd).buf.tolist(), [False, False] + 5 * [True])

    def test_disassemble(self):
        " d = disassemble(g); reassemble(d) "
        # gvar
        g = gvar('1(2)')
        gn = reassemble(disassemble(g), gvar.cov)
        d = gn - g
        self.assertEqual(d.mean, 0.0)
        self.assertEqual(d.sdev, 0.0)
        # array
        g = gvar([['1(2)', '2(3)'], ['3(4)', '4(5)']])
        gn = reassemble(disassemble(g), gvar.cov)
        self.assertEqual(g.shape, gn.shape)
        d = gn - g
        self.assertTrue(np.all(gv.mean(d) == 0.0))
        self.assertTrue(np.all(gv.sdev(d) == 0.0))
        # dict
        g = gvar(
            dict(s=gvar('1(2)'), a=gvar([['1(2)', '2(3)'], ['3(4)', '4(5)']]))
            )
        gn = reassemble(disassemble(g), gvar.cov)
        for k in g:
            d = gn[k] - g[k]
            self.assertTrue(np.all(gv.mean(d) == 0.0))
            self.assertTrue(np.all(gv.mean(d) == 0.0))

    @unittest.skipIf(FAST,"skipping test_pdfstats for speed")
    def test_pdfstats(self):
        " PDFStatistics(moments) "
        x = gv.gvar('3.0(4)')
        avgs = np.zeros((10,4), float)
        for i in range(10):
            moments = np.zeros(4, float)
            for xi in gv.raniter(x, 100):
                moments += xi ** np.arange(1, 5)
            s = PDFStatistics(moments / 100.)
            avgs[i] = [s.mean, s.sdev, s.skew, s.ex_kurt]
        mean = np.mean(avgs,axis=0)
        sdev = np.std(avgs, axis=0)
        diff = gvar(mean, sdev) - [x.mean, x.sdev, 0., 0.]
        self.assertTrue(
            np.all(np.fabs(gv.mean(diff)) < 5 * gv.sdev(diff))
            )


    @unittest.skipIf(not have_vegas, "vegas not installed")
    @unittest.skipIf(FAST,"skipping test_pdfstatshist for speed")
    def test_pdfstatshist(self):
        " PDFStatistics(histogram) "
        g = gv.gvar('2(1.0)')
        hist = PDFHistogram(g + 0.1, nbin=50, binwidth=0.2)
        integ = vegas.PDFIntegrator(g)
        integ(neval=1000, nitn=5)
        def f(p):
            return hist.count(p)
        results = integ(f, neval=1000, nitn=5,adapt=False)
        for stats in [
            PDFStatistics(histogram=(hist.bins, results)),
            hist.analyze(results).stats
            ]:
            self.assertTrue(
                abs(stats.median.mean - g.mean) < 5 * stats.median.sdev
                )
            self.assertTrue(
                abs(stats.plus.mean - g.sdev) < 5 * stats.plus.sdev
                )
            self.assertTrue(
                abs(stats.minus.mean - g.sdev) < 5 * stats.minus.sdev
                )

    def test_regulate(self):
        D = np.array([1., 2., 3.])
        corr = np.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
        cov = D[:, None] * corr * D[None, :]
        g1 = gvar(1, 10)
        g2 = gvar(3 * [2], cov)
        g3 = gvar(3 * [3], 2 * cov)
        g = np.concatenate(([g1], g2, g3))
        cov = evalcov(g)
        eps = 0.25
        norm = np.linalg.norm(evalcorr(g), np.inf)
        
        gr = regulate(g, eps=eps / norm, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        self.assertEqual(g.size, gr.dof)
        self.assertEqual(g.size - 1, gr.nmod)
        self.assertAlmostEqual(gr.eps, eps / norm)
        self.assertEqual(gr.svdcut, None)
        covr = evalcov(gr)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])
        
        gr, dummy = regulate(g, eps=eps / norm, wgts=True)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        self.assertEqual(g.size - 1, gr.nmod)
        self.assertEqual(g.size, gr.dof)
        covr = evalcov(gr)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])

    def test_regulate_svdcut(self):
        " regulate -> svd "
        D = np.array([1., 2., 3.])
        corr = np.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
        cov = D[:, None] * corr * D[None, :]
        g1 = gvar(1, 10)
        g2 = gvar(3 * [2], cov)
        g3 = gvar(3 * [3], 2 * cov)
        g = np.concatenate(([g1], g2, g3))
        svdcut = 0.25
        # verify that svd is being called in each case
        gr = regulate(g, svdcut=svdcut, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        self.assertEqual(gr.svdcut, svdcut)
        self.assertEqual(gr.eps, None)

        gr = regulate(g, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        self.assertEqual(gr.svdcut, 1e-12) # default
        self.assertEqual(gr.eps, None)

        gr = regulate(g, svdcut=svdcut, eps=svdcut, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        self.assertEqual(gr.svdcut, svdcut)
        self.assertEqual(gr.eps, None)

    def test_regulate_singular(self):
        D = np.array([1., 2., 3.])
        # two zero eigenvalues
        corr = np.array([[1., 1., 1.], [1., 1., 1.], [1.,1.,1.]])
        cov = D[:, None] * corr * D[None, :]
        g1 = gvar(1, 10)
        g2 = gvar(3 * [2], cov)
        g3 = gvar(3 * [3], 2 * cov)
        g = np.concatenate(([g1], g2, g3))
        cov = evalcov(g)
        corr = evalcorr(g)
        eps = 0.1
        norm = np.linalg.norm(evalcorr(g), np.inf)
        gr = regulate(g, eps=eps / norm, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        covr = evalcov(gr)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])
    
        gr, dummy = regulate(g, eps=eps / norm, wgts=True)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        covr = evalcov(gr)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])
        with self.assertRaises(np.linalg.LinAlgError):
            # det(corr)=0, so this should trigger an error
            gr, dummy = regulate(g, eps=0, wgts=True)

    def test_regulate_dict(self):
        D = np.array([1., 2., 3.])
        corr = np.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
        cov = D[:, None] * corr * D[None, :]
        g = BufferDict()
        g[1] = gvar(1, 10)
        g[2] = gvar(3 * [2], cov)
        g[3] = gvar(3 * [3], 2 * cov)
        cov = evalcov(g.flat)
        eps = 0.1
        norm = np.linalg.norm(evalcorr(g.flat), np.inf)
        gr = regulate(g, eps=eps / norm, wgts=False)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        covr = evalcov(gr.flat)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])

        gr, dummy = regulate(g, eps=eps / norm, wgts=True)
        self.assertTrue(gv.equivalent(gr - gr.correction, g))
        covr = evalcov(gr.flat)
        np.testing.assert_allclose(covr[0, :], cov[0, :])
        np.testing.assert_allclose(covr[:, 0], cov[:, 0])
        covr[1:, 1:][np.diag_indices_from(covr[1:, 1:])] -= eps * cov[1:, 1:].diagonal()
        np.testing.assert_allclose(covr[1:, 1:], cov[1:, 1:])
    
    def test_regulate_wgts(self):
        D = np.array([1., 2., 3.])
        corr = np.array([[1., .1, .2], [.1, 1., .3], [.2, .3, 1.]])
        cov = D[:, None] * corr * D[None, :]
        g1 = gvar(1, 10)
        g2 = gvar(3 * [2], cov)
        g3 = gvar(3 * [3], 2 * cov)
        g = np.concatenate(([g1], g2, g3))
        gr, i_wgts = regulate(g, eps=1e-15, wgts=1)
        covr = np.zeros((g.size, g.size), dtype=float) 
        i, wgts = i_wgts[0]
        if len(i) > 0:
            covr[i, i] = np.array(wgts) ** 2
        for i, wgts in i_wgts[1:]:
            covr[i[:, None], i] = (wgts.T).dot(wgts) # wgts.T @ wgts
        np.testing.assert_allclose(numpy.log(numpy.linalg.det(covr)), gr.logdet)
        self.assertEqual(gr.nmod, 6)
        np.testing.assert_allclose(covr[0,0], 100.)
        np.testing.assert_allclose(covr[1:4,1:4], cov)
        np.testing.assert_allclose(covr[4:7,4:7], 2 * cov)
        gr, i_wgts = regulate(g, eps=1e-15, wgts=-1)
        invcovr = np.zeros((g.size, g.size), dtype=float) 
        i, wgts = i_wgts[0]
        if len(i) > 0:
            invcovr[i, i] = np.array(wgts) ** 2
        for i, wgts in i_wgts[1:]:
                invcovr[i[:, None], i] += (wgts.T).dot(wgts) # wgts.T @ wgts
        np.testing.assert_allclose(invcovr[0,0], 1/100.)
        np.testing.assert_allclose(invcovr[1:4,1:4], np.linalg.inv(cov))
        np.testing.assert_allclose(invcovr[4:7,4:7], 0.5 * np.linalg.inv(cov))

class C(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return str(self.__dict__)

class CS(object):
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return str((self.x, self.y, self.z))
    def __repr__(self):
        return str((self.x, self.y, self.z))

class CC(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return str(self.__dict__)
    def _remove_gvars(self, gvlist):
        c = copy.copy(self)
        c.z = None
        c.__dict__ = gv.remove_gvars(c.__dict__, gvlist)
        return c
    def _distribute_gvars(self, gvlist):
        self.__dict__ = gv.distribute_gvars(self.__dict__, gvlist)
        return self 


if __name__ == '__main__':
	unittest.main()

