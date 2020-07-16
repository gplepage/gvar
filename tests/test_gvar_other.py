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
import numpy as np
import random
import gvar as gv
from gvar import *
from gvar.powerseries import PowerSeries, multiseries, multivar 
from gvar.pade import Pade, pade_gvar, pade_svd
try:
    import scipy.interpolate as scipy_interpolate
except:
    scipy_interpolate = None

def optprint(*args):
    pass

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

class test_ode(unittest.TestCase,ArrayTests):
    def setUp(self): pass

    def tearDown(self): pass

    def test_scalar(self):
        # exponential (scalar)
        def f(x, y):
            return y
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-13)
        y0 = 1
        y1 = odeint(y0, (0, 1))
        exact = numpy.exp(1)
        self.assertAlmostEqual(y1, exact)
        yN = odeint(y0, [0.0, 0.5, 1.0])
        self.assert_arraysclose(yN, numpy.exp([0.5, 1.0]))

    def test_hmax_maxstep_hmin(self):
        " ode with hmax, hmin, maxstep "
        # exponential (scalar)
        def f(x, y):
            return y
        sol = ode.Solution()
        odeint = ode.Integrator(
            deriv=f, h=1, hmax=0.005, maxstep = 1000, 
            analyzer=sol, tol=1e-13
            )
        y0 = 1
        y1 = odeint(y0, (0, 1))
        x = numpy.array(sol.x)
        self.assertAlmostEqual(max(x[1:] - x[:-1]), 0.005)
        exact = numpy.exp(1)
        self.assertAlmostEqual(y1, exact)
        odeint = ode.Integrator(deriv=f, h=1, maxstep=10, tol=1e-13)
        y0 = 1
        with self.assertRaises(RuntimeError):
            y1 = odeint(y0, (0, 1))
        sol = ode.Solution()
        odeint = ode.Integrator(deriv=f, h=1, hmin=0.1, tol=1e-13)
        if hasattr(self, 'assertWarns'): # for Python2.7
            with self.assertWarns(UserWarning):
                y1 = odeint(y0, (0, 1))

    def test_gvar_scalar(self):
        # exponential with errors
        gam = gv.gvar('1.0(1)')
        def f(x, y):
            return gam * y
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-10)
        y0 = gv.gvar('1.0(1)')
        y1 = odeint(y0, (0, 2))
        exact = y0 * np.exp(gam * 2)
        self.assertAlmostEqual((y1 / exact).mean, 1.)
        self.assertGreater(1e-8, (y1 / exact).sdev)
        self.assertTrue(
            gv.equivalent(odeint(y1, (2, 0)), y0, rtol=1e-6, atol=1e-6)
            )

    def test_vector(self):
        # harmonic oscillator with vectors
        def f(x, y):
            return numpy.array([y[1], -y[0]])
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-10)
        y0 = [0., 1.]
        y1 = odeint(y0, (0, 1))
        exact = [numpy.sin(1), numpy.cos(1)]
        self.assert_arraysclose(y1, exact)
        yN = odeint(y0, [0., 0.5, 1.0])
        self.assert_arraysclose(
            yN,
            [[numpy.sin(0.5), numpy.cos(0.5)], [numpy.sin(1.0), numpy.cos(1.0)]],
            )

    def test_vector_dict(self):
        # harmonic oscillator with vectors
        def f(x, y):
            deriv = {}
            deriv['y'] = y['dydx']
            deriv['dydx'] = - y['y']
            return deriv
        odeint = ode.DictIntegrator(deriv=f, h=1, tol=1e-10)
        y0 = dict(y=0., dydx=1.)
        y1 = odeint(y0, (0, 1))
        exact = [numpy.sin(1), numpy.cos(1)]
        self.assert_arraysclose([y1['y'], y1['dydx']], exact)
        yN = odeint(y0, [0., 0.5, 1.0])
        self.assert_arraysclose(
            [[yN[0]['y'], yN[0]['dydx']], [yN[1]['y'], yN[1]['dydx']]],
            [[numpy.sin(0.5), numpy.cos(0.5)], [numpy.sin(1.0), numpy.cos(1.0)]],
            )

    def test_gvar_dict(self):
        # harmonic oscillator with dictionaries and errors
        w2 = gv.gvar('1.00(2)')
        w = w2 ** 0.5
        def f(x, y):
            deriv = {}
            deriv['y'] = y['dydx']
            deriv['dydx'] =  -w2 * y['y']
            return deriv
        odeint = ode.DictIntegrator(deriv=f, h=1, tol=1e-10)
        x0 = 0
        y0 = dict(y=numpy.sin(w*x0), dydx=w * numpy.cos(w*x0))
        x1 = 10
        y1 = odeint(y0, (x0,x1))
        exact = dict(y=numpy.sin(w * x1), dydx=w * numpy.cos(w * x1))
        self.assert_gvclose(y1, exact)
        self.assertTrue(
            gv.equivalent(odeint(y1, (x1, x0)), y0, rtol=1e-6, atol=1e-6)
            )

    def test_delta(self):
        def delta(yerr, y, delta_y):
            return np.max(
                np.abs(yerr) / (np.abs(y) + np.abs(delta_y))
                )
        def f(x, y):
            return y * (1 + 0.1j)
        # without delta
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-13, delta=None) 
        y0 = 1
        y1 = odeint(y0, (0, 1))
        exact = numpy.exp(1 + 0.1j)
        self.assertAlmostEqual(y1, exact)
        # now with delta
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-13, delta=delta)
        y0 = 1
        y1 = odeint(y0, (0, 1))
        exact = numpy.exp(1 + 0.1j)
        self.assertAlmostEqual(y1, exact)

    def test_integral(self):
        def f(x):
            return 3 * x**2
        ans = ode.integral(f, (1,2), tol=1e-10)
        self.assertAlmostEqual(ans, 7)
        def f(x):
            return numpy.array([[1.], [2 * x], [3 * x**2]])
        ans = ode.integral(f, (1, 2))
        self.assert_arraysclose(ans, [[1], [3], [7]])
        def f(x):
            return dict(a=1., b=[2 * x, 3 * x**2])
        ans = ode.integral(f, (1, 2))
        self.assertAlmostEqual(ans['a'], 1.)
        self.assert_arraysclose(ans['b'], [3, 7])

    def test_integral_gvar(self):
        def f(x, a=gvar(1,1)):
            return a * 3 * x**2
        ans = ode.integral(f, (1,2), tol=1e-10)
        self.assertAlmostEqual(ans.mean, 7)
        self.assertAlmostEqual(ans.sdev, 7)
        a = gvar(1, 1)
        def f(x, a=a):
            return a * numpy.array([[1.], [2 * x], [3 * x**2]])
        ans = ode.integral(f, (1, 2))
        self.assert_arraysclose(mean(ans), [[1], [3], [7]])
        self.assert_arraysclose(sdev(ans), [[1], [3], [7]])
        def f(x, a=a):
            return dict(a=a, b=[2 * a * x, 3 * a * x**2])
        ans = ode.integral(f, (1, 2))
        self.assertAlmostEqual(ans['a'].mean, 1.)
        self.assert_arraysclose(mean(ans['b']), [3, 7])
        self.assertAlmostEqual(ans['a'].sdev, 1.)
        self.assert_arraysclose(sdev(ans['b']), [3, 7])
        self.assertTrue(gv.equivalent(
            ans,
            dict(a=a, b=[3 * a, 7 * a]),
            rtol=1e-6, atol=1e-6)
            )

    def test_solution(self):
        def f(x, y):
            return y
        y = ode.Integrator(deriv=f, h=1, tol=1e-13).solution(0., 1.)
        self.assertAlmostEqual(y(1.), np.exp(1.))
        self.assertEqual(y.x, 1)
        self.assertAlmostEqual(y.y, np.exp(1))
        self.assertAlmostEqual(y(1.5), np.exp(1.5))
        self.assertEqual(y.x, 1.5)
        self.assertAlmostEqual(y.y, np.exp(1.5))
        self.assert_arraysclose(y([2.0, 2.5]), np.exp([2.0, 2.5]))
        self.assertAlmostEqual(y.x, 2.5)
        self.assertAlmostEqual(y.y, np.exp(2.5))

    def test_solution_dict(self):
        def f(x, y):
            return dict(y=y['y'])
        y = ode.DictIntegrator(deriv=f, h=1, tol=1e-13).solution(0., dict(y=1.))
        y1 = y(1.)
        self.assertAlmostEqual(y1['y'], np.exp(1.))
        yN = y([0.5, 1.])
        self.assertAlmostEqual(yN[0]['y'], np.exp(0.5))
        self.assertAlmostEqual(yN[1]['y'], np.exp(1.0))

class test_cspline(unittest.TestCase,ArrayTests):
    def setUp(self): pass

    def tearDown(self): pass

    def f(self, x, c=[1, 2., 3., 0.]):
        return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3

    def Df(self, x, c=[1, 2., 3., 0.]):
        return c[1] + 2 * c[2] * x + 3 * c[3] * x ** 2

    def D2f(self, x, c=[1, 2., 3., 0.]):
        return 2 * c[2] + 6 *  c[3] * x

    def D3f(self, x, c=[1, 2., 3., 0.]):
        return 6 * c[3] + 0 * x 

    def integf(self, x, x0=0, c=[1, 2., 3., 0.]):
        return (
            c[0] * x + c[1] * x**2 / 2 + c[2] * x**3 / 3 + c[3] * x**4 / 4
            - (c[0] * x0 + c[1] * x0**2 / 2 + c[2] * x0**3 / 3 + c[3] * x0**4 / 4)
            )

    def test_shape(self):
        " shape of spline == shape of argument "
        xx = np.array([0, 1., 3.])
        yy = self.f(xx)
        yp= self.Df(xx)
        for extrap_order in [1,3]:
            s = cspline.CSpline(
                xx, yy,
                deriv=[yp[0], yp[-1]],
                extrap_order=extrap_order
                )
            for x in [-1, 0.5, 4.]:
                self.assertEqual(np.shape(x), np.shape(s(x)))
            for x in [
                [-1,0.5], [0.2, 0.5], [0.5, 4.], [-1,4.],
                [-1, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.],
                [0.5, 1.0, 1.5, 2.0, 2.5],
                [[-1, 1.1],[2.3, 4.]], [[2.]],
                ]:
                self.assertEqual(np.shape(x), np.shape(s(x)))

    def test_normal(self):
        " normal range "
        x = np.array([0, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]])
        x = np.arange(0.4, 3., 0.4)
        for xi in x:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))

    def test_2knot(self):
        " 2 knot spline "
        x = np.array([1., 3.])
        x0 = x[0]
        xtest = np.array([-1.5, 1., 2.2, 3., 3.5])
        # linear 
        c = [2., 2.7, 0., 0.]
        y = self.f(x, c=c)
        s = cspline.CSpline(x, y)
        for xi in xtest:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))
        # quadratic, left
        c = [2., 2.7, -1.1, 0.]
        y = self.f(x, c=c)
        s = cspline.CSpline(x, y, deriv=[self.Df(x[0], c=c), None])
        for xi in xtest:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))      
        # quadratic, right
        c = [2., -2.7, -0.7, 0.]
        y = self.f(x, c=c)
        s = cspline.CSpline(x, y, deriv=[None, self.Df(x[-1], c=c)])
        for xi in xtest:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))      
        # cubic
        c = [-2., -1.1, 0.5, 0.25]
        y = self.f(x, c=c)
        s = cspline.CSpline(x, y, deriv=[self.Df(x[0], c=c), self.Df(x[-1], c=c)])
        for xi in xtest:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))      
  
    def test_out_of_range(self):
        " out of range, extrap_order=3 "
        x = np.array([0, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]])
        for xi in [ -1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))
        for xx in [s([]), s.D([]), s.D2([]), s.integ([])]:
            self.assertEqual(list(xx), [])

    def test_out_of_range0(self):
        " out of range, extrap_order=0 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f0(x):
            if np.shape(x) == ():
                return f0(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = self.f(xl)
            ans[x>xr] = self.f(xr)
            return ans
        def Df0(x):
            if np.shape(x) == ():
                return Df0(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans
        def D2f0(x):
            if np.shape(x) == ():
                return D2f0(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans
        def integf0(x):
            if np.shape(x) == ():
                return integf0(np.array([x]))[0]
            ans = self.integf(x)
            ans[x<xl] = (x[x<xl] - xl) * self.f(xl)
            ans[x>xr] = self.integf(xr, xl) + (x[x>xr] - xr) * self.f(xr)
            return ans
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=0)
        for xi in [ -1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f0(xi), s(xi))
            self.assertAlmostEqual(Df0(xi), s.D(xi))
            self.assertAlmostEqual(D2f0(xi), s.D2(xi))
            self.assertAlmostEqual(integf0(xi), s.integ(xi))

    def test_out_of_range1(self):
        " out of range, extrap_order=1 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f1(x):
            if np.shape(x) == ():
                return f1(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = self.f(xl) + (x[x<xl] - xl) * self.Df(xl)
            ans[x>xr] = self.f(xr) + (x[x>xr] - xr) * self.Df(xr)
            return ans
        def Df1(x):
            if np.shape(x) == ():
                return Df1(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = self.Df(xl)
            ans[x>xr] = self.Df(xr)
            return ans
        def D2f1(x):
            if np.shape(x) == ():
                return D2f1(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans
        def integf1(x):
            if np.shape(x) == ():
                return integf1(np.array([x]))[0]
            ans = self.integf(x)
            dx = x[x<xl] - xl
            ans[x<xl] = dx * self.f(xl) + dx**2 * self.Df(xl) / 2.
            dx = x[x>xr] - xr
            ans[x>xr] = self.integf(xr, xl) + dx * self.f(xr) + dx**2 * self.Df(xr) / 2.
            return ans
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=1)
        for xi in [-1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f1(xi), s(xi))
            self.assertAlmostEqual(Df1(xi), s.D(xi))
            self.assertAlmostEqual(D2f1(xi), s.D2(xi))
            self.assertAlmostEqual(integf1(xi), s.integ(xi))

    def test_out_of_range2(self):
        " out of range, extrap_order=2 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f2(x):
            if np.shape(x) == ():
                return f2(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = (
                self.f(xl) + (x[x<xl] - xl) * self.Df(xl)
                + 0.5 * (x[x<xl] - xl) ** 2 * self.D2f(xl)
                )
            ans[x>xr] = (
                self.f(xr) + (x[x>xr] - xr) * self.Df(xr)
                + 0.5 * (x[x>xr] - xr) ** 2 * self.D2f(xr)
                )
            return ans
        def Df2(x):
            if np.shape(x) == ():
                return Df2(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = self.Df(xl) + (x[x<xl] - xl) * self.D2f(xl)
            ans[x>xr] = self.Df(xr) + (x[x>xr] - xr) * self.D2f(xr)
            return ans
        def D2f2(x):
            if np.shape(x) == ():
                return D2f2(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = self.D2f(xl)
            ans[x>xr] = self.D2f(xr)
            return ans
        def integf2(x):
            if np.shape(x) == ():
                return integf2(np.array([x]))[0]
            ans = self.integf(x)
            dx = x[x<xl] - xl
            ans[x<xl] = (
                dx * self.f(xl) + dx**2 * self.Df(xl) / 2.
                + dx**3 * self.D2f(xl) / 6.
                )
            dx = x[x>xr] - xr
            ans[x>xr] = (
                self.integf(xr, xl) + dx * self.f(xr)
                + dx**2 * self.Df(xr) / 2. + dx**3 * self.D2f(xr) / 6.
                )
            return ans
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=2)
        for xi in [-1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f2(xi), s(xi))
            self.assertAlmostEqual(Df2(xi), s.D(xi))
            self.assertAlmostEqual(D2f2(xi), s.D2(xi))
            self.assertAlmostEqual(integf2(xi), s.integ(xi))

    def test_left_natural_bc(self):
        # choose left bdy so that self.D2f(xl) = 0 ==> get exact fcn
        x = np.array([-0.25, 1., 3.])
        c = [1, 2, 3, 4]
        x0 = x[0]
        y = self.f(x, c=c)
        yp= self.Df(x, c=c)
        s = cspline.CSpline(x, y, deriv=[None, yp[-1]], alg='cspline')
        x = [-1., 0., 0.5, 2., 3., 4.][1:-1]
        for xi in x:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))

    def test_right_natural_bc(self):
        # choose right bdy so that self.D2f(xr) = 0 ==> get exact fcn
        x = np.array([-3., -1. , -0.25])
        c = [1, 2, 3, 4]
        x0 = x[0]
        y = self.f(x, c=c)
        yp= self.Df(x, c=c)
        s = cspline.CSpline(x, y, deriv=[yp[0], None], alg='cspline')
        x = [-5., -2., -1., 0., 0.5, 2.]
        for xi in x:
            self.assertAlmostEqual(self.f(xi, c=c), s(xi))
            self.assertAlmostEqual(self.Df(xi, c=c), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi, c=c), s.D2(xi))
            self.assertAlmostEqual(self.D3f(xi, c=c), s.D3(xi))
            self.assertAlmostEqual(self.integf(xi, x0, c=c), s.integ(xi))

    def test_gvar(self):
        x = gvar(['0(1)', '1(1)', '3(1)'])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]])
        for xi in x:
            self.assert_gvclose(self.f(xi), s(xi))
            self.assert_gvclose(self.Df(xi), s.D(xi))
            self.assert_gvclose(self.integf(xi, x0), s.integ(xi))
        x = np.arange(0.4, 3., 0.4)
        for xi in x:
            self.assertAlmostEqual(self.f(xi), mean(s(xi)))
            self.assertGreater(1e-9, sdev(s(xi)))
            self.assertAlmostEqual(self.Df(xi), mean(s.D(xi)))
            self.assertGreater(1e-9, sdev(s.D(xi)))
            self.assertAlmostEqual(self.D2f(xi), mean(s.D2(xi)))
            self.assertGreater(1e-9, sdev(s.D2(xi)))
            self.assertAlmostEqual(mean(self.integf(xi, x0.mean)), mean(s.integ(xi)))
            self.assert_gvclose(s.integ(xi), self.integf(xi, x0))

    def test_sin(self):
        " test CSpline with real function "
        xx = np.linspace(-1.,1.,500)
        fs = cspline.CSpline(xx, np.sin(xx), deriv=[np.cos(xx[0]), np.cos(xx[-1])], alg='cspline')
        for x in [-0.666, -0.333, -0.123, 0.123, 0.333, 0.666]:
            self.assertAlmostEqual(fs(x), np.sin(x), places=9)
            self.assertAlmostEqual(fs.D(x), np.cos(x), places=7)
            self.assertAlmostEqual(fs.D2(x), -np.sin(x), places=4)
            self.assertAlmostEqual(fs.D3(x), -np.cos(x), places=2)
            self.assertAlmostEqual(fs.integ(x), -np.cos(x) + np.cos(-1.), places=9)

        fs = cspline.CSpline(xx, np.sin(xx)) 
        for x in [-0.666, -0.333, -0.123, 0.123, 0.333, 0.666]:
            self.assertAlmostEqual(fs(x), np.sin(x), places=7)
            self.assertAlmostEqual(fs.D(x), np.cos(x), places=4)
            self.assertAlmostEqual(fs.D2(x), -np.sin(x), places=2)
            self.assertAlmostEqual(fs.integ(x), -np.cos(x) + np.cos(-1.), places=8)

        fs = cspline.CSpline(xx, np.sin(xx), alg='pchip') 
        for x in [-0.666, -0.333, -0.123, 0.123, 0.333, 0.666]:
            self.assertAlmostEqual(fs(x), np.sin(x), places=7)
            self.assertAlmostEqual(fs.D(x), np.cos(x), places=4)
            self.assertAlmostEqual(fs.D2(x), -np.sin(x), places=2)
            self.assertAlmostEqual(fs.integ(x), -np.cos(x) + np.cos(-1.), places=8)


    @unittest.skipIf(scipy_interpolate is None,"need scipy for test; not installed")
    def test_against_scipy(self):
        " test CSpline against scipy equivalents "
        x = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        def f(x):
            return np.cos(x) 
        y = f(x)
        yg = cspline.CSpline(x, y, alg='pchip')
        ys = scipy_interpolate.PchipInterpolator(x, y)
        xx = [-0.4, 0.0, 0.4, np.pi/4, 1.0, 1.9, 3.0, 4.0]
        for xi in xx:
            self.assertAlmostEqual(yg(xi), ys(xi))
        yg = cspline.CSpline(x, y, alg='cspline')
        ys = scipy_interpolate.CubicSpline(x, y, bc_type='natural')
        for xi in xx:
            self.assertAlmostEqual(yg(xi), ys(xi))
        yg = cspline.CSpline(x, y, alg='cspline', deriv=[0,0])
        ys = scipy_interpolate.CubicSpline(x, y, bc_type='clamped')
        for xi in xx:
            self.assertAlmostEqual(yg(xi), ys(xi))

class PowerSeriesTests(object):
    def __init__(self):
        pass

    def assert_close(self, a, b, rtol=1e-8, atol=1e-8):
        np.testing.assert_allclose(a.c, b.c, rtol=rtol, atol=atol)

class test_powerseries(unittest.TestCase, PowerSeriesTests):
    """docstring for test_powerseries"""
    def setUp(self):
        self.order = 10
        self.x = PowerSeries([0.,1.], order=self.order)
        self.x2 = PowerSeries([0., 0., 1.], order=self.order)
        self.z = PowerSeries([0+0j, 1.+0j], order=self.order)
        self.one = PowerSeries([1.], order=self.order)
        self.zero = PowerSeries([0.], order=self.order)
        coef = 1. / np.cumprod([1.] + (1. + np.arange(0., self.order+1.)).tolist())
        self.exp_x = PowerSeries(coef, order=self.order)
        osc_coef = coef * (1j)**np.arange(len(coef))
        self.cos_x = PowerSeries([xi.real for xi in osc_coef], order=self.order)
        self.sin_x = PowerSeries([xi.imag for xi in osc_coef], order=self.order)

    def test_constructor(self):
        " PowerSeries(c, order) "
        x = PowerSeries(1. + np.arange(2*self.order), order=self.order)
        self.assertEqual(len(x.c), self.order + 1)
        np.testing.assert_allclose(x.c, 1. + np.arange(self.order + 1))
        x = PowerSeries(order=self.order)
        y = PowerSeries(numpy.zeros(self.order + 1, object), order=self.order)
        self.assertEqual(len(x.c), len(y.c))
        for xi, yi in zip(x.c, y.c):
            self.assertEqual(xi, yi)
        for i in range(self.order + 1):
            self.assertEqual(x.c[i], y.c[i])
        y = PowerSeries(self.exp_x)
        for i in range(self.order + 1):
            y.c[i] *= 2
        self.assert_close(y, 2 * self.exp_x)
        self.assertTrue(PowerSeries(order=0).c == PowerSeries([0]).c)
        with self.assertRaises(ValueError):
            PowerSeries([])
        with self.assertRaises(ValueError):
            PowerSeries(order=-1)
        with self.assertRaises(ValueError):
            PowerSeries([1,2], order=-1)

    def test_arith(self):
        " x+y x-y x*y x/y x**2 "
        x = self.x
        y = self.exp_x
        self.assert_close(x * x, self.x2)
        self.assert_close(y / y, self.one)
        self.assert_close((y * y) / y, y)
        self.assert_close((x * y) / y, x)
        self.assert_close(y - y, self.zero)
        self.assert_close((x + y) - x, y)
        self.assert_close(x + y - x - y, self.zero)
        self.assert_close(x ** 2, self.x2)
        self.assert_close((1 + x) ** 2., 1 + 2*x + self.x2)
        self.assert_close(y * y * y, y ** 3)
        self.assert_close(2 ** x, self.exp_x ** log(2.))
        self.assert_close(y + y, 2 * y)
        self.assert_close(y + y, y * 2)
        self.assert_close(x + 2, PowerSeries([2, 1], order=self.order))
        self.assert_close(2 + x, PowerSeries([2, 1], order=self.order))
        self.assert_close(x - 2, PowerSeries([-2, 1], order=self.order))
        self.assert_close(2 - x, PowerSeries([2, -1], order=self.order))
        self.assert_close(2 * (y / 2), y)
        self.assert_close(y * (2 / y), 2 * self.one)
        self.assertEqual(y ** 0, 1.)
        self.assert_close(y ** (-2), 1 / y / y)

        # check division where c[0] = 0
        self.assert_close(x / x, PowerSeries([1], order=self.order - 1))
        self.assert_close((x + x ** 2) / x, PowerSeries([1, 1], order=self.order-1))
        self.assert_close((x * x) / self.x2, PowerSeries([1], order=self.order - 2))

        # check error checks
        with self.assertRaises(ZeroDivisionError):
            self.x / self.zero

    def test_sqrt(self):
        " sqrt "
        y = self.exp_x
        self.assert_close(sqrt(y) ** 2, y)
        self.assert_close(sqrt(y ** 2), y)
        self.assert_close(sqrt(y), y ** 0.5)

    def test_exp(self):
        x = self.x
        y = self.exp_x
        self.assert_close(exp(x), self.exp_x)
        self.assert_close(log(exp(y)), y)
        self.assert_close(2 ** y, exp(log(2) * y))
        self.assert_close(y ** x, exp(log(y) * x))
        self.assert_close(y ** 2.5, exp(log(y) * 2.5))

    def test_trig(self):
        jx = self.x * 1j
        x = self.x * (1+0j)
        self.assert_close(sin(x), (exp(jx) - exp(-jx))/2j)
        self.assert_close(cos(x), (exp(jx) + exp(-jx))/2)
        x = self.x
        self.assert_close(self.sin_x, sin(self.x))
        self.assert_close(self.cos_x, cos(self.x))
        self.assert_close(tan(x), sin(x) / cos(x))
        self.assert_close(cos(arccos(x)), x)
        self.assert_close(arccos(cos(1 + x)), 1 + x)
        self.assert_close(sin(arcsin(x)), x)
        self.assert_close(arcsin(sin(1 + x)), 1 + x)
        self.assert_close(tan(arctan(x)), x)
        self.assert_close(arctan(tan(1 + x)), 1 + x)

    def test_hyp(self):
        x = self.x
        self.assert_close(sinh(x), (self.exp_x - 1 / self.exp_x)/2)
        self.assert_close(cosh(x), (self.exp_x + 1 / self.exp_x)/2)
        self.assert_close(tanh(x), sinh(x) / cosh(x))
        self.assert_close(arcsinh(sinh(x)), x)
        self.assert_close(sinh(arcsinh(x)), x)
        self.assert_close(arccosh(cosh(2 + x)), 2 + x)
        self.assert_close(cosh(arccosh(1.25 + x)), 1.25 + x)
        self.assert_close(arctanh(tanh(0.25 + x)), 0.25 + x)
        self.assert_close(tanh(arctanh(0.25 + x)), 0.25 + x)

    def test_call(self):
        self.assert_close(self.sin_x(self.x), sin(self.x))
        f = log(1 + self.x)
        self.assert_close(self.exp_x(log(1 + self.x)), 1 + self.x)

    def test_str(self):
        " str(p) repr(p) "
        self.assertEqual(
            str(self.x),
            str(np.array([ 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
            )
        y = eval(repr(self.exp_x))
        self.assert_close(y, self.exp_x)

    def test_deviv_integ(self):
        " p.deriv() p.integ() "
        self.assert_close(self.exp_x.integ().deriv(), self.exp_x)
        self.assert_close(self.exp_x.integ(2).deriv(2), self.exp_x)
        self.assert_close(self.exp_x.integ(0), self.exp_x)
        self.assert_close(self.exp_x.deriv(0), self.exp_x)
        self.assertTrue(self.x.deriv(self.order + 2) == 0.0)
        self.assert_close(
            self.exp_x.deriv(),
            PowerSeries(self.exp_x, order=self.order - 1)
            )
        self.assert_close(
            self.exp_x.integ(x0=0),
            exp(PowerSeries(self.x, order=self.order+1)) - 1
            )

    def test_multiseries(self):
        def layout_test(m):
            ndim = 0 
            c = m.c[0]
            while isinstance(c, PowerSeries):
                ndim += 1 
                c = c.c[0]
            shape = ndim * (m.order + 1, )
            for idx in np.ndindex(shape):
                try:
                    m[idx]
                    if sum(idx) > m.order:
                        raise AssertionError('sum(idx) > m.order')
                except:
                    self.assertGreater(sum(idx), m.order)
        # begin
        c = np.arange(2 * 3 * 4, dtype=float)
        c.shape = (2, 3, 4)

        # check order
        m = multiseries(c.tolist(), order=3)
        assert m.order == 3
        layout_test(m)
        for ijk in np.ndindex(7, 7, 7):
            try:
                assert m[ijk] == (c[ijk] if np.all(np.array(c.shape) > ijk) else 0.)
                if sum(ijk) > m.order:
                    raise AssertionError('sum(ijk) > m.order')
            except:
                self.assertGreater(sum(ijk), 3)

        # now without order; arithmetic; function evaluation
        m = multiseries(c)
        # check linear arithmetic 
        c = 2 * c + c / 2
        m = 2 * m + m / 2
        c[0, 0, 0] = 13.
        m[0, 0, 0] = 13.
        assert m.order == 6
        for ijk in np.ndindex(7, 7, 7):
            try:
                assert m[ijk] == (c[ijk] if np.all(np.array(c.shape) > ijk) else 0.)
                if sum(ijk) > m.order:
                    raise AssertionError('sum(ijk) > m.order')
            except:
                self.assertGreater(sum(ijk), m.order)
        def f(x, y, z):
            return sum(m[ijk] * x ** ijk[0] * y ** ijk[1] * z ** ijk[2] for ijk in np.ndindex(c.shape))
        self.assertEqual(f(1,2,3), m(1, 2, 3))

        # derivatives
        def Df(x, y, z):
            return [
                sum(ijk[0] * m[ijk] * x ** (ijk[0] - 1) * y ** ijk[1] * z ** ijk[2] for ijk in np.ndindex(c.shape)),
                sum(ijk[1] * m[ijk] * x ** ijk[0] * y ** (ijk[1] - 1) * z ** ijk[2] for ijk in np.ndindex(c.shape)),
                sum(ijk[2] * m[ijk] * x ** ijk[0] * y ** ijk[1] * z ** (ijk[2] - 1) for ijk in np.ndindex(c.shape)),
            ]
        self.assertEqual(Df(1,2,3), [m.deriv(1,0,0)(1,2,3), m.deriv(0,1,0)(1,2,3), m.deriv(0,0,1)(1,2,3)])
        self.assertEqual(Df(1,2,3), [m.deriv(1)(1,2,3), m.deriv(0,1)(1,2,3), m.deriv(0,0,1)(1,2,3)])
        layout_test(m.deriv(1))
        self.assertEqual(m.deriv(1).order, m.order - 1)
        layout_test(m.deriv(1, 1))
        layout_test(m.deriv(1, 2, 3))
        self.assertEqual(m.deriv(1, 2, 3).order, m.order - 6)
        self.assertEqual(m.deriv(1,1)(1,2,3), m.deriv(1,1,0)(1,2,3))
        def DDf(x, y, z):
            return [
                sum(ijk[0] * ijk[1] * m[ijk] * x ** (ijk[0] - 1) * y ** (ijk[1] - 1) * z ** ijk[2] for ijk in np.ndindex(c.shape)),
                sum(ijk[1] * ijk[2] * m[ijk] * x ** ijk[0] * y ** (ijk[1] - 1) * z ** (ijk[2] - 1) for ijk in np.ndindex(c.shape)),
                sum(ijk[2] * ijk[0] * m[ijk] * x ** (ijk[0] - 1) * y ** ijk[1] * z ** (ijk[2] - 1) for ijk in np.ndindex(c.shape)),
            ]
        self.assertEqual(DDf(1,2,3), [m.deriv(1,1,0)(1,2,3), m.deriv(0,1,1)(1,2,3), m.deriv(1,0,1)(1,2,3)])
        def D2f(x, y, z):
            return [
                sum(ijk[0] * (ijk[0] - 1) * m[ijk] * x ** (ijk[0] - 2) * y ** ijk[1] * z ** ijk[2] for ijk in np.ndindex(c.shape)),
                sum(ijk[1] * (ijk[1] - 1) * m[ijk] * x ** ijk[0] * y ** (ijk[1] - 2) * z ** ijk[2] for ijk in np.ndindex(c.shape)),
                sum(ijk[2] * (ijk[2] - 1) * m[ijk] * x ** ijk[0] * y ** ijk[1] * z ** (ijk[2] - 2) for ijk in np.ndindex(c.shape)),
            ]
        self.assertEqual(D2f(1,2,3), [m.deriv(2,0,0)(1,2,3), m.deriv(0,2,0)(1,2,3), m.deriv(0,0,2)(1,2,3)])
        self.assertEqual(D2f(1,2,3), [m.deriv(2)(1,2,3), m.deriv(0,2)(1,2,3), m.deriv(0,0,2)(1,2,3)])
        self.assertEqual([0.,0.,0.],[ m.deriv(m.order + 1), m.deriv(0, m.order + 1), m.deriv(0, 0, m.order+1)])

        # derivatives undo integrals (not true of reverse)
        self.assertEqual(m(1,2,3), m.integ(1,2,3).deriv(1,2,3)(1,2,3))
        self.assertEqual(m(1,2,3), m.integ(1,2,3, x0=[3,2,1]).deriv(1,2,3)(1,2,3))
        self.assertEqual(m(1,2,3), m.integ(0,0,3).deriv(0,0,3)(1,2,3))
        self.assertEqual(m(1,2,3), m.integ(3).deriv(3,0,0)(1,2,3))

        # integrate from x0=[0,0,0]
        mm = m.integ(1,2,3)
        self.assertNotEqual(mm(1,2,3), 0)
        assert np.allclose(
            0, 
            [mm(0,2,3), mm(1,0,3), mm(1,2,0), mm(1,0,0), mm(0,2,0), mm(0,0,3), mm(0,0,0)],
            )
        # integrate from x0=[1,2,3]
        mm = m.integ(1,2,3, x0=[1,2,3])
        assert np.allclose(
            0, 
            [mm(1,2,3), mm(0,2,3), mm(1,0,3), mm(1,2,0), mm(1,0,0), mm(0,2,0), mm(0,0,3)],
            )
        assert not np.allclose(m(0,0,0), 0)
        assert not np.allclose(m(2,0,0), 0)
        assert not np.allclose(m(0,3,0), 0)
        assert not np.allclose(m(0,0,1), 0)

        # multivar
        c = np.arange(1 * 2 * 3, dtype=float)
        c.shape = (1, 2, 3)
        m = multiseries(c)
        x, y, z = multivar(3, order=m.order)
        newm = sum(c[ijk] * x ** ijk[0] * y ** ijk[1] * z ** ijk[2] for ijk in np.ndindex(c.shape))
        self.assertEqual(newm.order, m.order)
        for ijk in np.ndindex(3 * (m.order,)):
            if sum(ijk) <= m.order:
                self.assertEqual(newm[ijk], m[ijk])
        exp1 = gv.exp(x + 2 * y + z)
        exp2 = gv.exp(x + 2 * y) * gv.exp(z)
        assert np.allclose(exp1(0,0).c, [1., 1., 0.5, 1/6.])
        assert exp1.order == 3 and exp2.order == 3
        assert np.allclose(exp1(0,0).c, [1., 1., 0.5, 1/6.])
        for ijk in np.ndindex(3 * (exp1.order,)):
            if sum(ijk) <= m.order:
                self.assertEqual(exp1[ijk], exp2[ijk])

class test_root(unittest.TestCase,ArrayTests):
    def setUp(self):
        pass

    def test_search(self):
        " root.search(fcn, x0) "
        interval = root.search(np.sin, 0.5, incr=0.5, fac=1.)
        np.testing.assert_allclose(interval, (3.0, 3.5))
        self.assertEqual(interval.nit, 6)
        interval = root.search(np.sin, 1.0, incr=0.0, fac=1.5)
        np.testing.assert_allclose(interval, (27./8., 9./4.))
        with self.assertRaises(RuntimeError):
            root.search(np.sin, 0.5, incr=0.5, fac=1., maxit=5)

    def test_refine(self):
        " root.refine(fcn, interval) "
        def fcn(x):
            return (x + 1) ** 3 * (x - 0.5) ** 11
        r = root.refine(fcn, (0.1, 2.1))
        self.assertAlmostEqual(r, 0.5)
        r = root.refine(fcn, (2.1, 0.1))
        self.assertAlmostEqual(r, 0.5)
        rtol = 0.1/100000
        nit = 0
        for rtol in [0.1, 0.01, 0.001, 0.0001]:
            r = root.refine(fcn, (0.1, 2.1), rtol=rtol)
            self.assertGreater(rtol * 0.5, abs(r - 0.5))
            self.assertGreater(r.nit, nit)
            nit = r.nit
        def f(x, w=gv.gvar(1,0.1)):
            return np.sin(w * x)
        r = root.refine(f, (1, 4))
        self.assertAlmostEqual(r.mean, np.pi)
        self.assertAlmostEqual(r.sdev, 0.1 * np.pi)

class test_linalg(unittest.TestCase, ArrayTests):
    def setUp(self):
        pass

    def make_random(self, a, g='1.0(1)'):
        a = numpy.asarray(a)
        ans = numpy.empty(a.shape, object)
        for i in numpy.ndindex(a.shape):
            ans[i] = a[i] * gvar(g)
        return ans

    def test_det(self):
        m = self.make_random([[1., 0.1], [0.1, 2.]])
        detm = linalg.det(m)
        self.assertTrue(gv.equivalent(detm, m[0,0] * m[1,1] - m[0,1] * m[1,0]))
        m = self.make_random([[1.0,2.,3.],[0,4,5], [0,0,6]])
        self.assertTrue(gv.equivalent(linalg.det(m), m[0, 0] * m[1, 1] * m[2, 2]))
        s, logdet = linalg.slogdet(m)
        self.assertTrue(gv.equivalent(s * gv.exp(logdet), linalg.det(m)))

    def test_inv(self):
        m = self.make_random([[1., 0.1], [0.1, 2.]])
        one = gv.gvar([['1(0)', '0(0)'], ['0(0)', '1(0)']])
        invm = linalg.inv(m)
        self.assertTrue(gv.equivalent(linalg.inv(invm), m))
        for mm in [invm.dot(m), m.dot(invm)]:
            np.testing.assert_allclose(
                gv.mean(mm), [[1, 0], [0, 1]], rtol=1e-10, atol=1e-10
                )
            np.testing.assert_allclose(
                gv.sdev(mm), [[0, 0], [0, 0]], rtol=1e-10, atol=1e-10
                )
        p = linalg.det(m) * linalg.det(invm)
        self.assertAlmostEqual(p.mean, 1.)
        self.assertGreater(1e-10, p.sdev)

    def test_solve(self):
        m = self.make_random([[1., 0.1], [0.2, 2.]])
        b = self.make_random([3., 4.])
        x = linalg.solve(m, b)
        self.assertTrue(gv.equivalent(m.dot(x), b))
        m = self.make_random([[1., 0.1], [0.2, 2.]])
        b = self.make_random([[3., 1., 0.], [4., 2., 1.]])
        x = linalg.solve(m, b)
        self.assertTrue(gv.equivalent(m.dot(x), b))

    def test_eigvalsh(self):
        " linalg.eigvalsh(a) "
        # just eigenvalues
        m = gv.gvar([['2.1(1)', '0(0)'], ['0(0)', '0.5(3)']])
        th = 0.92
        cth = numpy.cos(th)
        sth = numpy.sin(th)
        u = numpy.array([[cth, sth], [-sth, cth]])
        mrot = u.T.dot(m.dot(u))
        val =  linalg.eigvalsh(mrot)
        self.assertTrue(gv.equivalent(val[0], m[1, 1]))
        self.assertTrue(gv.equivalent(val[1], m[0, 0]))

    def test_eigh(self):
        " linalg.eigh(a) "
        m = gv.gvar([
            ['2.0000001(1)', '0(0)', '0(0)'],
            ['0(0)', '2.0(3)', '0(0)'],
            ['0(0)', '0(0)', '5.0(2)']])
        th = 0.92
        cth = numpy.cos(th) * gv.gvar('1.0(1)')
        sth = numpy.sin(th) * gv.gvar('1.0(1)')
        u = numpy.array([[cth, sth], [-sth, cth]])
        mrot = np.array(m)
        mrot[:2, :2] = u.T.dot(m[:2, :2].dot(u))
        mrot[-2:, -2:] = u.T.dot(mrot[-2:, -2:].dot(u))
        val, vec = linalg.eigh(mrot)
        self.assertTrue(gv.equivalent(
            mrot.dot(vec[:, 0]), val[0] * vec[:, 0]
            ))
        self.assertTrue(gv.equivalent(
            mrot.dot(vec[:, 1]), val[1] * vec[:, 1]
            ))
        # test against numpy
        valnp, vecnp = np.linalg.eigh(gv.mean(mrot))
        np.testing.assert_allclose(valnp, gv.mean(val))
        np.testing.assert_allclose(vecnp, gv.mean(vec))

    def test_svd(self):
        " linalg.svd(a) "
        def reassemble(u,s,vT):
            ans = 0
            for i in range(len(s)):
                ans += u[:, i][:, None] * s[i] * vT[i, :][None, :]
            return ans
        aa = np.array([[1.,2.,3.], [2.5, 4., 5.], [3., 5., 6.]])
        aa = self.make_random(aa)
        for a in [aa, aa[:, :2], aa[:2, :], aa[:, :1], aa[:1, :]]:
            u, s, vT = linalg.svd(a)
            self.assertTrue(gv.equivalent(reassemble(u,s,vT), a))
            # test against numpy
            unp, snp, vTnp = np.linalg.svd(gv.mean(a), full_matrices=False)
            np.testing.assert_allclose(unp, gv.mean(u))
            np.testing.assert_allclose(vTnp, gv.mean(vT))
            np.testing.assert_allclose(snp, gv.mean(s))

    def test_lstsq(self):
        " linalg.lstsq(a) "
        x = np.arange(0.1, 1.1, .1)
        y = self.make_random(2 + x, '1.00(1)') * gv.gvar('1.00(1)')
        x = self.make_random(x, '1.00(1)')
        M = np.array([np.ones(len(x), float), x]).T
        c = linalg.lstsq(M, y, extrainfo=False, rcond=0, weighted=True)
        self.assertTrue(abs(c[0].mean - 2) < 5 * c[0].sdev)
        self.assertTrue(abs(c[1].mean - 1) < 5 * c[1].sdev)
        c, residual, rank, s = linalg.lstsq(M, y, extrainfo=True,rcond=0, weighted=False)
        self.assertTrue(abs(c[0].mean - 2) < 5 * c[0].sdev)
        self.assertTrue(abs(c[1].mean - 1) < 5 * c[1].sdev)
        self.assertEqual(rank, 2)
        # test against numpy
        cnp, residual, rank, s = np.linalg.lstsq(gv.mean(M), gv.mean(y), rcond=None)
        np.testing.assert_allclose(cnp, gv.mean(c))

class test_pade(unittest.TestCase,ArrayTests):
    def setUp(self): pass

    def tearDown(self): pass

    def test_Pade(self):
        pass 

    def test_pade_svd(self):
        " pade_svd(tayl, n, m) "
        optprint('\n=========== Test pade_svd')
        # Taylor expansion for exp(x)
        e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]

        # test against scipy
        p0, q0 = Pade._scipy_pade(e_exp, 2)
        p, q = pade_svd(e_exp, 3, 2)
        assert numpy.allclose(p, p0)
        assert numpy.allclose(q, q0)
        optprint('(3,2) Pade of exp(x) - num:', p)
        optprint('(3,2) Pade of exp(x) - den:', q)
        e = sum(p) / sum(q)
        optprint('Pade(x=1) = {:.6}    error = {:7.2}'.format(
            e,
            abs(e/numpy.exp(1) - 1.),
            ))

        # now with 10% errors --- automatically reduces to (2,1)
        p0, q0 = Pade._scipy_pade(e_exp[:4], 1)
        p, q = pade_svd(e_exp, 3, 2, rtol=0.1)
        assert numpy.allclose(p, p0)
        assert numpy.allclose(q, q0)
        optprint('(2,1) Pade of exp(x) - num:', p)
        optprint('(2,1) Pade of exp(x) - den:', q)
        e = sum(p) / sum(q)
        optprint('Pade(x=1) = {:.6}    error = {:7.2}'.format(
            e,
            abs(e/numpy.exp(1) - 1.)
            ))

        # now with 90% errors --- automatically reduces to (1,0)
        p, q = pade_svd(e_exp, 3, 2, rtol=0.9)
        optprint('(1,0) Pade of exp(x) - num:', p)
        optprint('(1,0) Pade of exp(x) - den:', q)
        e = sum(p) / sum(q)
        optprint('Pade(x=1) = {:.6}    error = {:7.2}'.format(
            e,
            abs(e/numpy.exp(1) - 1.)
            ))
        assert numpy.allclose(p, [1., 1.])
        assert numpy.allclose(q, [1.])

    def test_pade_svd_consistency(self):
        " pade_svd self consistency "
        # high-order taylor series
        x = gv.powerseries.PowerSeries([0,1], order=20)
        f = np.exp(x).c
        # verify that reduced-order Pades are exact Pades
        m,n = 7,7
        for rtol in [1, 0.1, 0.01, 0.001]:
            a, b = pade_svd(f, m, n, rtol=rtol)
            mm = len(a) - 1
            nn = len(b) - 1
            if (m,n) != (mm,nn):
                aa, bb = pade_svd(f, mm, nn)
                self.assertTrue(np.allclose(aa, a))
                self.assertTrue(np.allclose(bb, b))

    def test_pade_gvar(self):
        " pade_gvar(tayl, m, n) and Pade(tayl, order=(m,n))"
        optprint('\n=========== Test pade_gvar')
        e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0, 1.0/720.]
        def _scipy_pade(m, n):
            return Pade._scipy_pade(e_exp[:m + n + 1], n)
        def print_result(p, q):
            optprint('num =', p)
            optprint('den =', q)
        def test_result(p, q, e_exp):
            m = len(p) - 1
            n = len(q) - 1
            # test against scipy
            p0, q0 = _scipy_pade(m, n)
            try:
                assert numpy.allclose(mean(p), p0)
            except:
                print (m,n, p0, p, q0, q)
            assert numpy.allclose(mean(q), q0)
            # test that errors correlate with input coefficients
            num = powerseries.PowerSeries(p, order=m + n)
            den = powerseries.PowerSeries(q, order=m + n)
            ratio = (num/den).c / e_exp[:m + n + 1]
            assert numpy.allclose(mean(ratio), 1.)
            assert numpy.allclose(sdev(ratio), 0.0)

        # print('scipy', _scipy_pade(1,1), pade_svd(e_exp, 3,2, rtol=0.01))
        # 1% noise --- automatically reduces to (2,1)
        e_exp_noise = [x * gvar('1.0(1)') for x in e_exp]
        p, q = pade_gvar(e_exp_noise, 3, 2)
        print_result(p, q)
        self.assertEqual(len(p), 3)
        self.assertEqual(len(q), 2)
        test_result(p, q, e_exp_noise)

        # 30% noise --- automatically reduces to (1,1)
        e_exp_noise = [x * gvar('1.0(3)') for x in e_exp]
        p, q = pade_gvar(e_exp_noise, 3, 2)
        self.assertEqual(len(p), 2)
        self.assertEqual(len(q), 2)
        test_result(p, q, e_exp_noise)

        # 30% noise, rtol=None --- no reduction
        e_exp_noise = [x * gvar('1.0(3)') for x in e_exp]
        p, q = pade_gvar(e_exp_noise, 3, 2, rtol=None)
        self.assertEqual(len(p), 4)
        self.assertEqual(len(q), 3)
        test_result(p, q, e_exp_noise)


if __name__ == '__main__':
    unittest.main()
