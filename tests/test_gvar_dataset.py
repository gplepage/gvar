#!/usr/bin/env python
# encoding: utf-8
"""
test-dataset.py

"""
# Copyright (c) 2012-17 G. Peter Lepage.
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

import pickle
import os
import unittest
import warnings
import numpy as np
import random
import gvar as gv
from gvar import *
from gvar.dataset import *

FAST = False

try:
    import h5py
    NO_H5PY = False
except:
    NO_H5PY = True

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



class test_dataset(unittest.TestCase,ArrayTests):
    def setUp(self): pass

    def tearDown(self): pass

    def test_bin_data(self):
        """ bin_data """
        self.assertEqual(bin_data([1,2,3,4]),[1.5,3.5])
        self.assertEqual(bin_data(np.array([1,2,3,4])),[1.5,3.5])
        self.assert_arraysequal(bin_data([[1,2],[3,4]]),[[2.,3.]])
        self.assert_arraysequal(bin_data([[[1,2]],[[3,4]]]),[[[2.,3.]]])
        self.assertEqual(bin_data([1]),[])
        self.assertEqual(bin_data([1,2,3,4,5,6,7],binsize=3),[2.,5.])
        data = dict(s=[1,2,3,4],
                    v=[[1,2],[3,4],[5,6,],[7,8],[9,10]])
        bd = bin_data(data)
        self.assertEqual(bd['s'],[1.5,3.5])
        self.assert_arraysequal(bd['v'],[[2,3],[6,7]])
        data = dict(s=[1,2,3,4],
                    v=[[1,2],[3,4],[5,6,],[7,8],[9,10]])
        bd = bin_data(data,binsize=3)
        self.assertEqual(bd['s'],[2])
        self.assert_arraysequal(bd['v'],[[3,4]])
        with self.assertRaises(ValueError):
            bd = bin_data([[1,2],[[3,4]]])
        self.assertEqual(bin_data([]),[])
        self.assertEqual(bin_data(dict()),Dataset())

    def test_avg_data(self):
        """ avg_data """
        self.assertTrue(avg_data([]) is None)
        self.assertEqual(avg_data(dict()),BufferDict())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            avg_data(dict(s=[1.],v=[1.,2.]))
            self.assertEqual(len(w), 1)
        with self.assertRaises(ValueError):
            avg_data(dict(s=[],v=[1.,2.]), warn=False)
        with self.assertRaises(ValueError):
            avg_data(dict(s=[], v=[]))
        with self.assertRaises(ValueError):
            avg_data([1,2,"s"])
        mean = avg_data([1])
        self.assertEqual(mean.mean,1.)
        self.assertEqual(mean.sdev,0.)
        #
        mean = avg_data([1,2])
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,sum((vi-1.5)**2
                               for vi in [1,2])/4.)
        mean2 = avg_data(np.array([1.,2.]))
        self.assertEqual(mean.mean,mean2.mean)
        self.assertEqual(mean.sdev,mean2.sdev)
        #
        mean = avg_data([1,2],spread=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,sum((vi-1.5)**2
                               for vi in [1,2])/2.)
        #
        mean = avg_data([1,2],median=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,0.5**2/2.)
        #
        mean = avg_data([1,2],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,0.5**2)
        #
        mean = avg_data([1,2,3])
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,sum((vi-2.)**2
                               for vi in [1,2,3])/9.)
        #
        mean = avg_data([1,2,3], noerror=True)
        self.assertAlmostEqual(mean, 2.0)
        #
        mean = avg_data([[1],[2],[3]])
        self.assertAlmostEqual(mean[0].mean,2.0)
        self.assertAlmostEqual(mean[0].var,sum((vi-2.)**2
                               for vi in [1,2,3])/9.)

        mean = avg_data([[1],[2],[3]], noerror=True)
        self.assertAlmostEqual(mean[0], 2.0)

        mean = avg_data([1,2,3],spread=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,sum((vi-2.)**2
                               for vi in [1,2,3])/3.)
        #
        mean = avg_data([1,2,3],median=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,1./3.)
        #
        mean = avg_data([[1],[2],[3]],median=True)
        self.assertAlmostEqual(mean[0].mean,2.0)
        self.assertAlmostEqual(mean[0].var,1./3.)
        #
        mean = avg_data([1,2,3],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,1.)
        #
        mean = avg_data([1,2,3,4,5,6,7,8,9],median=True)
        self.assertAlmostEqual(mean.mean,5)
        self.assertAlmostEqual(mean.var,3.**2/9.)
        #
        mean = avg_data([1,2,3,4,5,6,7,8,9],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,5.)
        self.assertAlmostEqual(mean.var,3.**2)
        #
        mean = avg_data([1,2,3,4,5,6,7,8,9,10],median=True)
        self.assertAlmostEqual(mean.mean,5.5)
        self.assertAlmostEqual(mean.var,3.5**2/10.)
        #
        mean = avg_data([1,2,3,4,5,6,7,8,9,10],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,5.5)
        self.assertAlmostEqual(mean.var,3.5**2)
        #
        data = dict(s=[1,2,3],v=[[1,1],[2,2],[3,3]])
        mean = avg_data(data,median=True,spread=True)
        self.assertAlmostEqual(mean['s'].mean,2.0)
        self.assertAlmostEqual(mean['s'].var,1.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_gvclose(mean['v'],[gvar(2,1),gvar(2,1)])

        mean = avg_data(data, median=True, noerror=True)
        self.assertAlmostEqual(mean['s'],2.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_arraysclose(mean['v'], [2,2])

        mean = avg_data(data, noerror=True)
        self.assertAlmostEqual(mean['s'],2.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_arraysclose(mean['v'], [2,2])

    def test_autocorr(self):
        """ dataset.autocorr """
        N = 10000
        eps = 10./float(N)**0.5
        x = gvar(2,0.1)
        a = np.array([x() for i in range(N)])
        a = (a[:-2]+a[1:-1]+a[2:])/3.
        ac_ex = np.zeros(a.shape,float)
        ac_ex[:3] = np.array([1.,0.66667,0.33333])
        ac_a = autocorr(a)
        self.assertLess(numpy.std(ac_a-ac_ex)*2,eps)
        b = np.array([[x(),x()] for i in range(N)])
        b = (b[:-2]+b[1:-1]+b[2:])/3.
        ac_ex = np.array(list(zip(ac_ex,ac_ex)))
        ac_b = autocorr(b)
        self.assertLess(numpy.std(ac_b-ac_ex),eps)
        c = dict(a=a,b=b)
        ac_c = autocorr(c)
        self.assert_arraysequal(ac_c['a'],ac_a)
        self.assert_arraysequal(ac_c['b'],ac_b)

    def test_dataset_append(self):
        """ Dataset.append() """
        data = Dataset()
        data.append(s=1,v=[10,100])
        self.assert_arraysequal(data['s'],[1.])
        self.assert_arraysequal(data['v'],[[10.,100.]])
        data.append(s=2,v=[20,200])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.append(dict(s=3,v=[30,300]))
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.append('s',4.)
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.append('v',[40.,400.])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal( #
            data['v'],[[10.,100.],[20.,200.],[30.,300.],[40.,400.]])
        with self.assertRaises(ValueError):
            data.append('v',5.)
        with self.assertRaises(ValueError):
            data.append('s',[5.])
        with self.assertRaises(ValueError):
            data.append('s',"s")
        with self.assertRaises(ValueError):
            data.append('v',[[5.,6.]])
        with self.assertRaises(ValueError):
            data.append('v',[.1],'s')
        with self.assertRaises(ValueError):
            data.append([1.])
        #
        data = Dataset()
        data.append('s',1)
        self.assertEqual(data['s'],[1.])
        data = Dataset()
        data.append(dict(s=1,v=[10,100]))
        self.assertEqual(data['s'],[1.])
        self.assert_arraysequal(data['v'],[[10.,100.]])

    def test_dataset_extend(self):
        """ Dataset.extend """
        data = Dataset()
        data.extend(s=[1,2],v=[[10.,100.],[20.,200.]])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(s=[])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(s=[3],v=[[30,300]])
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.extend('s',[4.])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.extend('v',[[40,400.]])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal( #
            data['v'],[[10.,100.],[20.,200.],[30.,300.],[40.,400.]])
        with self.assertRaises(TypeError):
            data.extend('s',5.)
        with self.assertRaises(ValueError):
            data.extend('v',[5.,6.])
        with self.assertRaises(ValueError):
            data.extend('s',"s")
        with self.assertRaises(ValueError):
            data.extend('v',[[[5.,6.]]])
        #
        with self.assertRaises(ValueError):
            data.extend('v',[[5,6],[[5.,6.]]])
        with self.assertRaises(ValueError):
            data.extend('v',[.1],'s')
        with self.assertRaises(ValueError):
            data.extend([1.])
        #
        data = Dataset()
        data.extend(dict(s=[1,2],v=[[10.,100.],[20.,200.]]))
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(dict(s=[3],v=[[30,300]]))
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data = Dataset()
        data.extend('s',[1,2])
        data.extend('v',[[10.,100.],[20.,200.]])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])

    def test_dataset_init(self):
        """ Dataset() """
        fin = ['test-gvar.input1','test-gvar.input2']
        with open(fin[0],"w") as f:
            f.write("""
                # first
                s 1
                v 10 100
                #second
                s 2
                v 20 200
                s 3
                v 30 300
                """)  # """
        with open(fin[1],"w") as f:
            f.write("""
                a [[1,10]]
                a [[2,20]]
                a [[3,30]]
                """)  # """
        data = Dataset(fin[0])
        self.assertEqual(data['s'],[1,2,3])
        self.assert_arraysequal(data['v'],[[10,100],[20,200],[30,300]])
        data = Dataset(fin)
        self.assertEqual(data['s'],[1,2,3])
        self.assert_arraysequal(data['v'],[[10,100],[20,200],[30,300]])
        self.assert_arraysequal(data['a'],[[[1,10]],[[2,20]],[[3,30]]])
        data = Dataset(fin[0],binsize=2)
        self.assertEqual(data['s'],[1.5])
        self.assert_arraysequal(data['v'],[[15,150]])
        data = Dataset(fin,keys=['s'])
        self.assertTrue('v' not in data)
        self.assertTrue('a' not in data)
        self.assertTrue('s' in data)
        self.assertEqual(data['s'],[1,2,3])
        with self.assertRaises(TypeError):
            data = Dataset("xxx.input1","xxx.input2")
        os.remove(fin[0])
        os.remove(fin[1])

    def test_dataset_init2(self):
        """ init from dictionaries or datasets """
        def assert_dset_equal(d1, d2):
            for k in d1:
                assert k in d2, 'key mismatch'
            for k in d2:
                assert k in d1, 'key mismatch'
                self.assertTrue(np.all(np.array(d1[k]) == np.array(d2[k])))
        data = Dataset(dict(a=[[1.,3.], [3.,4.]], b=[1., 2.]))
        data_reduced = Dataset(dict(a=[[1.,3.], [3.,4.]]))
        data_binned = Dataset(dict(a=[[2.,3.5]], b=[1.5]))
        data_empty = Dataset()
        self.assertEqual(data['a'], [[1.,3.], [3.,4.]])
        self.assertEqual(data['b'], [1., 2.])
        assert_dset_equal(data, Dataset(data))
        assert_dset_equal(data_reduced, Dataset(data,keys=['a']))
        assert_dset_equal(data,
            Dataset([('a', [[1.,3.], [3.,4.]]), ('b', [1., 2.])])
            )
        assert_dset_equal(data,
            Dataset([['a', [[1.,3.], [3.,4.]]], ['b', [1., 2.]]])
            )
        assert_dset_equal(data_reduced, Dataset(data, keys=['a']))
        assert_dset_equal(data_reduced, Dataset(data, grep='[^b]'))
        assert_dset_equal(data_empty, Dataset(data, grep='[^b]', keys=['b']))
        assert_dset_equal(data_binned, Dataset(data, binsize=2))
        assert_dset_equal(
            Dataset(data_binned, keys=['a']),
            Dataset(data, binsize=2, keys=['a'])
            )
        assert_dset_equal(
            Dataset(data_binned, keys=['a']),
            Dataset(data, binsize=2, grep='[^b]')
            )
        assert_dset_equal(
            Dataset(data_binned, keys=['a']),
            Dataset(data, binsize=2, grep='[^b]', keys=['a'])
            )
        s = pickle.dumps(data)
        assert_dset_equal(data, pickle.loads(s))

    def test_dataset_toarray(self):
        """ Dataset.toarray """
        data = Dataset()
        data.extend(s=[1,2],v=[[1,2],[2,3]])
        data = data.toarray()
        self.assert_arraysequal(data['s'],[1,2])
        self.assert_arraysequal(data['v'],[[1,2],[2,3]])
        self.assertEqual(data['s'].shape,(2,))
        self.assertEqual(data['v'].shape,(2,2))

    def test_dataset_slice(self):
        """ Dataset.slice """
        data = Dataset()
        data.extend(a=[1,2,3,4],b=[[1],[2],[3],[4]])
        ndata = data.slice(slice(0,None,2))
        self.assert_arraysequal(ndata['a'],[1,3])
        self.assert_arraysequal(ndata['b'],[[1],[3]])

    def test_dataset_grep(self):
        """ Dataset.grep """
        data = Dataset()
        data.extend(aa=[1,2,3,4],ab=[[1],[2],[3],[4]])
        ndata = data.grep("a")
        self.assertTrue('aa' in ndata and 'ab' in ndata)
        self.assert_arraysequal(ndata['ab'],data['ab'])
        self.assert_arraysequal(ndata['aa'],data['aa'])
        ndata = data.grep("b")
        self.assertTrue('aa' not in ndata and 'ab' in ndata)
        self.assert_arraysequal(ndata['ab'],data['ab'])

    def test_dataset_samplesize(self):
        """ Dataset.samplesize """
        data = Dataset()
        data.extend(aa=[1,2,3,4],ab=[[1],[2],[3]])
        self.assertEqual(data.samplesize,3)

    def test_dataset_trim(self):
        """ Dataset.trim """
        data = Dataset()
        data.append(a=1,b=10)
        data.append(a=2,b=20)
        data.append(a=3)
        ndata = data.trim()
        self.assertEqual(ndata.samplesize,2)
        self.assert_arraysequal(ndata['a'],[1,2])
        self.assert_arraysequal(ndata['b'],[10,20])

    def test_dataset_arrayzip(self):
        """ Dataset.arrayzip """
        data = Dataset()
        data.extend(a=[1,2,3], b=[10,20,30])
        a = data.arrayzip([['a'], ['b']])
        self.assert_arraysequal(a, [[[1],[10]],[[2],[20]],[[3],[30]]])
        with self.assertRaises(ValueError):
            data.append(a=4)
            a = data.arrayzip(['a','b'])

    def test_dataset_bootstrap_iter(self):
        """ bootstrap_iter(data_dict) """
        # make data
        N = 100
        a0 = dict(n=gvar(1,1),a=[gvar(2,2),gvar(100,100)])
        dset = Dataset()
        for ai in raniter(a0,30):
            dset.append(ai)
        a = avg_data(dset)

        # do bootstrap -- calculate means
        bs_mean = Dataset()
        for ai in bootstrap_iter(dset,N):
            for k in ai:
                bs_mean.append(k,np.average(ai[k],axis=0))
                for x in ai[k]:
                    self.assertTrue(   #
                        x in numpy.asarray(dset[k]),
                        "Bootstrap element not in original dataset.")
        a_bs = avg_data(bs_mean,bstrap=True)

        # 6 sigma tests
        an_mean = a['n'].mean
        an_sdev = a['n'].sdev
        self.assertGreater(6*an_sdev/N**0.5,abs(an_mean-a_bs['n'].mean))
        self.assertGreater(6*an_sdev/N**0.5,abs(an_sdev-a_bs['n'].sdev))

    def test_array_bootstrap_iter(self):
        """ bootstrap_iter(data_array) """
        N = 100
        a0 = [[1,2],[3,4],[5,6]]
        for ai in bootstrap_iter(a0,N):
            self.assertTrue(len(ai)==len(a0),"Bootstrap copy wrong length.")
            for x in ai:
                self.assertTrue(    #
                    x in numpy.asarray(a0),
                    "Bootstrap element not in original dataset.")

    @unittest.skipIf(NO_H5PY,"skipping test_dataset_hdf5 --- no h5py modules")
    def test_dataset_hdf5(self):
        # make hdf5 file
        s = [1., 2., 3., 4.]
        v = list(np.array([[10.,11.], [12., 13.], [14., 15.], [16., 17.]]))
        ref_dset = dict(s=s, v=v)
        with h5py.File('test-gvar.h5', 'w') as h5file:
            h5file['/run1/s'] = s
            h5file['/run2/v'] = v
        # everything
        dset = Dataset('test-gvar.h5', h5group=['/run1', '/run2'])
        self.assertEqual(list(dset.keys()), ['s', 'v'])
        for k in dset:
            self.assertEqual(str(dset[k]), str(ref_dset[k]))
        # s only
        dset = Dataset('test-gvar.h5', h5group=['/run1', '/run2'], grep='[^v]')
        self.assertEqual(list(dset.keys()), ['s'])
        for k in ['s']:
            self.assertEqual(str(dset[k]), str(ref_dset[k]))
        # v only
        dset = Dataset('test-gvar.h5', h5group=['/run1', '/run2'], keys=['v'])
        self.assertEqual(list(dset.keys()), ['v'])
        for k in ['v']:
            self.assertEqual(str(dset[k]), str(ref_dset[k]))
        # binsize=2
        dset = Dataset('test-gvar.h5', h5group=['/run1', '/run2'], binsize=2)
        self.assertEqual(list(dset.keys()), ['s', 'v'])
        self.assertEqual(dset['s'], [1.5, 3.5])
        self.assertEqual(
            str(dset['v']),
            str([np.array([11., 12.]), np.array([15., 16.])])
            )
        os.remove('test-gvar.h5')

    def test_svd_diagnosis(self):
        " svd_diagnosis "
        # random correlated data (10x10 correlation matrix)
        chebval = np.polynomial.chebyshev.chebval
        gv.ranseed(1)
        x = np.linspace(-.9, .9, 10)
        c = gv.raniter(gv.gvar(len(x) * ['0(1)']))

        # small dataset (big svdcut)
        dset = []
        for n in range(15):
            dset.append(chebval(x, next(c)))
        gv.ranseed(2)
        s = gv.dataset.svd_diagnosis(dset)
        self.assertGreater(s.svdcut, 0.01)
        # print(s.svdcut)
        # s.plot_ratio(show=True)
        # test with dictionary
        gv.ranseed(2)
        sd = gv.dataset.svd_diagnosis(dict(a=dset))
        self.assertEqual(s.svdcut, sd.svdcut)

        # large dataset (small or no svdcut)
        dset = []
        for n in range(100):
            dset.append(chebval(x, next(c)))
        gv.ranseed(3)
        s = svd_diagnosis(dset)
        self.assertGreater(0.01, s.svdcut)
        # print(s.svdcut)
        # s.plot_ratio(show=True)

        # with models (only if lsqfit installed)
        try:
            import lsqfit
        except:
            return
        class Linear(lsqfit.MultiFitterModel):
            def __init__(self, datatag, x, intercept, slope):
                super(Linear, self).__init__(datatag)
                self.x = np.array(x)
                self.intercept = intercept
                self.slope = slope
            def fitfcn(self, p):
                return p[self.intercept] + p[self.slope] * self.x
            def buildprior(self, prior, mopt=None, extend=False):
                " Extract the model's parameters from prior. "
                newprior = {}
                newprior[self.intercept] = prior[self.intercept]
                newprior[self.slope] = prior[self.slope]
                return newprior
            def builddata(self, data):
                " Extract the model's fit data from data. "
                return data[self.datatag]
            def builddataset(self, dset):
                " Extract the model's fit data from a dataset. "
                return dset[self.datatag]
        x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        y_samples = [
            [2.8409,   4.8393,   6.8403,   8.8377,  10.8356,  12.8389,  14.8356,  16.8362,  18.8351,  20.8341],
            [2.8639,   4.8612,   6.8597,   8.8559,  10.8537,  12.8525,  14.8498,  16.8487,  18.8460,  20.8447],
            [3.1048,   5.1072,   7.1071,   9.1076,  11.1090,  13.1107,  15.1113,  17.1134,  19.1145,  21.1163],
            [3.0710,   5.0696,   7.0708,   9.0705,  11.0694,  13.0681,  15.0693,  17.0695,  19.0667,  21.0678],
            [3.0241,   5.0223,   7.0198,   9.0204,  11.0191,  13.0193,  15.0198,  17.0163,  19.0154,  21.0155],
            [2.9719,   4.9700,   6.9709,   8.9706,  10.9707,  12.9705,  14.9699,  16.9686,  18.9676,  20.9686],
            [3.0688,   5.0709,   7.0724,   9.0730,  11.0749,  13.0776,  15.0790,  17.0800,  19.0794,  21.0795],
            [3.1471,   5.1468,   7.1452,   9.1451,  11.1429,  13.1445,  15.1450,  17.1435,  19.1425,  21.1432],
            [3.0233,   5.0233,   7.0225,   9.0224,  11.0225,  13.0216,  15.0224,  17.0217,  19.0208,  21.0222],
            [2.8797,   4.8792,   6.8803,   8.8794,  10.8800,  12.8797,  14.8801,  16.8797,  18.8803,  20.8812],
            [3.0388,   5.0407,   7.0409,   9.0439,  11.0443,  13.0459,  15.0455,  17.0479,  19.0493,  21.0505],
            [3.1353,   5.1368,   7.1376,   9.1367,  11.1360,  13.1377,  15.1369,  17.1400,  19.1384,  21.1396],
            [3.0051,   5.0063,   7.0022,   9.0052,  11.0040,  13.0033,  15.0007,  16.9989,  18.9994,  20.9995],
            [3.0221,   5.0197,   7.0193,   9.0183,  11.0179,  13.0184,  15.0164,  17.0177,  19.0159,  21.0155],
            [3.0188,   5.0200,   7.0184,   9.0183,  11.0189,  13.0188,  15.0191,  17.0183,  19.0177,  21.0186],
            ]
        dset = dict(y=y_samples)
        model = Linear('y', x, intercept='y0', slope='s')
        prior = gv.gvar(dict(y0='1(1)', s='2(2)'))
        gv.ranseed(4)
        s = svd_diagnosis(dset , models=[model])
        self.assertGreater(s.nmod, 0)
        self.assertGreater(s.svdcut, s.val[s.nmod - 1] / s.val[-1])
        self.assertGreater(s.val[s.nmod] / s.val[-1], s.svdcut)
        return
        # skip rest
        fitter = lsqfit.MultiFitter(models=[model])
        fit = fitter.lsqfit(prior=prior, svdcut=s.svdcut, data=s.avgdata)
        print (fit)
        # s.avgdata = gv.gvar(gv.mean(s.avgdata), gv.sdev(s.avgdata))
        fit = fitter.lsqfit(prior=prior, data=s.avgdata)
        print (fit)
        s.plot_ratio(show=True)


if __name__ == '__main__':
	unittest.main()

