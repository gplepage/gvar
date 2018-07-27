#!/usr/bin/env python
# encoding: utf-8
"""
test-bufferdict.py

"""
# Copyright (c) 2012-2018 G. Peter Lepage.
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

import unittest
import pickle as pckl
import numpy as np
import gvar as gv
from gvar import BufferDict, add_parameter_parentheses, trim_redundant_keys
from gvar import nonredundant_keys

class ArrayTests(object):
    def __init__(self):
        pass
    ##
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



class test_bufferdict(unittest.TestCase,ArrayTests):
    def setUp(self):
        global b,bkeys,bvalues,bslices,bbuf
        b = BufferDict()
        bkeys = ['scalar','vector','tensor']
        bvalues = [0.,np.array([1.,2.]),np.array([[3.,4.],[5.,6.]])]
        bslices = [0,slice(1, 3, None),slice(3, 7, None)]
        bbuf = np.arange(7.)
        b['scalar'] = 0.
        b['vector'] = [1.,2.]
        b['tensor'] = [[3.,4.],[5.,6.]]

    def tearDown(self):
        global b,bkeys,bvalues,bslices,bbuf
        b = None

    def test_flat(self):
        """ b.flat """
        global b,bkeys,bvalues,bslices,bbuf
        self.assert_arraysequal(b.flat,bbuf)
        self.assertEqual(b.size,len(bbuf))
        b.flat = 10.+bbuf
        self.assert_arraysequal(b.flat,10.+bbuf)
        for k,v in zip(bkeys,bvalues):
            self.assert_arraysequal(b[k],10.+v)
        b.flat = bbuf
        bbuf_save = np.array(bbuf)
        for k in b:
            b[k] = 10.
        self.assert_arraysequal(bbuf,bbuf_save)

    def test_buf(self):
        """ b.buf """
        global b,bkeys,bvalues,bslices,bbuf
        self.assert_arraysequal(b.flat,bbuf)
        self.assertEqual(b.size,len(bbuf))
        b.buf = 10.+bbuf
        self.assert_arraysequal(b.buf,10.+bbuf)
        for k,v in zip(bkeys,bvalues):
            self.assert_arraysequal(b[k],10.+v)
        b.buf = bbuf
        for k in b:
            b[k] = 10.
        self.assert_arraysequal(bbuf,np.zeros(bbuf.size)+10.)

    def test_keys(self):
        """ b.keys """
        global b,bkeys
        self.assertSequenceEqual(list(b), bkeys)

    def test_slice(self):
        """ b.slice(k) """
        global b,bkeys,bvalues,bslices,bbuf
        for k,sl in zip(bkeys,bslices):
            self.assertEqual(sl,b.slice(k))

    def test_getitem(self):
        """ v = b[k] """
        global b,bkeys,bvalues,bslices,bbuf
        for k,v in zip(bkeys,bvalues):
            self.assert_arraysequal(b[k],v)

    def test_setitem(self):
        """ b[k] = v """
        global b,bkeys,bvalues,bslices,bbuf
        for k,v in zip(bkeys,bvalues):
            b[k] = v + 10.
            self.assert_arraysequal(b[k],v+10.)
            self.assert_arraysequal(b.flat[b.slice(k)],((v+10.).flatten()
                                                if k=='tensor' else v+10.))
        b['pseudoscalar'] = 11.
        self.assertEqual(b['pseudoscalar'],11)
        self.assertEqual(b.flat[-1],11)
        self.assertSequenceEqual(list(b.keys()),bkeys+['pseudoscalar'])

    def test_str(self):
        """ str(b) repr(b) """
        outstr = "{'scalar':0.0,'vector':array([1.,2.]),"
        outstr += "'tensor':array([[3.,4.],[5.,6.]])}"
        self.assertEqual(''.join(str(b).split()),outstr)
        outstr = "BufferDict([('scalar',0.0),('vector',array([1.,2.])),"
        outstr += "('tensor',array([[3.,4.],[5.,6.]]))])"
        self.assertEqual(''.join(repr(b).split()),outstr)

    def testufferdict_b(self):
        """ BufferDict(b) """
        global b,bkeys,bvalues,bslices,bbuf
        nb = BufferDict(b)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
            nb[k] += 10.
            self.assert_arraysequal(nb[k] , b[k]+10.)
        nb = BufferDict(nb,buf=b.flat)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
        self.assertEqual(b.size,nb.size)
        b.flat[-1] = 130.
        self.assertEqual(nb.flat[-1],130.)
        with self.assertRaises(ValueError):
            nb = BufferDict(b,buf=nb.flat[:-1])
        nb = BufferDict(b, keys=reversed(bkeys))
        nbkeys = list(nb.keys())
        self.assertEqual(nbkeys, list(reversed(bkeys)))

    def test_update(self):
        """ b.add(dict(..)) """
        global b,bkeys,bvalues,bslices,bbuf
        nb = BufferDict()
        nb.update(b)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
            nb[k] += 10.
            self.assert_arraysequal(nb[k] , b[k]+10.)

    def test_del(self):
        """ del b[k] """
        global b
        for k in b:
            newb = BufferDict(b)
            keys = list(b.keys())
            del newb[k]
            keys.remove(k)
            self.assertEqual(keys, list(newb.keys()))
            size = np.size(b[k])
            self.assertEqual(len(b.buf) - size, len(newb.buf))
            for l in newb:
                self.assertTrue(np.all(newb[l] == b[l]))

    def test_dtype(self):
        """ BufferDict(d, dtype=X) """
        global b
        ib = BufferDict(b, dtype=np.intp)
        self.assertTrue(np.all(ib.buf == b.buf))
        self.assertTrue(isinstance(ib.buf[0], np.intp))
        ib = BufferDict([(k, b[k]) for k in b], dtype=np.intp)
        self.assertTrue(np.all(ib.buf == b.buf))
        self.assertTrue(isinstance(ib.buf[0], np.intp))
        ob = BufferDict([], dtype=object)
        ob[0] = 1
        ob[0] = gv.gvar('1(1)')

    def test_get(self):
        """ g.get(k) """
        global b
        for k in b:
            self.assertTrue(np.all(b.get(k,133) == b[k]))
        self.assertEqual(b.get(None, 12), 12)

    def test_add_err(self):
        """ b.add err """
        global b
        with self.assertRaises(ValueError):
            b.add(bkeys[1],10.)

    def test_getitem_err(self):
        """ b[k] err """
        global b
        with self.assertRaises(KeyError):
            x = b['pseudoscalar']

    def test_buf_err(self):
        """ b.flat assignment err """
        global b,bbuf
        with self.assertRaises(ValueError):
            b.buf = bbuf[:-1]

    def test_pickle(self):
        global b
        sb = pckl.dumps(b)
        c = pckl.loads(sb)
        for k in b:
            self.assert_arraysequal(b[k],c[k])

    # def test_dump(self):
    #     global b
    #     b[0,1] = 2.
    #     for use_json in [True,False]:
    #         sb = b.dumps(use_json=use_json)
    #         c = BufferDict.loads(sb, use_json=use_json)
    #         for k in b:
    #             self.assert_arraysequal(b[k],c[k])

    def test_pickle_gvar(self):
        b = BufferDict(dict(a=gv.gvar(1,2),b=[gv.gvar(3,4),gv.gvar(5,6)]))
        sb = pckl.dumps(b)
        c = pckl.loads(sb)
        for k in b:
            self.assert_gvclose(b[k],c[k],rtol=1e-6)

    # def test_dump_gvar(self):
    #     b = BufferDict(dict(a=gv.gvar(1,2),b=[gv.gvar(3,4),gv.gvar(5,6)]))
    #     b[0,1] = b['a']+10*b['b'][0]
    #     for use_json in [True,False]:
    #         sb = b.dumps(use_json=use_json)
    #         c = BufferDict.loads(sb, use_json=use_json)
    #         for k in b:
    #             self.assert_gvclose(b[k],c[k],rtol=1e-6)
    #         self.assert_gvclose((c[0,1]-10*c['b'][0])/c['a'],
    #                             gv.gvar(1.0,0.0), rtol=1e-6)

    def test_extension_mapping(self):
        " BufferDict extension and mapping properties  "
        p = BufferDict()
        p['a'] = 1.
        p['b'] = [2., 3.]
        p['log(c)'] = 0.
        p['sqrt(d)'] = [5., 6.]
        p['erfinv(e)'] = [[33.]]
        newp = BufferDict(p)
        for i in range(2):
            for k in p:
                assert np.all(p[k] == newp[k])
            assert newp['c'] == np.exp(newp['log(c)'])
            assert np.all(newp['d'] == np.square(newp['sqrt(d)']))
            assert np.all(newp['e'] == gv.erf(newp['erfinv(e)']))
            assert np.all(p.buf == newp.buf)
            p.buf[:] = [10., 20., 30., 1., 2., 3., 4.]
            newp.buf = np.array(p.buf.tolist())
        self.assertEqual(
            gv.get_dictkeys(p, ['c', 'a', 'log(c)', 'e', 'd']),
            ['log(c)', 'a', 'log(c)', 'erfinv(e)', 'sqrt(d)']
            )
        self.assertEqual(
            [gv.dictkey(p, k)  for k in [
                'c', 'a', 'log(c)', 'e', 'd'
                ]],
            ['log(c)', 'a', 'log(c)', 'erfinv(e)', 'sqrt(d)']
            )
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'a'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'b'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'c'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'd'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'e'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'log(c)'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'sqrt(d)'))
        self.assertTrue(gv.BufferDict.has_dictkey(p, 'erfinv(e)'))
        self.assertTrue(not gv.BufferDict.has_dictkey(p, 'log(a)'))
        self.assertTrue(not gv.BufferDict.has_dictkey(p, 'sqrt(b)'))
        self.assertEqual(list(p), ['a', 'b', 'log(c)', 'sqrt(d)', 'erfinv(e)'])
        np.testing.assert_equal(
            list(p.values()),
            [10.0, [20., 30.], 1.0, [2., 3.], [[4.]]]
            )
        self.assertEqual(p.get('c'), p['c'])

        # tracking?
        self.assertAlmostEqual(p['c'], np.exp(1))
        self.assertAlmostEqual(p['log(c)'], 1.)
        p['log(c)'] = 2.
        self.assertAlmostEqual(p['c'], np.exp(2))
        self.assertAlmostEqual(p['log(c)'], 2.)
        p['a'] = 12.
        self.assertAlmostEqual(p['c'], np.exp(2))
        self.assertAlmostEqual(p['log(c)'], 2.)
        self.assertEqual(
            list(p),
            ['a', 'b', 'log(c)', 'sqrt(d)', 'erfinv(e)'],
            )

        # the rest is not so important
        # trim redundant keys
        oldp = trim_redundant_keys(newp)
        assert 'c' not in oldp
        assert 'd' not in oldp
        assert np.all(oldp.buf == newp.buf)

        # nonredundant keys
        assert set(nonredundant_keys(newp.keys())) == set(p.keys())

        # stripkey
        for ks, f, k in [
            ('aa', np.exp, 'log(aa)'),
            ('aa', np.square, 'sqrt(aa)'),
            ]:
            assert (ks, f) == gv._bufferdict._stripkey(k)

        # addparentheses
        pvar = BufferDict()
        pvar['a'] = p['a']
        pvar['b'] = p['b']
        pvar['logc'] = p['log(c)']
        pvar['sqrtd'] = p['sqrt(d)']
        pvar['erfinv(e)'] = p['erfinv(e)']
        pvar = add_parameter_parentheses(pvar)
        for k in p:
            assert k in pvar
            assert np.all(p[k] == pvar[k])
        for k in pvar:
            assert k in p
        pvar = add_parameter_parentheses(pvar)
        for k in p:
            assert k in pvar
            assert np.all(p[k] == pvar[k])
        for k in pvar:
            assert k in p
        pvar['log(c(23))'] = 1.2
        pvar = BufferDict(pvar)
        assert 'c(23)' not in pvar
        assert 'log(c(23))' in pvar
        self.assertAlmostEqual(gv.exp(pvar['log(c(23))']), pvar['c(23)'])





if __name__ == '__main__':
    unittest.main()

