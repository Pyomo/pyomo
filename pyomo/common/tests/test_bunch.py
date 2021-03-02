#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the PyUtilib project
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  ___________________________________________________________________________

import pickle
import unittest
from pyomo.common.collections import Bunch

class Test(unittest.TestCase):
    def test_Bunch1(self):
        opt = Bunch('a=None c=d e="1 2 3"', foo=1, bar='x')
        self.assertEqual(opt.ll, None)
        self.assertEqual(opt.a, None)
        self.assertEqual(opt.c, 'd')
        self.assertEqual(opt.e, '1 2 3')
        self.assertEqual(opt.foo, 1)
        self.assertEqual(opt.bar, 'x')
        self.assertEqual(opt['bar'], 'x')
        opt.xx = 1
        opt['yy'] = 2
        self.assertEqual(
            set(opt.keys()), set(['a', 'bar', 'c', 'foo', 'e', 'xx', 'yy']))
        opt.x = Bunch(a=1, b=2)
        self.assertEqual(
            set(opt.keys()), set(
                ['a', 'bar', 'c', 'foo', 'e', 'xx', 'yy', 'x']))
        self.assertEqual(
            repr(opt),
            "Bunch(a = None, bar = 'x', c = 'd', e = '1 2 3', foo = 1, x = Bunch(a = 1, b = 2), xx = 1, yy = 2)")
        self.assertEqual(
            str(opt), """a: None
bar: 'x'
c: 'd'
e: '1 2 3'
foo: 1
x:
    a: 1
    b: 2
xx: 1
yy: 2""")
        opt._name_ = 'BUNCH'
        self.assertEqual(
            set(opt.keys()), set(
                ['a', 'bar', 'c', 'foo', 'e', 'xx', 'yy', 'x']))
        self.assertEqual(
            repr(opt),
            "Bunch(a = None, bar = 'x', c = 'd', e = '1 2 3', foo = 1, x = Bunch(a = 1, b = 2), xx = 1, yy = 2)")
        self.assertEqual(
            str(opt), """a: None
bar: 'x'
c: 'd'
e: '1 2 3'
foo: 1
x:
    a: 1
    b: 2
xx: 1
yy: 2""")

    def test_Container2(self):
        o1 = Bunch('a=None c=d e="1 2 3"', foo=1, bar='x')
        s = pickle.dumps(o1)
        o2 = pickle.loads(s)
        self.assertEqual(o1, o2)
