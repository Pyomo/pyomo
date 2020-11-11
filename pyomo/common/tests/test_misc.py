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
import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__)) + os.sep
import pyutilib.th as unittest
import pyomo.common.misc


class Test(unittest.TestCase):

    def test_tostr(self):
        # Verify that tostr() generates a string
        str = pyomo.common.misc.tostr([0.0, 1])
        self.assertEqual(str, "0.0 1")
        str = pyomo.common.misc.tostr([])
        self.assertEqual(str, "")

    def test_flatten_tuple1(self):
        # Verify that flatten_tuple() flattens a normal tuple
        tmp = (1, "2", 3.0)
        ans = pyomo.common.misc.flatten_tuple(tmp)
        self.assertEqual(ans, tmp)

    def test_flatten_tuple2(self):
        # Verify that flatten_tuple() flattens a nested tuple
        tmp = (1, "2", (4, ("5.0", (6))), 3.0)
        ans = pyomo.common.misc.flatten_tuple(tmp)
        target = (1, "2", 4, "5.0", 6, 3.0)
        self.assertEqual(ans, target)

    def test_flatten_tuple3(self):
        # Verify that flatten_tuple() returns a non-tuple
        tmp = [1, "2", 3.0]
        ans = pyomo.common.misc.flatten_tuple(tmp)
        self.assertEqual(ans, tmp)

    def test_flatten_tuple4(self):
        # Verify that flatten_tuple() removes empty tuples
        tmp = ((), 1, (), "2", ((), 4, ((), "5.0", (6), ()), ()), 3.0, ())
        ans = pyomo.common.misc.flatten_tuple(tmp)
        target = (1, "2", 4, "5.0", 6, 3.0)
        self.assertEqual(ans, target)

    def test_flatten_tuple5(self):
        # Verify that flatten_tuple() can collapse to a single empty tuple
        self.assertEqual((1, 2, 3, 4, 5), pyomo.common.misc.flatten_tuple((
            (), 1, (), 2, ((), 3, ((), 4, ()), ()), 5, ())))
        self.assertEqual((), pyomo.common.misc.flatten_tuple(((((), ()), ()), ())))
        self.assertEqual((), pyomo.common.misc.flatten_tuple(((), ((), ((),)))))

    def test_flatten_list1(self):
        # Verify that flatten_list() flattens a normal list
        tmp = [1, "2", 3.0]
        ans = pyomo.common.misc.flatten_list(tmp)
        self.assertEqual(ans, tmp)

    def test_flatten_list2(self):
        # Verify that flatten_list() flattens a nested list
        tmp = [1, "2", [4, ["5.0", [6]]], 3.0]
        ans = pyomo.common.misc.flatten_list(tmp)
        target = [1, "2", 4, "5.0", 6, 3.0]
        self.assertEqual(ans, target)

    def test_flatten_list3(self):
        # Verify that flatten_list() returns a non-list
        tmp = (1, "2", 3.0)
        ans = pyomo.common.misc.flatten_list(tmp)
        self.assertEqual(ans, tmp)

    def test_flatten_list4(self):
        # Verify that flatten_list() removes empty lists
        tmp = [[], 1, [], "2", [[], 4, [[], "5.0", [6], []], []], 3.0, []]
        ans = pyomo.common.misc.flatten_list(tmp)
        target = [1, "2", 4, "5.0", 6, 3.0]
        self.assertEqual(ans, target)

    def test_flatten_list5(self):
        # Verify that flatten_list() can collapse to a single empty list
        self.assertEqual([1, 2, 3, 4, 5], pyomo.common.misc.flatten_list(
            [[], 1, [], 2, [[], 3, [[], 4, []], []], 5, []]))
        self.assertEqual([], pyomo.common.misc.flatten_list([[[[], []], []], []]))
        self.assertEqual([], pyomo.common.misc.flatten_list([[], [[], [[],]]]))

    def test_Bunch(self):
        a = 1
        b = "b"
        tmp = pyomo.common.misc.Bunch(a=a, b=b)
        self.assertEqual(tmp.a, a)
        self.assertEqual(tmp.b, b)

    def test_Container1(self):
        opt = pyomo.common.misc.Container('a=None c=d e="1 2 3"', foo=1, bar='x')
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
        opt.x = pyomo.common.misc.Container(a=1, b=2)
        self.assertEqual(
            set(opt.keys()), set(
                ['a', 'bar', 'c', 'foo', 'e', 'xx', 'yy', 'x']))
        self.assertEqual(
            repr(opt),
            "Container(a = None, bar = 'x', c = 'd', e = '1 2 3', foo = 1, x = Container(a = 1, b = 2), xx = 1, yy = 2)")
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
        opt._name_ = 'CONTAINER'
        self.assertEqual(
            set(opt.keys()), set(
                ['a', 'bar', 'c', 'foo', 'e', 'xx', 'yy', 'x']))
        self.assertEqual(
            repr(opt),
            "Container(a = None, bar = 'x', c = 'd', e = '1 2 3', foo = 1, x = Container(a = 1, b = 2), xx = 1, yy = 2)")
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
        o1 = pyomo.common.misc.Container('a=None c=d e="1 2 3"', foo=1, bar='x')
        s = pickle.dumps(o1)
        o2 = pickle.loads(s)
        self.assertEqual(o1, o2)

    def test_flatten1(self):
        # Test that flatten works correctly
        self.assertEqual("abc", pyomo.common.misc.flatten("abc"))
        self.assertEqual(1, pyomo.common.misc.flatten(1))
        self.assertEqual([1, 2, 3], pyomo.common.misc.flatten((1, 2, 3)))
        self.assertEqual([1, 2, 3], pyomo.common.misc.flatten([1, 2, 3]))
        self.assertEqual([1, 2, 3, 4], pyomo.common.misc.flatten((1, 2, [3, 4])))
        self.assertEqual([1, 2, 'abc'], pyomo.common.misc.flatten((1, 2, 'abc')))
        self.assertEqual([1, 2, 'abc'], pyomo.common.misc.flatten((1, 2, ('abc',))))
        a = [0, 9, 8]
        self.assertEqual([1, 2, 0, 9, 8], pyomo.common.misc.flatten((1, 2, a)))
        self.assertEqual([1, 2, 3, 4, 5], pyomo.common.misc.flatten(
            [[], 1, [], 2, [[], 3, [[], 4, []], []], 5, []]))
        self.assertEqual([], pyomo.common.misc.flatten([[[[], []], []], []]))
        self.assertEqual([], pyomo.common.misc.flatten([[], [[], [[],]]]))

    def test_quote_split(self):
        ans = pyomo.common.misc.quote_split("[ ]+", "a bb ccc")
        self.assertEqual(ans, ["a", "bb", "ccc"])
        ans = pyomo.common.misc.quote_split("[ ]+", "")
        self.assertEqual(ans, [""])
        ans = pyomo.common.misc.quote_split("[ ]+", 'a "bb ccc"')
        self.assertEqual(ans, ["a", "\"bb ccc\""])
        ans = pyomo.common.misc.quote_split("[ ]+", "a 'bb ccc'")
        self.assertEqual(ans, ["a", "'bb ccc'"])
        ans = pyomo.common.misc.quote_split("[ ]+", "a X\"bb ccc\"Y")
        self.assertEqual(ans, ["a", "X\"bb ccc\"Y"])
        ans = pyomo.common.misc.quote_split("[ ]+", "a X'bb ccc'Y")
        self.assertEqual(ans, ["a", "X'bb ccc'Y"])
        ans = pyomo.common.misc.quote_split("[ ]+", "a X'bb ccc'Y A")
        self.assertEqual(ans, ["a", "X'bb ccc'Y", "A"])
        try:
            ans = pyomo.common.misc.quote_split("[ ]+", 'a "bb ccc')
            self.fail(
                "test_quote_split - failed to detect unterminated quotation")
        except ValueError:
            pass

        ans = pyomo.common.misc.quote_split("a bb\\\" ccc")
        self.assertEqual(ans, ["a", "bb\\\"", "ccc"])
        self.assertRaises(ValueError, pyomo.common.misc.quote_split,
                          ("a bb\\\\\" ccc"))
        ans = pyomo.common.misc.quote_split("a \"bb  ccc\"")
        self.assertEqual(ans, ["a", "\"bb  ccc\""])
        ans = pyomo.common.misc.quote_split("a 'bb \" ccc'")
        self.assertEqual(ans, ["a", "'bb \" ccc'"])
        ans = pyomo.common.misc.quote_split("a \"bb ' ccc\"")
        self.assertEqual(ans, ["a", "\"bb ' ccc\""])
        ans = pyomo.common.misc.quote_split("a \"bb \\\\\\\" ccc\"")
        self.assertEqual(ans, ["a", "\"bb \\\\\\\" ccc\""])
        ans = pyomo.common.misc.quote_split('b', "abbbccc")
        self.assertEqual(ans, ["a", '', '', 'ccc'])
        ans = pyomo.common.misc.quote_split('b+', "abbbccc")
        self.assertEqual(ans, ["a", 'ccc'])
        ans = pyomo.common.misc.quote_split(' ', "a b\ c")
        self.assertEqual(ans, ["a", 'b\ c'])

    def test_sort_index1(self):
        # Test that sort_index returns the correct value for a sorted array
        ans = pyomo.common.misc.sort_index(range(0, 10))
        self.assertEqual(ans, list(range(0, 10)))

    def test_sort_index2(self):
        # Test that sort_index returns an array that can be used to sort the data
        data = [4, 2, 6, 8, 1, 9, 3, 10, 7, 5]
        ans = pyomo.common.misc.sort_index(data)
        sorted = []
        for i in range(0, len(data)):
            sorted.append(data[ans[i]])
        data.sort()
        self.assertEqual(data, sorted)


if __name__ == "__main__":
    unittest.main()

    
