#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
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
        opt = Bunch('a=None c=d e="1 2 3" f=" 5 "', foo=1, bar='x')
        self.assertEqual(opt.ll, None)
        self.assertEqual(opt.a, None)
        self.assertEqual(opt.c, 'd')
        self.assertEqual(opt.e, '1 2 3')
        self.assertEqual(opt.f, 5)
        self.assertEqual(opt.foo, 1)
        self.assertEqual(opt.bar, 'x')
        self.assertEqual(opt['bar'], 'x')
        opt.xx = 1
        opt['yy'] = 2
        self.assertEqual(
            set(opt.keys()), set(['a', 'bar', 'c', 'f', 'foo', 'e', 'xx', 'yy'])
        )
        opt.x = Bunch(a=1, b=2)
        self.assertEqual(
            set(opt.keys()), set(['a', 'bar', 'c', 'f', 'foo', 'e', 'xx', 'yy', 'x'])
        )
        self.assertEqual(
            repr(opt),
            "Bunch(a = None, bar = 'x', c = 'd', e = '1 2 3', f = 5, "
            "foo = 1, x = Bunch(a = 1, b = 2), xx = 1, yy = 2)",
        )
        self.assertEqual(
            str(opt),
            """a: None
bar: 'x'
c: 'd'
e: '1 2 3'
f: 5
foo: 1
x:
    a: 1
    b: 2
xx: 1
yy: 2""",
        )
        opt._name_ = 'BUNCH'
        self.assertEqual(
            set(opt.keys()), set(['a', 'bar', 'c', 'f', 'foo', 'e', 'xx', 'yy', 'x'])
        )
        self.assertEqual(
            repr(opt),
            "Bunch(a = None, bar = 'x', c = 'd', e = '1 2 3', f = 5, "
            "foo = 1, x = Bunch(a = 1, b = 2), xx = 1, yy = 2)",
        )
        self.assertEqual(
            str(opt),
            """a: None
bar: 'x'
c: 'd'
e: '1 2 3'
f: 5
foo: 1
x:
    a: 1
    b: 2
xx: 1
yy: 2""",
        )

        with self.assertRaisesRegex(
            TypeError, r"Bunch\(\) positional arguments must be strings"
        ):
            Bunch(5)

        with self.assertRaisesRegex(
            ValueError,
            r"Bunch\(\) positional arguments must be space "
            "separated strings of form 'key=value', got 'foo'",
        ):
            Bunch('a=5 foo = 6')

    def test_pickle(self):
        o1 = Bunch('a=None c=d e="1 2 3"', foo=1, bar='x')
        s = pickle.dumps(o1)
        o2 = pickle.loads(s)
        self.assertEqual(o1, o2)

    def test_attr_methods(self):
        b = Bunch()
        b.foo = 5
        self.assertEqual(list(b.keys()), ['foo'])
        self.assertEqual(b.foo, 5)
        b._foo = 50
        self.assertEqual(list(b.keys()), ['foo'])
        self.assertEqual(b.foo, 5)
        self.assertEqual(b._foo, 50)

        del b.foo
        self.assertEqual(list(b.keys()), [])
        self.assertEqual(b.foo, None)
        self.assertEqual(b._foo, 50)

        del b._foo
        self.assertEqual(list(b.keys()), [])
        self.assertEqual(b.foo, None)
        with self.assertRaisesRegex(AttributeError, "Unknown attribute '_foo'"):
            b._foo

    def test_item_methods(self):
        b = Bunch()
        b['foo'] = 5
        self.assertEqual(list(b.keys()), ['foo'])
        self.assertEqual(b['foo'], 5)
        b['_foo'] = 50
        self.assertEqual(list(b.keys()), ['foo'])
        self.assertEqual(b['foo'], 5)
        self.assertEqual(b['_foo'], 50)

        del b['foo']
        self.assertEqual(list(b.keys()), [])
        self.assertEqual(b['foo'], None)
        self.assertEqual(b['_foo'], 50)

        del b['_foo']
        self.assertEqual(list(b.keys()), [])
        self.assertEqual(b['foo'], None)
        with self.assertRaisesRegex(AttributeError, "Unknown attribute '_foo'"):
            b['_foo']

        with self.assertRaisesRegex(ValueError, r"Bunch keys must be str \(got int\)"):
            b[5]

        with self.assertRaisesRegex(ValueError, r"Bunch keys must be str \(got int\)"):
            b[5] = 5

        with self.assertRaisesRegex(ValueError, r"Bunch keys must be str \(got int\)"):
            del b[5]

    def test_update(self):
        data = {
            'a': 1,
            'b': [2, {'bb': 3}, [4, {'bbb': 5}]],
            'c': {'cc': 6, 'ccc': {'e': 7}},
            'd': [],
        }

        # Test passing a dict
        b = Bunch()
        b.update(data)
        self.assertEqual(
            repr(b),
            'Bunch(a = 1, '
            'b = [2, Bunch(bb = 3), [4, Bunch(bbb = 5)]], '
            'c = Bunch(cc = 6, ccc = Bunch(e = 7)), '
            'd = [])',
        )

        # Test passing a generator
        b = Bunch()
        b.update(data.items())
        self.assertEqual(
            repr(b),
            'Bunch(a = 1, '
            'b = [2, Bunch(bb = 3), [4, Bunch(bbb = 5)]], '
            'c = Bunch(cc = 6, ccc = Bunch(e = 7)), '
            'd = [])',
        )

        # Test passing a list
        b = Bunch()
        b.update(list(data.items()))
        self.assertEqual(
            repr(b),
            'Bunch(a = 1, '
            'b = [2, Bunch(bb = 3), [4, Bunch(bbb = 5)]], '
            'c = Bunch(cc = 6, ccc = Bunch(e = 7)), '
            'd = [])',
        )

    def test_str(self):
        data = {
            'a': 1,
            'b': [2, {'bb': 3}, [4, {'bbb': 5}]],
            'c': {'cc': 6, 'ccc': {'e': 7}},
            'd': [],
        }

        b = Bunch()
        b.update(data)
        self.assertEqual(
            str(b),
            '''
a: 1
b:
- 2
-
    bb: 3
- [4, Bunch(bbb = 5)]
c:
    cc: 6
    ccc:
        e: 7
d: []
'''.strip(),
        )

    def test_set_name(self):
        b = Bunch()
        self.assertEqual(b._name_, 'Bunch')
        b.set_name('TEST')
        self.assertEqual(b._name_, 'TEST')
