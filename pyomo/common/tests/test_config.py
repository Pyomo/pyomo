#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import pyutilib.th as unittest

from pyomo.common.config import (
    ConfigBlock, ConfigList, ConfigValue,
    PositiveInt, NegativeInt, NonPositiveInt, NonNegativeInt,
    PositiveFloat, NegativeFloat, NonPositiveFloat, NonNegativeFloat,
    In, Path, PathList
)

class TestConfig(unittest.TestCase):
    def test_PositiveInt(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(5, PositiveInt))
        self.assertEqual(c.a, 5)
        c.a = 4.
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 6)

    def test_NegativeInt(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(-5, NegativeInt))
        self.assertEqual(c.a, -5)
        c.a = -4.
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -6)

    def test_NonPositiveInt(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(-5, NonPositiveInt))
        self.assertEqual(c.a, -5)
        c.a = -4.
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = -2.6
        self.assertEqual(c.a, -6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeInt(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(5, NonNegativeInt))
        self.assertEqual(c.a, 5)
        c.a = 4.
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 2.6
        self.assertEqual(c.a, 6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_PositiveFloat(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(5, PositiveFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 2.6)

    def test_NegativeFloat(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(-5, NegativeFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 0
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, -2.6)

    def test_NonPositiveFloat(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(-5, NonPositiveFloat))
        self.assertEqual(c.a, -5)
        c.a = -4.
        self.assertEqual(c.a, -4)
        c.a = -6
        self.assertEqual(c.a, -6)
        c.a = -2.6
        self.assertEqual(c.a, -2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, -2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = 4
        self.assertEqual(c.a, 0)

    def test_NonNegativeFloat(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(5, NonNegativeFloat))
        self.assertEqual(c.a, 5)
        c.a = 4.
        self.assertEqual(c.a, 4)
        c.a = 6
        self.assertEqual(c.a, 6)
        c.a = 2.6
        self.assertEqual(c.a, 2.6)
        with self.assertRaises(ValueError):
            c.a = 'a'
        self.assertEqual(c.a, 2.6)
        c.a = 0
        self.assertEqual(c.a, 0)
        with self.assertRaises(ValueError):
            c.a = -4
        self.assertEqual(c.a, 0)

    def test_In(self):
        c = ConfigBlock()
        c.declare('a', ConfigValue(None, In([1,3,5])))
        self.assertEqual(c.a, None)
        c.a = 3
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = 2
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = {}
        self.assertEqual(c.a, 3)
        with self.assertRaises(ValueError):
            c.a = '1'
        self.assertEqual(c.a, 3)

        c.declare('b', ConfigValue(None, In([1,3,5], int)))
        self.assertEqual(c.b, None)
        c.b = 3
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = 2
        self.assertEqual(c.b, 3)
        with self.assertRaises(ValueError):
            c.b = {}
        self.assertEqual(c.b, 3)
        c.b = '1'
        self.assertEqual(c.b, 1)


    def test_Path(self):
        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/',os.path.sep)
        cwd = os.getcwd() + os.path.sep
        c = ConfigBlock()

        c.declare('a', ConfigValue(None, Path()))
        self.assertEqual(c.a, None)
        c.a = "/a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm('/a/b/c'))
        c.a = "a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd+'a/b/c'))
        c.a = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.a)
        self.assertEqual(c.a, norm(cwd+'a/b/c'))
        c.a = None
        self.assertIs(c.a, None)

        c.declare('b', ConfigValue(None, Path('rel/path')))
        self.assertEqual(c.b, None)
        c.b = "/a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm('/a/b/c'))
        c.b = "a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd+'rel/path/a/b/c'))
        c.b = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.b)
        self.assertEqual(c.b, norm(cwd+'a/b/c'))
        c.b = None
        self.assertIs(c.b, None)

        c.declare('c', ConfigValue(None, Path('/my/dir')))
        self.assertEqual(c.c, None)
        c.c = "/a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/a/b/c'))
        c.c = "a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm('/my/dir/a/b/c'))
        c.c = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.c)
        self.assertEqual(c.c, norm(cwd+'a/b/c'))
        c.c = None
        self.assertIs(c.c, None)

        c.declare('d_base', ConfigValue("${CWD}", str))
        c.declare('d', ConfigValue(None, Path(c.get('d_base'))))
        self.assertEqual(c.d, None)
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd+'a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd+'a/b/c'))
        
        c.d_base = '/my/dir'
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/my/dir/a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd+'a/b/c'))

        c.d_base = 'rel/path'
        c.d = "/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm('/a/b/c'))
        c.d = "a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd+'rel/path/a/b/c'))
        c.d = "${CWD}/a/b/c"
        self.assertTrue(os.path.sep in c.d)
        self.assertEqual(c.d, norm(cwd+'a/b/c'))

        try:
            Path.SuppressPathExpansion = True
            c.d = "/a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, '/a/b/c')
            c.d = "a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, 'a/b/c')
            c.d = "${CWD}/a/b/c"
            self.assertTrue('/' in c.d)
            self.assertTrue('\\' not in c.d)
            self.assertEqual(c.d, "${CWD}/a/b/c")
        finally:
            Path.SuppressPathExpansion = False

    def test_PathList(self):
        def norm(x):
            if cwd[1] == ':' and x[0] == '/':
                x = cwd[:2] + x
            return x.replace('/',os.path.sep)
        cwd = os.getcwd() + os.path.sep
        c = ConfigBlock()

        c.declare('a', ConfigValue(None, PathList()))
        self.assertEqual(c.a, None)
        c.a = "/a/b/c"
        self.assertEqual(len(c.a), 1)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm('/a/b/c'))

        c.a = ["a/b/c", "/a/b/c", "${CWD}/a/b/c"]
        self.assertEqual(len(c.a), 3)
        self.assertTrue(os.path.sep in c.a[0])
        self.assertEqual(c.a[0], norm(cwd+'a/b/c'))
        self.assertTrue(os.path.sep in c.a[1])
        self.assertEqual(c.a[1], norm('/a/b/c'))
        self.assertTrue(os.path.sep in c.a[2])
        self.assertEqual(c.a[2], norm(cwd+'a/b/c'))

        c.a = ()
        self.assertEqual(len(c.a), 0)
        self.assertIs(type(c.a), list)
