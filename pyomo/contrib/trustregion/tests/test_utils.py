#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.utils import (
    copyVector, minIgnoreNone, maxIgnoreNone,
    IterationLog, Logger)

class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_copyVector(self):
        x = [1, 2, 3]
        y = [2, 7, 9]
        z = [1.0, 4.0, 10.0]
        c_x, c_y, c_z = copyVector(x, y, z)
        self.assertTrue(x[i] == c_x[i] for i in range(len(x)))
        self.assertTrue(y[j] == c_y[j] for j in range(len(y)))
        self.assertTrue(z[k] == c_z[k] for k in range(len(z)))
        self.assertFalse(id(x) == id(c_x))
        self.assertFalse(id(y) == id(c_y))
        self.assertFalse(id(z) == id(c_z))

    def test_minIgnoreNone(self):
        a = 1
        b = 2
        self.assertEqual(minIgnoreNone(a, b), a)
        a = None
        self.assertEqual(minIgnoreNone(a, b), b)
        a = 1
        b = None
        self.assertEqual(minIgnoreNone(a, b), a)
        a = None
        self.assertEqual(minIgnoreNone(a, b), None)

    def test_maxIgnoreNone(self):
        a = 1
        b = 2
        self.assertEqual(maxIgnoreNone(a, b), b)
        a = None
        self.assertEqual(maxIgnoreNone(a, b), b)
        a = 1
        b = None
        self.assertEqual(maxIgnoreNone(a, b), a)
        a = None
        self.assertEqual(maxIgnoreNone(a, b), None)
