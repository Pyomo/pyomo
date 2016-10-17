#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#

import pyutilib.th as unittest

from pyomo.core.base import IntegerSet
from pyomo.environ import *
from pyomo.core.base.external import (PythonCallbackFunction,
                                      AMPLExternalFunction)

def _h(*args):
    return 2 + sum(args)

class TestPythonCallbackFunction(unittest.TestCase):

    def test_call(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_h)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f(), 2.0)
        self.assertEqual(m.f(1), 3.0)
        self.assertEqual(m.f(1,2), 5.0)

    def test_getname(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_h)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f.name, "f")
        self.assertEqual(m.f.local_name, "f")
        self.assertEqual(m.f.getname(), "f")
        self.assertEqual(m.f.getname(True), "f")

        M = ConcreteModel()
        M.m = m
        self.assertEqual(M.m.f.name, "m.f")
        self.assertEqual(M.m.f.local_name, "f")
        self.assertEqual(M.m.f.getname(), "f")
        self.assertEqual(M.m.f.getname(True), "m.f")

class TestAMPLExternalFunction(unittest.TestCase):

    def test_getname(self):
        m = ConcreteModel()
        m.f = ExternalFunction(library="junk.so", function="junk")
        self.assertIsInstance(m.f, AMPLExternalFunction)
        self.assertEqual(m.f.name, "f")
        self.assertEqual(m.f.local_name, "f")
        self.assertEqual(m.f.getname(), "f")
        self.assertEqual(m.f.getname(True), "f")

        M = ConcreteModel()
        M.m = m
        self.assertEqual(M.m.f.name, "m.f")
        self.assertEqual(M.m.f.local_name, "f")
        self.assertEqual(M.m.f.getname(), "f")
        self.assertEqual(M.m.f.getname(True), "m.f")

if __name__ == "__main__":
    unittest.main()
