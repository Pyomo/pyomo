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
import os
import pyutilib.th as unittest

from pyomo.core.base import IntegerSet
from pyomo.environ import *
from pyomo.core.base.external import (PythonCallbackFunction,
                                      AMPLExternalFunction)
from pyomo.opt import check_available_solvers

def _g(*args):
    return len(args)

def _h(*args):
    return 2 + sum(args)

class TestPythonCallbackFunction(unittest.TestCase):

    def test_call_countArgs(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_g)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f(), 0)
        self.assertEqual(m.f(2), 1)
        self.assertEqual(m.f(2,3), 2)

    def test_call_sumfcn(self):
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

    def getGSL(self):
        # Find the GSL DLL
        DLL = None
        for path in [os.getcwd()] + os.environ['PATH'].split(os.pathsep):
            test = os.path.join(path, 'amplgsl.dll')
            if os.path.isfile(test):
                DLL = test
                break
        print("GSL found here: %s" % DLL)
        return DLL

    def test_eval_gsl_function(self):
        DLL = self.getGSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library in the PATH")
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.x = Var(initialize=3, bounds=(1e-5,None))
        model.o = Objective(expr=model.z_func(model.x))
        self.assertAlmostEqual(value(model.o), 2.0, 7)

    @unittest.skipIf(not check_available_solvers('ipopt'),
                     "The 'ipopt' solver is not available")
    def test_solve_gsl_function(self):
        DLL = self.getGSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library in the PATH")
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.x = Var(initialize=3, bounds=(1e-5,None))
        model.o = Objective(expr=model.z_func(model.x))
        opt = SolverFactory('ipopt')
        res = opt.solve(model, tee=True)
        self.assertAlmostEqual(value(model.o), 0.885603194411, 7)


if __name__ == "__main__":
    unittest.main()
