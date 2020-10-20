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
from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")
from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.common.getGSL import find_GSL
from pyomo.environ import ConcreteModel, ExternalFunction, Var, Objective

class TestAMPLExternalFunction(unittest.TestCase):
    def assertListsAlmostEqual(self, first, second, places=7, msg=None):
        self.assertEqual(len(first), len(second))
        msg = "lists %s and %s differ at item " % (
            first, second)
        for i,a in enumerate(first):
            self.assertAlmostEqual(a, second[i], places, msg + str(i))

    def test_solve_gsl_function(self):
        DLL = find_GSL()
        if not DLL:
            self.skipTest("Could not find the amplgsl.dll library")
        model = ConcreteModel()
        model.z_func = ExternalFunction(library=DLL, function="gsl_sf_gamma")
        model.x = Var(initialize=3, bounds=(1e-5,None))
        model.o = Objective(expr=model.z_func(model.x))
        nlp = PyomoNLP(model)
        self.assertAlmostEqual(nlp.evaluate_objective(), 2, 7)
        assert "AMPLFUNC" not in os.environ


if __name__ == "__main__":
    unittest.main()
