#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run CyIpopt tests")

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run CyIpopt tests")

from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
    cyipopt_available,
    CyIpoptProblemInterface,
)

if not cyipopt_available:
    raise unittest.SkipTest("CyIpopt is not available")


class TestSubclassCyIpoptInterface(unittest.TestCase):
    def test_subclass_no_init(self):
        class MyCyIpoptProblem(CyIpoptProblemInterface):
            def __init__(self):
                # This subclass implements __init__ but does not call
                # super().__init__
                pass

            def x_init(self):
                pass

            def x_lb(self):
                pass

            def x_ub(self):
                pass

            def g_lb(self):
                pass

            def g_ub(self):
                pass

            def scaling_factors(self):
                pass

            def objective(self, x):
                pass

            def gradient(self, x):
                pass

            def constraints(self, x):
                pass

            def jacobianstructure(self):
                pass

            def jacobian(self, x):
                pass

            def hessianstructure(self):
                pass

            def hessian(self, x, y, obj_factor):
                pass

        problem = MyCyIpoptProblem()
        x0 = []
        msg = "__init__ has not been called"
        with self.assertRaisesRegex(RuntimeError, msg):
            problem.solve(x0)


if __name__ == "__main__":
    unittest.main()
