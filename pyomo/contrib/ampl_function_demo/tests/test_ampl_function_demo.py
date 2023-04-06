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

import platform
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import find_library
from pyomo.opt import check_available_solvers
from pyomo.core.base.external import nan

flib = find_library("asl_external_demo")
is_pypy = platform.python_implementation().lower().startswith('pypy')


@unittest.skipUnless(flib, 'Could not find the "asl_external_demo.so" library')
class TestAMPLExternalFunction(unittest.TestCase):
    @unittest.skipIf(is_pypy, 'Cannot evaluate external functions under pypy')
    def test_eval_function(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="demo_function")
        self.assertAlmostEqual(m.tf("sum", 1, 2, 3)(), 6, 4)
        self.assertAlmostEqual(m.tf("inv", 1, 2, 3)(), 1.8333333, 4)
        m.cbrt = pyo.ExternalFunction(library=flib, function="safe_cbrt")
        self.assertAlmostEqual(m.cbrt(6)(), 1.81712059, 4)
        self.assertStructuredAlmostEqual(
            m.cbrt.evaluate_fgh([0]), (0, [100951], [-1.121679e13]), reltol=1e-5
        )

    @unittest.skipIf(is_pypy, 'Cannot evaluate external functions under pypy')
    def test_eval_function_fgh(self):
        m = pyo.ConcreteModel()
        m.tf = pyo.ExternalFunction(library=flib, function="demo_function")

        f, g, h = m.tf.evaluate_fgh(("sum", 1, 2, 3))
        self.assertEqual(f, 6)
        self.assertEqual(g, [nan, 1, 1, 1])
        self.assertEqual(h, [nan, nan, 0, nan, 0, 0, nan, 0, 0, 0])

        f, g, h = m.tf.evaluate_fgh(("inv", 1, 2, 3))
        self.assertAlmostEqual(f, 1.8333333, 4)
        self.assertStructuredAlmostEqual(g, [nan, -1, -1 / 4, -1 / 9])
        self.assertStructuredAlmostEqual(
            h, [nan, nan, 2, nan, 0, 1 / 4, nan, 0, 0, 2 / 27]
        )

    @unittest.skipUnless(
        check_available_solvers('ipopt'), "The 'ipopt' solver is not available"
    )
    def test_solve_function(self):
        m = pyo.ConcreteModel()
        m.sum_it = pyo.ExternalFunction(library=flib, function="demo_function")
        m.cbrt = pyo.ExternalFunction(library=flib, function="safe_cbrt")
        m.x = pyo.Var(initialize=4.0)
        m.y = pyo.Var(initialize=0)
        m.y.fix()
        # Note: this also tests passing constant expressions to external
        # functions in the NL writer
        m.c = pyo.Constraint(expr=1.5 == m.sum_it("inv", 3, m.x, 1 / (m.y + 1)))
        m.o = pyo.Objective(expr=m.cbrt(m.x))
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m, tee=True)
        self.assertAlmostEqual(m.x(), 6, 4)
