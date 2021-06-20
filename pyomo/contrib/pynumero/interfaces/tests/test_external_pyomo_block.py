#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy, scipy_available
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.common.dependencies.scipy import sparse as sps

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
)
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
    get_hessian_of_constraint,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
    ScalarExternalGreyBoxBlock,
    IndexedExternalGreyBoxBlock,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest(
        "Need IPOPT to run ExternalPyomoModel tests"
        )


def make_external_model():
    m = pyo.ConcreteModel()
    m.a = pyo.Var()
    m.b = pyo.Var()
    m.r = pyo.Var()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.x_out = pyo.Var()
    m.y_out = pyo.Var()
    m.c_out_1 = pyo.Constraint(expr=m.x_out == m.x)
    m.c_out_2 = pyo.Constraint(expr=m.y_out == m.y)
    m.c_ex_1 = pyo.Constraint(expr=
            m.x**3 - 2*m.y == m.a**2 + m.b**3 - m.r**3 - 2
            )
    m.c_ex_2 = pyo.Constraint(expr=
            m.x + m.y**3 == m.a**3 + 2*m.b**2 + m.r**2 + 1
            )
    return m


class TestExternalGreyBoxBlock(unittest.TestCase):

    def test_construct_scalar(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        self.assertIs(type(block), ScalarExternalGreyBoxBlock)

        m_ex = make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
                input_vars,
                external_vars,
                residual_cons,
                external_cons,
                )
        block.set_external_model(ex_model)

        self.assertEqual(len(block.inputs), len(input_vars))
        self.assertEqual(len(block.outputs), 0)
        self.assertEqual(len(block._equality_constraint_names), 2)

#    def test_construct_indexed(self):
#        block = ExternalGreyBoxBlock([0, 1, 2], concrete=True)
#        self.assertIs(type(block), IndexedExternalGreyBoxBlock)
#
#        m = make_external_model()
#        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
#        external_vars = [m_ex.x, m_ex.y]
#        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
#        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
#        ex_model = ExternalPyomoModel(
#                input_vars,
#                external_vars,
#                residual_cons,
#                external_cons,
#                )
#
#        for i in block:
#            b = block[i]
#            b.set_external_model(ex_model)
#            self.assertEqual(len(b.inputs), len(input_vars))
#            self.assertEqual(len(b.outputs), 0)
#            self.assertEqual(len(b._equality_constraint_names), 2)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_solve_square(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
                input_vars,
                external_vars,
                residual_cons,
                external_cons,
                )
        block.set_external_model(ex_model)

        m.a = pyo.Var()
        m.b = pyo.Var()
        m.r = pyo.Var()

        n_inputs = 3

        def linking_constraint_rule(m, i):
            if i == 0:
                return m.a == m.ex_block.inputs["input_0"]
            elif i == 1:
                return m.b == m.ex_block.inputs["input_1"]
            elif i == 2:
                return m.r == m.ex_block.inputs["input_2"]

        m.linking_constraint = pyo.Constraint(range(3),
                rule=linking_constraint_rule)

        m.a.fix(1)
        m.b.fix(2)
        m.r.fix(3)

        m.obj = pyo.Objective(expr=0)

        solver = pyo.SolverFactory("cyipopt")
        solver.solve(m)

        self.assertFalse(m_ex.a.fixed)
        self.assertFalse(m_ex.b.fixed)
        self.assertFalse(m_ex.r.fixed)

        m_ex.a.fix(1)
        m_ex.b.fix(2)
        m_ex.r.fix(3)
        ipopt = pyo.SolverFactory("ipopt")
        ipopt.solve(m_ex)

        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-8)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_optimize(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
                input_vars,
                external_vars,
                residual_cons,
                external_cons,
                )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(expr=
                (x-2.0)**2 + (y-2.0)**2 + (a-2.0)**2 + (b-2.0)**2 + (r-2.0)**2
                )

        # Solve with external model embedded
        solver = pyo.SolverFactory("cyipopt")
        solver.solve(m, tee=True)

        m_ex.obj = pyo.Objective(expr=
                (m_ex.x-2.0)**2 + (m_ex.y-2.0)**2 + (m_ex.a-2.0)**2 +
                (m_ex.b-2.0)**2 + (m_ex.r-2.0)**2
                )
        m_ex.a.set_value(0.0)
        m_ex.b.set_value(0.0)
        m_ex.r.set_value(0.0)
        m_ex.y.set_value(0.0)
        m_ex.x.set_value(0.0)

        # Solve external model, now with same objective function
        ipopt = pyo.SolverFactory("ipopt")
        ipopt.solve(m_ex, tee=True)

        # Make sure full space and reduced space solves give same
        # answers.
        self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-8)


"""
Test solve:
    - given inputs, solve for outputs (square problem)
    - solve an optimization problem with the embedded external model
"""

"""
The ExternalGreyBoxModel doesn't do much with the external model. It
is mostly an intermediate between the external model and the NLP, which
is created by the CyIpopt solver?
^ This work is done during the cyipopt solver's solve method, which is
unfortunate. It makes this somewhat hard to test...

What data from the ExternalGreyBoxBlock is used where?
"""


if __name__ == '__main__':
    unittest.main()
