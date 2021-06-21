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
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
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
    m.c_out_1 = pyo.Constraint(expr=m.x_out - m.x == 0)
    m.c_out_2 = pyo.Constraint(expr=m.y_out - m.y == 0)
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

        m.linking_constraint = pyo.Constraint(range(n_inputs),
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
        solver.solve(m)

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
        ipopt.solve(m_ex)

        # Make sure full space and reduced space solves give same
        # answers.
        self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-8)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-8)

class TestPyomoNLPWithGreyBoxBLocks(unittest.TestCase):

    def test_set_and_evaluate(self):
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

        m.a = pyo.Var()
        m.b = pyo.Var()
        m.r = pyo.Var()

        n_inputs = 3

        def linking_constraint_rule(m, i):
            if i == 0:
                return m.a - m.ex_block.inputs["input_0"] == 0
            elif i == 1:
                return m.b - m.ex_block.inputs["input_1"] == 0
            elif i == 2:
                return m.r - m.ex_block.inputs["input_2"] == 0

        m.linking_constraint = pyo.Constraint(range(n_inputs),
                rule=linking_constraint_rule)

        nlp = PyomoNLPWithGreyBoxBlocks(m)

        # Set primals in model, get primals in nlp
        # set/get duals
        # evaluate constraints
        # evaluate Jacobian
        # evaluate Hessian
        self.assertEqual(nlp.n_primals(), 8)

        # PyomoNLPWithGreyBoxBlocks sorts variables by name
        primals_names = [
                "a",
                "b",
                "ex_block.inputs[input_0]",
                "ex_block.inputs[input_1]",
                "ex_block.inputs[input_2]",
                "ex_block.inputs[input_3]",
                "ex_block.inputs[input_4]",
                "r",
                ]
        self.assertEqual(nlp.primals_names(), primals_names)
        np.testing.assert_equal(np.zeros(8), nlp.get_primals())

        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        nlp.set_primals(primals)
        np.testing.assert_equal(primals, nlp.get_primals())
        nlp.load_state_into_pyomo()

        for name, val in zip(primals_names, primals):
            var = m.find_component(name)
            self.assertEqual(var.value, val)

        constraint_names = [
                "linking_constraint[0]",
                "linking_constraint[1]",
                "linking_constraint[2]",
                "ex_block.residual_0",
                "ex_block.residual_1",
                ]
        self.assertEqual(constraint_names, nlp.constraint_names())
        residuals = np.array([
                -2.0,
                -2.0,
                3.0,
                # These values were obtained by solving the same system
                # with Ipopt in another script. It may be better to do
                # the solve in this test in case the system changes.
                5.0-(-3.03051522),
                6.0-3.583839997,
                ])
        np.testing.assert_allclose(residuals, nlp.evaluate_constraints(),
                rtol=1e-8)

        duals = np.array([1, 2, 3, 4, 5])
        nlp.set_duals(duals)

        self.assertEqual(ex_model.residual_con_multipliers, [4, 5])
        np.testing.assert_equal(nlp.get_duals(), duals)

    def test_jacobian(self):
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

        m.a = pyo.Var()
        m.b = pyo.Var()
        m.r = pyo.Var()

        n_inputs = 3

        def linking_constraint_rule(m, i):
            if i == 0:
                return m.a - m.ex_block.inputs["input_0"] == 0
            elif i == 1:
                return m.b - m.ex_block.inputs["input_1"] == 0
            elif i == 2:
                return m.r - m.ex_block.inputs["input_2"] == 0

        m.linking_constraint = pyo.Constraint(range(n_inputs),
                rule=linking_constraint_rule)

        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        nlp.set_primals(primals)
        jac = nlp.evaluate_jacobian()

        # Variable and constraint orders (verified by previous test):
        # Rows:
        # [
        #     "linking_constraint[0]",
        #     "linking_constraint[1]",
        #     "linking_constraint[2]",
        #     "ex_block.residual_0",
        #     "ex_block.residual_1",
        # ]
        # Cols:
        # [
        #     "a",
        #     "b",
        #     "ex_block.inputs[input_0]",
        #     "ex_block.inputs[input_1]",
        #     "ex_block.inputs[input_2]",
        #     "ex_block.inputs[input_3]",
        #     "ex_block.inputs[input_4]",
        #     "r",
        # ]
        row = [
                0, 0,
                1, 1,
                2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,
                ]
        col = [
                0, 2,
                1, 3,
                7, 4,
                2, 3, 4, 5, 6,
                2, 3, 4, 5, 6,
                ]
        data = [
                1, -1,
                1, -1,
                1, -1,
                -0.16747094, -1.00068434, 1.72383729, 1, 0,
                -0.30708535, -0.28546127, -0.25235924, 0, 1,
                ]
        self.assertEqual(len(row), len(jac.row))
        rcd_dict = dict(((i, j), val) for i, j, val in zip(row, col, data))
        for i, j, val in zip(jac.row, jac.col, jac.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-8)

    def test_hessian(self):
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

        m.a = pyo.Var()
        m.b = pyo.Var()
        m.r = pyo.Var()

        n_inputs = 3

        def linking_constraint_rule(m, i):
            # Nonlinear linking constraints are unusual. They are useful
            # here to test correct combination of Hessians from multiple
            # sources, however.
            if i == 0:
                return m.a**2 - 0.5*m.ex_block.inputs["input_0"]**2 == 0
            elif i == 1:
                return m.b**2 - 0.5*m.ex_block.inputs["input_1"]**2 == 0
            elif i == 2:
                return m.r**2 - 0.5*m.ex_block.inputs["input_2"]**2 == 0

        m.linking_constraint = pyo.Constraint(range(n_inputs),
                rule=linking_constraint_rule)

        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        duals = np.array([1, 1, 1, 1, 1])
        nlp.set_primals(primals)
        nlp.set_duals(duals)
        hess = nlp.evaluate_hessian_lag()

        # Variable order (verified by a previous test):
        # [
        #     "a",
        #     "b",
        #     "ex_block.inputs[input_0]",
        #     "ex_block.inputs[input_1]",
        #     "ex_block.inputs[input_2]",
        #     "ex_block.inputs[input_3]",
        #     "ex_block.inputs[input_4]",
        #     "r",
        # ]
        row = [0, 1, 7]
        col = [0, 1, 7]
        # Data entries are influenced by multiplier values.
        # Here these are just ones.
        data = [2.0, 2.0, 2.0]
        # ^ These variables only appear in linking constraints
        rcd_dict = dict(((i, j), val) for i, j, val in zip(row, col, data))

        # These are the coordinates of the Hessian corresponding to
        # external variables with true nonzeros. The coordinates have
        # terms due to objective, linking constraints, and external
        # constraints. Values were extracted from the external model
        # while writing this test, which is just meant to verify
        # that the different Hessians combined properly.
        ex_block_nonzeros = {
                (2, 2): 2.0 + (-1.0) + (-0.10967928) + (-0.25595929),
                (2, 3): (-0.10684633) + (0.05169308),
                (3, 2): (-0.10684633) + (0.05169308),
                (2, 4): (0.19329898) + (0.03823075),
                (4, 2): (0.19329898) + (0.03823075),
                (3, 3): 2.0 + (-1.0) + (-1.31592135) + (-0.0241836),
                (3, 4): (1.13920361) + (0.01063667),
                (4, 3): (1.13920361) + (0.01063667),
                (4, 4): 2.0 + (-1.0) + (-1.0891866) + (0.01190218),
                (5, 5): 2.0,
                (6, 6): 2.0,
                }
        rcd_dict.update(ex_block_nonzeros)

        # Because "external Hessians" are computed by factorizing matrices,
        # we have dense blocks in the Hessian for now.
        ex_block_coords = [2, 3, 4, 5, 6]
        for i, j in itertools.product(ex_block_coords, ex_block_coords):
            row.append(i)
            col.append(j)
            if (i, j) not in rcd_dict:
                rcd_dict[i, j] = 0.0

        self.assertEqual(len(row), len(hess.row))
        for i, j, val in zip(hess.row, hess.col, hess.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-8)


if __name__ == '__main__':
    unittest.main()
