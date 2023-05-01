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

import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.common.dependencies.scipy import sparse as sps

from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
    CyIpoptSolverWrapper,
    ImplicitFunctionSolver,
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
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
    PressureDropTwoOutputsWithHessian,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest("Need IPOPT to run ExternalPyomoModel tests")


def _make_external_model():
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
    m.c_ex_1 = pyo.Constraint(
        expr=m.x**3 - 2 * m.y == m.a**2 + m.b**3 - m.r**3 - 2
    )
    m.c_ex_2 = pyo.Constraint(
        expr=m.x + m.y**3 == m.a**3 + 2 * m.b**2 + m.r**2 + 1
    )
    return m


def _add_linking_constraints(m):
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

    m.linking_constraint = pyo.Constraint(range(n_inputs), rule=linking_constraint_rule)


def _add_nonlinear_linking_constraints(m):
    # Nonlinear linking constraints are unusual. They are useful
    # here to test correct combination of Hessians from multiple
    # sources, however.
    m.a = pyo.Var()
    m.b = pyo.Var()
    m.r = pyo.Var()

    n_inputs = 3

    def linking_constraint_rule(m, i):
        if i == 0:
            return m.a**2 - 0.5 * m.ex_block.inputs["input_0"] ** 2 == 0
        elif i == 1:
            return m.b**2 - 0.5 * m.ex_block.inputs["input_1"] ** 2 == 0
        elif i == 2:
            return m.r**2 - 0.5 * m.ex_block.inputs["input_2"] ** 2 == 0

    m.linking_constraint = pyo.Constraint(range(n_inputs), rule=linking_constraint_rule)


def make_dynamic_model():
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[0, 1, 2])
    m = pyo.ConcreteModel()

    m.time = pyo.Set(initialize=[0, 1, 2])
    t0 = m.time.first()

    m.h = pyo.Var(m.time, initialize=1.0)
    m.dhdt = pyo.Var(m.time, initialize=1.0)
    m.flow_in = pyo.Var(m.time, bounds=(0, None), initialize=1.0)
    m.flow_out = pyo.Var(m.time, initialize=1.0)

    m.flow_coef = pyo.Param(initialize=2.0, mutable=True)

    def h_diff_eqn_rule(m, t):
        return m.dhdt[t] - (m.flow_in[t] - m.flow_out[t]) == 0

    m.h_diff_eqn = pyo.Constraint(m.time, rule=h_diff_eqn_rule)

    def dhdt_disc_eqn_rule(m, t):
        if t == m.time.first():
            return pyo.Constraint.Skip
        else:
            t_prev = m.time.prev(t)
            delta_t = t - t_prev
            return m.dhdt[t] - delta_t * (m.h[t] - m.h[t_prev]) == 0

    m.dhdt_disc_eqn = pyo.Constraint(m.time, rule=dhdt_disc_eqn_rule)

    def flow_out_eqn(m, t):
        return m.flow_out[t] == m.flow_coef * m.h[t] ** 0.5

    m.flow_out_eqn = pyo.Constraint(m.time, rule=flow_out_eqn)

    return m


class TestExternalGreyBoxBlock(unittest.TestCase):
    def test_construct_scalar(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        self.assertIs(type(block), ScalarExternalGreyBoxBlock)

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        self.assertEqual(len(block.inputs), len(input_vars))
        self.assertEqual(len(block.outputs), 0)
        self.assertEqual(len(block._equality_constraint_names), 2)

    def test_construct_indexed(self):
        block = ExternalGreyBoxBlock([0, 1, 2], concrete=True)
        self.assertIs(type(block), IndexedExternalGreyBoxBlock)

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )

        for i in block:
            b = block[i]
            b.set_external_model(ex_model)
            self.assertEqual(len(b.inputs), len(input_vars))
            self.assertEqual(len(b.outputs), 0)
            self.assertEqual(len(b._equality_constraint_names), 2)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_solve_square(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        _add_linking_constraints(m)

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

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        # Solve with external model embedded
        solver = pyo.SolverFactory("cyipopt")
        solver.solve(m)

        m_ex.obj = pyo.Objective(
            expr=(m_ex.x - 2.0) ** 2
            + (m_ex.y - 2.0) ** 2
            + (m_ex.a - 2.0) ** 2
            + (m_ex.b - 2.0) ** 2
            + (m_ex.r - 2.0) ** 2
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

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_optimize_with_cyipopt_for_inner_problem(self):
        # Use CyIpopt, rather than the default SciPy solvers,
        # for the inner problem
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]

        # This passes options to the internal ImplicitFunctionSolver,
        # which by default is SccImplicitFunctionSolver.
        # This option tells it what solver to use subsystems in its
        # decomposition.
        solver_options = dict(solver_class=CyIpoptSolverWrapper)
        ex_model = ExternalPyomoModel(
            input_vars,
            external_vars,
            residual_cons,
            external_cons,
            solver_options=solver_options,
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        # Solve with external model embedded
        solver = pyo.SolverFactory("cyipopt")
        solver.solve(m)

        m_ex.obj = pyo.Objective(
            expr=(m_ex.x - 2.0) ** 2
            + (m_ex.y - 2.0) ** 2
            + (m_ex.a - 2.0) ** 2
            + (m_ex.b - 2.0) ** 2
            + (m_ex.r - 2.0) ** 2
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

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_optimize_no_decomposition(self):
        # This is a test that does not use the SCC decomposition
        # to converge the implicit function. We do this by passing
        # solver_class=ImplicitFunctionSolver rather than the default,
        # SccImplicitFunctionSolver
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars,
            external_vars,
            residual_cons,
            external_cons,
            solver_class=ImplicitFunctionSolver,
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        # Solve with external model embedded
        solver = pyo.SolverFactory("cyipopt")
        solver.solve(m)

        m_ex.obj = pyo.Objective(
            expr=(m_ex.x - 2.0) ** 2
            + (m_ex.y - 2.0) ** 2
            + (m_ex.a - 2.0) ** 2
            + (m_ex.b - 2.0) ** 2
            + (m_ex.r - 2.0) ** 2
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

    def test_construct_dynamic(self):
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()

        inputs = [m.h, m.dhdt, m.flow_in]
        ext_vars = [m.flow_out]
        residuals = [m.h_diff_eqn]
        ext_cons = [m.flow_out_eqn]

        external_model_dict = {
            t: ExternalPyomoModel(
                [var[t] for var in inputs],
                [var[t] for var in ext_vars],
                [con[t] for con in residuals],
                [con[t] for con in ext_cons],
            )
            for t in time
        }

        reduced_space = pyo.Block(concrete=True)
        reduced_space.external_block = ExternalGreyBoxBlock(
            time, external_model=external_model_dict
        )
        block = reduced_space.external_block
        block[t0].deactivate()
        self.assertIs(type(block), IndexedExternalGreyBoxBlock)

        for t in time:
            b = block[t]
            self.assertEqual(len(b.inputs), len(inputs))
            self.assertEqual(len(b.outputs), 0)
            self.assertEqual(len(b._equality_constraint_names), len(residuals))

        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eqn = pyo.Reference(m.dhdt_disc_eqn)

        pyomo_vars = list(reduced_space.component_data_objects(pyo.Var))
        pyomo_cons = list(reduced_space.component_data_objects(pyo.Constraint))
        # NOTE: Variables in the EGBB are not found by component_data_objects
        self.assertEqual(len(pyomo_vars), len(inputs) * len(time))
        # "Constraints" defined by the EGBB are not found either, although
        # this is expected.
        self.assertEqual(len(pyomo_cons), len(time) - 1)

        reduced_space._obj = pyo.Objective(expr=0)

        # This is required to avoid a failure in the implicit function
        # evaluation when "initializing" (?) the PNLPwGBB.
        # Why exactly is function evaluation necessary for this
        # initialization again?
        block[:].inputs[:].set_value(1.0)

        # This is necessary for these variables to appear in the PNLPwGBB.
        # Otherwise they don't appear in any "real" constraints of the
        # PyomoNLP.
        reduced_space.const_input_eqn = pyo.Constraint(
            expr=reduced_space.input_var[2] - reduced_space.input_var[1] == 0
        )

        nlp = PyomoNLPWithGreyBoxBlocks(reduced_space)
        self.assertEqual(
            nlp.n_primals(),
            # EGBB "inputs", dhdt, and flow_in exist for t != t0.
            # h exists for all time.
            (2 + len(inputs)) * (len(time) - 1) + len(time),
        )
        self.assertEqual(
            nlp.n_constraints(),
            # EGBB equality constraints and disc_eqn exist for t != t0.
            # const_input_eqn is a single constraint
            (len(residuals) + 1) * (len(time) - 1) + 1,
        )

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_solve_square_dynamic(self):
        # Create the "external model"
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in.fix(1.5)

        # Create the block that will hold the reduced space model.
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)

        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            # TODO: skipping time.first() necessary?
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(
                    input_vars, external_vars, residual_cons, external_cons
                )
                block[t].set_external_model(external_model)

        n_inputs = len(input_vars)

        def linking_constraint_rule(m, i, t):
            if t == t0:
                return pyo.Constraint.Skip
            if i == 0:
                return m.diff_var[t] == m.external_block[t].inputs["input_0"]
            elif i == 1:
                return m.deriv_var[t] == m.external_block[t].inputs["input_1"]

        reduced_space.linking_constraint = pyo.Constraint(
            range(n_inputs), time, rule=linking_constraint_rule
        )
        # Initialize new variables
        for t in time:
            if t != t0:
                block[t].inputs["input_0"].set_value(m.h[t].value)
                block[t].inputs["input_1"].set_value(m.dhdt[t].value)

        reduced_space._obj = pyo.Objective(expr=0)

        solver = pyo.SolverFactory("cyipopt")
        results = solver.solve(reduced_space, tee=True)

        # Full space square model was solved in a separate script
        # to obtain these values.
        h_target = [1.2, 0.852923, 0.690725]
        dhdt_target = [-0.690890, -0.347077, -0.162198]
        flow_out_target = [2.190980, 1.847077, 1.662198]
        for t in time:
            if t == t0:
                continue
            values = [m.h[t].value, m.dhdt[t].value, m.flow_out[t].value]
            target_values = [h_target[t], dhdt_target[t], flow_out_target[t]]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-5)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_optimize_dynamic(self):
        # Create the "external model"
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in[t0].fix(1.5)

        m.obj = pyo.Objective(expr=sum((m.h[t] - 2.0) ** 2 for t in m.time if t != t0))

        # Create the block that will hold the reduced space model.
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
        reduced_space.objective = pyo.Reference(m.obj)

        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            # TODO: skipping time.first() necessary?
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t], m.flow_in[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(
                    input_vars, external_vars, residual_cons, external_cons
                )
                block[t].set_external_model(external_model)

        n_inputs = len(input_vars)

        def linking_constraint_rule(m, i, t):
            if t == t0:
                return pyo.Constraint.Skip
            if i == 0:
                return m.diff_var[t] == m.external_block[t].inputs["input_0"]
            elif i == 1:
                return m.deriv_var[t] == m.external_block[t].inputs["input_1"]
            elif i == 2:
                return m.input_var[t] == m.external_block[t].inputs["input_2"]

        reduced_space.linking_constraint = pyo.Constraint(
            range(n_inputs), time, rule=linking_constraint_rule
        )
        # Initialize new variables
        for t in time:
            if t != t0:
                block[t].inputs["input_0"].set_value(m.h[t].value)
                block[t].inputs["input_1"].set_value(m.dhdt[t].value)
                block[t].inputs["input_2"].set_value(m.flow_in[t].value)

        solver = pyo.SolverFactory("cyipopt")
        results = solver.solve(reduced_space)

        # These values were obtained by solving this problem in the full
        # space in a separate script.
        h_target = [1.2, 2.0, 2.0]
        dhdt_target = [-0.690890, 0.80, 0.0]
        flow_in_target = [1.5, 3.628427, 2.828427]
        flow_out_target = [2.190890, 2.828427, 2.828427]
        for t in time:
            if t == t0:
                continue
            values = [
                m.h[t].value,
                m.dhdt[t].value,
                m.flow_out[t].value,
                m.flow_in[t].value,
            ]
            target_values = [
                h_target[t],
                dhdt_target[t],
                flow_out_target[t],
                flow_in_target[t],
            ]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-5)

    @unittest.skipUnless(cyipopt_available, "cyipopt is not available")
    def test_optimize_dynamic_references(self):
        """
        When when pre-existing variables are attached to the EGBB
        as references, linking constraints are no longer necessary.
        """
        # Create the "external model"
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in[t0].fix(1.5)

        m.obj = pyo.Objective(expr=sum((m.h[t] - 2.0) ** 2 for t in m.time if t != t0))

        # Create the block that will hold the reduced space model.
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
        reduced_space.objective = pyo.Reference(m.obj)

        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            # TODO: is skipping time.first() necessary?
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t], m.flow_in[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(
                    input_vars, external_vars, residual_cons, external_cons
                )
                block[t].set_external_model(external_model, inputs=input_vars)

        solver = pyo.SolverFactory("cyipopt")
        results = solver.solve(reduced_space)

        # These values were obtained by solving this problem in the full
        # space in a separate script.
        h_target = [1.2, 2.0, 2.0]
        dhdt_target = [-0.690890, 0.80, 0.0]
        flow_in_target = [1.5, 3.628427, 2.828427]
        flow_out_target = [2.190890, 2.828427, 2.828427]
        for t in time:
            if t == t0:
                continue
            values = [
                m.h[t].value,
                m.dhdt[t].value,
                m.flow_out[t].value,
                m.flow_in[t].value,
            ]
            target_values = [
                h_target[t],
                dhdt_target[t],
                flow_out_target[t],
                flow_in_target[t],
            ]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-5)


class TestPyomoNLPWithGreyBoxBLocks(unittest.TestCase):
    def test_set_and_evaluate(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        _add_linking_constraints(m)

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
        residuals = np.array(
            [
                -2.0,
                -2.0,
                3.0,
                # These values were obtained by solving the same system
                # with Ipopt in another script. It may be better to do
                # the solve in this test in case the system changes.
                5.0 - (-3.03051522),
                6.0 - 3.583839997,
            ]
        )
        np.testing.assert_allclose(residuals, nlp.evaluate_constraints(), rtol=1e-8)

        duals = np.array([1, 2, 3, 4, 5])
        nlp.set_duals(duals)

        self.assertEqual(ex_model.residual_con_multipliers, [4, 5])
        np.testing.assert_equal(nlp.get_duals(), duals)

    def test_jacobian(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        _add_linking_constraints(m)

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
        row = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        col = [0, 2, 1, 3, 7, 4, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
        data = [
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -0.16747094,
            -1.00068434,
            1.72383729,
            1,
            0,
            -0.30708535,
            -0.28546127,
            -0.25235924,
            0,
            1,
        ]
        self.assertEqual(len(row), len(jac.row))
        rcd_dict = dict(((i, j), val) for i, j, val in zip(row, col, data))
        for i, j, val in zip(jac.row, jac.col, jac.data):
            self.assertIn((i, j), rcd_dict)
            self.assertAlmostEqual(rcd_dict[i, j], val, delta=1e-8)

    def test_hessian_1(self):
        # Test with duals equal vector of ones.
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        _add_nonlinear_linking_constraints(m)

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

    def test_hessian_2(self):
        # Test with duals different than vector of ones
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block

        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(
            input_vars, external_vars, residual_cons, external_cons
        )
        block.set_external_model(ex_model)

        a = m.ex_block.inputs["input_0"]
        b = m.ex_block.inputs["input_1"]
        r = m.ex_block.inputs["input_2"]
        x = m.ex_block.inputs["input_3"]
        y = m.ex_block.inputs["input_4"]
        m.obj = pyo.Objective(
            expr=(x - 2.0) ** 2
            + (y - 2.0) ** 2
            + (a - 2.0) ** 2
            + (b - 2.0) ** 2
            + (r - 2.0) ** 2
        )

        _add_nonlinear_linking_constraints(m)

        nlp = PyomoNLPWithGreyBoxBlocks(m)
        primals = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        duals = np.array([4.4, -3.3, 2.2, -1.1, 0.0])
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
        data = [4.4 * 2.0, -3.3 * 2.0, 2.2 * 2.0]
        # ^ These variables only appear in linking constraints
        rcd_dict = dict(((i, j), val) for i, j, val in zip(row, col, data))

        # These are the coordinates of the Hessian corresponding to
        # external variables with true nonzeros. The coordinates have
        # terms due to objective, linking constraints, and external
        # constraints. Values were extracted from the external model
        # while writing this test, which is just meant to verify
        # that the different Hessians combined properly.
        ex_block_nonzeros = {
            (2, 2): 2.0 + 4.4 * (-1.0) + -1.1 * (-0.10967928),
            (2, 3): -1.1 * (-0.10684633),
            (3, 2): -1.1 * (-0.10684633),
            (2, 4): -1.1 * (0.19329898),
            (4, 2): -1.1 * (0.19329898),
            (3, 3): 2.0 + (-3.3) * (-1.0) + -1.1 * (-1.31592135),
            (3, 4): -1.1 * (1.13920361),
            (4, 3): -1.1 * (1.13920361),
            (4, 4): 2.0 + 2.2 * (-1.0) + -1.1 * (-1.0891866),
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


class TestExternalGreyBoxBlockWithReferences(unittest.TestCase):
    """
    Tests for ExternalGreyBoxBlock with existing variables used
    as inputs and outputs
    """

    def _create_pressure_drop_model(self):
        """
        Create a Pyomo model with pure ExternalGreyBoxModel embedded.
        """
        m = pyo.ConcreteModel()

        # Create variables that the external block will use
        m.Pin = pyo.Var()
        m.c = pyo.Var()
        m.F = pyo.Var()
        m.P2 = pyo.Var()
        m.Pout = pyo.Var()

        # Create some random constraints and objective. These variables
        # need to appear somewhere other than the external block.
        m.Pin_con = pyo.Constraint(expr=m.Pin == 5.0)
        m.c_con = pyo.Constraint(expr=m.c == 1.0)
        m.F_con = pyo.Constraint(expr=m.F == 10.0)
        m.P2_con = pyo.Constraint(expr=m.P2 <= 5.0)
        m.obj = pyo.Objective(expr=(m.Pout - 3.0) ** 2)

        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]

        # This is "model 3" from the external_grey_box_models.py file.
        ex_model = PressureDropTwoOutputsWithHessian()
        m.egb = ExternalGreyBoxBlock()
        m.egb.set_external_model(ex_model, inputs=inputs, outputs=outputs)
        return m

    def test_pressure_drop_model(self):
        m = self._create_pressure_drop_model()

        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]

        pyomo_variables = list(m.component_data_objects(pyo.Var))
        pyomo_constraints = list(m.component_data_objects(pyo.Constraint))

        # The references to inputs and outputs are not picked up twice,
        # as EGBB does not have ctype Block
        self.assertEqual(len(pyomo_variables), len(inputs) + len(outputs))
        self.assertEqual(len(pyomo_constraints), len(cons))

        # Test the inputs and outputs attributes on egb
        self.assertIs(m.egb.inputs.ctype, pyo.Var)
        self.assertIs(m.egb.outputs.ctype, pyo.Var)

        self.assertEqual(len(m.egb.inputs), len(inputs))
        self.assertEqual(len(m.egb.outputs), len(outputs))

        for i in range(len(inputs)):
            self.assertIs(inputs[i], m.egb.inputs[i])
        for i in range(len(outputs)):
            self.assertIs(outputs[i], m.egb.outputs[i])

    def test_pressure_drop_model_nlp(self):
        m = self._create_pressure_drop_model()

        cons = [m.c_con, m.F_con, m.Pin_con, m.P2_con]
        inputs = [m.Pin, m.c, m.F]
        outputs = [m.P2, m.Pout]

        nlp = PyomoNLPWithGreyBoxBlocks(m)

        n_primals = len(inputs) + len(outputs)
        n_eq_con = len(cons) + len(outputs)
        self.assertEqual(nlp.n_primals(), n_primals)
        self.assertEqual(nlp.n_constraints(), n_eq_con)

        constraint_names = [
            "c_con",
            "F_con",
            "Pin_con",
            "P2_con",
            "egb.output_constraints[P2]",
            "egb.output_constraints[Pout]",
        ]
        primals = inputs + outputs
        nlp_constraints = nlp.constraint_names()
        nlp_vars = nlp.primals_names()

        con_idx_map = {}
        for name in constraint_names:
            # Quadratic scan to get constraint indices is not ideal.
            # Could this map be created while PyNLPwGBB is being constructed?
            con_idx_map[name] = nlp_constraints.index(name)

        var_idx_map = ComponentMap()
        for var in primals:
            name = var.name
            var_idx_map[var] = nlp_vars.index(name)

        incident_vars = {con.name: list(identify_variables(con.expr)) for con in cons}
        incident_vars["egb.output_constraints[P2]"] = inputs + [outputs[0]]
        incident_vars["egb.output_constraints[Pout]"] = inputs + [outputs[1]]

        expected_nonzeros = set()
        for con, varlist in incident_vars.items():
            i = con_idx_map[con]
            for var in varlist:
                j = var_idx_map[var]
                expected_nonzeros.add((i, j))

        self.assertEqual(len(expected_nonzeros), nlp.nnz_jacobian())

        jac = nlp.evaluate_jacobian()
        for i, j in zip(jac.row, jac.col):
            self.assertIn((i, j), expected_nonzeros)


if __name__ == "__main__":
    unittest.main()
