#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager

from pyomo.contrib.pynumero.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)
from pyomo.common.dependencies.scipy import sparse as spa

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.exceptions import PyNumeroEvaluationError
from pyomo.contrib.pynumero.asl import AmplInterface

if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests"
    )

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

from pyomo.contrib.pynumero.interfaces.cyipopt_interface import (
    cyipopt,
    cyipopt_available,
    CyIpoptNLP,
)

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver

if cyipopt_available:
    # We don't raise unittest.SkipTest if not cyipopt_available as there is a
    # test below that tests an exception when cyipopt is unavailable.
    cyipopt_ge_1_3 = hasattr(cyipopt, "CyIpoptEvaluationError")
    ipopt_ge_3_14 = cyipopt.IPOPT_VERSION >= (3, 14, 0)


def create_model1():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=4.0)
    m.c = pyo.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = pyo.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    m.o = pyo.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model2():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=4.0)
    m.d = pyo.Constraint(expr=m.x[1] + m.x[2] <= 5)
    m.o = pyo.Objective(expr=m.x[1] ** 2 + 4 * m.x[2] ** 2 - 8 * m.x[1] - 16 * m.x[2])
    m.x[1].setub(3.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model3(G, A, b, c):
    nx = G.shape[0]
    nl = A.shape[0]

    model = pyo.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)

    model.x = pyo.Var(model.var_ids, initialize=0.0)
    model.hessian_f = pyo.Param(
        model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j]
    )
    model.jacobian_c = pyo.Param(
        model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j]
    )
    model.rhs = pyo.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = pyo.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]

    model.equalities = pyo.Constraint(model.con_ids, rule=equality_constraint_rule)

    def objective_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def create_model4():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=1.0)
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
    m.obj = pyo.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)
    return m


def create_model6():
    model = pyo.ConcreteModel()

    model.S = [1, 2]
    model.x = pyo.Var(model.S, initialize=1.0)

    def f(model):
        return (
            model.x[1] ** 4
            + (model.x[1] + model.x[2]) ** 2
            + (-1.0 + pyo.exp(model.x[2])) ** 2
        )

    model.f = pyo.Objective(rule=f)
    return model


def create_model9():
    # clplatea OXR2-MN-V-0
    model = pyo.ConcreteModel()

    p = 71
    wght = -0.1
    hp2 = 0.5 * p**2

    model.x = pyo.Var(pyo.RangeSet(1, p), pyo.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(
            0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2
            + 0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2
            + hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4
            + hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4
            for i in range(2, p + 1)
            for j in range(2, p + 1)
        ) + (wght * model.x[p, p])

    model.f = pyo.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


def make_hs071_model():
    # This is a model that is mathematically equivalent to the Hock-Schittkowski
    # test problem 071, but that will trigger an evaluation error if x[0] goes
    # above 1.1.
    m = pyo.ConcreteModel()
    m.x = pyo.Var([0, 1, 2, 3], bounds=(1.0, 5.0))
    m.x[0] = 1.0
    m.x[1] = 5.0
    m.x[2] = 5.0
    m.x[3] = 1.0
    m.obj = pyo.Objective(expr=m.x[0] * m.x[3] * (m.x[0] + m.x[1] + m.x[2]) + m.x[2])
    # This expression evaluates to zero, but is not well defined when x[0] > 1.1
    trivial_expr_with_eval_error = (pyo.sqrt(1.1 - m.x[0])) ** 2 + m.x[0] - 1.1
    m.ineq1 = pyo.Constraint(expr=m.x[0] * m.x[1] * m.x[2] * m.x[3] >= 25.0)
    m.eq1 = pyo.Constraint(
        expr=(
            m.x[0] ** 2 + m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2
            == 40.0 + trivial_expr_with_eval_error
        )
    )
    return m


@unittest.skipIf(cyipopt_available, "cyipopt is available")
class TestCyIpoptNotAvailable(unittest.TestCase):
    def test_not_available_exception(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        msg = "cyipopt is required"
        with self.assertRaisesRegex(RuntimeError, msg):
            solver = CyIpoptSolver(CyIpoptNLP(nlp))


@unittest.skipUnless(cyipopt_available, "cyipopt is not available")
class TestCyIpoptSolver(unittest.TestCase):
    def test_model1(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
        y_sol = np.array([-1.0, 53.90357665])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -428.6362455416348, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_model1_with_scaling(self):
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.o] = 1e-6  # scale the objective
        m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        m.scaling_factor[m.x[1]] = 4.0  # scale one of the x variables

        with TempfileManager.new_context() as temp:
            cynlp = CyIpoptNLP(PyomoNLP(m))
            logfile = temp.create_tempfile('_cyipopt-scaling.log')
            options = {
                'nlp_scaling_method': 'user-scaling',
                'output_file': logfile,
                'file_print_level': 10,
                'max_iter': 0,
            }
            solver = CyIpoptSolver(cynlp, options=options)
            x, info = solver.solve()
            cynlp.close()

            with open(logfile, 'r') as fd:
                solver_trace = fd.read()

        # check for the following strings in the log
        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn(f"output_file = {logfile}", solver_trace)
        self.assertIn('objective scaling factor = 1e-06', solver_trace)
        self.assertIn('x scaling provided', solver_trace)
        self.assertIn('c scaling provided', solver_trace)
        self.assertIn('d scaling provided', solver_trace)
        self.assertIn('DenseVector "x scaling vector" with 3 elements:', solver_trace)
        self.assertIn('x scaling vector[    1]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    2]= 1.0000000000000000e+00', solver_trace)
        self.assertIn('x scaling vector[    3]= 4.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "c scaling vector" with 1 elements:', solver_trace)
        self.assertIn('c scaling vector[    1]= 2.0000000000000000e+00', solver_trace)
        self.assertIn('DenseVector "d scaling vector" with 1 elements:', solver_trace)
        self.assertIn('d scaling vector[    1]= 3.0000000000000000e+00', solver_trace)

    def test_model2(self):
        model = create_model2()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.0, 1.99997807])
        y_sol = np.array([0.00017543])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -31.000000057167462, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_model3(self):
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])

        model = create_model3(G, A, b, c)
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp))
        x, info = solver.solve(tee=False)
        x_sol = np.array([2.0, -1.0, 1.0])
        y_sol = np.array([-3.0, 2.0])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        nlp.set_primals(x)
        nlp.set_duals(y_sol)
        self.assertAlmostEqual(nlp.evaluate_objective(), -3.5, places=5)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_options(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(CyIpoptNLP(nlp), options={'max_iter': 1})
        x, info = solver.solve(tee=False)
        nlp.set_primals(x)
        self.assertAlmostEqual(nlp.evaluate_objective(), -5.0879028e02, places=5)

    @unittest.skipUnless(
        cyipopt_available and cyipopt_ge_1_3, "cyipopt version < 1.3.0"
    )
    def test_hs071_evalerror(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory("cyipopt")
        res = solver.solve(m, tee=True)

        x = list(m.x[:].value)
        expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
        np.testing.assert_allclose(x, expected_x)

    def test_hs071_evalerror_halt(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory("cyipopt", halt_on_evaluation_error=True)
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            res = solver.solve(m, tee=True)

    @unittest.skipIf(
        not cyipopt_available or cyipopt_ge_1_3, "cyipopt version >= 1.3.0"
    )
    def test_hs071_evalerror_old_cyipopt(self):
        m = make_hs071_model()
        solver = pyo.SolverFactory("cyipopt")
        msg = "Error in AMPL evaluation"
        with self.assertRaisesRegex(PyNumeroEvaluationError, msg):
            res = solver.solve(m, tee=True)

    def test_solve_without_objective(self):
        m = create_model1()
        m.o.deactivate()
        m.x[2].fix(0.0)
        m.x[3].fix(4.0)
        solver = pyo.SolverFactory("cyipopt")
        res = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(res)
        self.assertAlmostEqual(m.x[1].value, 9.0)

    def test_solve_13arg_callback(self):
        m = create_model1()

        iterate_data = []

        def intermediate(
            nlp,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials,
        ):
            x = nlp.get_primals()
            y = nlp.get_duals()
            iterate_data.append((x, y))

        x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
        y_sol = np.array([-1.0, 53.90357665])

        solver = pyo.SolverFactory("cyipopt", intermediate_callback=intermediate)
        res = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(res)

        # Make sure iterate vectors have the right shape and that the final
        # iterate contains the primal solution we expect.
        for x, y in iterate_data:
            self.assertEqual(x.shape, (3,))
            self.assertEqual(y.shape, (2,))
        x, y = iterate_data[-1]
        self.assertTrue(np.allclose(x_sol, x))
        # Note that we can't assert that dual variables in the NLP are those
        # at the solution because, at this point in the algorithm, the NLP
        # only has access to the *previous iteration's* dual values.

    # The 13-arg callback works with cyipopt < 1.3, but we will use the
    # get_current_iterate method, which is only available in 1.3+ and IPOPT 3.14+
    @unittest.skipIf(
        not cyipopt_available or not cyipopt_ge_1_3 or not ipopt_ge_3_14,
        "cyipopt version < 1.3.0",
    )
    def test_solve_get_current_iterate(self):
        m = create_model1()

        iterate_data = []

        def intermediate(
            nlp,
            problem,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials,
        ):
            iterate = problem.get_current_iterate()
            x = iterate["x"]
            y = iterate["mult_g"]
            iterate_data.append((x, y))

        x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
        y_sol = np.array([-1.0, 53.90357665])

        solver = pyo.SolverFactory("cyipopt", intermediate_callback=intermediate)
        res = solver.solve(m, tee=True)
        pyo.assert_optimal_termination(res)

        # Make sure iterate vectors have the right shape and that the final
        # iterate contains the primal and dual solution we expect.
        for x, y in iterate_data:
            self.assertEqual(x.shape, (3,))
            self.assertEqual(y.shape, (2,))
        x, y = iterate_data[-1]
        self.assertTrue(np.allclose(x_sol, x))
        self.assertTrue(np.allclose(y_sol, y))
