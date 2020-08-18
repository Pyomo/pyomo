#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
import pyomo.environ as pyo
import os

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_sparse as spa, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

try:
    import ipopt
except ImportError:
    raise unittest.SkipTest("Pynumero needs cyipopt to run CyIpoptSolver tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptSolver, CyIpoptNLP
)


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
    model.hessian_f = pyo.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = pyo.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
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
        return model.x[1] ** 4 + (model.x[1] + model.x[2]) ** 2 + (-1.0 + pyo.exp(model.x[2])) ** 2

    model.f = pyo.Objective(rule=f)
    return model


def create_model9():
    # clplatea OXR2-MN-V-0
    model = pyo.ConcreteModel()

    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2

    model.x = pyo.Var(pyo.RangeSet(1, p), pyo.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 \
                   for i in range(2, p + 1) for j in range(2, p + 1)) + (wght * model.x[p, p])

    model.f = pyo.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


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
        m.scaling_factor[m.o] = 1e-6 # scale the objective
        m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        m.scaling_factor[m.x[1]] = 4.0  # scale one of the x variables

        cynlp = CyIpoptNLP(PyomoNLP(m))
        options={'nlp_scaling_method': 'user-scaling',
                 'output_file': '_cyipopt-scaling.log',
                 'file_print_level':10,
                 'max_iter': 0}
        solver = CyIpoptSolver(cynlp, options=options)
        x, info = solver.solve()

        with open('_cyipopt-scaling.log', 'r') as fd:
            solver_trace = fd.read()
        os.remove('_cyipopt-scaling.log')

        # check for the following strings in the log and then delete the log
        self.assertIn('nlp_scaling_method = user-scaling', solver_trace)
        self.assertIn('output_file = _cyipopt-scaling.log', solver_trace)
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
        y_sol = np.array([-3.,  2.])
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
        self.assertAlmostEqual(nlp.evaluate_objective(), -5.0879028e+02, places=5)
