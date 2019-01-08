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
import pyomo.environ as aml

try:
    import scipy.sparse as spa
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

from pyomo.contrib.pynumero.interfaces import PyomoNLP, TwoStageStochasticNLP

try:
    import ipopt
    from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
except ImportError:
    raise unittest.SkipTest("Pynumero needs cyipopt to run CyIpoptSolver tests")


def create_model1():
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    # m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m

def create_model2():
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2], initialize=4.0)
    m.d = aml.Constraint(expr=m.x[1] + m.x[2] <= 5)
    m.o = aml.Objective(expr=m.x[1] ** 2 + 4 * m.x[2] ** 2 - 8 * m.x[1] - 16 * m.x[2])
    m.x[1].setub(3.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model3(G, A, b, c):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)
    model.hessian_f = aml.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = aml.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = aml.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = aml.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def objective_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.obj = aml.Objective(rule=objective_rule, sense=aml.minimize)

    return model

def create_model4():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2], initialize=1.0)
    m.c1 = aml.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
    m.obj = aml.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)
    return m


def create_model6():
    model = aml.ConcreteModel()

    model.S = [1, 2]
    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return model.x[1] ** 4 + (model.x[1] + model.x[2]) ** 2 + (-1.0 + aml.exp(model.x[2])) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model9():

    # clplatea OXR2-MN-V-0
    model = aml.ConcreteModel()

    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 \
                   for i in range(2, p + 1) for j in range(2, p + 1)) + (wght * model.x[p, p])

    model.f = aml.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model


class TestCyIpoptSolver(unittest.TestCase):

    def test_model1(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(nlp)
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.85958688, 4.67936007, 3.10358931])
        y_sol = np.array([-1.0, 53.90357665])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        self.assertAlmostEqual(nlp.objective(x), -428.6362455416348)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_model2(self):
        model = create_model2()
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(nlp)
        x, info = solver.solve(tee=False)
        x_sol = np.array([3.0, 1.99997807])
        y_sol = np.array([0.00017543])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        self.assertAlmostEqual(nlp.objective(x), -31.000000057167462, 3)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_model3(self):
        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])

        model = create_model3(G, A, b, c)
        nlp = PyomoNLP(model)
        solver = CyIpoptSolver(nlp)
        x, info = solver.solve(tee=False)
        x_sol = np.array([2.0, -1.0, 1.0])
        y_sol = np.array([-3.,  2.])
        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        self.assertAlmostEqual(nlp.objective(x), -3.5, 3)
        self.assertTrue(np.allclose(info['mult_g'], y_sol, rtol=1e-4))

    def test_composite_nlp(self):

        G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = np.array([[1, 0, 1], [0, 1, 1]])
        b = np.array([3, 0])
        c = np.array([-8, -3, -3])

        scenarios = dict()
        coupling_vars = dict()
        n_scenarios = 2
        np.random.seed(seed=985739465)
        bs = [b, b+0.001]

        for i in range(n_scenarios):
            instance = create_model3(G, A, bs[i], c)
            nlp = PyomoNLP(instance)
            scenario_name = "s{}".format(i)
            scenarios[scenario_name] = nlp
            coupling_vars[scenario_name] = [nlp.variable_idx(instance.x[0])]

        nlp = TwoStageStochasticNLP(scenarios, coupling_vars)

        solver = CyIpoptSolver(nlp)
        x, info = solver.solve(tee=False)
        x_sol = np.array([2.00003846,
                          -0.99996154,
                          0.99996154,
                          2.00003846,
                          -0.99996154,
                          1.00096154,
                          2.00003846])

        self.assertTrue(np.allclose(x, x_sol, rtol=1e-4))
        self.assertAlmostEqual(nlp.objective(x), -6.99899, 3)



