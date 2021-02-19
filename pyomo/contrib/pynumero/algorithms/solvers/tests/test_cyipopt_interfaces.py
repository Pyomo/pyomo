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

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptNLP


def create_model1():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=4.0)
    m.d = pyo.Constraint(expr=m.x[1] + m.x[2] ** 2 <= 18.0)
    m.c = pyo.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.o = pyo.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)
    return m

class TestCyIpoptNLP(unittest.TestCase):

    def test_model1_CyIpoptNLP(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        cynlp = CyIpoptNLP(nlp)
        self._check_model1(nlp, cynlp)

    def test_model1_CyIpoptNLP_scaling(self):
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.scaling_factor[m.o] = 1e-6 # scale the objective
        m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        m.scaling_factor[m.x[1]] = 4.0  # scale one of the x variables

        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-6)
        self.assertTrue(len(x_scaling) == 3)
        # vars are in order x[2], x[3], x[1]
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        # assuming the order is d then c
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)

        # test missing obj
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        #m.scaling_factor[m.o] = 1e-6 # scale the objective
        m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        m.scaling_factor[m.x[1]] = 4.0  # scale the x variable

        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1.0)
        self.assertTrue(len(x_scaling) == 3)
        # vars are in order x[2], x[3], x[1]
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        # assuming the order is d then c
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)

        # test missing var
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.scaling_factor[m.o] = 1e-6 # scale the objective
        m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        #m.scaling_factor[m.x] = 4.0  # scale the x variable

        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-6)
        self.assertTrue(len(x_scaling) == 3)
        # vars are in order x[2], x[3], x[1]
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 1.0)
        self.assertTrue(len(g_scaling) == 2)
        # assuming the order is d then c
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 2.0)

        # test missing c
        m = create_model1()
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

        m.scaling_factor[m.o] = 1e-6 # scale the objective
        #m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        m.scaling_factor[m.x[1]] = 4.0  # scale the x variable

        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling == 1e-6)
        self.assertTrue(len(x_scaling) == 3)
        # vars are in order x[2], x[3], x[1]
        self.assertTrue(x_scaling[0] == 1.0)
        self.assertTrue(x_scaling[1] == 1.0)
        self.assertTrue(x_scaling[2] == 4.0)
        self.assertTrue(len(g_scaling) == 2)
        # assuming the order is d then c
        self.assertTrue(g_scaling[0] == 3.0)
        self.assertTrue(g_scaling[1] == 1.0)

        # test missing all
        m = create_model1()
        #m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        #m.scaling_factor[m.o] = 1e-6 # scale the objective
        #m.scaling_factor[m.c] = 2.0  # scale the equality constraint
        #m.scaling_factor[m.d] = 3.0  # scale the inequality constraint
        #m.scaling_factor[m.x] = 4.0  # scale the x variable

        cynlp = CyIpoptNLP(PyomoNLP(m))
        obj_scaling, x_scaling, g_scaling = cynlp.scaling_factors()
        self.assertTrue(obj_scaling is None)
        self.assertTrue(x_scaling is None)
        self.assertTrue(g_scaling is None)
        
    def _check_model1(self, nlp, cynlp):
        # test x_init
        expected_xinit = np.asarray([4.0, 4.0, 4.0], dtype=np.float64)
        xinit = cynlp.x_init()
        self.assertTrue(np.array_equal(xinit, expected_xinit))

        # test x_lb
        expected_xlb = list()
        for v in nlp.get_pyomo_variables():
            if v.lb == None:
                expected_xlb.append(-np.inf)
            else:
                expected_xlb.append(v.lb)
        expected_xlb = np.asarray(expected_xlb)
        xlb = cynlp.x_lb()
        self.assertTrue(np.array_equal(xlb, expected_xlb))

        # test x_ub
        expected_xub = list()
        for v in nlp.get_pyomo_variables():
            if v.ub == None:
                expected_xub.append(np.inf)
            else:
                expected_xub.append(v.ub)
        expected_xub = np.asarray(expected_xub)
        xub = cynlp.x_ub()
        self.assertTrue(np.array_equal(xub, expected_xub))

        # test g_lb
        expected_glb = np.asarray([-np.inf, 0.0], dtype=np.float64)
        glb = cynlp.g_lb()
        self.assertTrue(np.array_equal(glb, expected_glb))

        # test g_ub
        expected_gub = np.asarray([18, 0.0], dtype=np.float64)
        gub = cynlp.g_ub()
        print(expected_gub)
        print(gub)
        self.assertTrue(np.array_equal(gub, expected_gub))

        x = cynlp.x_init()
        # test objective
        self.assertEqual(cynlp.objective(x), -504)
        # test gradient
        expected = np.asarray([-576, 8, 64], dtype=np.float64)
        self.assertTrue(np.allclose(expected, cynlp.gradient(x)))
        # test constraints
        expected = np.asarray([20, -5], dtype=np.float64)
        constraints = cynlp.constraints(x)
        self.assertTrue(np.allclose(expected, constraints)) 
        
        # test jacobian
        expected = np.asarray([[8.0, 0, 1.0],[0.0, 8.0, 1.0]])
        spexpected = spa.coo_matrix(expected).todense()
        rows, cols = cynlp.jacobianstructure()
        values = cynlp.jacobian(x)
        jac = spa.coo_matrix((values, (rows,cols)), shape=(len(constraints), len(x))).todense()
        self.assertTrue(np.allclose(spexpected, jac))

        # test hessian
        y = constraints.copy()
        y.fill(1.0)
        rows, cols = cynlp.hessianstructure()
        values = cynlp.hessian(x, y, obj_factor=1.0)
        hess_lower = spa.coo_matrix((values, (rows,cols)), shape=(len(x), len(x))).todense()
        expected_hess_lower = np.asarray([[-286.0, 0.0, 0.0], [0.0, 4.0, 0.0], [-144.0, 0.0, 192.0]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_hess_lower, hess_lower))
