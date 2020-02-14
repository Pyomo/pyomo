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

from pyomo.contrib.pynumero import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import scipy.sparse as spa
import numpy as np

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run CyIpoptSolver tests")

import scipy.sparse as sp
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

    def test_model1(self):
        model = create_model1()
        nlp = PyomoNLP(model)
        cynlp = CyIpoptNLP(nlp)

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
        spexpected = sp.coo_matrix(expected).todense()
        rows, cols = cynlp.jacobianstructure()
        values = cynlp.jacobian(x)
        jac = sp.coo_matrix((values, (rows,cols)), shape=(len(constraints), len(x))).todense()
        self.assertTrue(np.allclose(spexpected, jac))

        # test hessian
        y = constraints.copy()
        y.fill(1.0)
        rows, cols = cynlp.hessianstructure()
        values = cynlp.hessian(x, y, obj_factor=1.0)
        hess_lower = sp.coo_matrix((values, (rows,cols)), shape=(len(x), len(x))).todense()
        expected_hess_lower = np.asarray([[-286.0, 0.0, 0.0], [0.0, 4.0, 0.0], [-144.0, 0.0, 192.0]], dtype=np.float64)
        self.assertTrue(np.allclose(expected_hess_lower, hess_lower))
