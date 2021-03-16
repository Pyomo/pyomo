#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import os

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import RenamedNLP, ProjectedNLP

def create_pyomo_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=(-10,10), initialize={0:1.0, 1:2.0, 2:4.0})

    m.obj = pyo.Objective(expr=m.x[0]**2 + m.x[0]*m.x[1] + m.x[0]*m.x[2] + m.x[2]**2)

    m.con1 = pyo.Constraint(expr=m.x[0]*m.x[1] + m.x[0]*m.x[2] == 4)
    m.con2 = pyo.Constraint(expr=m.x[0] + m.x[2] == 4)

    return m

class TestRenamedNLP(unittest.TestCase):
    def test_rename(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(nlp.primals_names(), expected_names)
        renamed_nlp = RenamedNLP(nlp, {'x[0]': 'y[0]', 'x[1]':'y[1]', 'x[2]':'y[2]'})
        expected_names = ['y[0]', 'y[1]', 'y[2]']
        
class TestProjectedNLP(unittest.TestCase):
    def test_projected(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(),
                                       np.asarray([1.0, 2.0, 4.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 1.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0],[1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test a reordering
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0, 2.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0],[1.0, 1.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0],[1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'y', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'y', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(),np.asarray([1.0, 4.0, np.nan, 2.0]))
        
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0, 0.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 0.0, 1.0],[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]'])
        expected_names = ['x[0]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(),np.asarray([1.0, 4.0]))
        
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 4)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 4)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 4)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0],[1.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 4)
        expectedH = np.asarray([[2.0, 1.0],[1.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

if __name__ == '__main__':
    TestRenamedNLP().test_rename()
    TestProjectedNLP().test_projected()
    
    
