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
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'inverse_reduced_hessian numpy',
                                     minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'inverse_reduced_hessian requires scipy')

if numpy_available:
    from pyomo.contrib.pynumero.asl import AmplInterface
    asl_available = AmplInterface.available()
else:
    asl_available=False

if not (numpy_available and scipy_available and asl_available):
    raise unittest.SkipTest('inverse_reduced_hessian tests require numpy, scipy, and asl')
from pyomo.common.dependencies import (pandas as pd, pandas_available)

ipopt_solver = pyo.SolverFactory('ipopt')
if not ipopt_solver.available(exception_flag=False):
    raise unittest.SkipTest('ipopt is not available')

numdiff_available = True
try:
    import numdifftools as nd
except:
    numdiff_available = False

from pyomo.contrib.interior_point.inverse_reduced_hessian import inv_reduced_hessian_barrier
                 
class TestInverseReducedHessian(unittest.TestCase):
    # the original test
    def test_invrh_zavala_thesis(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1,2,3]) 
        m.obj = pyo.Objective(expr=(m.x[1]-1)**2 + (m.x[2]-2)**2 + (m.x[3]-3)**2)
        m.c1 = pyo.Constraint(expr=m.x[1] + 2*m.x[2] + 3*m.x[3]==0)

        status, invrh = inv_reduced_hessian_barrier(m, [m.x[2], m.x[3]])
        expected_invrh = np.asarray([[ 0.35714286, -0.21428571],
                                     [-0.21428571, 0.17857143]])
        np.testing.assert_array_almost_equal(invrh, expected_invrh)

    # test by DLW, April 2020
    def _simple_model(self, add_constraint=False):
        # Hardwired to have two x columns and one y
        # if add_constraint is true, there is a binding constraint on b0
        data = pd.DataFrame([[1, 1.1, 0.365759306],
                             [2, 1.2, 4],
                             [3, 1.3, 4.8876684],
                             [4, 1.4, 5.173455561],
                             [5, 1.5, 2.093799081],
                             [6, 1.6, 9],
                             [7, 1.7, 6.475045106],
                             [8, 1.8, 8.127111268],
                             [9, 1.9, 6],
                             [10, 1.21, 10.20642714],
                             [11, 1.22, 13.08211636],
                             [12, 1.23, 10],
                             [13, 1.24, 15.38766047],
                             [14, 1.25, 14.6587746],
                             [15, 1.26, 13.68608604],
                             [16, 1.27, 14.70707893],
                             [17, 1.28, 18.46192779],
                             [18, 1.29, 15.60649164]],
                             columns=['tofu','chard', 'y'])

        model = pyo.ConcreteModel()

        model.b0 = pyo.Var(initialize = 0)
        model.bindexes = pyo.Set(initialize=['tofu', 'chard'])
        model.b = pyo.Var(model.bindexes, initialize = 1)

        # try to make trouble
        if add_constraint:
            model.binding_constraint = pyo.Constraint(expr=model.b0>=10)

        # The columns need to have unique values (or you get warnings)
        def response_rule(m, t, c):
            expr = m.b0 + m.b['tofu']*t + m.b['chard']*c
            return expr
        model.response_function = pyo.Expression(data.tofu, data.chard, rule = response_rule)

        def SSE_rule(m):
            return sum((data.y[i] - m.response_function[data.tofu[i], data.chard[i]])**2\
                            for i in data.index)
        model.SSE = pyo.Objective(rule = SSE_rule, sense=pyo.minimize)

        return model

    @unittest.skipIf(not numdiff_available, "numdifftools missing")
    @unittest.skipIf(not pandas_available, "pandas missing")
    def test_3x3_using_linear_regression(self):
        """ simple linear regression with two x columns, so 3x3 Hessian"""        

        model = self._simple_model()
        solver = pyo.SolverFactory("ipopt")
        status = solver.solve(model)
        self.assertTrue(check_optimal_termination(status))
        tstar = [pyo.value(model.b0),
                 pyo.value(model.b['tofu']), pyo.value(model.b['chard'])]

        def _ndwrap(x):
            # wrapper for numdiff call
            model.b0.fix(x[0])
            model.b["tofu"].fix(x[1])
            model.b["chard"].fix(x[2])
            rval = pyo.value(model.SSE)
            return rval

        H = nd.Hessian(_ndwrap)(tstar)
        HInv = np.linalg.inv(H)

        model.b0.fixed = False
        model.b["tofu"].fixed = False
        model.b["chard"].fixed = False
        status, H_inv_red_hess = inv_reduced_hessian_barrier(model,
                                                             [model.b0,
                                                              model.b["tofu"],
                                                              model.b["chard"]])
        # this passes at decimal=6, BTW
        np.testing.assert_array_almost_equal(HInv, H_inv_red_hess, decimal=3)


    @unittest.skipIf(not numdiff_available, "numdifftools missing")
    @unittest.skipIf(not pandas_available, "pandas missing")
    def test_with_binding_constraint(self):
        """ there is a binding constraint"""        

        model = self._simple_model(add_constraint=True)

        status, H_inv_red_hess = inv_reduced_hessian_barrier(model,
                                                             [model.b0,
                                                              model.b["tofu"],
                                                              model.b["chard"]])
        print("test_with_binding_constraint should see an error raised.")


if __name__ == '__main__':
    unittest.main()
