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
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy_available, attempt_import

import pyomo.environ as pyo

# Use attempt_import here due to unguarded NumPy import in this file
nlp = attempt_import("pyomo.contrib.pynumero.interfaces.pyomo_nlp")[0]
import pyomo.contrib.sensitivity_toolbox.pynumero as pnsens
from pyomo.contrib.pynumero.asl import AmplInterface

if not scipy_available or not numpy_available:
    raise unittest.SkipTest("scipy or numpy is not available")

if not AmplInterface.available():
    raise unittest.SkipTest("Pynumero needs the ASL extension to run NLP tests")


class TestSeriesData(unittest.TestCase):
    def test_dsdp_dfdp_pyomo(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(initialize=200)
        m.x2 = pyo.Var(initialize=5)
        m.p1 = pyo.Var(initialize=10)
        m.p2 = pyo.Var(initialize=5)
        m.obj = pyo.Objective(
            expr=m.x1 * m.p1 + m.x2 * m.x2 * m.p2 + m.p1 * m.p2, sense=pyo.minimize
        )
        m.c1 = pyo.Constraint(expr=m.x1 == 2 * m.p1**2)
        m.c2 = pyo.Constraint(expr=m.x2 == m.p2)
        theta = [m.p1, m.p2]

        dsdp, dfdp, rmap, cmap = pnsens.get_dsdp_dfdp(m, theta)

        # Since x1 = p1
        np.testing.assert_almost_equal(dsdp[rmap[m.x1], cmap[m.p1]], 40.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x1], cmap[m.p2]], 0.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x2], cmap[m.p1]], 0.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x2], cmap[m.p2]], 1.0)

        # if x1 = 2 * p1 and x2 = p2 then
        #   df/dp1 = 6 p1**2 + p2 = 45.0
        #   df/dp2 = 3 p2 + p1 = 85.0
        np.testing.assert_almost_equal(dfdp[0, cmap[m.p1]], 605.0)
        np.testing.assert_almost_equal(dfdp[0, cmap[m.p2]], 85.0)

    def test_dsdp_dfdp_pyomo_nlp(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(initialize=200)
        m.x2 = pyo.Var(initialize=5)
        m.p1 = pyo.Var(initialize=10)
        m.p2 = pyo.Var(initialize=5)
        m.obj = pyo.Objective(
            expr=m.x1 * m.p1 + m.x2 * m.x2 * m.p2 + m.p1 * m.p2, sense=pyo.minimize
        )
        m.c1 = pyo.Constraint(expr=m.x1 == 2 * m.p1**2)
        m.c2 = pyo.Constraint(expr=m.x2 == m.p2)
        theta = [m.p1, m.p2]

        m2 = nlp.PyomoNLP(m)
        dsdp, dfdp, rmap, cmap = pnsens.get_dsdp_dfdp(m2, theta)

        # Since x1 = p1
        assert dsdp.shape == (4, 2)
        np.testing.assert_almost_equal(dsdp[rmap[m.x1], cmap[m.p1]], 40.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x1], cmap[m.p2]], 0.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x2], cmap[m.p1]], 0.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.x2], cmap[m.p2]], 1.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.p1], cmap[m.p2]], 0.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.p2], cmap[m.p2]], 1.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.p1], cmap[m.p1]], 1.0)
        np.testing.assert_almost_equal(dsdp[rmap[m.p2], cmap[m.p1]], 0.0)

        # if x1 = 2 * p1 and x2 = p2 then
        #   df/dp1 = 6 p1**2 + p2 = 45.0
        #   df/dp2 = 3 p2 + p1 = 85.0
        assert dfdp.shape == (1, 2)
        np.testing.assert_almost_equal(dfdp[0, cmap[m.p1]], 605.0)
        np.testing.assert_almost_equal(dfdp[0, cmap[m.p2]], 85.0)

    def test_dydp_pyomo(self):
        m = pyo.ConcreteModel()
        m.x1 = pyo.Var(initialize=200)
        m.x2 = pyo.Var(initialize=5)
        m.p1 = pyo.Var(initialize=10)
        m.p2 = pyo.Var(initialize=5)
        m.obj = pyo.Objective(
            expr=m.x1 * m.p1 + m.x2 * m.x2 * m.p2 + m.p1 * m.p2, sense=pyo.minimize
        )
        m.c1 = pyo.Constraint(expr=m.x1 == 2 * m.p1**2)
        m.c2 = pyo.Constraint(expr=m.x2 == m.p2)

        m.e1 = pyo.Expression(expr=2 * m.p1**2)
        m.e2 = pyo.Expression(expr=m.p2)
        m.e3 = pyo.Expression(expr=m.x1 * m.p1 + m.x2 * m.x2 * m.p2 + m.p1 * m.p2)

        theta = [m.p1, m.p2]
        dsdp, dfdp, rmap, cmap = pnsens.get_dsdp_dfdp(m, theta)

        dydp, rmap = pnsens.get_dydp([m.e1, m.x1, m.e2, m.x2, m.e3], dsdp, rmap, cmap)

        assert dydp.shape == (5, 2)
        np.testing.assert_almost_equal(dydp[rmap[m.x1], cmap[m.p1]], 40.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e1], cmap[m.p1]], 40.0)
        np.testing.assert_almost_equal(dydp[rmap[m.x1], cmap[m.p2]], 0.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e1], cmap[m.p2]], 0.0)
        np.testing.assert_almost_equal(dydp[rmap[m.x2], cmap[m.p1]], 0.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e2], cmap[m.p1]], 0.0)
        np.testing.assert_almost_equal(dydp[rmap[m.x2], cmap[m.p2]], 1.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e2], cmap[m.p2]], 1.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e3], cmap[m.p1]], 605.0)
        np.testing.assert_almost_equal(dydp[rmap[m.e3], cmap[m.p2]], 85.0)

        # make sure the rows are in the order of y_list
        assert rmap[m.e1] == 0
        assert rmap[m.x1] == 1
        assert rmap[m.e2] == 2
        assert rmap[m.x2] == 3
        assert rmap[m.e3] == 4
