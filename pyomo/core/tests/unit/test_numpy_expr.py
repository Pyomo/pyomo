#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
import pyomo.common.unittest as unittest

from pyomo.common.dependencies import (
    numpy as np, numpy_available,
    pandas as pd, pandas_available,
)

from pyomo.environ import (
    ConcreteModel, Var, RangeSet, Param, Objective, Set, Constraint,
)
from pyomo.core.expr.current import MonomialTermExpression
from pyomo.core.expr.numvalue import NumericNDArray

@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestNumPy(unittest.TestCase):
    def test_numpy_scalar_times_scalar_var(self):
        # Test issue #685
        m = ConcreteModel()
        m.x = Var()
        e = np.float64(5) * m.x
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(str(e), "5.0*x")

        e = m.x * np.float64(5)
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(str(e), "5.0*x")

    def test_numpy_float(self):
        # Test issue #31
        m = ConcreteModel()

        m.T = Set(initialize=range(3))
        m.v = Var(initialize=1, bounds=(0,None))
        m.c = Var(m.T, initialize=20)
        h = [np.float32(1.0), 1.0, 1]

        def rule(m, t):
            return m.c[0] == h[t] * m.c[0]
        m.x = Constraint(m.T, rule=rule)

        def rule(m, t):
            return m.c[0] == h[t] * m.c[0] * m.v
        m.y = Constraint(m.T, rule=rule)

        def rule(m, t):
            return m.c[0] == h[t] * m.v
        m.z = Constraint(m.T, rule=rule)

        #m.pprint()
        for t in m.T:
            self.assertEqual(str(m.x[0].expr), str(m.x[t].expr))
            self.assertEqual(str(m.y[0].expr), str(m.y[t].expr))
            self.assertEqual(str(m.z[0].expr), str(m.z[t].expr))

    def test_indexed_constraint(self):
        m = ConcreteModel()
        m.x = Var([0,1,2,3])
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([10, 20])
        m.c = Constraint([0,1], expr=A @ m.x <= b)
        self.assertEqual(
            str(m.c[0].expr),
            "x[0] + 2*x[1] + 3*x[2] + 4*x[3]  <=  10.0"
        )
        self.assertEqual(
            str(m.c[1].expr),
            "5*x[0] + 6*x[1] + 7*x[2] + 8*x[3]  <=  20.0"
        )
