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
from pyomo.core.expr.compare import compare_expressions

@unittest.skipUnless(numpy_available, 'numpy is not available')
class TestNumPy(unittest.TestCase):
    def test_numpy_scalar_times_scalar_var(self):
        # Test issue #685
        m = ConcreteModel()
        m.x = Var()
        e = np.float64(5) * m.x
        self.assertIs(type(e), MonomialTermExpression)
        self.assertTrue(compare_expressions(e, 5.0*m.x))

        e = m.x * np.float64(5)
        self.assertIs(type(e), MonomialTermExpression)
        self.assertTrue(compare_expressions(e, 5.0*m.x))

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
            self.assertTrue(compare_expressions(m.x[0].expr, m.x[t].expr))
            self.assertTrue(compare_expressions(m.y[0].expr, m.y[t].expr))
            self.assertTrue(compare_expressions(m.z[0].expr, m.z[t].expr))

    def test_indexed_constraint(self):
        m = ConcreteModel()
        m.x = Var([0,1,2,3])
        A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([10, 20])
        m.c = Constraint([0,1], expr=A @ m.x <= b)
        self.assertTrue(compare_expressions(
            m.c[0].expr,
            m.x[0] + 2*m.x[1] + 3*m.x[2] + 4*m.x[3] <= 10))
        self.assertTrue(compare_expressions(
            m.c[1].expr,
            5*m.x[0] + 6*m.x[1] + 7*m.x[2] + 8*m.x[3] <= 20))
