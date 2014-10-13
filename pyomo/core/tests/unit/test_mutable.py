#
# Unit Tests for Elements of a Model
#
# TestSimpleCon                Class for testing single constraint
# TestArrayCon                Class for testing array of constraint
#

import logging
import os
import sys

from six import StringIO

from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

from pyomo.core.base import IntegerSet
from pyomo.environ import *
from pyomo.opt import *
import pyutilib.th as unittest
import pyutilib.services


class TestMutable(unittest.TestCase):
    def test_mutable_constraint_upper(self):
        model = AbstractModel()
        model.P = Param(initialize=2.0, mutable=True)
        model.X = Var()

        def constraint_rule(m):
            return m.X <= m.P
        model.C = Constraint(rule=constraint_rule)

        instance = model.create()

        self.assertEqual(value(instance.C.upper), 2.0)

        instance.P = 4.0
        instance.preprocess()

        self.assertEqual(value(instance.C.upper), 4.0)


    def test_mutable_constraint_lower(self):
        model = AbstractModel()
        model.Q = Param(initialize=2.0, mutable=True)
        model.X = Var()

        def constraint_rule(m):
            return m.X >= m.Q
        model.C = Constraint(rule=constraint_rule)

        instance = model.create()

        self.assertEqual(value(instance.C.lower), 2.0)

        instance.Q = 4.0
        instance.preprocess()

        self.assertEqual(value(instance.C.lower), 4.0)


    def test_mutable_constraint_both(self):
        model = AbstractModel()
        model.P = Param(initialize=4.0, mutable=True)
        model.Q = Param(initialize=2.0, mutable=True)
        model.X = Var()

        def constraint_rule(m):
            return m.Q <= m.X <= m.P
        model.C = Constraint(rule=constraint_rule)

        instance = model.create()

        self.assertEqual(value(instance.C.lower), 2.0)
        self.assertEqual(value(instance.C.upper), 4.0)

        instance.P = 8.0
        instance.Q = 1.0
        instance.preprocess()

        self.assertEqual(value(instance.C.lower), 1.0)
        self.assertEqual(value(instance.C.upper), 8.0)



    def test_mutable_var_bounds_lower(self):
        model = AbstractModel()
        model.P = Param(initialize=2.0, mutable=True)
        model.X = Var(bounds=(model.P,None))

        instance = model.create()

        self.assertEqual(instance.X.bounds, (2.0, None))

        instance.P = 4.0
        instance.preprocess()

        self.assertEqual(instance.X.bounds, (4.0, None))


    def test_mutable_var_bounds_upper(self):
        model = AbstractModel()
        model.Q = Param(initialize=2.0, mutable=True)
        model.X = Var(bounds=(model.Q,None))

        instance = model.create()

        self.assertEqual(instance.X.bounds, (2.0, None))

        instance.Q = 4.0
        instance.preprocess()

        self.assertEqual(instance.X.bounds, (4.0, None))


    def test_mutable_var_bounds_both(self):
        model = AbstractModel()
        model.P = Param(initialize=4.0, mutable=True)
        model.Q = Param(initialize=2.0, mutable=True)
        model.X = Var(bounds=(model.P,model.Q))

        instance = model.create()

        self.assertEqual(value(instance.X.lb), 4.0)
        self.assertEqual(value(instance.X.ub), 2.0)

        instance.P = 8.0
        instance.Q = 1.0
        instance.preprocess()

        self.assertEqual(value(instance.X.lb), 8.0)
        self.assertEqual(value(instance.X.ub), 1.0)

if __name__ == "__main__":
    unittest.main()

