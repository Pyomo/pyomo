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
# Unit Tests for Integral Objects
#

import os
from os.path import abspath, dirname

import pyutilib.th as unittest

from pyomo.environ import (ConcreteModel, Var, Set, TransformationFactory,
                           Expression)
from pyomo.dae import ContinuousSet, Integral
from pyomo.dae.diffvar import DAE_Error

from pyomo.repn import generate_standard_repn

from six import StringIO

currdir = dirname(abspath(__file__)) + os.sep


class TestIntegral(unittest.TestCase):

    # test valid declarations
    def test_valid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.t, m.x)

        def _int1(m, t):
            return m.v[t]

        m.int1 = Integral(m.t, rule=_int1)

        def _int2(m, s, t):
            return m.v2[s, t]

        m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

        def _int3(m, t, x):
            return m.v3[t, x]

        m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

        def _int4(m, x):
            return m.int3[x]

        m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

        self.assertTrue(isinstance(m.int1, Expression))
        self.assertTrue(isinstance(m.int2, Expression))
        self.assertTrue(isinstance(m.int3, Expression))
        self.assertTrue(isinstance(m.int4, Expression))
        self.assertTrue(m.int1.get_continuousset() is m.t)
        self.assertTrue(m.int2.get_continuousset() is m.t)
        self.assertTrue(m.int3.get_continuousset() is m.t)
        self.assertTrue(m.int4.get_continuousset() is m.x)
        self.assertEqual(len(m.int1), 1)
        self.assertEqual(len(m.int2), 3)
        self.assertEqual(len(m.int3), 2)
        self.assertEqual(len(m.int4), 1)
        self.assertTrue(m.int1.type() is Integral)
        self.assertTrue(m.int2.type() is Integral)
        self.assertTrue(m.int3.type() is Integral)
        self.assertTrue(m.int4.type() is Integral)

        repn = generate_standard_repn(m.int1.expr)
        self.assertEqual(repn.linear_coefs, (0.5, 0.5))
        self.assertTrue(repn.linear_vars[0] is m.v[1])
        self.assertTrue(repn.linear_vars[1] is m.v[0])

        repn = generate_standard_repn(m.int2[1].expr)
        self.assertEqual(repn.linear_coefs, (0.5, 0.5))
        self.assertTrue(repn.linear_vars[0] is m.v2[1, 1])
        self.assertTrue(repn.linear_vars[1] is m.v2[1, 0])

        repn = generate_standard_repn(m.int4.expr)
        self.assertEqual(repn.linear_coefs, (1.25, 1.25, 1.25, 1.25))
        self.assertTrue(repn.linear_vars[0] is m.v3[1, 10])
        self.assertTrue(repn.linear_vars[1] is m.v3[0, 10])
        self.assertTrue(repn.linear_vars[2] is m.v3[1, 5])
        self.assertTrue(repn.linear_vars[3] is m.v3[0, 5])

    # test invalid declarations
    def test_invalid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.x, m.t)

        def _int(m, t):
            return m.v[t]

        def _int2(m, x, t):
            return m.v3[x, t]

        def _int3(m, s, t):
            return m.v2[s,t]

        # Integrals must be indexed by a ContinuousSet
        try:
            m.int = Integral(rule=_int)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # Specifying multiple aliases of same option
        try:
            m.int = Integral(m.t, wrt=m.t, withrespectto=m.t, rule=_int)
            self.fail('Expected TypeError')
        except TypeError:
            pass

        # No ContinuousSet specified
        try:
            m.int2 = Integral(m.x, m.t, rule= _int2)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # 'wrt' is not a ContinuousSet
        try:
            m.int = Integral(m.s, m.t, wrt=m.s, rule=_int2)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # 'wrt' is not in argument list
        try:
            m.int = Integral(m.t, wrt=m.x, rule=_int)
            self.fail('Expected ValueError')
        except ValueError:
            pass

        # 'bounds' not supported
        try:
            m.int = Integral(m.t, wrt=m.t, rule=_int, bounds=(0,0.5))
            self.fail('Expected DAE_Error')
        except DAE_Error:
            pass

        # No rule specified
        try:
            m.int = Integral(m.t, wrt=m.t)
            self.fail('Expected ValueError')
        except ValueError:
            pass

            # test DerivativeVar reclassification after discretization

        def test_reclassification_finite_difference(self):
            m = ConcreteModel()
            m.t = ContinuousSet(bounds=(0, 1))
            m.x = ContinuousSet(bounds=(5, 10))
            m.s = Set(initialize=[1, 2, 3])
            m.v = Var(m.t)
            m.v2 = Var(m.s, m.t)
            m.v3 = Var(m.t, m.x)

            def _int1(m, t):
                return m.v[t]

            m.int1 = Integral(m.t, rule=_int1)

            def _int2(m, s, t):
                return m.v2[s, t]

            m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

            def _int3(m, t, x):
                return m.v3[t, x]

            m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

            def _int4(m, x):
                return m.int3[x]

            m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

            self.assertFalse(m.int1.is_fully_discretized())
            self.assertFalse(m.int2.is_fully_discretized())
            self.assertFalse(m.int3.is_fully_discretized())
            self.assertFalse(m.int4.is_fully_discretized())

            TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)

            self.assertTrue(m.int1.is_fully_discretized())
            self.assertTrue(m.int2.is_fully_discretized())
            self.assertFalse(m.int3.is_fully_discretized())
            self.assertFalse(m.int4.is_fully_discretized())

            self.assertTrue(m.int1.type() is Integral)
            self.assertTrue(m.int2.type() is Integral)
            self.assertTrue(m.int3.type() is Integral)
            self.assertTrue(m.int4.type() is Integral)

            TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.x)

            self.assertTrue(m.int3.is_fully_discretized())
            self.assertTrue(m.int4.is_fully_discretized())

            self.assertTrue(m.int1.type() is Expression)
            self.assertTrue(m.int2.type() is Expression)
            self.assertTrue(m.int3.type() is Expression)
            self.assertTrue(m.int4.type() is Expression)

    # test DerivativeVar reclassification after discretization
    def test_reclassification_collocation(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.t, m.x)

        def _int1(m, t):
            return m.v[t]

        m.int1 = Integral(m.t, rule=_int1)

        def _int2(m, s, t):
            return m.v2[s, t]

        m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

        def _int3(m, t, x):
            return m.v3[t, x]

        m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

        def _int4(m, x):
            return m.int3[x]

        m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

        self.assertFalse(m.int1.is_fully_discretized())
        self.assertFalse(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)

        self.assertTrue(m.int1.is_fully_discretized())
        self.assertTrue(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.type() is Integral)
        self.assertTrue(m.int2.type() is Integral)
        self.assertTrue(m.int3.type() is Integral)
        self.assertTrue(m.int4.type() is Integral)

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.x)

        self.assertTrue(m.int3.is_fully_discretized())
        self.assertTrue(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.type() is Expression)
        self.assertTrue(m.int2.type() is Expression)
        self.assertTrue(m.int3.type() is Expression)
        self.assertTrue(m.int4.type() is Expression)


if __name__ == "__main__":
    unittest.main()
