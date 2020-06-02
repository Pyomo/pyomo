#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


# -*- coding: UTF-8 -*-
"""Tests disjunct fixing."""
import pyutilib.th as unittest
from pyomo.environ import (Block,
                           Constraint, ConcreteModel, TransformationFactory,
                           NonNegativeReals)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error


class TestFixDisjuncts(unittest.TestCase):
    """Tests fixing of disjuncts."""

    def test_fix_disjunct(self):
        """Test for deactivation of trivial constraints."""
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d2.c = Constraint()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.d1.indicator_var.set_value(1)
        m.d2.indicator_var.set_value(0)

        TransformationFactory('gdp.fix_disjuncts').apply_to(m)
        self.assertTrue(m.d1.indicator_var.fixed)
        self.assertTrue(m.d1.active)
        self.assertTrue(m.d2.indicator_var.fixed)
        self.assertFalse(m.d2.active)
        self.assertEqual(m.d1.ctype, Block)
        self.assertEqual(m.d2.ctype, Block)
        self.assertTrue(m.d2.c.active)

    def test_xor_not_sum_to_1(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2], xor=True)
        m.d1.indicator_var.set_value(1)
        m.d2.indicator_var.set_value(1)
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def test_disjunction_not_sum_to_1(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2], xor=False)
        m.d1.indicator_var.set_value(0)
        m.d2.indicator_var.set_value(0)
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def test_disjunct_not_binary(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.d1.indicator_var.domain = NonNegativeReals
        m.d2.indicator_var.domain = NonNegativeReals
        m.d1.indicator_var.set_value(0.5)
        m.d2.indicator_var.set_value(0.5)
        with self.assertRaises(ValueError):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)


if __name__ == '__main__':
    unittest.main()
