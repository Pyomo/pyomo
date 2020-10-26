# -*- coding: UTF-8 -*-
"""Tests disjunct reclassifier transformation."""
import pyutilib.th as unittest
from pyomo.core import (Block, ConcreteModel, TransformationFactory, RangeSet, Constraint, Var)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error


class TestDisjunctReclassify(unittest.TestCase):
    """Tests reclassification of disjuncts."""

    def test_deactivated_parent_disjunct(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        m.d1.deactivate()
        TransformationFactory('gdp.reclassify').apply_to(m)
        self.assertIs(m.d1.ctype, Block)
        self.assertIs(m.d1.sub1.ctype, Block)
        self.assertIs(m.d1.sub2.ctype, Block)

    def test_deactivated_parent_block(self):
        m = ConcreteModel()
        m.d1 = Block()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        m.d1.deactivate()
        TransformationFactory('gdp.reclassify').apply_to(m)
        self.assertIs(m.d1.ctype, Block)
        self.assertIs(m.d1.sub1.ctype, Block)
        self.assertIs(m.d1.sub2.ctype, Block)

    def test_active_parent_disjunct(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.reclassify').apply_to(m)

    def test_active_parent_disjunct_target(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        TransformationFactory('gdp.bigm').apply_to(m, targets=m.d1.disj)
        m.d1.indicator_var.fix(1)
        TransformationFactory('gdp.reclassify').apply_to(m)
        self.assertIs(m.d1.ctype, Block)
        self.assertIs(m.d1.sub1.ctype, Block)
        self.assertIs(m.d1.sub2.ctype, Block)

    def test_active_parent_block(self):
        m = ConcreteModel()
        m.d1 = Block()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.reclassify').apply_to(m)

    def test_deactivate_nested_disjunction(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.d1 = Disjunct()
        m.d1.d2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.d1, m.d1.d2])
        m.d2 = Disjunct()
        m.disj = Disjunction(expr=[m.d1, m.d2])
        m.d1.deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)
        # for disj in m.component_data_objects(Disjunction, active=True):
        #     print(disj.name)
        # There should be no active Disjunction objects.
        self.assertIsNone(
            next(m.component_data_objects(Disjunction, active=True), None))

    def test_do_not_reactivate_disjuncts_with_abandon(self):
        m = ConcreteModel()
        m.x = Var()
        m.s = RangeSet(4)
        m.d = Disjunct(m.s)
        m.d[2].bad_constraint_should_not_be_active = Constraint(expr=m.x >= 1)
        m.disj1 = Disjunction(expr=[m.d[1], m.d[2]])
        m.disj2 = Disjunction(expr=[m.d[3], m.d[4]])
        m.d[1].indicator_var.fix(1)
        m.d[2].deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)
        self.assertFalse(m.d[2].active)


if __name__ == '__main__':
    unittest.main()
