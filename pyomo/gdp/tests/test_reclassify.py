# -*- coding: UTF-8 -*-
"""Tests disjunct reclassifier transformation."""
import pyutilib.th as unittest
from pyomo.core import (Block, ConcreteModel, TransformationFactory)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier
from pyomo.gdp.plugins import bigm


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
        self.assertIs(m.d1.type(), Block)
        self.assertIs(m.d1.sub1.type(), Block)
        self.assertIs(m.d1.sub2.type(), Block)

    def test_deactivated_parent_block(self):
        m = ConcreteModel()
        m.d1 = Block()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        m.d1.deactivate()
        TransformationFactory('gdp.reclassify').apply_to(m)
        self.assertIs(m.d1.type(), Block)
        self.assertIs(m.d1.sub1.type(), Block)
        self.assertIs(m.d1.sub2.type(), Block)

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
        self.assertIs(m.d1.type(), Block)
        self.assertIs(m.d1.sub1.type(), Block)
        self.assertIs(m.d1.sub2.type(), Block)

    def test_active_parent_block(self):
        m = ConcreteModel()
        m.d1 = Block()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.reclassify').apply_to(m)


if __name__ == '__main__':
    unittest.main()
