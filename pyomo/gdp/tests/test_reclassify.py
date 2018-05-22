# -*- coding: UTF-8 -*-
"""Tests disjunct reclassifier transformation."""
import pyutilib.th as unittest
from pyomo.core import (Block, ConcreteModel, TransformationFactory)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.gdp_var_mover import HACK_GDP_Disjunct_Reclassifier


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
        # This should not raise an error

    def test_deactivated_parent_block(self):
        m = ConcreteModel()
        m.d1 = Block()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        m.d1.deactivate()
        TransformationFactory('gdp.reclassify').apply_to(m)
        # This should not raise an error

    def test_active_parent_disjunct(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d1.sub1 = Disjunct()
        m.d1.sub2 = Disjunct()
        m.d1.disj = Disjunction(expr=[m.d1.sub1, m.d1.sub2])
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.reclassify').apply_to(m)

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
