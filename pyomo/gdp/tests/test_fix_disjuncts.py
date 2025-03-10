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


# -*- coding: UTF-8 -*-
"""Tests disjunct fixing."""
import pyomo.common.unittest as unittest
from pyomo.environ import (
    Block,
    Constraint,
    ConcreteModel,
    TransformationFactory,
    NonNegativeReals,
    BooleanVar,
    LogicalConstraint,
    SolverFactory,
    Objective,
    value,
    Var,
    implies,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers

solvers = check_available_solvers('gurobi')


class TestFixDisjuncts(unittest.TestCase):
    """Tests fixing of disjuncts."""

    def test_fix_disjunct(self):
        """Test for deactivation of trivial constraints."""
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d2.c = Constraint()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(False)

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
        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(True)
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def test_disjunction_not_sum_to_1(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2], xor=False)
        m.d1.indicator_var.set_value(False)
        m.d2.indicator_var.set_value(False)
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def test_disjunct_not_binary(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.d1.binary_indicator_var.domain = NonNegativeReals
        m.d2.binary_indicator_var.domain = NonNegativeReals
        m.d1.binary_indicator_var.set_value(0.5)
        m.d2.binary_indicator_var.set_value(0.5)
        with self.assertRaisesRegex(
            GDP_Error,
            "The value of the indicator_var of "
            "Disjunct 'd1' is None. All indicator_vars "
            "must have values before calling "
            "'fix_disjuncts'.",
        ):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    def test_disjuncts_partially_fixed(self):
        m = ConcreteModel()
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.another1 = Disjunct()
        m.another2 = Disjunct()
        m.another = Disjunction(expr=[m.another1, m.another2])

        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(False)

        with self.assertRaisesRegex(
            GDP_Error,
            "The value of the indicator_var of "
            "Disjunct 'another1' is None. All indicator_vars "
            "must have values before calling "
            "'fix_disjuncts'.",
        ):
            TransformationFactory('gdp.fix_disjuncts').apply_to(m)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_logical_constraints_transformed(self):
        """It is expected that the result of this transformation is a MI(N)LP,
        so check that LogicalConstraints are handled correctly"""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.d2.c = Constraint()
        m.d = Disjunction(expr=[m.d1, m.d2])
        m.another = Disjunction(expr=[[m.x == 3], [m.x == 0]])
        m.Y = BooleanVar()
        m.global_logical = LogicalConstraint(expr=m.Y.xor(m.d1.indicator_var))
        m.d1.logical = LogicalConstraint(
            expr=implies(~m.Y, m.another.disjuncts[0].indicator_var)
        )
        m.obj = Objective(expr=m.x)

        m.d1.indicator_var.set_value(True)
        m.d2.indicator_var.set_value(False)
        m.another.disjuncts[0].indicator_var.set_value(True)
        m.another.disjuncts[1].indicator_var.set_value(False)

        TransformationFactory('gdp.fix_disjuncts').apply_to(m)

        # Make sure there are no active LogicalConstraints
        self.assertEqual(
            len(
                list(
                    m.component_data_objects(
                        LogicalConstraint, active=True, descend_into=(Block, Disjunct)
                    )
                )
            ),
            0,
        )
        # See that it solves as expected
        SolverFactory('gurobi').solve(m)
        self.assertTrue(value(m.d1.indicator_var))
        self.assertFalse(value(m.d2.indicator_var))
        self.assertTrue(value(m.another.disjuncts[0].indicator_var))
        self.assertFalse(value(m.another.disjuncts[1].indicator_var))
        self.assertEqual(value(m.Y.get_associated_binary()), 0)
        self.assertEqual(value(m.x), 3)

    def test_reclassify_deactivated_disjuncts(self):
        m = ConcreteModel()
        m.d = Disjunct([1, 2, 3])
        m.disjunction = Disjunction(expr=[m.d[1], m.d[2], m.d[3]])
        m.d[1].deactivate()
        m.d[2].indicator_var = True
        m.d[3].indicator_var = False

        TransformationFactory('gdp.fix_disjuncts').apply_to(m)

        self.assertTrue(m.d[1].indicator_var.fixed)
        self.assertFalse(value(m.d[1].indicator_var))
        self.assertFalse(m.d[1].active)
        self.assertEqual(m.d[1].ctype, Block)

        self.assertTrue(m.d[2].indicator_var.fixed)
        self.assertTrue(value(m.d[2].indicator_var))
        self.assertTrue(m.d[2].active)
        self.assertTrue(m.d[3].indicator_var.fixed)
        self.assertFalse(value(m.d[3].indicator_var))
        self.assertFalse(m.d[3].active)
        self.assertEqual(m.d[1].ctype, Block)
        self.assertEqual(m.d[2].ctype, Block)


if __name__ == '__main__':
    unittest.main()
