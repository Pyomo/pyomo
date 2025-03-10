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

# -*- coding: utf-8 -*-
"""Tests deactivation of trivial constraints."""
import pyomo.common.unittest as unittest
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.environ import Constraint, ConcreteModel, TransformationFactory, Var


class TestTrivialConstraintDeactivator(unittest.TestCase):
    """Tests deactivation of trivial constraints."""

    def test_deactivate_trivial_constraints(self):
        """Test for deactivation of trivial constraints."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.c = Constraint(expr=m.v1 <= m.v2)
        m.c2 = Constraint(expr=m.v2 >= m.v3)
        m.c3 = Constraint(expr=m.v1 <= 5)
        m.v1.fix()

        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
        self.assertTrue(m.c.active)
        self.assertTrue(m.c2.active)
        self.assertFalse(m.c3.active)

    def test_deactivate_trivial_constraints_return_list(self):
        """Test for deactivation of trivial constraints."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.c = Constraint(expr=m.v1 <= m.v2)
        m.c2 = Constraint(expr=m.v2 >= m.v3)
        m.c3 = Constraint(expr=m.v1 <= 5)
        m.v1.fix()

        trivial = []
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            m, return_trivial=trivial
        )
        self.assertTrue(m.c.active)
        self.assertTrue(m.c2.active)
        self.assertFalse(m.c3.active)
        self.assertEqual(len(trivial), 1)
        self.assertIs(trivial[0], m.c3)

    def test_deactivate_trivial_constraints_revert(self):
        """Test for reversion of trivial constraint deactivation."""
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.c = Constraint(expr=m.v1 <= m.v2)
        m.c2 = Constraint(expr=m.v2 >= m.v3)
        m.c3 = Constraint(expr=m.v1 <= 5)
        m.v1.fix()

        xfrm = TransformationFactory('contrib.deactivate_trivial_constraints')
        xfrm.apply_to(m, tmp=True)
        self.assertTrue(m.c.active)
        self.assertTrue(m.c2.active)
        self.assertFalse(m.c3.active)

        xfrm.revert(m)
        self.assertTrue(m.c3.active)

    def test_trivial_constraints_lb_conflict(self):
        """Test for violated trivial constraint lower bound."""
        with self.assertRaisesRegex(
            InfeasibleConstraintException,
            "Trivial constraint c violates LB 2.0 ≤ BODY 1.",
        ):
            self._trivial_constraints_lb_conflict()

    def _trivial_constraints_lb_conflict(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.c = Constraint(expr=m.v1 >= 2)
        m.v1.fix()
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)

    def test_trivial_constraints_ub_conflict(self):
        """Test for violated trivial constraint upper bound."""
        with self.assertRaisesRegex(
            InfeasibleConstraintException,
            "Trivial constraint c violates BODY 1 ≤ UB 0.0.",
        ):
            self._trivial_constraints_ub_conflict()

    def _trivial_constraints_ub_conflict(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.c = Constraint(expr=m.v1 <= 0)
        m.v1.fix()
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)

    def test_trivial_constraint_due_to_0_coefficient(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.y.fix(0)
        m.c = Constraint(expr=m.x * m.y >= 0)

        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)

        self.assertFalse(m.c.active)

    def test_higher_degree_trivial_constraint(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(expr=(m.x**2 + m.y) * m.z >= -8)
        m.z.fix(0)
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)
        self.assertFalse(m.c.active)

    def test_trivial_linear_constraint_due_to_cancellation(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x - m.x <= 0)

        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(m)

        self.assertFalse(m.c.active)


if __name__ == '__main__':
    unittest.main()
