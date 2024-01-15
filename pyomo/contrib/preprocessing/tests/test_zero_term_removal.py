"""Tests detection of zero terms."""
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Constraint, TransformationFactory, Var
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn


class TestRemoveZeroTerms(unittest.TestCase):
    """Tests removal of zero terms."""

    def test_zero_term_removal(self):
        """Test for removing zero terms from linear constraints."""
        m = ConcreteModel()
        m.v0 = Var()
        m.v1 = Var()
        m.v2 = Var()
        m.v3 = Var()
        m.c = Constraint(expr=m.v0 == m.v1 * m.v2 + m.v3)
        m.c2 = Constraint(expr=m.v1 * m.v2 + m.v3 <= m.v0)
        m.c3 = Constraint(expr=m.v0 <= m.v1 * m.v2 + m.v3)
        m.c4 = Constraint(expr=EXPR.inequality(1, m.v1 * m.v2 + m.v3, 3))
        m.v1.fix(0)

        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        m.v1.unfix()
        # Check that the term no longer exists
        self.assertFalse(
            any(id(m.v1) == id(v) for v in EXPR.identify_variables(m.c.body))
        )
        self.assertFalse(
            any(id(m.v1) == id(v) for v in EXPR.identify_variables(m.c2.body))
        )
        self.assertFalse(
            any(id(m.v1) == id(v) for v in EXPR.identify_variables(m.c3.body))
        )
        self.assertFalse(
            any(id(m.v1) == id(v) for v in EXPR.identify_variables(m.c4.body))
        )

    def test_trivial_constraints_skipped(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(expr=(m.x + m.y) * m.z >= 8)
        m.z.fix(0)
        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        m.z.unfix()
        # check constraint is unchanged
        self.assertEqual(m.c.lower, 8)
        self.assertIsNone(m.c.upper)
        repn = generate_standard_repn(m.c.body)
        self.assertTrue(repn.is_quadratic())
        self.assertEqual(repn.quadratic_coefs[0], 1)
        self.assertEqual(repn.quadratic_coefs[1], 1)
        self.assertIs(repn.quadratic_vars[0][0], m.x)
        self.assertIs(repn.quadratic_vars[0][1], m.z)
        self.assertIs(repn.quadratic_vars[1][0], m.y)
        self.assertIs(repn.quadratic_vars[1][1], m.z)
        self.assertEqual(repn.constant, 0)


if __name__ == '__main__':
    unittest.main()
