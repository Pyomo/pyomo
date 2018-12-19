"""Tests detection of zero terms."""
import pyutilib.th as unittest
from pyomo.environ import (ConcreteModel, Constraint, TransformationFactory,
                           Var)
from pyomo.core.expr import current as EXPR


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
        self.assertFalse(any(id(m.v1) == id(v)
                             for v in EXPR.identify_variables(m.c.body)))
        self.assertFalse(any(id(m.v1) == id(v)
                             for v in EXPR.identify_variables(m.c2.body)))
        self.assertFalse(any(id(m.v1) == id(v)
                             for v in EXPR.identify_variables(m.c3.body)))
        self.assertFalse(any(id(m.v1) == id(v)
                             for v in EXPR.identify_variables(m.c4.body)))


if __name__ == '__main__':
    unittest.main()
