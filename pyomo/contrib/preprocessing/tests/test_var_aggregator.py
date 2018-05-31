"""Tests the variable aggregation module."""
import pyutilib.th as unittest
from pyomo.contrib.preprocessing.plugins.var_aggregator import (max_if_not_None,
                                                                min_if_not_None,
                                                                _get_equality_linked_variables,
                                                                _build_equality_set)
from pyomo.core.kernel import ComponentSet
from pyomo.environ import (ConcreteModel, Constraint, RangeSet,
                           TransformationFactory, Var)


class TestVarAggregate(unittest.TestCase):
    """Tests variable aggregation."""

    def build_model(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1)
        m.v2 = Var(initialize=2)
        m.v3 = Var(initialize=3)
        m.v4 = Var(initialize=4)
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v2.fix()

        m.s = RangeSet(4)
        m.x = Var(m.s, initialize=5)
        m.c = Constraint(m.s)
        m.c.add(1, expr=m.x[1] == m.x[3])
        m.c.add(2, expr=m.x[2] == m.x[4])
        m.c.add(3, expr=m.x[2] == m.x[3])
        m.c.add(4, expr=m.x[1] == 1)

        m.y = Var([1, 2], initialize=3)
        m.c_too = Constraint(expr=m.y[1] == m.y[2])

        m.z1 = Var()
        m.z2 = Var()
        m.ignore_me = Constraint(expr=m.y[1] + m.z1 + m.z2 <= 0)

        return m

    def test_equality_linked_variables(self):
        """Test for equality-linked variable detection."""
        m = self.build_model()
        self.assertEquals(_get_equality_linked_variables(m.c1), ())
        self.assertEquals(_get_equality_linked_variables(m.c2), ())
        c3 = _get_equality_linked_variables(m.c3)
        self.assertIn(m.v3, c3)
        self.assertIn(m.v4, c3)
        self.assertEquals(len(c3), 2)
        self.assertEquals(_get_equality_linked_variables(m.ignore_me), ())

    def test_equality_set(self):
        """Test for equality set map generation."""
        m = self.build_model()
        eq_var_map = _build_equality_set(m)
        self.assertIsNone(eq_var_map.get(m.z1, None))
        self.assertIsNone(eq_var_map.get(m.v1, None))
        self.assertIsNone(eq_var_map.get(m.v2, None))
        self.assertEquals(eq_var_map[m.v3], ComponentSet([m.v3, m.v4]))
        self.assertEquals(eq_var_map[m.v4], ComponentSet([m.v3, m.v4]))
        self.assertEquals(
            eq_var_map[m.x[1]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEquals(
            eq_var_map[m.x[2]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEquals(
            eq_var_map[m.x[3]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEquals(
            eq_var_map[m.x[4]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEquals(eq_var_map[m.y[1]], ComponentSet([m.y[1], m.y[2]]))
        self.assertEquals(eq_var_map[m.y[2]], ComponentSet([m.y[1], m.y[2]]))

    def test_var_aggregate(self):
        """Test for transitivity in a variable equality set."""
        m = self.build_model()

        TransformationFactory('contrib.aggregate_vars').apply_to(m)
        z_to_vars = m._var_aggregator_info.z_to_vars
        var_to_z = m._var_aggregator_info.var_to_z
        self.assertEquals(
            z_to_vars[m._var_aggregator_info.z[1]],
            ComponentSet([m.v3, m.v4]))
        self.assertEquals(
            z_to_vars[m._var_aggregator_info.z[2]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEquals(
            z_to_vars[m._var_aggregator_info.z[3]],
            ComponentSet([m.y[1], m.y[2]]))
        self.assertIs(var_to_z[m.v3], m._var_aggregator_info.z[1])
        self.assertIs(var_to_z[m.v4], m._var_aggregator_info.z[1])
        self.assertIs(var_to_z[m.x[1]], m._var_aggregator_info.z[2])
        self.assertIs(var_to_z[m.x[2]], m._var_aggregator_info.z[2])
        self.assertIs(var_to_z[m.x[3]], m._var_aggregator_info.z[2])
        self.assertIs(var_to_z[m.x[4]], m._var_aggregator_info.z[2])
        self.assertIs(var_to_z[m.y[1]], m._var_aggregator_info.z[3])
        self.assertIs(var_to_z[m.y[2]], m._var_aggregator_info.z[3])

    def test_min_if_not_None(self):
        self.assertEquals(min_if_not_None([1, 2, None, 3, None]), 1)
        self.assertEquals(min_if_not_None([None, None, None]), None)
        self.assertEquals(min_if_not_None([]), None)
        self.assertEquals(min_if_not_None([None, 3, -1, 2]), -1)
        self.assertEquals(min_if_not_None([0]), 0)
        self.assertEquals(min_if_not_None([0, None]), 0)

    def test_max_if_not_None(self):
        self.assertEquals(max_if_not_None([1, 2, None, 3, None]), 3)
        self.assertEquals(max_if_not_None([None, None, None]), None)
        self.assertEquals(max_if_not_None([]), None)
        self.assertEquals(max_if_not_None([None, 3, -1, 2]), 3)
        self.assertEquals(max_if_not_None([0]), 0)
        self.assertEquals(max_if_not_None([0, None]), 0)


if __name__ == '__main__':
    unittest.main()
