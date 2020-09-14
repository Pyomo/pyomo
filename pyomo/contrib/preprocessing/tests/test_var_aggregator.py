"""Tests the variable aggregation module."""
import pyutilib.th as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
    _build_equality_set,
    _get_equality_linked_variables,
    max_if_not_None,
    min_if_not_None
)
from pyomo.environ import (ConcreteModel, Constraint, ConstraintList,
                           Objective, RangeSet, SolverFactory,
                           TransformationFactory, Var)


class TestVarAggregate(unittest.TestCase):
    """Tests variable aggregation."""

    def build_model(self):
        m = ConcreteModel()
        m.v1 = Var(initialize=1, bounds=(1, 8))
        m.v2 = Var(initialize=2, bounds=(0, 3))
        m.v3 = Var(initialize=3, bounds=(-7, 4))
        m.v4 = Var(initialize=4, bounds=(2, 6))
        m.c1 = Constraint(expr=m.v1 == m.v2)
        m.c2 = Constraint(expr=m.v2 == m.v3)
        m.c3 = Constraint(expr=m.v3 == m.v4)
        m.v2.fix()

        m.s = RangeSet(5)
        m.x = Var(m.s, initialize=5)
        m.c = Constraint(m.s)
        m.c.add(1, expr=m.x[1] == m.x[3])
        m.c.add(2, expr=m.x[2] == m.x[4])
        m.c.add(3, expr=m.x[2] == m.x[3])
        m.c.add(4, expr=m.x[1] == 1)
        m.c.add(5, expr=(2, m.x[5], 3))

        m.y = Var([1, 2], initialize={1: 3, 2: 4})
        m.c_too = Constraint(expr=m.y[1] == m.y[2])

        m.z1 = Var()
        m.z2 = Var()
        m.ignore_me = Constraint(expr=m.y[1] + m.z1 + m.z2 <= 0)
        m.ignore_me_too = Constraint(expr=m.y[1] * m.y[2] == 0)
        m.multiple = Constraint(expr=m.y[1] == 2 * m.y[2])

        return m

    def test_aggregate_fixed_var_diff_values(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.v = Var(m.s, bounds=(0, 5))
        m.c = ConstraintList()
        m.c.add(expr=m.v[1] == m.v[2])
        m.c.add(expr=m.v[2] == m.v[3])
        m.c.add(expr=m.v[1] == 1)
        m.c.add(expr=m.v[3] == 3)
        with self.assertRaises(ValueError):
            TransformationFactory('contrib.aggregate_vars').apply_to(m)

    def test_fixed_var_out_of_bounds_lb(self):
        m = ConcreteModel()
        m.s = RangeSet(2)
        m.v = Var(m.s, bounds=(0, 5))
        m.c = ConstraintList()
        m.c.add(expr=m.v[1] == m.v[2])
        m.c.add(expr=m.v[1] == -1)
        with self.assertRaises(ValueError):
            TransformationFactory('contrib.aggregate_vars').apply_to(m)

    def test_fixed_var_out_of_bounds_ub(self):
        m = ConcreteModel()
        m.s = RangeSet(2)
        m.v = Var(m.s, bounds=(0, 5))
        m.c = ConstraintList()
        m.c.add(expr=m.v[1] == m.v[2])
        m.c.add(expr=m.v[1] == 6)
        with self.assertRaises(ValueError):
            TransformationFactory('contrib.aggregate_vars').apply_to(m)

    def test_do_not_tranform_deactivated_constraints(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.c1 = Constraint(expr=m.x == m.y)
        m.c2 = Constraint(expr=(2, m.x, 3))
        m.c3 = Constraint(expr=m.x == 0)
        m.c3.deactivate()
        TransformationFactory('contrib.aggregate_vars').apply_to(m)
        self.assertIs(m.c2.body, m._var_aggregator_info.z[1])
        self.assertIs(m.c3.body, m.x)

    def test_equality_linked_variables(self):
        """Test for equality-linked variable detection."""
        m = self.build_model()
        self.assertEqual(_get_equality_linked_variables(m.c1), ())
        self.assertEqual(_get_equality_linked_variables(m.c2), ())
        c3 = _get_equality_linked_variables(m.c3)
        self.assertIn(m.v3, c3)
        self.assertIn(m.v4, c3)
        self.assertEqual(len(c3), 2)
        self.assertEqual(_get_equality_linked_variables(m.ignore_me), ())
        self.assertEqual(_get_equality_linked_variables(m.ignore_me_too), ())
        self.assertEqual(_get_equality_linked_variables(m.multiple), ())

    def test_equality_set(self):
        """Test for equality set map generation."""
        m = self.build_model()
        eq_var_map = _build_equality_set(m)
        self.assertIsNone(eq_var_map.get(m.z1, None))
        self.assertIsNone(eq_var_map.get(m.v1, None))
        self.assertIsNone(eq_var_map.get(m.v2, None))
        self.assertEqual(eq_var_map[m.v3], ComponentSet([m.v3, m.v4]))
        self.assertEqual(eq_var_map[m.v4], ComponentSet([m.v3, m.v4]))
        self.assertEqual(
            eq_var_map[m.x[1]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEqual(
            eq_var_map[m.x[2]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEqual(
            eq_var_map[m.x[3]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEqual(
            eq_var_map[m.x[4]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEqual(eq_var_map[m.y[1]], ComponentSet([m.y[1], m.y[2]]))
        self.assertEqual(eq_var_map[m.y[2]], ComponentSet([m.y[1], m.y[2]]))

    def test_var_aggregate(self):
        """Test for transitivity in a variable equality set."""
        m = self.build_model()

        TransformationFactory('contrib.aggregate_vars').apply_to(m)
        z_to_vars = m._var_aggregator_info.z_to_vars
        var_to_z = m._var_aggregator_info.var_to_z
        z = m._var_aggregator_info.z
        self.assertEqual(
            z_to_vars[z[1]], ComponentSet([m.v3, m.v4]))
        self.assertEqual(
            z_to_vars[z[2]],
            ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
        self.assertEqual(
            z_to_vars[z[3]],
            ComponentSet([m.y[1], m.y[2]]))
        self.assertIs(var_to_z[m.v3], z[1])
        self.assertIs(var_to_z[m.v4], z[1])
        self.assertIs(var_to_z[m.x[1]], z[2])
        self.assertIs(var_to_z[m.x[2]], z[2])
        self.assertIs(var_to_z[m.x[3]], z[2])
        self.assertIs(var_to_z[m.x[4]], z[2])
        self.assertIs(var_to_z[m.y[1]], z[3])
        self.assertIs(var_to_z[m.y[2]], z[3])

        self.assertEqual(z[1].value, 2)
        self.assertEqual(z[1].lb, 2)
        self.assertEqual(z[1].ub, 4)

        self.assertEqual(z[3].value, 3.5)

    def test_min_if_not_None(self):
        self.assertEqual(min_if_not_None([1, 2, None, 3, None]), 1)
        self.assertEqual(min_if_not_None([None, None, None]), None)
        self.assertEqual(min_if_not_None([]), None)
        self.assertEqual(min_if_not_None([None, 3, -1, 2]), -1)
        self.assertEqual(min_if_not_None([0]), 0)
        self.assertEqual(min_if_not_None([0, None]), 0)

    def test_max_if_not_None(self):
        self.assertEqual(max_if_not_None([1, 2, None, 3, None]), 3)
        self.assertEqual(max_if_not_None([None, None, None]), None)
        self.assertEqual(max_if_not_None([]), None)
        self.assertEqual(max_if_not_None([None, 3, -1, 2]), 3)
        self.assertEqual(max_if_not_None([0]), 0)
        self.assertEqual(max_if_not_None([0, None]), 0)

    @unittest.skipIf(not SolverFactory('glpk').available(),
                     "GLPK solver is not available.")
    def test_var_update(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(bounds=(0, 1))
        m.c = Constraint(expr=m.x == m.y)
        m.o = Objective(expr=m.x)
        TransformationFactory('contrib.aggregate_vars').apply_to(m)
        SolverFactory('glpk').solve(m)
        z = m._var_aggregator_info.z
        self.assertEqual(z[1].value, 0)
        self.assertEqual(m.x.value, None)
        self.assertEqual(m.y.value, None)
        TransformationFactory('contrib.aggregate_vars').update_variables(m)
        self.assertEqual(z[1].value, 0)
        self.assertEqual(m.x.value, 0)
        self.assertEqual(m.y.value, 0)


if __name__ == '__main__':
    unittest.main()
