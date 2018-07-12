"""Tests the induced linearity module."""
import pyutilib.th as unittest
from pyomo.core.kernel import ComponentSet
from pyomo.environ import (ConcreteModel, Constraint, ConstraintList,
                           Objective, RangeSet, SolverFactory,
                           TransformationFactory, Var, exp, Binary, Integers, summation, NonNegativeReals)
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
    _bilinear_expressions, detect_effectively_discrete_vars, _reformulate_case_1, determine_valid_values)
from pyomo.gdp import Disjunct


class TestInducedLinearity(unittest.TestCase):
    """Tests induced linearity."""

    def test_detect_bilinear_vars(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(
            expr=(m.x - 3) * (m.y + 2) - (m.z + 4) * m.y + (m.x + 2) ** 2
            + exp(m.y ** 2) * m.x <= m.z)
        m.c2 = Constraint(expr=m.x * m.y == 3)
        bilinear_map = _bilinear_expressions(m)
        self.assertEqual(len(bilinear_map), 3)
        self.assertEqual(len(bilinear_map[m.x]), 2)
        self.assertEqual(len(bilinear_map[m.y]), 2)
        self.assertEqual(len(bilinear_map[m.z]), 1)
        self.assertEqual(bilinear_map[m.x][m.x], ComponentSet([m.c]))
        self.assertEqual(bilinear_map[m.x][m.y], ComponentSet([m.c, m.c2]))
        self.assertEqual(bilinear_map[m.y][m.x], ComponentSet([m.c, m.c2]))
        self.assertEqual(bilinear_map[m.y][m.z], ComponentSet([m.c]))
        self.assertEqual(bilinear_map[m.z][m.y], ComponentSet([m.c]))

    def test_detect_effectively_discrete_vars(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(domain=Binary)
        m.z = Var(domain=Integers)
        m.constr = Constraint(expr=m.x == m.y + m.z)
        m.a = Var()
        m.b = Var(domain=Binary)
        m.c = Var(domain=Integers)
        m.disj = Disjunct()
        m.disj.constr = Constraint(expr=m.a == m.b + m.c)
        effectively_discrete = detect_effectively_discrete_vars(m, 1E-6)
        self.assertEqual(len(effectively_discrete), 1)
        self.assertEqual(effectively_discrete[m.x], [m.constr])
        effectively_discrete = detect_effectively_discrete_vars(m.disj, 1E-6)
        self.assertEqual(len(effectively_discrete), 1)
        self.assertEqual(effectively_discrete[m.a], [m.disj.constr])

    def test_determine_valid_values(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.logical.add(expr=m.y[3] + m.y[4] == 1)
        m.logical.add(expr=m.y[2] + m.y[4] <= 1)
        var_to_values_map = determine_valid_values(
            m, detect_effectively_discrete_vars(m, 1E-6))
        print(var_to_values_map)
        # valid_values = set([1, 2, 3, 4, 5, 6])
        # self.assertEqual(set(var_to_values_map[m.x]), valid_values)

    # def test_induced_linearity_case1(self):
    #     m = ConcreteModel()
    #     m.x = Var(domain=NonNegativeReals)
    #     m.v = Var()
    #     m.y = Var([1, 2, 3], domain=Binary)
    #     m.c = Constraint(expr=m.v * m.x <= 0)
    #     m.c2 = Constraint(expr=m.x == m.y[1] + 2 * m.y[2] + 3 * m.y[3])
    #     m.c3 = Constraint(expr=summation(m.y) == 1)
    #     with self.assertRaises(NotImplementedError):
    #         _reformulate_case_1(m.x, m.v, m.c2, m.c)
    #
    # def test_induced_linearity_case2(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
