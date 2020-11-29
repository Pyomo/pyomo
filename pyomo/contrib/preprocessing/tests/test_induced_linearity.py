#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests the induced linearity module."""
import pyutilib.th as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
    _bilinear_expressions,
    detect_effectively_discrete_vars,
    determine_valid_values)
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (Binary, ConcreteModel, Constraint, ConstraintList,
                           Integers, RangeSet, SolverFactory,
                           TransformationFactory, Var, exp)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn

glpk_available = SolverFactory('glpk').available()


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
        m.ignore_inequality = Constraint(expr=m.x <= m.y + m.z)
        m.ignore_nonlinear = Constraint(expr=m.x ** 2 == m.y + m.z)
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

    @unittest.skipIf(not glpk_available, 'GLPK not available')
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
            m, detect_effectively_discrete_vars(m, 1E-6), Bunch(
                equality_tolerance=1E-6,
                pruning_solver='glpk'))
        valid_values = set([1, 2, 3, 4, 5])
        self.assertEqual(set(var_to_values_map[m.x]), valid_values)

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_induced_linearity_case2(self):
        m = ConcreteModel()
        m.x = Var([0], bounds=(-3, 8))
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.logical.add(expr=m.y[3] + m.y[4] == 1)
        m.logical.add(expr=m.y[2] + m.y[4] <= 1)
        m.b = Var(bounds=(-2, 7))
        m.c = Var()
        m.bilinear = Constraint(
            expr=(m.x[0] - 3) * (m.b + 2) - (m.c + 4) * m.b +
            exp(m.b ** 2) * m.x[0] <= m.c)
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        xfrmed_blk = m._induced_linearity_info.x0_b_bilinear
        self.assertSetEqual(
            set(xfrmed_blk.valid_values), set([1, 2, 3, 4, 5]))
        select_one_repn = generate_standard_repn(
            xfrmed_blk.select_one_value.body)
        self.assertEqual(
            ComponentSet(select_one_repn.linear_vars),
            ComponentSet(xfrmed_blk.x_active[i] for i in xfrmed_blk.valid_values))

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_bilinear_in_disjuncts(self):
        m = ConcreteModel()
        m.x = Var([0], bounds=(-3, 8))
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z)
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.logical.add(expr=m.y[3] + m.y[4] == 1)
        m.logical.add(expr=m.y[2] + m.y[4] <= 1)
        m.v = Var([1, 2])
        m.v[1].setlb(-2)
        m.v[1].setub(7)
        m.v[2].setlb(-4)
        m.v[2].setub(5)
        m.bilinear = Constraint(
            expr=(m.x[0] - 3) * (m.v[1] + 2) - (m.v[2] + 4) * m.v[1] +
            exp(m.v[1] ** 2) * m.x[0] <= m.v[2])
        m.disjctn = Disjunction(expr=[
            [m.x[0] * m.v[1] <= 4],
            [m.x[0] * m.v[2] >= 6]
        ])
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        self.assertEqual(
            m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1)
        self.assertEqual(
            m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 1)

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_induced_linear_in_disjunct(self):
        m = ConcreteModel()
        m.x = Var([0], bounds=(-3, 8))
        m.y = Var(RangeSet(2), domain=Binary)
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.v = Var([1])
        m.v[1].setlb(-2)
        m.v[1].setub(7)
        m.bilinear_outside = Constraint(
            expr=m.x[0] * m.v[1] >= 2)
        m.disjctn = Disjunction(expr=[
            [m.x[0] * m.v[1] == 3,
             2 * m.x[0] == m.y[1] + m.y[2]],
            [m.x[0] * m.v[1] == 4]
        ])
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        self.assertEqual(
            m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1)
        self.assertEqual(
            m.bilinear_outside.body.polynomial_degree(), 2)
        self.assertEqual(
            m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 2)


if __name__ == '__main__':
    unittest.main()
