# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Tests the induced linearity module."""

import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
    _bilinear_expressions,
    _process_bilinear_constraints,
    detect_effectively_discrete_vars,
    determine_valid_values,
)
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
    Binary,
    Block,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Integers,
    NegativeReals,
    PositiveIntegers,
    RangeSet,
    SolverFactory,
    TransformationFactory,
    Var,
    exp,
    value,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn

glpk_available = SolverFactory('glpk').available()


def _assign_increment(block, chosen, total):
    for val in block.valid_values:
        block.x_active[val].set_value(int(val == chosen))
        # Check the complete resulting assignment below instead of emitting
        # Pyomo domain warnings while constructing deliberately invalid points.
        block.v_increment[val].set_value(
            total if val == chosen else 0, skip_validation=True
        )


def _assignment_is_feasible(model):
    for var in model.component_data_objects(Var, active=True):
        val = value(var)
        if (
            val not in var.domain
            or (var.has_lb() and val < var.lb)
            or (var.has_ub() and val > var.ub)
        ):
            return False
    for con in model.component_data_objects(Constraint, active=True):
        val = value(con.body)
        if (con.has_lb() and val < value(con.lower)) or (
            con.has_ub() and val > value(con.upper)
        ):
            return False
    return True


class TestInducedLinearity(unittest.TestCase):
    """Tests induced linearity."""

    def test_detect_bilinear_vars(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(
            expr=(m.x - 3) * (m.y + 2)
            - (m.z + 4) * m.y
            + (m.x + 2) ** 2
            + exp(m.y**2) * m.x
            <= m.z
        )
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
        m.ignore_nonlinear = Constraint(expr=m.x**2 == m.y + m.z)
        m.a = Var()
        m.b = Var(domain=Binary)
        m.c = Var(domain=Integers)
        m.disj = Disjunct()
        m.disj.constr = Constraint(expr=m.a == m.b + m.c)
        effectively_discrete = detect_effectively_discrete_vars(m, 1e-6)
        self.assertEqual(len(effectively_discrete), 1)
        self.assertEqual(effectively_discrete[m.x], [m.constr])
        effectively_discrete = detect_effectively_discrete_vars(m.disj, 1e-6)
        self.assertEqual(len(effectively_discrete), 1)
        self.assertEqual(effectively_discrete[m.a], [m.disj.constr])

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_determine_valid_values(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z
        )
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.logical.add(expr=m.y[3] + m.y[4] == 1)
        m.logical.add(expr=m.y[2] + m.y[4] <= 1)
        var_to_values_map = determine_valid_values(
            m,
            detect_effectively_discrete_vars(m, 1e-6),
            Bunch(equality_tolerance=1e-6, pruning_solver='glpk'),
        )
        valid_values = set([1, 2, 3, 4, 5])
        self.assertEqual(set(var_to_values_map[m.x]), valid_values)

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_induced_linearity_case2(self):
        m = ConcreteModel()
        m.x = Var([0], bounds=(-3, 8))
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z
        )
        m.logical = ConstraintList()
        m.logical.add(expr=m.y[1] + m.y[2] == 1)
        m.logical.add(expr=m.y[3] + m.y[4] == 1)
        m.logical.add(expr=m.y[2] + m.y[4] <= 1)
        m.b = Var(bounds=(-2, 7))
        m.c = Var()
        m.bilinear = Constraint(
            expr=(m.x[0] - 3) * (m.b + 2) - (m.c + 4) * m.b + exp(m.b**2) * m.x[0]
            <= m.c
        )
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        xfrmed_blk = m._induced_linearity_info.x0_b_bilinear
        self.assertSetEqual(set(xfrmed_blk.valid_values), set([1, 2, 3, 4, 5]))
        select_one_repn = generate_standard_repn(xfrmed_blk.select_one_value.body)
        self.assertEqual(
            ComponentSet(select_one_repn.linear_vars),
            ComponentSet(xfrmed_blk.x_active[i] for i in xfrmed_blk.valid_values),
        )

    @unittest.skipIf(not glpk_available, 'GLPK not available')
    def test_bilinear_in_disjuncts(self):
        m = ConcreteModel()
        m.x = Var([0], bounds=(-3, 8))
        m.y = Var(RangeSet(4), domain=Binary)
        m.z = Var(domain=Integers, bounds=(-1, 2))
        m.constr = Constraint(
            expr=m.x[0] == m.y[1] + 2 * m.y[2] + m.y[3] + 2 * m.y[4] + m.z
        )
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
            expr=(m.x[0] - 3) * (m.v[1] + 2)
            - (m.v[2] + 4) * m.v[1]
            + exp(m.v[1] ** 2) * m.x[0]
            <= m.v[2]
        )
        m.disjctn = Disjunction(expr=[[m.x[0] * m.v[1] <= 4], [m.x[0] * m.v[2] >= 6]])
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        self.assertEqual(
            m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1
        )
        self.assertEqual(
            m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 1
        )

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
        m.bilinear_outside = Constraint(expr=m.x[0] * m.v[1] >= 2)
        m.disjctn = Disjunction(
            expr=[
                [m.x[0] * m.v[1] == 3, 2 * m.x[0] == m.y[1] + m.y[2]],
                [m.x[0] * m.v[1] == 4],
            ]
        )
        TransformationFactory('contrib.induced_linearity').apply_to(m)
        self.assertEqual(
            m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1
        )
        self.assertEqual(m.bilinear_outside.body.polynomial_degree(), 2)
        self.assertEqual(
            m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 2
        )

    def test_reformulation_rejects_inconsistent_selected_value(self):
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.bit = Var(domain=Binary, initialize=0)
        m.y = Var(bounds=(-2, 4), initialize=2)
        m.defx = Constraint(expr=m.x == 1 + m.bit)
        m.product = Constraint(expr=m.x * m.y == 4)
        self.assertFalse(_assignment_is_feasible(m))

        m._induced_linearity_info = Block()
        _process_bilinear_constraints(m, m.x, m.y, {1, 2}, [m.product])
        blk = m._induced_linearity_info.x_y_bilinear
        _assign_increment(blk, chosen=2, total=2)

        # The rewritten product row alone matches 4, but selecting 2
        # while the original variable is 1 must be infeasible.
        self.assertEqual(value(m.product.body), 4)
        self.assertFalse(_assignment_is_feasible(m))

        m.x.set_value(2)
        m.bit.set_value(1)
        self.assertTrue(_assignment_is_feasible(m))

    def test_inactive_copy_allows_zero_with_strictly_signed_partner(self):
        for domain, bounds, yval in (
            (PositiveIntegers, (2, 5), 3),
            (NegativeReals, (-5, -2), -3),
        ):
            with self.subTest(domain=domain):
                m = ConcreteModel()
                m.x = Var(initialize=1)
                m.bit = Var(domain=Binary, initialize=0)
                m.y = Var(domain=domain, bounds=bounds, initialize=yval)
                m.defx = Constraint(expr=m.x == 1 + m.bit)
                m.product = Constraint(expr=m.x * m.y == yval)
                self.assertTrue(_assignment_is_feasible(m))

                m._induced_linearity_info = Block()
                _process_bilinear_constraints(m, m.x, m.y, {1, 2}, [m.product])
                blk = m._induced_linearity_info.x_y_bilinear
                _assign_increment(blk, chosen=1, total=yval)
                self.assertTrue(_assignment_is_feasible(m))
                self.assertTrue(blk.v_defn.equality)
                self.assertIs(m.y.domain, domain)

    def test_shared_effectively_discrete_variable_cannot_have_two_values(self):
        m = ConcreteModel()
        m.x = Var(initialize=-1)
        m.bit = Var(domain=Binary, initialize=0)
        m.y = Var(bounds=(-3, 3), initialize=2)
        m.z = Var(bounds=(-3, 3), initialize=-1)
        m.defx = Constraint(expr=m.x == -1 + 3 * m.bit)
        m.xy = Constraint(expr=m.x * m.y == -2)
        m.xz = Constraint(expr=m.x * m.z == -2)
        m._induced_linearity_info = Block()
        _process_bilinear_constraints(m, m.x, m.y, {-1, 2}, [m.xy])
        _process_bilinear_constraints(m, m.x, m.z, {-1, 2}, [m.xz])
        blocks = m._induced_linearity_info
        _assign_increment(blocks.x_y_bilinear, chosen=-1, total=2)
        _assign_increment(blocks.x_z_bilinear, chosen=2, total=-1)
        self.assertFalse(_assignment_is_feasible(m))

        m.z.set_value(2)
        _assign_increment(blocks.x_z_bilinear, chosen=-1, total=2)
        self.assertTrue(_assignment_is_feasible(m))

    def test_reformulated_self_product_keeps_squared_value(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-2, 3), initialize=-2)
        m.bit = Var(domain=Binary, initialize=0)
        m.defx = Constraint(expr=m.x == -2 + 5 * m.bit)
        m.square = Constraint(expr=m.x**2 == 4)
        m._induced_linearity_info = Block()
        _process_bilinear_constraints(m, m.x, m.x, {-2, 3}, [m.square])
        _assign_increment(m._induced_linearity_info.x_x_bilinear, chosen=-2, total=-2)
        self.assertTrue(_assignment_is_feasible(m))


if __name__ == '__main__':
    unittest.main()
