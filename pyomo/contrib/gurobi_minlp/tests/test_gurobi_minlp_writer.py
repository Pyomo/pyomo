#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.gurobi_minlp.repn.gurobi_direct_minlp import (
    GurobiMINLPVisitor,
)
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Integers,
    log,
    NonNegativeIntegers,
    NonNegativeReals,
    NonPositiveIntegers,
    NonPositiveReals,
    Param,
    Reals,
    sqrt,
    Var,
)

# TODO: Need to check major version >=12 too.
gurobipy, gurobipy_available = attempt_import('gurobipy')

if gurobipy_available:
    from gurobipy import GRB

## DEBUG
from pytest import set_trace


class CommonTest(unittest.TestCase):
    def get_model(self):
        m = ConcreteModel()
        m.x1 = Var(domain=NonNegativeReals)
        m.x2 = Var(domain=Reals)
        m.x3 = Var(domain=NonPositiveReals)
        m.y1 = Var(domain=Integers)
        m.y2 = Var(domain=NonNegativeIntegers)
        m.y3 = Var(domain=NonPositiveIntegers)
        m.z1 = Var(domain=Binary)

        return m

    def get_visitor(self):
        grb_model = gurobipy.Model()
        return GurobiMINLPVisitor(grb_model, symbolic_solver_labels=True)


@unittest.skipUnless(gurobipy_available, "gurobipy is not available")
class TestGurobiMINLPWalker(CommonTest):
    def test_var_domains(self):
        m = self.get_model()
        e = m.x1 + m.x2 + m.x3 + m.y1 + m.y2 + m.y3 + m.z1
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]
        x3 = visitor.var_map[id(m.x3)]
        y1 = visitor.var_map[id(m.y1)]
        y2 = visitor.var_map[id(m.y2)]
        y3 = visitor.var_map[id(m.y3)]
        z1 = visitor.var_map[id(m.z1)]

        self.assertEqual(x1.lb, 0)
        self.assertEqual(x1.ub, float('inf'))
        self.assertEqual(x1.vtype, GRB.CONTINUOUS)

        self.assertEqual(x2.lb, -float('inf'))
        self.assertEqual(x2.ub, float('inf'))
        self.assertEqual(x2.vtype, GRB.CONTINUOUS)
        
        self.assertEqual(x3.lb, -float('inf'))
        self.assertEqual(x3.ub, 0)
        self.assertEqual(x3.vtype, GRB.CONTINUOUS)

        self.assertEqual(y1.lb, -float('inf'))
        self.assertEqual(y1.ub, float('inf'))
        self.assertEqual(y1.vtype, GRB.INTEGER)

        self.assertEqual(y2.lb, 0)
        self.assertEqual(y2.ub, float('inf'))
        self.assertEqual(y2.vtype, GRB.INTEGER)

        self.assertEqual(y3.lb, -float('inf'))
        self.assertEqual(y3.ub, 0)
        self.assertEqual(y3.vtype, GRB.INTEGER)

        self.assertEqual(z1.vtype, GRB.BINARY)

    def test_var_bounds(self):
        m = self.get_model()
        m.x2.setlb(-34)
        m.x2.setub(45)
        m.x3.setub(5)
        m.y1.setlb(-2)
        m.y1.setub(3)
        m.y2.setlb(-5)
        m.z1.setub(4)
        m.z1.setlb(-3)

        e = m.x1 + m.x2 + m.x3 + m.y1 + m.y2 + m.y3 + m.z1
        visitor = self.get_visitor()
        expr = visitor.walk_expression((e, e, 0))

        x2 = visitor.var_map[id(m.x2)]
        x3 = visitor.var_map[id(m.x3)]
        y1 = visitor.var_map[id(m.y1)]
        y2 = visitor.var_map[id(m.y2)]
        z1 = visitor.var_map[id(m.z1)]

        self.assertEqual(x2.lb, -34)
        self.assertEqual(x2.ub, 45)
        self.assertEqual(x2.vtype, GRB.CONTINUOUS)
        
        self.assertEqual(x3.lb, -float('inf'))
        self.assertEqual(x3.ub, 0)
        self.assertEqual(x3.vtype, GRB.CONTINUOUS)

        self.assertEqual(y1.lb, -2)
        self.assertEqual(y1.ub, 3)
        self.assertEqual(y1.vtype, GRB.INTEGER)

        self.assertEqual(y2.lb, 0)
        self.assertEqual(y2.ub, float('inf'))
        self.assertEqual(y2.vtype, GRB.INTEGER)

        self.assertEqual(z1.vtype, GRB.BINARY)

    def test_write_addition(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 + m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_subtraction(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 - m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_product(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 * m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO        

    def test_write_power_expression_var_const(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 ** 2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_power_expression_var_var(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 ** m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_power_expression_const_var(self):
        m = self.get_model()
        m.c = Constraint(expr=2 ** m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_absolute_value_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=abs(m.x1) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_write_expression_with_mutable_param(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)
        m.c = Constraint(expr=m.p ** m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    def test_monomial_expression(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)

        const_expr = 3 * m.x1
        nested_expr = (1 / m.p) * m.x1
        pow_expr = (m.p ** (0.5)) * m.x1

        visitor = self.get_visitor()
        expr = visitor.walk_expression((const_expr, const_expr, 0))
        expr = visitor.walk_expression((nested_expr, nested_expr, 0))
        expr = visitor.walk_expression((pow_expr, pow_expr, 0))

        # TODO

    def test_log_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=log(m.x1) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

    # TODO: what other unary expressions?

    def test_handle_complex_number_sqrt(self):
        m = self.get_model()
        m.p = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=sqrt(-m.p) + m.x1 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression((m.c.body, m.c, 0))

        # TODO

