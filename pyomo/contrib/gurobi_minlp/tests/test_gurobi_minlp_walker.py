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
    Objective,
    Param,
    Reals,
    sqrt,
    Var,
)

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')

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
        expr = visitor.walk_expression(e)

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
        expr = visitor.walk_expression(e)

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
        expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        # This is a linear expression
        self.assertEqual(expr.size(), 2)
        self.assertEqual(expr.getCoeff(0), 1.0)
        self.assertEqual(expr.getCoeff(1), 1.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertIs(expr.getVar(1), x2)
        self.assertEqual(expr.getConstant(), 0.0)

    def test_write_subtraction(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 - m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        # Also linear, whoot!
        self.assertEqual(expr.size(), 2)
        self.assertEqual(expr.getCoeff(0), 1.0)
        self.assertEqual(expr.getCoeff(1), -1.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertIs(expr.getVar(1), x2)
        self.assertEqual(expr.getConstant(), 0.0)

    def test_write_product(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1 * m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        # This is quadratic
        self.assertEqual(expr.size(), 1)
        lin_expr = expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 0)
        self.assertIs(expr.getVar1(0), x1)
        self.assertIs(expr.getVar2(0), x2)
        self.assertEqual(expr.getCoeff(0), 1.0)

    def test_write_product_with_fixed_var(self):
        m = self.get_model()
        m.x2.fix(4)
        m.c = Constraint(expr=m.x1 * m.x2 == 1)

        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]

        # this is linear
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getCoeff(0), 4.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getConstant(), 0.0)

    def test_write_division(self):
        m = self.get_model()
        m.c = Constraint(expr=1 / m.x1 == 1)

        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # three nodes
        self.assertEqual(len(opcode), 3)
        # the root is a division expression
        self.assertEqual(parent[0], -1)  # root
        self.assertEqual(opcode[0], GRB.OPCODE_DIVIDE)
        # divide has no additional data
        self.assertEqual(data[0], -1)

        # first arg is 1
        self.assertEqual(parent[1], 0)
        self.assertEqual(opcode[1], GRB.OPCODE_CONSTANT)
        self.assertEqual(data[1], 1)

        # second arg is x1
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_VARIABLE)
        self.assertIs(data[2], x1)

    def test_write_quadratic_power_expression_var_const(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # This is also quadratic
        x1 = visitor.var_map[id(m.x1)]

        self.assertEqual(expr.size(), 1)
        lin_expr = expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 0)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertIs(expr.getVar1(0), x1)
        self.assertIs(expr.getVar2(0), x1)
        self.assertEqual(expr.getCoeff(0), 1.0)

    def _get_nl_expr_tree(self, visitor, expr):
        # This is a bit hacky, but the only way that I know to get the expression tree
        # publicly is from a general nonlinear constraint. So we can create it, and
        # then pull out the expression we just used to test it
        grb_model = visitor.grb_model
        aux = grb_model.addVar()
        grb_model.addConstr(aux == expr)
        grb_model.update()
        constrs = grb_model.getGenConstrs()
        self.assertEqual(len(constrs), 1)

        aux_var, opcode, data, parent = grb_model.getGenConstrNLAdv(constrs[0])
        self.assertIs(aux_var, aux)
        return opcode, data, parent

    def test_write_nonquadratic_power_expression_var_const(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**3 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # This is general nonlinear
        x1 = visitor.var_map[id(m.x1)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # three nodes
        self.assertEqual(len(opcode), 3)

        # the root is a power expression
        self.assertEqual(parent[0], -1)  # means root
        self.assertEqual(opcode[0], GRB.OPCODE_POW)
        # pow has no additional data
        self.assertEqual(data[0], -1)

        # first child is x1
        self.assertEqual(parent[1], 0)
        self.assertIs(data[1], x1)
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)

        # second child is 3
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_CONSTANT)
        self.assertEqual(data[2], 3.0)  # the data is the constant's value

    def test_write_power_expression_var_var(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # You can't actually use this in a model in Gurobi 12, but you can build the
        # expression... (It fails during the solve for some reason.)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # three nodes
        self.assertEqual(len(opcode), 3)

        # the root is a power expression
        self.assertEqual(parent[0], -1)  # means root
        self.assertEqual(opcode[0], GRB.OPCODE_POW)
        # pow has no additional data
        self.assertEqual(data[0], -1)

        # first child is x1
        self.assertEqual(parent[1], 0)
        self.assertIs(data[1], x1)
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)

        # second child is x2
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_VARIABLE)
        self.assertIs(data[2], x2)

    def test_write_power_expression_const_var(self):
        m = self.get_model()
        m.c = Constraint(expr=2**m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # three nodes
        self.assertEqual(len(opcode), 3)

        # the root is a power expression
        self.assertEqual(parent[0], -1)  # means root
        self.assertEqual(opcode[0], GRB.OPCODE_POW)
        # pow has no additional data
        self.assertEqual(data[0], -1)

        # first child is 2
        self.assertEqual(parent[1], 0)
        self.assertEqual(data[1], 2.0)
        self.assertEqual(opcode[1], GRB.OPCODE_CONSTANT)

        # second child is x2
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_VARIABLE)
        self.assertIs(data[2], x2)

    def test_write_absolute_value_of_var(self):
        # Gurobi doesn't support abs of expressions, so we have to do a factorable
        # programming thing...
        m = self.get_model()
        m.c = Constraint(expr=abs(m.x1) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # expr is actually an auxiliary variable. We should
        # get a constraint:
        # expr == abs(x1)

        self.assertIsInstance(expr, gurobipy.Var)
        grb_model = visitor.grb_model
        self.assertEqual(grb_model.numVars, 2)
        self.assertEqual(grb_model.numGenConstrs, 1)
        self.assertEqual(grb_model.numConstrs, 0)
        self.assertEqual(grb_model.numQConstrs, 0)

        # we're going to have to write the resulting model to an lp file to test that we
        # have what we expect

        # TODO

    def test_write_absolute_value_of_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=abs(m.x1 + 2 * m.x2) >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # expr is actually an auxiliary variable. We should
        # get three constraints:
        # aux1 == x1 + 2 * x2
        # expr == abs(aux1)

        # we're going to have to write the resulting model to an lp file to test that we
        # have what we expect
        self.assertIsInstance(expr, gurobipy.Var)
        grb_model = visitor.grb_model
        self.assertEqual(grb_model.numVars, 4)
        self.assertEqual(grb_model.numGenConstrs, 1)
        self.assertEqual(grb_model.numConstrs, 1)
        self.assertEqual(grb_model.numQConstrs, 0)

        # TODO

    def test_write_expression_with_mutable_param(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)
        m.c = Constraint(expr=m.p**m.x2 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # expr is nonlinear
        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # three nodes
        self.assertEqual(len(opcode), 3)

        # the root is a power expression
        self.assertEqual(parent[0], -1)  # means root
        self.assertEqual(opcode[0], GRB.OPCODE_POW)
        # pow has no additional data
        self.assertEqual(data[0], -1)

        # first child is 4
        self.assertEqual(parent[1], 0)
        self.assertEqual(data[1], 4.0)
        self.assertEqual(opcode[1], GRB.OPCODE_CONSTANT)

        # second child is x2
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_VARIABLE)
        self.assertIs(data[2], x2)

    def test_monomial_expression(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)

        const_expr = 3 * m.x1
        nested_expr = (1 / m.p) * m.x1
        pow_expr = (m.p ** (0.5)) * m.x1

        visitor = self.get_visitor()
        expr = visitor.walk_expression(const_expr)
        x1 = visitor.var_map[id(m.x1)]
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getCoeff(0), 3)

        expr = visitor.walk_expression(nested_expr)
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertAlmostEqual(expr.getCoeff(0), 1 / 4)

        expr = visitor.walk_expression(pow_expr)
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getCoeff(0), 2)

    def test_log_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=log(m.x1) >= 3)
        m.pprint()
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # expr is nonlinear
        x1 = visitor.var_map[id(m.x1)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        # two nodes
        self.assertEqual(len(opcode), 2)

        # the root is a power expression
        self.assertEqual(parent[0], -1)  # means root
        self.assertEqual(opcode[0], GRB.OPCODE_LOG)
        self.assertEqual(data[0], -1)

        # child is x1
        self.assertEqual(parent[1], 0)
        self.assertIs(data[1], x1)
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)

    # TODO: what other unary expressions?

    def test_handle_complex_number_sqrt(self):
        m = self.get_model()
        m.p = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=sqrt(-m.p) + m.x1 >= 3)
        visitor = self.get_visitor()
        expr = visitor.walk_expression(m.c.body)

        # TODO

