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

from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr import ProductExpression, SumExpression
from pyomo.common.errors import InvalidValueError
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiMINLPVisitor
from pyomo.contrib.solver.tests.solvers.gurobi_to_pyomo_expressions import (
    grb_nl_to_pyo_expr,
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

    def test_var_domains(self):
        m = self.get_model()
        e = m.x1 + m.x2 + m.x3 + m.y1 + m.y2 + m.y3 + m.z1
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(e)

        # We don't call update in walk expression for performance reasons, but
        # we need to update here in order to be able to test expr.
        visitor.grb_model.update()

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
        _, expr = visitor.walk_expression(e)

        # We don't call update in walk expression for performance reasons, but
        # we need to update here in order to be able to test expr.
        visitor.grb_model.update()

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
        _, expr = visitor.walk_expression(m.c.body)

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
        _, expr = visitor.walk_expression(m.c.body)

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
        _, expr = visitor.walk_expression(m.c.body)

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
        _, expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]

        # this is linear
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getCoeff(0), 4.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getConstant(), 0.0)

    def test_write_product_with_0(self):
        m = self.get_model()
        m.c = Constraint(expr=(0 * m.x1 * m.x2) * m.x3 == 0)

        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # this is a "nonlinear"
        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(
            self,
            pyo_expr,
            ProductExpression((ProductExpression((0.0, m.x1, m.x2, m.x3)),)),
        )

    def test_write_division(self):
        m = self.get_model()
        m.c = Constraint(expr=1 / m.x1 == 1)

        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        visitor.grb_model.update()
        grb_to_pyo_var_map = {
            grb_var: py_var for py_var, grb_var in visitor.var_map.items()
        }

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, grb_to_pyo_var_map)
        assertExpressionsEqual(self, pyo_expr, 1.0 / m.x1)

    def test_write_division_linear(self):
        m = self.get_model()
        m.p = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=(m.x1 + m.x2) * m.p / 10 == 1)

        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        # linear
        self.assertEqual(expr.size(), 2)
        self.assertEqual(expr.getConstant(), 0)
        self.assertAlmostEqual(expr.getCoeff(0), 3 / 10)
        self.assertIs(expr.getVar(0), x1)
        self.assertAlmostEqual(expr.getCoeff(1), 3 / 10)
        self.assertIs(expr.getVar(1), x2)

    def test_write_linear_power_expression_var_const(self):
        m = self.get_model()
        m.devious = Param(initialize=1, mutable=True)
        m.c = Constraint(expr=m.x1**m.devious >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[id(m.x1)]

        # It's just a single var
        self.assertIs(expr, x1)
        self.assertEqual(len(visitor.grb_model.getGenConstrs()), 0)

        # now try a linear expression
        m.c2 = Constraint(expr=(m.x1 + 2 * m.x2) ** m.devious >= 5)
        _, lin_expr = visitor.walk_expression(m.c2.body)
        self.assertEqual(len(visitor.grb_model.getGenConstrs()), 0)
        x2 = visitor.var_map[m.x2]
        self.assertEqual(lin_expr.size(), 2)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertIs(lin_expr.getVar(0), x1)
        self.assertIs(lin_expr.getVar(1), x2)
        self.assertEqual(lin_expr.getCoeff(0), 1.0)
        self.assertEqual(lin_expr.getCoeff(1), 2.0)

        # now do a quadratic expression
        m.c3 = Constraint(expr=(m.x1**2 + 5.4) ** m.devious >= 8)
        _, quad_expr = visitor.walk_expression(m.c3.body)
        self.assertEqual(len(visitor.grb_model.getGenConstrs()), 0)
        self.assertEqual(quad_expr.size(), 1)
        expr = quad_expr.getLinExpr()
        # no vars in linear part, just the constant
        self.assertEqual(expr.size(), 0)
        self.assertEqual(expr.getConstant(), 5.4)
        self.assertIs(quad_expr.getVar1(0), x1)
        self.assertIs(quad_expr.getVar2(0), x1)
        self.assertEqual(quad_expr.getCoeff(0), 1.0)

    def test_write_quadratic_power_expression_var_const(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**2 >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # This is quadratic
        x1 = visitor.var_map[id(m.x1)]

        self.assertEqual(expr.size(), 1)
        lin_expr = expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 0)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertIs(expr.getVar1(0), x1)
        self.assertIs(expr.getVar2(0), x1)
        self.assertEqual(expr.getCoeff(0), 1.0)

    def test_write_quadratic_constant_pow_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=(m.x1**2 + 2 * m.x2 + 3) ** 2 <= 7)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # This is general nonlinear
        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(
            self,
            pyo_expr,
            SumExpression((3.0, ProductExpression((2.0, m.x2)), m.x1**2)) ** 2,
        )

    def test_write_nonquadratic_power_expression_var_const(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**3 >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # This is general nonlinear
        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(self, pyo_expr, m.x1**3.0)

    def test_write_power_expression_var_var(self):
        m = self.get_model()
        m.c = Constraint(expr=m.x1**m.x2 >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # You can't actually use this in a model in Gurobi 12, but you can build the
        # expression... (It fails during the solve.)

        x1 = visitor.var_map[id(m.x1)]
        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(self, pyo_expr, m.x1**m.x2)

    def test_write_power_expression_const_var(self):
        m = self.get_model()
        m.c = Constraint(expr=2**m.x2 >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(self, pyo_expr, 2.0**m.x2)

    def test_write_absolute_value_of_constant(self):
        m = self.get_model()
        m.tricky = Param(initialize=-3.4, mutable=True)
        m.c = Constraint(expr=abs(m.tricky + m.x2) + m.x1 <= 7)
        m.x2.fix(1)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        x1 = visitor.var_map[m.x1]

        self.assertEqual(len(visitor.grb_model.getGenConstrs()), 0)
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 2.4)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getCoeff(0), 1.0)

    def test_write_absolute_value_of_var(self):
        # Gurobi doesn't support abs of expressions, so we have to do a factorable
        # programming thing...
        m = self.get_model()
        m.c = Constraint(expr=abs(m.x1) >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # expr is actually an auxiliary variable. We should
        # get a constraint:
        # expr == abs(x1)
        x1 = visitor.var_map[id(m.x1)]

        self.assertIsInstance(expr, gurobipy.Var)
        grb_model = visitor.grb_model
        # We don't call update in walk expression for performance reasons, but
        # we need to update here in order to be able to test expr.
        grb_model.update()
        self.assertEqual(grb_model.numVars, 2)
        self.assertEqual(grb_model.numGenConstrs, 1)
        self.assertEqual(grb_model.numConstrs, 0)
        self.assertEqual(grb_model.numQConstrs, 0)

        cons = grb_model.getGenConstrs()[0]
        aux, v = grb_model.getGenConstrAbs(cons)
        self.assertIs(aux, expr)
        self.assertIs(v, x1)

    def test_write_absolute_value_of_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=abs(m.x1 + 2 * m.x2) >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # expr is actually an auxiliary variable. We should
        # get three constraints:
        # aux1 == x1 + 2 * x2
        # expr == abs(aux1)

        x1 = visitor.var_map[m.x1]
        x2 = visitor.var_map[m.x2]

        # we're going to have to write the resulting model to an lp file to test that we
        # have what we expect
        self.assertIsInstance(expr, gurobipy.Var)
        grb_model = visitor.grb_model
        # We don't call update in walk expression for performance reasons, but
        # we need to update here in order to be able to test expr.
        grb_model.update()
        self.assertEqual(grb_model.numVars, 4)
        self.assertEqual(grb_model.numGenConstrs, 1)
        self.assertEqual(grb_model.numConstrs, 1)
        self.assertEqual(grb_model.numQConstrs, 0)

        cons = grb_model.getGenConstrs()[0]
        aux2, aux1 = grb_model.getGenConstrAbs(cons)
        self.assertIs(aux2, expr)

        cons = grb_model.getConstrs()[0]
        # this guy is linear equality
        self.assertEqual(cons.RHS, 0)
        self.assertEqual(cons.Sense, '=')
        linexpr = grb_model.getRow(cons)
        self.assertEqual(linexpr.getConstant(), 0)
        self.assertEqual(linexpr.size(), 3)
        self.assertEqual(linexpr.getCoeff(0), -1)
        self.assertIs(linexpr.getVar(0), x1)
        self.assertEqual(linexpr.getCoeff(1), -2)
        self.assertIs(linexpr.getVar(1), x2)
        self.assertEqual(linexpr.getCoeff(2), 1)
        self.assertIs(linexpr.getVar(2), aux1)

    def test_write_expression_with_mutable_param(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)
        m.c = Constraint(expr=m.p**m.x2 >= 3)
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # expr is nonlinear
        x2 = visitor.var_map[id(m.x2)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(self, pyo_expr, 4.0**m.x2)

    def test_monomial_expression(self):
        m = self.get_model()
        m.p = Param(initialize=4, mutable=True)

        const_expr = 3 * m.x1
        nested_expr = (1 / m.p) * m.x1
        pow_expr = (m.p ** (0.5)) * m.x1

        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(const_expr)
        x1 = visitor.var_map[id(m.x1)]
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getCoeff(0), 3)

        _, expr = visitor.walk_expression(nested_expr)
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertAlmostEqual(expr.getCoeff(0), 1 / 4)

        _, expr = visitor.walk_expression(pow_expr)
        self.assertEqual(expr.size(), 1)
        self.assertEqual(expr.getConstant(), 0.0)
        self.assertIs(expr.getVar(0), x1)
        self.assertEqual(expr.getCoeff(0), 2)

    def test_log_expression(self):
        m = self.get_model()
        m.c = Constraint(expr=log(m.x1) >= 3)
        m.pprint()
        visitor = self.get_visitor()
        _, expr = visitor.walk_expression(m.c.body)

        # expr is nonlinear
        x1 = visitor.var_map[id(m.x1)]

        opcode, data, parent = self._get_nl_expr_tree(visitor, expr)

        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in visitor.var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(self, pyo_expr, log(m.x1))

    def test_handle_complex_number_sqrt(self):
        m = self.get_model()
        m.p = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=sqrt(-m.p) + m.x1 >= 3)

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Invalid number encountered evaluating constant unary expression "
            r"sqrt\(- p\): math domain error",
        ):
            _, expr = visitor.walk_expression(m.c.body)

    def test_handle_invalid_log(self):
        m = self.get_model()
        m.p = Param(initialize=0, mutable=True)
        m.c = Constraint(expr=log(m.p) + m.x1 >= 3)

        visitor = self.get_visitor()
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Invalid number encountered evaluating constant unary expression "
            r"log\(p\): math domain error",
        ):
            _, expr = visitor.walk_expression(m.c.body)
