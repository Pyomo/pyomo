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

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import numpy as np, numpy_available

from pyomo.contrib.solver.tests.solvers.gurobi_to_pyomo_expressions import (
    grb_nl_to_pyo_expr,
)
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.expr.numeric_expr import SumExpression, ProductExpression
from pyomo.environ import (
    Binary,
    BooleanVar,
    ConcreteModel,
    Constraint,
    Expression,
    Integers,
    log,
    LogicalConstraint,
    maximize,
    NonNegativeIntegers,
    NonNegativeReals,
    NonPositiveIntegers,
    NonPositiveReals,
    Objective,
    Param,
    Reals,
    value,
    Var,
)
from pyomo.gdp import Disjunction
from pyomo.opt import WriterFactory
from pyomo.contrib.solver.solvers.gurobi_direct_minlp import (
    GurobiDirectMINLP,
    GurobiMINLPVisitor,
)
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import TerminationCondition
from pyomo.contrib.solver.tests.solvers.test_gurobi_minlp_walker import CommonTest

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')
if gurobipy_available:
    from gurobipy import GRB


def make_model():
    m = ConcreteModel()
    m.x1 = Var(domain=NonNegativeReals, bounds=(0, 10))
    m.x2 = Var(domain=Reals, bounds=(-3, 4))
    m.x3 = Var(domain=NonPositiveReals, bounds=(-13, 0))
    m.y1 = Var(domain=Integers, bounds=(4, 14))
    m.y2 = Var(domain=NonNegativeIntegers, bounds=(5, 16))
    m.y3 = Var(domain=NonPositiveIntegers, bounds=(-13, 0))
    m.z1 = Var(domain=Binary)

    m.c1 = Constraint(expr=2**m.x2 >= m.x3)
    m.c2 = Constraint(expr=m.y1**2 <= 7)
    m.c3 = Constraint(expr=m.y2 + m.y3 + 5 * m.z1 >= 17)

    m.obj = Objective(expr=log(m.x1))

    return m


@unittest.skipUnless(gurobipy_available, "Gurobipy 12 is not available")
class TestGurobiMINLPWriter(CommonTest):
    def test_small_model(self):
        grb_model = gurobipy.Model()
        visitor = GurobiMINLPVisitor(grb_model, symbolic_solver_labels=True)

        m = make_model()

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 7)
        x1 = var_map[id(m.x1)]
        x2 = var_map[id(m.x2)]
        x3 = var_map[id(m.x3)]
        y1 = var_map[id(m.y1)]
        y2 = var_map[id(m.y2)]
        y3 = var_map[id(m.y3)]
        z1 = var_map[id(m.z1)]

        self.assertEqual(grb_model.numVars, 9)
        self.assertEqual(grb_model.numIntVars, 4)
        self.assertEqual(grb_model.numBinVars, 1)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 2)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 1)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 2)

        ## linear constraints

        # this is the linear piece of c1
        c = lin_constrs[0]
        c_expr = grb_model.getRow(c)
        self.assertEqual(c.RHS, 0)
        self.assertEqual(c.Sense, '<')
        self.assertEqual(c_expr.size(), 1)
        self.assertEqual(c_expr.getCoeff(0), 1)
        self.assertEqual(c_expr.getConstant(), 0)
        aux_var = c_expr.getVar(0)

        c3 = lin_constrs[1]
        c3_expr = grb_model.getRow(c3)
        self.assertEqual(c3_expr.size(), 3)
        self.assertIs(c3_expr.getVar(0), y2)
        self.assertEqual(c3_expr.getCoeff(0), 1)
        self.assertIs(c3_expr.getVar(1), y3)
        self.assertEqual(c3_expr.getCoeff(1), 1)
        self.assertIs(c3_expr.getVar(2), z1)
        self.assertEqual(c3_expr.getCoeff(2), 5)
        self.assertEqual(c3_expr.getConstant(), 0)
        self.assertEqual(c3.RHS, 17)
        self.assertEqual(c3.Sense, '>')

        ## quadratic constraint
        c2 = quad_constrs[0]
        c2_expr = grb_model.getQCRow(c2)
        lin_expr = c2_expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 0)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertEqual(c2.QCRHS, 7)
        self.assertEqual(c2.QCSense, '<')
        self.assertEqual(c2_expr.size(), 1)
        self.assertIs(c2_expr.getVar1(0), y1)
        self.assertIs(c2_expr.getVar2(0), y1)
        self.assertEqual(c2_expr.getCoeff(0), 1)

        ## general nonlinear constraints
        obj_cons = nonlinear_constrs[0]
        res_var, opcode, data, parent = grb_model.getGenConstrNLAdv(obj_cons)
        self.assertEqual(len(opcode), 2)  # two nodes in the expression tree
        self.assertEqual(opcode[0], GRB.OPCODE_LOG)
        # log has no data
        self.assertEqual(parent[0], -1)  # it's the root
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)
        self.assertIs(data[1], x1)
        self.assertEqual(parent[1], 0)

        # we can check that res_var is the objective
        self.assertEqual(grb_model.ModelSense, 1)  # minimizing
        obj = grb_model.getObjective()
        self.assertEqual(obj.size(), 1)
        self.assertEqual(obj.getCoeff(0), 1)
        self.assertIs(obj.getVar(0), res_var)

        c1 = nonlinear_constrs[1]
        res_var, opcode, data, parent = grb_model.getGenConstrNLAdv(c1)
        # This is where we link into the linear inequality constraint
        self.assertIs(res_var, aux_var)
        # test the tree for the expression x3  + (- (2 ** x2))
        self.assertEqual(len(opcode), 6)
        self.assertEqual(opcode[0], GRB.OPCODE_PLUS)
        # plus has no data
        self.assertEqual(parent[0], -1)  # root
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)
        self.assertIs(data[1], x3)
        self.assertEqual(parent[1], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_UMINUS)  # negation
        # negation has no data
        self.assertEqual(parent[2], 0)
        self.assertEqual(opcode[3], GRB.OPCODE_POW)
        # pow has no data
        self.assertEqual(parent[3], 2)
        self.assertEqual(opcode[4], GRB.OPCODE_CONSTANT)
        self.assertEqual(data[4], 2)
        self.assertEqual(parent[4], 3)
        self.assertEqual(opcode[5], GRB.OPCODE_VARIABLE)
        self.assertIs(data[5], x2)
        self.assertEqual(parent[5], 3)

    def test_write_NPV_negation_in_RHS(self):
        m = ConcreteModel()
        m.x1 = Var()
        m.p1 = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=-m.x1 == m.p1)
        m.obj = Objective(expr=m.x1)

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 1)
        x1 = var_map[id(m.x1)]

        self.assertEqual(grb_model.numVars, 1)
        self.assertEqual(grb_model.numIntVars, 0)
        self.assertEqual(grb_model.numBinVars, 0)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 1)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 0)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 0)

        # constraint
        c = lin_constrs[0]
        c_expr = grb_model.getRow(c)
        self.assertEqual(c.RHS, 3)
        self.assertEqual(c.Sense, '=')
        self.assertEqual(c_expr.size(), 1)
        self.assertEqual(c_expr.getCoeff(0), -1)
        self.assertEqual(c_expr.getConstant(), 0)
        self.assertIs(c_expr.getVar(0), x1)

        # objective
        self.assertEqual(grb_model.ModelSense, 1)  # minimizing
        obj = grb_model.getObjective()
        self.assertEqual(obj.size(), 1)
        self.assertEqual(obj.getCoeff(0), 1)
        self.assertIs(obj.getVar(0), x1)

    def test_writer_ignores_deactivated_logical_constraints(self):
        m = ConcreteModel()
        m.x1 = Var()
        m.p1 = Param(initialize=3, mutable=True)
        m.c = Constraint(expr=-m.x1 == m.p1)
        m.obj = Objective(expr=m.x1)

        m.b = BooleanVar()
        m.whatever = LogicalConstraint(expr=~m.b)
        m.whatever.deactivate()

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 1)
        x1 = var_map[id(m.x1)]

        self.assertEqual(grb_model.numVars, 1)
        self.assertEqual(grb_model.numIntVars, 0)
        self.assertEqual(grb_model.numBinVars, 0)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 1)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 0)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 0)

        # constraint
        c = lin_constrs[0]
        c_expr = grb_model.getRow(c)
        self.assertEqual(c.RHS, 3)
        self.assertEqual(c.Sense, '=')
        self.assertEqual(c_expr.size(), 1)
        self.assertEqual(c_expr.getCoeff(0), -1)
        self.assertEqual(c_expr.getConstant(), 0)
        self.assertIs(c_expr.getVar(0), x1)

        # objective
        self.assertEqual(grb_model.ModelSense, 1)  # minimizing
        obj = grb_model.getObjective()
        self.assertEqual(obj.size(), 1)
        self.assertEqual(obj.getCoeff(0), 1)
        self.assertIs(obj.getVar(0), x1)

    def test_named_expression_quadratic(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=m.x**2 + m.y)
        m.c = Constraint(expr=m.e <= 7)
        m.c2 = Constraint(expr=m.e >= -3)
        m.obj = Objective(expr=0)

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 2)
        x = var_map[id(m.x)]
        y = var_map[id(m.y)]

        self.assertEqual(grb_model.numVars, 2)
        self.assertEqual(grb_model.numIntVars, 0)
        self.assertEqual(grb_model.numBinVars, 0)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 0)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 2)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 0)

        # constraint 1
        c1 = quad_constrs[0]
        expr = grb_model.getQCRow(c1)
        lin_expr = expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 1)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertEqual(lin_expr.getCoeff(0), 1)
        self.assertIs(lin_expr.getVar(0), y)
        self.assertEqual(c1.QCRHS, 7)
        self.assertEqual(c1.QCSense, '<')
        self.assertEqual(expr.size(), 1)
        self.assertIs(expr.getVar1(0), x)
        self.assertIs(expr.getVar2(0), x)
        self.assertEqual(expr.getCoeff(0), 1)

        # constraint 2
        c2 = quad_constrs[1]
        expr = grb_model.getQCRow(c1)
        lin_expr = expr.getLinExpr()
        self.assertEqual(lin_expr.size(), 1)
        self.assertEqual(lin_expr.getConstant(), 0)
        self.assertEqual(lin_expr.getCoeff(0), 1)
        self.assertIs(lin_expr.getVar(0), y)
        self.assertEqual(c2.QCRHS, -3)
        self.assertEqual(c2.QCSense, '>')
        self.assertEqual(expr.size(), 1)
        self.assertIs(expr.getVar1(0), x)
        self.assertIs(expr.getVar2(0), x)
        self.assertEqual(expr.getCoeff(0), 1)

        # objective
        self.assertEqual(grb_model.ModelSense, 1)  # minimizing
        obj = grb_model.getObjective()
        self.assertEqual(obj.size(), 0)

    def test_named_expression_nonlinear(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=log(m.x) ** 2 + m.y)
        m.c = Constraint(expr=m.e <= 7)
        m.c2 = Constraint(expr=m.e + m.y**3 + log(m.x + m.y) >= -3)
        m.obj = Objective(expr=0)

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 2)
        x = var_map[id(m.x)]
        y = var_map[id(m.y)]
        reverse_var_map = {grbv: pyov for pyov, grbv in var_map.items()}

        self.assertEqual(grb_model.numVars, 4)
        self.assertEqual(grb_model.numIntVars, 0)
        self.assertEqual(grb_model.numBinVars, 0)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 2)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 0)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 2)

        c1 = lin_constrs[0]
        aux1 = grb_model.getRow(c1)
        self.assertEqual(c1.RHS, 7)
        self.assertEqual(c1.Sense, '<')
        self.assertEqual(aux1.size(), 1)
        self.assertEqual(aux1.getCoeff(0), 1)
        self.assertEqual(aux1.getConstant(), 0)
        aux1 = aux1.getVar(0)

        c2 = lin_constrs[1]
        aux2 = grb_model.getRow(c2)
        self.assertEqual(c2.RHS, -3)
        self.assertEqual(c2.Sense, '>')
        self.assertEqual(aux2.size(), 1)
        self.assertEqual(aux2.getCoeff(0), 1)
        self.assertEqual(aux2.getConstant(), 0)
        aux2 = aux2.getVar(0)

        # log(x)**2 + y
        g1 = nonlinear_constrs[0]
        aux_var, opcode, data, parent = grb_model.getGenConstrNLAdv(g1)
        self.assertIs(aux_var, aux1)
        assertExpressionsEqual(
            self,
            grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map),
            log(m.x) ** 2 + m.y,
        )

        # log(x)**2 + y + y**3 + log(x + y)
        g2 = nonlinear_constrs[1]
        aux_var, opcode, data, parent = grb_model.getGenConstrNLAdv(g2)
        self.assertIs(aux_var, aux2)
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)
        assertExpressionsEqual(
            self,
            pyo_expr,
            SumExpression(
                (
                    SumExpression((SumExpression((log(m.x) ** 2, m.y)), m.y**3.0)),
                    log(SumExpression((m.x, m.y))),
                )
            ),
        )

        # objective
        self.assertEqual(grb_model.ModelSense, 1)  # minimizing
        obj = grb_model.getObjective()
        self.assertEqual(obj.size(), 0)

    def test_solve_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1))
        m.y = Var()
        m.c = Constraint(expr=m.y == m.x**2)
        m.obj = Objective(expr=m.x + m.y, sense=maximize)

        results = SolverFactory('gurobi_direct_minlp').solve(m, tee=True)

        self.assertEqual(value(m.obj.expr), 2)

        self.assertEqual(value(m.x), 1)
        self.assertEqual(value(m.y), 1)

        self.assertEqual(
            results.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(results.incumbent_objective, 2)
        self.assertEqual(results.objective_bound, 2)

    def test_unbounded_because_of_multiplying_by_0(self):
        # Gurobi believes that the expression in m.c is nonlinear, so we have
        # to pass it that way for this to work. Because this is in fact an
        # unbounded model.

        m = ConcreteModel()
        m.x1 = Var()
        m.x2 = Var()
        m.x3 = Var()
        m.c = Constraint(expr=(0 * m.x1 * m.x2) * m.x3 == 0)
        m.obj = Objective(expr=m.x1)

        grb_model, var_map, obj, grb_cons, pyo_cons = WriterFactory(
            'gurobi_minlp'
        ).write(m, symbolic_solver_labels=True)

        self.assertEqual(len(var_map), 3)
        x1 = var_map[id(m.x1)]
        x2 = var_map[id(m.x2)]
        x3 = var_map[id(m.x3)]

        self.assertEqual(grb_model.numVars, 4)
        self.assertEqual(grb_model.numIntVars, 0)
        self.assertEqual(grb_model.numBinVars, 0)

        lin_constrs = grb_model.getConstrs()
        self.assertEqual(len(lin_constrs), 1)
        quad_constrs = grb_model.getQConstrs()
        self.assertEqual(len(quad_constrs), 0)
        nonlinear_constrs = grb_model.getGenConstrs()
        self.assertEqual(len(nonlinear_constrs), 1)

        # this is the auxiliary variable equality
        c = lin_constrs[0]
        c_expr = grb_model.getRow(c)
        self.assertEqual(c.RHS, 0)
        self.assertEqual(c.Sense, '=')
        self.assertEqual(c_expr.size(), 1)
        self.assertEqual(c_expr.getCoeff(0), 1)
        self.assertEqual(c_expr.getConstant(), 0)
        aux_var = c_expr.getVar(0)

        # this is the nonlinear equality
        c = nonlinear_constrs[0]
        res_var, opcode, data, parent = grb_model.getGenConstrNLAdv(c)
        # This is where we link into the linear inequality constraint
        self.assertIs(res_var, aux_var)
        reverse_var_map = {grb_v: pyo_v for pyo_v, grb_v in var_map.items()}
        pyo_expr = grb_nl_to_pyo_expr(opcode, data, parent, reverse_var_map)

        assertExpressionsEqual(
            self,
            pyo_expr,
            ProductExpression((ProductExpression((0.0, m.x1, m.x2, m.x3)),)),
        )

        opt = SolverFactory('gurobi_direct_minlp')
        opt.config.raise_exception_on_nonoptimal_result = False
        results = opt.solve(m)
        # model is unbounded
        self.assertEqual(results.termination_condition, TerminationCondition.unbounded)

    @unittest.skipUnless(numpy_available, "Numpy is not available")
    def test_numpy_trivially_true_constraint(self):
        m = ConcreteModel()
        m.x1 = Var()
        m.x2 = Var()
        m.x1.fix(np.float64(0))
        m.x2.fix(np.float64(0))
        m.c = Constraint(expr=m.x1 == m.x2)
        m.obj = Objective(expr=m.x1)
        results = SolverFactory('gurobi_direct_minlp').solve(m)

        self.assertEqual(
            results.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(value(m.obj), 0)
        self.assertEqual(results.incumbent_objective, 0)
        self.assertEqual(results.objective_bound, 0)

    def test_trivially_true_constraint(self):
        """
        We can pass trivially true things to Gurobi and it's fine
        """
        m = ConcreteModel()
        m.x1 = Var()
        m.x2 = Var()
        m.x1.fix(2)
        m.x2.fix(2)
        m.c = Constraint(expr=m.x1 <= m.x2)
        m.obj = Objective(expr=m.x1)
        results = SolverFactory('gurobi_direct_minlp').solve(m, tee=True)

        self.assertEqual(
            results.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(value(m.obj), 2)
        self.assertEqual(results.incumbent_objective, 2)
        self.assertEqual(results.objective_bound, 2)

    def test_multiple_objective_error(self):
        m = make_model()
        m.obj2 = Objective(expr=m.x1 + m.x2)

        with self.assertRaisesRegex(
            ValueError,
            "More than one active objective defined for input model 'unknown': "
            "Cannot write to gurobipy",
        ):
            results = SolverFactory('gurobi_direct_minlp').solve(m)

    def test_unrecognized_component_error(self):
        m = make_model()
        m.disj = Disjunction(expr=[[m.x1 + m.x2 == 3], [m.x1 + m.x2 >= 7]])

        with self.assertRaisesRegex(
            ValueError,
            r"The model \('unknown'\) contains the following active components "
            r"that the Gurobi MINLP writer does not know how to process:"
            + "\n\t"
            + r"\<class 'pyomo.gdp.disjunct.Disjunct'\>:"
            + "\n\t\t"
            + r"disj_disjuncts\[0\]"
            + "\n\t\t"
            + r"disj_disjuncts\[1\]"
            + "\n\t"
            + r"<class 'pyomo.gdp.disjunct.Disjunction'>:"
            + "\n\t\t"
            + "disj",
        ):
            results = SolverFactory('gurobi_direct_minlp').solve(m)
