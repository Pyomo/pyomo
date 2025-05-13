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

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
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
    Reals,
    Var,
)
from pyomo.opt import WriterFactory
from pyomo.contrib.gurobi_minlp.repn.gurobi_direct_minlp import GurobiMINLPVisitor
from pyomo.contrib.gurobi_minlp.tests.test_gurobi_minlp_walker import CommonTest

## DEBUG
from pytest import set_trace

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

        grb_model, var_map = WriterFactory('gurobi_minlp').write(
            m, symbolic_solver_labels=True
        )

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
        self.assertEqual(len(opcode), 2) # two nodes in the expression tree
        self.assertEqual(opcode[0], GRB.OPCODE_LOG)
        # log has no data
        self.assertEqual(parent[0], -1) # it's the root
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)
        self.assertIs(data[1], x1)
        self.assertEqual(parent[1], 0)
        
        # we can check that res_var is the objective
        self.assertEqual(grb_model.ModelSense, 1) # minimizing
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
        self.assertEqual(parent[0], -1) # root
        self.assertEqual(opcode[1], GRB.OPCODE_VARIABLE)
        self.assertIs(data[1], x3)
        self.assertEqual(parent[1], 0)
        self.assertEqual(opcode[2], GRB.OPCODE_UMINUS) # negation
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


# ESJ: Note: It appears they don't allow x1 ** x2...?  Well, they wait and give the
# error in the solver log, so not sure what we want to do about that?
