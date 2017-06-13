import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

import pyomo.opt
from pyomo.util.plugin import Plugin
from pyomo.environ import *
from pyomo.core.base import expr_common, expr as EXPR

solvers = pyomo.opt.check_available_solvers('glpk')


class TestCoopr3(unittest.TestCase):

    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    def test_add_slack_vars(self):
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=model.x <= 5)
        model.rule2 = Constraint(expr=1 <= model.y <= 3)
        model.rule3 = Constraint(expr=model.x >= 0.1)
        model.obj = Objective(expr=-model.x-model.y)
        inst = model
        xinst = TransformationFactory('core.add_slack_variables').create_using(model)

        # should have new block
        xblock = xinst.component("_core_add_slack_variables")
        self.assertIsInstance(xblock, Block)
        # should have new variables on new block
        self.assertTrue(hasattr(xblock, "_slack_minus_rule1"))
        self.assertFalse(hasattr(xblock, "_slack_plus_rule1"))
        self.assertTrue(hasattr(xblock, "_slack_minus_rule2"))
        self.assertTrue(hasattr(xblock, "_slack_plus_rule2"))
        self.assertFalse(hasattr(xblock, "_slack_minus_rule3"))
        self.assertTrue(hasattr(xblock, "_slack_plus_rule3"))
        # all new variables in non-negative reals
        self.assertEqual(xblock._slack_minus_rule1.bounds, (0, None))
        self.assertEqual(xblock._slack_minus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule3.bounds, (0, None))
        # should have modified constraints
        # check all original variables still there:
        oldCons = model.rule1
        xCons = xinst.rule1
        print inst.rule1.body
        print xinst.rule1.body._args[0]
        # TODO: I really think this should be true...
        self.assertIs(inst.rule1.body, xinst.rule1.body._args[0])
        # print xCons.expr
        # print type(instance.rule1.body)
        # for i, arg in enumerate(instance.rule1.body._args):
        #     self.assertIs(arg, xinst.rule1.body._args[i])
        #     self.assertEqual(instance.rule1.body._coef[i], xinst.rule1.body._coef[i])
        self.assertIs(xblock._slack_minus_rule1, xinst.rule1.body._args[-1])
        self.assertEqual(xinst.rule1.body._coef[-1], -1)
        # should have old objective deactivated.
        self.assertFalse(xinst.obj.active)
        # active objective should minimize sum of slacks
        self.assertTrue(hasattr(xblock, "_slack_objective"))
        self.assertTrue(xblock._slack_objective.active)
        # TODO: check that the objective is actually right when I know how to check an expression.

    def test_add_slack_vars_badModel(self):
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=6 <= model.x <= 5)
        instance = model
        xfrm = TransformationFactory('core.add_slack_variables')
        with self.assertRaises(RuntimeError) as err:
            xinst = xfrm.create_using(instance)
        self.assertEqual(err.exception.message, "Lower bound exceeds upper bound in constraint rule1")
