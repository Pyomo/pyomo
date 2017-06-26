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

# DEBUG
from nose.tools import set_trace


class TestAddSlacks_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=model.x <= 5)
        model.rule2 = Constraint(expr=1 <= model.y <= 3)
        model.rule3 = Constraint(expr=model.x >= 0.1)
        model.obj = Objective(expr=-model.x-model.y)
        return model

    def test_add_trans_block(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        xblock = m.component("_core_add_slack_variables")
        self.assertIsInstance(xblock, Block)

    def test_slack_vars_added(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        xblock = m.component("_core_add_slack_variables")
        
        # should have new variables on new block
        self.assertIsInstance(xblock.component("_slack_minus_rule1"), Var)
        self.assertFalse(hasattr(xblock, "_slack_plus_rule1"))
        self.assertIsInstance(xblock.component("_slack_minus_rule2"), Var)
        self.assertIsInstance(xblock.component("_slack_plus_rule2"), Var)
        self.assertFalse(hasattr(xblock, "_slack_minus_rule3"))
        self.assertIsInstance(xblock.component("_slack_plus_rule3"), Var)
        # all new variables in non-negative reals
        self.assertEqual(xblock._slack_minus_rule1.bounds, (0, None))
        self.assertEqual(xblock._slack_minus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule2.bounds, (0, None))
        self.assertEqual(xblock._slack_plus_rule3.bounds, (0, None))

    def test_ub_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        # check all original variables still there:
        cons = m.rule1
        transBlock = m.component("_core_add_slack_variables")
        
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 5)
        
        self.assertIs(cons.body._args[0], m.x)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_minus_rule1)
        self.assertEqual(cons.body._coef[1], -1)
        
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertEqual(cons.body._const, 0)

    def test_lb_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        # check all original variables still there:
        cons = m.rule3
        transBlock = m.component("_core_add_slack_variables")

        self.assertIsNone(cons.upper)
        self.assertEqual(cons.lower, 0.1)
        
        self.assertIs(cons.body._args[0], m.x)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_plus_rule3)
        self.assertEqual(cons.body._coef[1], 1)
        
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertEqual(cons.body._const, 0)

    def test_both_bounds_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        # check all original variables still there:
        cons = m.rule2
        transBlock = m.component("_core_add_slack_variables")
        
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)

        self.assertIs(cons.body._args[0], m.y)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_plus_rule2)
        self.assertEqual(cons.body._coef[1], 1)
        self.assertIs(cons.body._args[2], transBlock._slack_minus_rule2)
        self.assertEqual(cons.body._coef[2], -1)
        
        self.assertEqual(len(cons.body._args), 3)
        self.assertEqual(len(cons.body._coef), 3)
        self.assertEqual(cons.body._const, 0)

    def test_obj_deactivated(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        # should have old objective deactivated.
        self.assertFalse(m.obj.active)

    def test_new_obj_created(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        
        transBlock = m.component("_core_add_slack_variables")

        # active objective should minimize sum of slacks
        obj = transBlock.component("_slack_objective")
        self.assertIsInstance(obj, Objective)
        self.assertTrue(obj.active)
        
        self.assertIs(obj.expr._args[0], transBlock._slack_minus_rule1)
        self.assertIs(obj.expr._args[1], transBlock._slack_plus_rule2)
        self.assertIs(obj.expr._args[2], transBlock._slack_minus_rule2)
        self.assertIs(obj.expr._args[3], transBlock._slack_plus_rule3)
        for i in range(0, 4):
            self.assertEqual(obj.expr._coef[i], 1)
        self.assertEqual(obj.expr._const, 0)
        self.assertEqual(len(obj.expr._args), 4)
        self.assertEqual(len(obj.expr._coef), 4)

    def test_badModel_err(self):
        model = ConcreteModel()
        model.x = Var(within=NonNegativeReals)
        model.rule1 = Constraint(expr=6 <= model.x <= 5)
        with self.assertRaises(RuntimeError) as err:
            TransformationFactory('core.add_slack_variables').apply_to(model)
        # TODO: test this with reg expr.
        self.assertEqual(err.exception.message, "Lower bound exceeds upper bound in constraint rule1")

    def test_leave_deactivated_constraints(self):
        m = self.makeModel()
        m.rule2.deactivate()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        
        cons = m.rule2
        self.assertFalse(cons.active)
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        self.assertIs(cons.body._args[0], m.y)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertEqual(len(cons.body._args), 1)
        self.assertEqual(len(cons.body._coef), 1)
        self.assertEqual(cons.body._const, 0)

    def test_Constraint_targets(self):
        m = self.makeModel()
        m.rule2.deactivate()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        set_trace()
        # TODO

    def test_IndexedConstraint_targets(self):
        pass

    def test_ConstraintData_targets(self):
        pass

    def test_err_for_bogus_kwds(self):
        pass
