import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

from six import StringIO
from pyomo.util.log import LoggingIntercept

import pyutilib.th as unittest
import pyutilib.services
import random

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
        # set seed so we can test name collisions predictably
        random.seed(666)

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

    def test_trans_block_name_collision(self):
        m = self.makeModel()
        m._core_add_slack_variables = Block()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        xblock = m.component("_core_add_slack_variables_4")
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

    # wrapping this as a method because I use it again later when I test
    # targets
    def checkRule1(self, m):
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
        
    def checkRule3(self, m):
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

    def test_ub_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule1(m)

    def test_lb_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule3(m)
        
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
        self.assertRaisesRegexp(
            RuntimeError, 
            "Lower bound exceeds upper bound in constraint rule1*", 
            TransformationFactory('core.add_slack_variables').apply_to, 
            model)

    def test_leave_deactivated_constraints(self):
        m = self.makeModel()
        m.rule2.deactivate()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        
        cons = m.rule2
        self.assertFalse(cons.active)
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        # cons.body is a SimpleVar
        self.assertIs(cons.body, m.y)

    def checkTargetSlackVars(self, transBlock):
        self.assertIsInstance(transBlock.component("_slack_minus_rule1"), Var)
        self.assertFalse(hasattr(transBlock, "_slack_plus_rule1"))
        self.assertIsNone(transBlock.component("_slack_minus_rule2"))
        self.assertIsNone(transBlock.component("_slack_plus_rule2"))
        self.assertFalse(hasattr(transBlock, "_slack_minus_rule3"))
        self.assertIsInstance(transBlock.component("_slack_plus_rule3"), Var)

    def test_only_targets_have_slack_vars(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])
        
        transBlock = m.component("_core_add_slack_variables")
        # check that we only made slack vars for targets
        self.checkTargetSlackVars(transBlock)

    def test_only_targets_have_slack_vars_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        transBlock = m2.component("_core_add_slack_variables")
        # check that we only made slack vars for targets
        self.checkTargetSlackVars(transBlock)
        
    def checkNonTargetCons(self, m):
        cons = m.rule2
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        # cons.body is a SimpleVar
        self.assertIs(cons.body, m.y)

    def test_nontarget_constraint_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])
        
        self.checkNonTargetCons(m)

    def test_nontarget_constraint_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])
        
        self.checkNonTargetCons(m2)

    def test_target_constraints_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.checkRule1(m)
        self.checkRule3(m)

    def test_target_constraints_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.checkRule1(m2)
        self.checkRule3(m2)

    def checkTargetsObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertEqual(len(obj.expr._args), 2)
        self.assertEqual(len(obj.expr._coef), 2)
        self.assertIs(obj.expr._args[0], transBlock._slack_minus_rule1)
        self.assertEqual(obj.expr._coef[0], 1)
        self.assertIs(obj.expr._args[1], transBlock._slack_plus_rule3)
        self.assertEqual(obj.expr._coef[1], 1)
        self.assertEqual(obj.expr._const, 0)

    def test_target_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.assertFalse(m.obj.active)
        self.checkTargetsObj(m)

    def test_target_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.assertFalse(m2.obj.active)
        self.checkTargetsObj(m2)

    def test_err_for_bogus_kwds(self):
        m = self.makeModel()
        OUTPUT = StringIO()
        with LoggingIntercept(OUTPUT, 'pyomo.core'):
            TransformationFactory('core.add_slack_variables').apply_to(
                m,
                notakwd="I want a feasible model")
        self.assertIn("Unrecognized keyword arguments in add slack variable "
                      "transformation:\nnotakwd",
                      OUTPUT.getvalue())

    def test_transformed_constraints_sumexpression_body(self):
        m = self.makeModel()
        m.rule4 = Constraint(expr=5 <= m.x - 2*m.y <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule4)])

        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertEqual(c.lower, 5)
        self.assertEqual(c.upper, 9)
        self.assertEqual(len(c.body._args), 4)
        self.assertEqual(len(c.body._coef), 4)
        self.assertIs(c.body._args[0], m.x)
        self.assertIs(c.body._args[1], m.y)
        self.assertIs(c.body._args[2], transBlock._slack_plus_rule4)
        self.assertIs(c.body._args[3], transBlock._slack_minus_rule4)
        self.assertEqual(c.body._coef[0], 1)
        self.assertEqual(c.body._coef[1], -2)
        self.assertEqual(c.body._coef[2], 1)
        self.assertEqual(c.body._coef[3], -1)

    def test_transformed_constraint_scalar_body(self):
        m = self.makeModel()
        m.p = Param(initialize=6)
        m.rule4 = Constraint(expr=m.p <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule4)])
        
        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertIsNone(c.lower)
        self.assertEqual(c.upper, 9)
        self.assertEqual(len(c.body._args), 1)
        self.assertEqual(len(c.body._coef), 1)
        self.assertEqual(c.body._const, 6)
        self.assertIs(c.body._args[0], transBlock._slack_minus_rule4)
        self.assertEqual(c.body._coef[0], -1)
       

# TODO: QUESTION: Is this bad? I just copied all of the tests that involve 
# expressions and changed what I needed to. But the only thing different is the
# coefficients as dictionaries rather than constraints... So... the code is 
# nearly identical... 
class TestAddSlacks_pyomo4(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.pyomo4_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        return TestAddSlacks_coopr3.makeModel()

    # wrapping this as a method because I use it again later when I test
    # targets
    def checkRule1(self, m):
        # check all original variables still there:
        cons = m.rule1
        transBlock = m.component("_core_add_slack_variables")
        
        self.assertIsNone(cons.lower)
        self.assertEqual(cons.upper, 5)
        
        self.assertIs(cons.body._args[0], m.x)
        self.assertEqual(cons.body._coef[id(cons.body._args[0])], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_minus_rule1)
        self.assertEqual(cons.body._coef[id(cons.body._args[1])], -1)
        
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertEqual(cons.body._const, 0)
        
    def checkRule3(self, m):
        # check all original variables still there:
        cons = m.rule3
        transBlock = m.component("_core_add_slack_variables")

        self.assertIsNone(cons.upper)
        self.assertEqual(cons.lower, 0.1)
        
        self.assertIs(cons.body._args[0], m.x)
        self.assertEqual(cons.body._coef[id(cons.body._args[0])], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_plus_rule3)
        self.assertEqual(cons.body._coef[id(cons.body._args[1])], 1)
        
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertEqual(cons.body._const, 0)

    def test_ub_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule1(m)

    def test_lb_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        self.checkRule3(m)
        
    def test_both_bounds_constraint_modified(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(m)

        # check all original variables still there:
        cons = m.rule2
        transBlock = m.component("_core_add_slack_variables")
        
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)

        self.assertIs(cons.body._args[0], m.y)
        self.assertEqual(cons.body._coef[id(cons.body._args[0])], 1)
        self.assertIs(cons.body._args[1], transBlock._slack_plus_rule2)
        self.assertEqual(cons.body._coef[id(cons.body._args[1])], 1)
        self.assertIs(cons.body._args[2], transBlock._slack_minus_rule2)
        self.assertEqual(cons.body._coef[id(cons.body._args[2])], -1)
        
        self.assertEqual(len(cons.body._args), 3)
        self.assertEqual(len(cons.body._coef), 3)
        self.assertEqual(cons.body._const, 0)

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
            self.assertEqual(obj.expr._coef[id(obj.expr._args[i])], 1)
        self.assertEqual(obj.expr._const, 0)
        self.assertEqual(len(obj.expr._args), 4)
        self.assertEqual(len(obj.expr._coef), 4)

    def test_leave_deactivated_constraints(self):
        m = self.makeModel()
        m.rule2.deactivate()
        TransformationFactory('core.add_slack_variables').apply_to(m)
        
        cons = m.rule2
        self.assertFalse(cons.active)
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        self.assertIs(cons.body, m.y)

    def test_nontarget_constraint_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])
        
        cons = m.rule2
        self.assertEqual(cons.lower, 1)
        self.assertEqual(cons.upper, 3)
        # cons.body is a SimpleVar
        self.assertIs(cons.body, m.y)

    def test_target_constraints_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.checkRule1(m)
        self.checkRule3(m)

    def test_target_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m, 
            targets=[ComponentUID(m.rule1), ComponentUID(m.rule3)])

        self.assertFalse(m.obj.active)

        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertEqual(len(obj.expr._args), 2)
        self.assertEqual(len(obj.expr._coef), 2)
        self.assertIs(obj.expr._args[0], transBlock._slack_minus_rule1)
        self.assertEqual(obj.expr._coef[id(obj.expr._args[0])], 1)
        self.assertIs(obj.expr._args[1], transBlock._slack_plus_rule3)
        self.assertEqual(obj.expr._coef[id(obj.expr._args[1])], 1)
        self.assertEqual(obj.expr._const, 0)

    def test_transformed_constraints_linearexpression_body(self):
        m = self.makeModel()
        m.rule4 = Constraint(expr=5 <= m.x - 2*m.y <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule4)])

        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertEqual(c.lower, 5)
        self.assertEqual(c.upper, 9)
        self.assertEqual(len(c.body._args), 4)
        self.assertEqual(len(c.body._coef), 4)
        self.assertIs(c.body._args[0], m.x)
        self.assertIs(c.body._args[1], m.y)
        self.assertIs(c.body._args[2], transBlock._slack_plus_rule4)
        self.assertIs(c.body._args[3], transBlock._slack_minus_rule4)
        self.assertEqual(c.body._coef[id(c.body._args[0])], 1)
        self.assertEqual(c.body._coef[id(c.body._args[1])], -2)
        self.assertEqual(c.body._coef[id(c.body._args[2])], 1)
        self.assertEqual(c.body._coef[id(c.body._args[3])], -1)

    def test_transformed_constraints_sumexpression_body(self):
        m = self.makeModel()
        m.rule4 = Constraint(expr=5 <= m.x**2 - 2*m.y <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule4)])

        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertEqual(c.lower, 5)
        self.assertEqual(c.upper, 9)
        self.assertEqual(len(c.body._args), 4)
        # this is a PowExpression now
        self.assertIs(c.body._args[0]._args[0], m.x)
        self.assertEqual(c.body._args[0]._args[1], 2)
        # this is a linear expression
        term2 = c.body._args[1]
        self.assertEqual(len(term2._args), 1)
        self.assertEqual(len(term2._coef), 1)
        self.assertIs(term2._args[0], m.y)
        self.assertEqual(term2._coef[id(term2._args[0])], -2)
        # this is just a variable
        self.assertIs(c.body._args[2], transBlock._slack_plus_rule4)
        # this is a linear expression
        term4 = c.body._args[3]
        self.assertEqual(len(term4._args), 1)
        self.assertEqual(len(term4._coef), 1)
        self.assertIs(term4._args[0], transBlock._slack_minus_rule4)
        self.assertEqual(term4._coef[id(term4._args[0])], -1)

    def test_transformed_constraint_scalar_body(self):
        m = self.makeModel()
        m.p = Param(initialize=6)
        m.rule4 = Constraint(expr=m.p <= 9)
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule4)])
        
        transBlock = m._core_add_slack_variables
        c = m.rule4
        self.assertIsNone(c.lower)
        self.assertEqual(c.upper, 9)
        self.assertEqual(len(c.body._args), 1)
        self.assertEqual(len(c.body._coef), 1)
        self.assertEqual(c.body._const, 6)
        self.assertIs(c.body._args[0], transBlock._slack_minus_rule4)
        self.assertEqual(c.body._coef[id(c.body._args[0])], -1)


class TestAddSlacks_IndexedConstraints_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.S = Set(initialize=[1,2,3])
        m.x = Var(m.S)
        m.y = Var()
        def rule1_rule(m, s):
            return 2*m.x[s] >= 4
        m.rule1 = Constraint(m.S, rule=rule1_rule)
        m.rule2 = Constraint(expr=m.y <= 6)
        m.obj = Objective(expr=sum(m.x[s] for s in m.S) - m.y)
        return m
    
    def checkSlackVars_indexedtarget(self, transBlock):
        self.assertIsInstance(transBlock.component("_slack_plus_rule1[1]"), Var)
        self.assertIsInstance(transBlock.component("_slack_plus_rule1[2]"), Var)
        self.assertIsInstance(transBlock.component("_slack_plus_rule1[3]"), Var)
        self.assertIsNone(transBlock.component("_slack_minus_rule2"))

    def test_indexedtarget_only_create_slackvars_for_targets(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])

        transBlock = m.component("_core_add_slack_variables")
        # TODO: So, right now indexed constraints don't result in indexed
        # slack variables. They could... But I don't know if it matters much?
        # They are named sensibly either way... Dunno.
        self.checkSlackVars_indexedtarget(transBlock)
    
    def test_indexedtarget_only_create_slackvars_for_targets_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1)])

        transBlock = m2.component("_core_add_slack_variables")
        self.checkSlackVars_indexedtarget(transBlock)
        
    def checkRule2(self, m):
        cons = m.rule2
        self.assertEqual(cons.upper, 6)
        self.assertIsNone(cons.lower)
        self.assertIs(cons.body, m.y)
        
    def test_indexedtarget_nontarget_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])

        self.checkRule2(m)

    def test_indexedtarget_nontarget_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1)])

        self.checkRule2(m2)

    def checkTargetObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertIsInstance(obj, Objective)
        self.assertEqual(len(obj.expr._args), 3)
        self.assertEqual(len(obj.expr._coef), 3)
        self.assertEqual(obj.expr._const, 0)
        self.assertIs(obj.expr._args[0], 
                      transBlock.component("_slack_plus_rule1[1]"))
        self.assertIs(obj.expr._args[1], 
                      transBlock.component("_slack_plus_rule1[2]"))
        self.assertIs(obj.expr._args[2], 
                      transBlock.component("_slack_plus_rule1[3]"))
        self.assertIs(obj.expr._coef[0], 1) 
        self.assertIs(obj.expr._coef[1], 1) 
        self.assertIs(obj.expr._coef[2], 1) 

    def test_indexedtarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])
        
        self.assertFalse(m.obj.active)
        self.checkTargetObj(m)

    def test_indexedtarget_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1)])
        
        self.assertFalse(m2.obj.active)
        self.checkTargetObj(m2)
        
    def checkTransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        self.assertEqual(len(c.body._args), 2)
        self.assertEqual(len(c.body._coef), 2)
        self.assertIs(c.body._args[0], m.x[i])
        self.assertIs(
            c.body._args[1], 
            m._core_add_slack_variables.component(
                "_slack_plus_rule1[%s]" % i))
        self.assertEqual(c.body._coef[0], 2)
        self.assertEqual(c.body._coef[1], 1)
        self.assertEqual(c.body._const, 0)

    def test_indexedtarget_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])
        
        for i in [1,2,3]:
            self.checkTransformedRule1(m, i)

    def test_indexedtarget_targets_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1)])
        
        for i in [1,2,3]:
            self.checkTransformedRule1(m2, i)

    def checkSlackVars_constraintDataTarget(self, transBlock):
        self.assertIsInstance(transBlock.component("_slack_plus_rule1[2]"), Var)
        self.assertIsNone(transBlock.component("_slack_plus_rule1[1]"))
        self.assertIsNone(transBlock.component("_slack_plus_rule1[3]"))
        self.assertIsNone(transBlock.component("_slack_minus_rule2"))

    def test_ConstraintDatatarget_only_add_slackvars_for_targets(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])

        transBlock = m._core_add_slack_variables
        self.checkSlackVars_constraintDataTarget(transBlock)

    def test_ConstraintDatatarget_only_add_slackvars_for_targets_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1[2])])

        transBlock = m2._core_add_slack_variables
        self.checkSlackVars_constraintDataTarget(transBlock)

    def checkUntransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        self.assertEqual(len(c.body._numerator), 1)
        self.assertEqual(len(c.body._denominator), 0)
        self.assertIs(c.body._numerator[0], m.x[i])
        self.assertEqual(c.body._coef, 2)

    def test_ConstraintDatatarget_nontargets_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])
        
        self.checkUntransformedRule1(m, 1)
        self.checkUntransformedRule1(m, 3)
        self.checkRule2(m)

    def test_ConstraintDatatarget_nontargets_same_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1[2])])
        
        self.checkUntransformedRule1(m2, 1)
        self.checkUntransformedRule1(m2, 3)
        self.checkRule2(m2)

    def test_ConstraintDatatarget_target_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.checkTransformedRule1(m, 2)

    def test_ConstraintDatatarget_target_transformed_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.checkTransformedRule1(m2, 2)

    def checkConstraintDataObj(self, m):
        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertIsInstance(obj, Objective)
        self.assertIs(obj.expr, transBlock.component("_slack_plus_rule1[2]"))

    def test_ConstraintDatatarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.assertFalse(m.obj.active)
        self.checkConstraintDataObj(m)

    def test_ConstraintDatatarget_objective_create_using(self):
        m = self.makeModel()
        m2 = TransformationFactory('core.add_slack_variables').create_using(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.assertFalse(m2.obj.active)
        self.checkConstraintDataObj(m2)


# TODO: QUESTION: same as above here...
class TestAddSlacks_IndexedConstraints_pyomo4(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.pyomo4_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        return TestAddSlacks_IndexedConstraints_coopr3.makeModel()

    def checkRule2(self, m):
        cons = m.rule2
        self.assertEqual(cons.upper, 6)
        self.assertIsNone(cons.lower)
        self.assertIs(cons.body, m.y)
        
    def test_indexedtarget_nontarget_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])

        self.checkRule2(m)

    def test_indexedtarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])
        
        self.assertFalse(m.obj.active)
        
        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertIsInstance(obj, Objective)
        self.assertEqual(len(obj.expr._args), 3)
        self.assertEqual(len(obj.expr._coef), 3)
        self.assertEqual(obj.expr._const, 0)
        self.assertIs(obj.expr._args[0], 
                      transBlock.component("_slack_plus_rule1[1]"))
        self.assertIs(obj.expr._args[1], 
                      transBlock.component("_slack_plus_rule1[2]"))
        self.assertIs(obj.expr._args[2], 
                      transBlock.component("_slack_plus_rule1[3]"))
        self.assertIs(obj.expr._coef[id(obj.expr._args[0])], 1) 
        self.assertIs(obj.expr._coef[id(obj.expr._args[1])], 1) 
        self.assertIs(obj.expr._coef[id(obj.expr._args[2])], 1) 

    def checkTransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        self.assertEqual(len(c.body._args), 2)
        self.assertEqual(len(c.body._coef), 2)
        self.assertIs(c.body._args[0], m.x[i])
        self.assertIs(
            c.body._args[1], 
            m._core_add_slack_variables.component(
                "_slack_plus_rule1[%s]" % i))
        self.assertEqual(c.body._coef[id(c.body._args[0])], 2)
        self.assertEqual(c.body._coef[id(c.body._args[1])], 1)
        self.assertEqual(c.body._const, 0)

    def test_indexedtarget_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1)])
        
        for i in [1,2,3]:
            self.checkTransformedRule1(m, i)

    def checkUntransformedRule1(self, m, i):
        c = m.rule1[i]
        self.assertEqual(c.lower, 4)
        self.assertIsNone(c.upper)
        self.assertEqual(len(c.body._args), 1)
        self.assertEqual(len(c.body._coef), 1)
        self.assertIs(c.body._args[0], m.x[i])
        self.assertEqual(c.body._coef[id(c.body._args[0])], 2)

    def test_ConstraintDatatarget_nontargets_same(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])
        
        self.checkUntransformedRule1(m, 1)
        self.checkUntransformedRule1(m, 3)
        self.checkRule2(m)

    def test_ConstraintDatatarget_target_transformed(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.checkTransformedRule1(m, 2)

    def test_ConstraintDatatarget_objective(self):
        m = self.makeModel()
        TransformationFactory('core.add_slack_variables').apply_to(
            m,
            targets=[ComponentUID(m.rule1[2])])

        self.assertFalse(m.obj.active)
        
        transBlock = m._core_add_slack_variables
        obj = transBlock.component("_slack_objective")
        self.assertIsInstance(obj, Objective)
        self.assertIs(obj.expr, transBlock.component("_slack_plus_rule1[2]"))
