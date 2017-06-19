import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random

# DEBUG
from nose.tools import set_trace

class TestBigM_2TermDisj_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)
    
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        # m.BigM = Suffix(direction=Suffix.LOCAL)
        # m.BigM[None] = 78
        m.a = Var(bounds=(2,7))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a == 0)
            else:
                disjunct.c = Constraint(expr=m.a >= 5)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        m.disjunction = Disjunction(rule=disj_rule)
        return m
    
    def test_new_block_created(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        # we have a transformation block
        gdpblock = m.component("_pyomo_gdp_relaxation")
        self.assertIsInstance(gdpblock, Block)
        # it has the disjunct on it
        self.assertIsInstance(m._pyomo_gdp_relaxation.component("d"), Block)

    def test_old_block_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        oldblock = m.component("disjunction")
        self.assertIsInstance(oldblock, Disjunction)
        self.assertFalse(oldblock.active)

    def test_new_block_nameCollision(self):
        m = self.makeModel()
        m._pyomo_gdp_relaxation = Block()
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m.component("_pyomo_gdp_relaxation4")
        self.assertIsInstance(gdpblock, Block)
        # it has the disjunct on it
        self.assertIsInstance(m._pyomo_gdp_relaxation4.component("d"), Block)

    def test_indicator_vars(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        oldblock = m.component("d")
        # have indicator variables
        self.assertIsInstance(oldblock[0].indicator_var, Var)
        self.assertTrue(oldblock[0].indicator_var.is_binary())
        self.assertIsInstance(oldblock[1].indicator_var, Var)
        self.assertTrue(oldblock[1].indicator_var.is_binary())

    def test_xor_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        xor = m.component("_pyomo_gdp_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertIs(m.d[0].indicator_var, xor.body._args[0])
        self.assertIs(m.d[1].indicator_var, xor.body._args[1])
        self.assertEqual(1, xor.body._coef[0])
        self.assertEqual(1, xor.body._coef[1])
        self.assertEqual(xor.lower, 1)
        self.assertEqual(xor.upper, 1)

    def test_or_constraints(self):
        m = ConcreteModel()
        # m.BigM = Suffix(direction=Suffix.LOCAL)
        # m.BigM[None] = 78
        m.a = Var(bounds=(2,7))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a == 0)
            else:
                disjunct.c = Constraint(expr=m.a >= 5)
        m.d = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.d[0], m.d[1]]
        # ask for an or constraint
        m.disjunction = Disjunction(rule=disj_rule, xor=False)
        TransformationFactory('gdp.bigm').apply_to(m)
        orcons = m.component("_pyomo_gdp_relaxation_disjunction_or")
        self.assertIsInstance(orcons, Constraint)
        self.assertIs(m.d[0].indicator_var, orcons.body._args[0])
        self.assertIs(m.d[1].indicator_var, orcons.body._args[1])
        self.assertEqual(1, orcons.body._coef[0])
        self.assertEqual(1, orcons.body._coef[1])
        self.assertEqual(orcons.lower, 1)
        self.assertIsNone(orcons.upper)
    
    def test_deactivated_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        oldblock = m.component("d")
        # old constraint there, deactivated
        for i in [0,1]:
            oldc = oldblock[i].component("c")
            self.assertIsInstance(oldc, Constraint)
            self.assertFalse(oldc.active)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m._pyomo_gdp_relaxation.component("d")

        oldc = m.d[0].component("c")
        newc = gdpblock[0].component("c_eq")
        self.assertIsInstance(newc, Constraint)
        self.assertTrue(newc.active)

        # new constraint is right
        self.assertIs(oldc.lower, newc.lower)
        self.assertIs(oldc.upper, newc.upper)
        self.assertIs(oldc.body, newc.body._args[0])
        self.assertIs(m.d[0].indicator_var, newc.body._args[1]._args[0])
        # TODO: finish checking everything
        
    def test_indexedDisjunction(self):
        # TODO: I think this is going to belong in another class...
        # since its a new model...
        m = ConcreteModel()
        m.s = Set(initialize=[1, 2, 3])
        m.a = Var(m.s, bounds=(2,7))
        def d_rule(disjunct, flag, s):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a[s] == 0)
            else:
                disjunct.c = Constraint(expr=m.a[s] >= 5)
        m.disjunct = Disjunct([0,1], m.s, rule=d_rule)
        def disj_rule(m, s):
            return [m.disjunct[0, s], m.disjunct[1, s]]
        m.disjunction = Disjunction(m.s, rule=disj_rule)
        
        TransformationFactory('gdp.bigm').apply_to(m)
