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
        m.disj = Disjunction(rule=disj_rule)
        return m
    
    def test_new_transformation_block(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        # we have a transformation block
        gdpblock = m.component("_pyomo_gdp_relaxation")
        self.assertIsInstance(gdpblock, Block)
        # it has the disjunct on it
        self.assertIsInstance(m._pyomo_gdp_relaxation.component("d"), Block)

    def test_new_trans_block_nameCollision(self):
        m = self.makeModel()
        m._pyomo_gdp_relaxation = Block()
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m.component("_pyomo_gdp_relaxation4")
        self.assertIsInstance(gdpblock, Block)
        # it has the disjunct on it
        self.assertIsInstance(m._pyomo_gdp_relaxation.component("d"), Block)

    def test_still_have_indVars(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        dblock = m._gdp_relax.component("d")
        self.assertIsInstance(dblock[0].indicator_var, Var)
        self.assertTrue(dblock[0].indicator_var.is_binary())
        self.assertIsInstance(dblock[1].indicator_var, Var)
        self.assertTrue(dblock[1].indicator_var.is_binary())
    
    def test_deactivated_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        dblock = m._gdp_relax.component("d")

        # old constraint there, deactivated, new constraint active
        oldc = dblock[0].component("c")
        self.assertIsInstance(oldc, Constraint)
        self.assertFalse(oldc.active)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        dblock = m._gdp_relax.component("d")

        oldc = dblock[0].component("c")
        newc = dblock[0].component("c_eq")
        self.assertIsInstance(newc, Constraint)
        self.assertTrue(newc.active)

        # new constraint is right
        self.assertIs(oldc.lower, newc.lower)
        self.assertIs(oldc.upper, newc.upper)
        self.assertIs(oldc.body, newc.body._args[0])
        self.assertIs(dblock[0].indicator_var, newc.body._args[1]._args[0])
        set_trace()
        
