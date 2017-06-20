import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random

# DEBUG
from nose.tools import set_trace

class TwoTermDisj_coopr3(unittest.TestCase):
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

    def test_disjunction_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        oldblock = m.component("disjunction")
        self.assertIsInstance(oldblock, Disjunction)
        self.assertFalse(oldblock.active)

    def test_disjunctdatas_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        oldblock = m.component("disjunction")
        self.assertFalse(oldblock.disjuncts[0].active)
        self.assertFalse(oldblock.disjuncts[1].active)

    # TODO: do I need this to be true?? It's not at the moment,but I'm
    # confused.
    def test_disjunct_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        #set_trace()
        self.assertFalse(m.d.active)

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
        # bounds
        self.assertIs(oldc.lower, newc.lower)
        self.assertIs(oldc.upper, newc.upper)
        # body
        self.assertIs(oldc.body, newc.body._args[0])
        self.assertEqual(newc.body._coef[0], 1)
        self.assertEqual(newc.body._coef[1], 3)
        self.assertIs(m.d[0].indicator_var, newc.body._args[1]._args[0])
        self.assertEqual(newc.body._args[1]._coef[0], -1)
        self.assertEqual(newc.body._args[1]._const, 1)
        # and there isn't any more...
        self.assertEqual(len(newc.body._args), 2)
        self.assertEqual(len(newc.body._coef), 2)
        self.assertEqual(len(newc.body._args[1]._args), 1)
        self.assertEqual(len(newc.body._args[1]._coef), 1)
        
        oldc = m.d[1].component("c")
        newc_lo = gdpblock[1].component("c_lo")
        newc_hi = gdpblock[1].component("c_hi")

        self.assertIsInstance(newc_lo, Constraint)
        self.assertIsInstance(newc_hi, Constraint)
        self.assertTrue(newc_lo.active)
        self.assertTrue(newc_hi.active)

        # new constraint is right
        # bounds
        self.assertIs(oldc.lower, newc_lo.lower)
        self.assertIsNone(newc_lo.upper)
        self.assertIsNone(newc_hi.lower)
        self.assertIs(oldc.upper, newc_hi.upper)
        # body
        self.assertIs(oldc.body, newc_lo.body._args[0])
        self.assertEqual(newc_lo.body._coef[0], 1)
        self.assertEqual(newc_lo.body._coef[1], -2)
        self.assertIs(newc_lo.body._args[1]._args[0], m.d[1].indicator_var)
        self.assertEqual(newc_lo.body._args[1]._coef[0], -1)
        self.assertEqual(newc_lo.body._args[1]._const, 1)
        
        self.assertEqual(len(newc_lo.body._args), 2)
        self.assertEqual(len(newc_lo.body._coef), 2)
        self.assertEqual(len(newc_lo.body._args[1]._args), 1)
        self.assertEqual(len(newc_lo.body._args[1]._coef), 1)
        
        self.assertIs(oldc.body, newc_hi.body._args[0])
        self.assertEqual(newc_hi.body._coef[0], 1)
        self.assertEqual(newc_hi.body._coef[1], -7)
        self.assertIs(m.d[1].indicator_var, newc_hi.body._args[1]._args[0])
        self.assertEqual(newc_hi.body._args[1]._coef[0], -1)
        self.assertEqual(newc_hi.body._args[1]._const, 1)

        self.assertEqual(len(newc_hi.body._args), 2)
        self.assertEqual(len(newc_hi.body._coef), 2)
        self.assertEqual(len(newc_hi.body._args[1]._args), 1)
        self.assertEqual(len(newc_hi.body._args[1]._coef), 1)


class TwoTermIndexedDisj_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)
    
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.s = Set(initialize=[1, 2, 3])
        m.t = Set(initialize=['A','B'])
        m.a = Var(m.s, m.t, bounds=(2,7))
        def d_rule(disjunct, flag, s, t):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a[s, t] == 0)
            else:
                disjunct.c = Constraint(expr=m.a[s, t] >= 5)
        m.disjunct = Disjunct([0,1], m.s, m.t, rule=d_rule)
        def disj_rule(m, s, t):
            return [m.disjunct[0, s, t], m.disjunct[1, s, t]]
        m.disjunction = Disjunction(m.s, m.t, rule=disj_rule)
        return m

    def test_xor_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor = m.component("_pyomo_gdp_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        for i in m.disjunction.index_set():
            self.assertEqual(xor[i].body._coef[0], 1)
            self.assertEqual(xor[i].body._coef[1], 1)
            self.assertIs(xor[i].body._args[0], 
                          m.disjunction[i].disjuncts[0].indicator_var)
            self.assertIs(xor[i].body._args[1], 
                          m.disjunction[i].disjuncts[1].indicator_var)
            self.assertEqual(xor[i].body._const, 0)
            self.assertEqual(xor[i].lower, 1)
            self.assertEqual(xor[i].upper, 1)
            self.assertEqual(len(xor[i].body._coef), 2)
            self.assertEqual(len(xor[i].body._args), 2)
    
    def test_deactivated_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        for i in m.disjunct.index_set():
            self.assertFalse(m.disjunct[i].c.active)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        # TODO
        #set_trace()
