import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random
import weakref

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
        self.assertIsInstance(
            m._pyomo_gdp_relaxation[0].component("c"),
            Constraint)

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

    def test_disjunct_weakrefs(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)        
        transblock = m.component("_pyomo_gdp_relaxation")
        oldblock = m.component("d")

        # we should have a dictionary on each _DisjunctData and similarly
        # on each _BlockData of the corresponding disjunct block on the
        # transformation block
        for i in [0,1]:
            infodict = getattr(oldblock[i], "_gdp_trans_info")
            self.assertIsInstance(infodict, dict)
            self.assertIs(infodict['bigm'](), transblock[i])
        
            infodict2 = getattr(transblock[i], "_gdp_trans_info")
            self.assertIsInstance(infodict2, dict)
            self.assertIs(infodict2['src'](), oldblock[i])

    def test_new_block_nameCollision(self):
        m = self.makeModel()
        m._pyomo_gdp_relaxation = Block()
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m.component("_pyomo_gdp_relaxation4")
        self.assertIsInstance(gdpblock, Block)
        # it has the disjunct on it
        self.assertIsInstance(
            m._pyomo_gdp_relaxation4[0].component("c"), 
            Constraint)

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
        gdpblock = m._pyomo_gdp_relaxation

        oldc = m.d[0].component("c")
        # we have an indexed constraint called c.
        newcons = gdpblock[0].component("c")
        self.assertIsInstance(newcons, Constraint)
        self.assertTrue(newcons.active)

        # c[0] is the only constraint we had to use since original had
        # only LB
        newc = newcons[0]
        self.assertTrue(newc.active)

        # test new constraint is right
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
        

        # second constraint
        oldc = m.d[1].component("c")
        newc = gdpblock[1].component("c")
        self.assertIsInstance(newc, Constraint)
        # now we've used both indices since original constraint was equality
        newc_lo = newc[0]
        newc_hi = newc[1]

        self.assertTrue(newc_lo.active)
        self.assertTrue(newc_hi.active)

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
        m.s = Set(initialize=[1, 2])
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

    def test_transformed_block_structure(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m.component("_pyomo_gdp_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock[0].c, Constraint)
        self.assertIsInstance(transBlock[2].c, Constraint)
        self.assertIsInstance(transBlock[4].c, Constraint)
        self.assertIsInstance(transBlock[6].c, Constraint)

        self.assertIsInstance(transBlock[1].c, Constraint)
        self.assertIsInstance(transBlock[3].c, Constraint)
        self.assertIsInstance(transBlock[5].c, Constraint)
        self.assertIsInstance(transBlock[7].c, Constraint)

    def test_disjunct_weakrefs(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_relaxation")
        oldblock = m.component("disjunct")
        
        # let the record show that this was a terrible idea...
        infodict = getattr(transBlock[0], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[0,1,'A'])
        infodict2 = getattr(oldblock[0,1,'A'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[0])

        infodict = getattr(transBlock[1], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[1,1,'A'])
        infodict2 = getattr(oldblock[1,1,'A'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[1])

        infodict = getattr(transBlock[2], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[0,1,'B'])
        infodict2 = getattr(oldblock[0,1,'B'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[2])

        infodict = getattr(transBlock[3], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[1,1,'B'])
        infodict2 = getattr(oldblock[1,1,'B'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[3])

        infodict = getattr(transBlock[4], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[0,2,'A'])
        infodict2 = getattr(oldblock[0,2,'A'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[4])

        infodict = getattr(transBlock[5], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[1,2,'A'])
        infodict2 = getattr(oldblock[1,2,'A'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[5])

        infodict = getattr(transBlock[6], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[0,2,'B'])
        infodict2 = getattr(oldblock[0,2,'B'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[6])

        infodict = getattr(transBlock[7], "_gdp_trans_info")
        self.assertIsInstance(infodict, dict)
        self.assertIs(infodict['src'](), oldblock[1,2,'B'])
        infodict2 = getattr(oldblock[1,2,'B'], "_gdp_trans_info")
        self.assertIsInstance(infodict2, dict)
        self.assertIs(infodict2['bigm'](), transBlock[7])

    def test_deactivated_disjuncts(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        for i in m.disjunct.index_set():
            self.assertFalse(m.disjunct[i].active)

    def test_deactivated_disjunction(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertFalse(m.disjunction.active)
        for i in m.disjunction.index_set():
            self.assertFalse(m.disjunction[i].active)


class DisjOnBlock(unittest.TestCase):
    # when the disjunction is on a block, we want the xor constraint
    # on its parent block, but the transformation block still on the
    # model.
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.b = Block()
        m.a = Var(bounds=(0,5))
        def d_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a<=3)
            else:
                disjunct.c = Constraint(expr=m.a==0)
        m.b.disjunct = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.disjunct[0], m.disjunct[1]]
        m.b.disjunction = Disjunction(rule=disj_rule)
        return m

    def test_xor_constraint_added(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
       
        # TODO: this is an interestingly evil result of doing names
        # the way I am... That I get a period in there. Is that OK?
        self.assertIsInstance(
            m.b.component('_pyomo_gdp_relaxation_b.disjunction_xor'), 
            Constraint)

    def test_trans_block_created(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        self.assertIsInstance(m.component('_pyomo_gdp_relaxation'), Block)
        self.assertFalse(hasattr(m.b, '_pyomo_gdp_relaxation'))


class MultiTermDisj_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.s = Set(initialize=[1, 2])
        m.a = Var(m.s, bounds=(2,7))
        def d_rule(disjunct, flag, s):
            m = disjunct.model()
            if flag==0:
                disjunct.c = Constraint(expr=m.a[s] == 0)
            elif flag==1:
                disjunct.c = Constraint(expr=m.a[s] >= 5)
            else:
                disjunct.c = Constraint(expr=2<=m.a[s] <= 4)
        m.disjunct = Disjunct([0,1,2], m.s, rule=d_rule)
        def disj_rule(m, s):
            return [m.disjunct[0, s], m.disjunct[1, s], m.disjunct[2,s]]
        m.disjunction = Disjunction(m.s, rule=disj_rule)
        return m
    
    def test_xor_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor = m.component("_pyomo_gdp_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertEqual(xor[1].lower, 1)
        self.assertEqual(xor[1].upper, 1)
        self.assertEqual(xor[2].lower, 1)
        self.assertEqual(xor[2].upper, 1)
        self.assertEqual(len(xor[1].body._args), 3)
        self.assertEqual(len(xor[2].body._args), 3)
        self.assertEqual(len(xor[1].body._coef), 3)
        self.assertEqual(len(xor[1].body._coef), 3)
        self.assertEqual(xor[1].body._const, 0)
        self.assertEqual(xor[2].body._const, 0)
 
        self.assertIs(xor[1].body._args[0], m.disjunct[0,1].indicator_var)
        self.assertEqual(xor[1].body._coef[0], 1)
        self.assertIs(xor[1].body._args[1], m.disjunct[1,1].indicator_var)
        self.assertEqual(xor[1].body._coef[1], 1)
        self.assertIs(xor[1].body._args[2], m.disjunct[2,1].indicator_var)
        self.assertEqual(xor[1].body._coef[2], 1)
        
        self.assertIs(xor[2].body._args[0], m.disjunct[0,2].indicator_var)
        self.assertEqual(xor[2].body._coef[0], 1)
        self.assertIs(xor[2].body._args[1], m.disjunct[1,2].indicator_var)
        self.assertEqual(xor[2].body._coef[1], 1)
        self.assertIs(xor[2].body._args[2], m.disjunct[2,2].indicator_var)
        self.assertEqual(xor[2].body._coef[2], 1)

    # TODO: I don't know, is anything else really different?


class IndexedConstraintsInDisj_coopr3(unittest.TestCase):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.s = Set(initialize=[1, 2])
        m.lbs = Param(m.s, initialize={1:2, 2:4})
        m.ubs = Param(m.s, initialize={1:7, 2:6})
        def bounds_rule(m, s):
            return (m.lbs[s], m.ubs[s])
        m.a = Var(m.s, bounds=bounds_rule)
        def d_rule(disjunct, flag):
            m = disjunct.model()
            def true_rule(d, s):
                return m.a[s] == 0
            def false_rule(d, s):
                return m.a[s] >= 5
            if flag:
                disjunct.c = Constraint(m.s, rule=true_rule)
            else:
                disjunct.c = Constraint(m.s, rule=false_rule)
        m.disjunct = Disjunct([0,1], rule=d_rule)
        def disj_rule(m):
            return [m.disjunct[0], m.disjunct[1]]
        m.disjunction = Disjunction(rule=disj_rule)
        return m

    def test_transformed_constraints_on_block(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_relaxation")
        self.assertIsInstance(transBlock, Block)
        
        cons1 = transBlock[0].component("c.1")
        self.assertIsInstance(cons1, Constraint)
        self.assertTrue(cons1.active)
        self.assertTrue(cons1[0].active)
        #self.assertTrue(cons1[1].active)
        # QUESTION: does the second index need to be not active?
        cons2 = transBlock[0].component("c.2")
        self.assertIsInstance(cons2, Constraint)
        self.assertTrue(cons2.active)
        self.assertTrue(cons2[0].active)
        # QUESTION: same as above
        #self.assertFalse(cons2[1].active)

        cons3 = transBlock[1].component("c.1")
        self.assertIsInstance(cons3, Constraint)
        self.assertTrue(cons3.active)
        self.assertTrue(cons3[0].active)
        self.assertTrue(cons3[1].active)

        cons4 = transBlock[1].component("c.2")
        self.assertIsInstance(cons4, Constraint)
        self.assertTrue(cons4.active)
        self.assertTrue(cons4[0].active)
        self.assertTrue(cons4[1].active)


class DisjunctInMultipleDisjunctions(unittest.TestCase):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.a = Var(bounds=(-10,50))

        def d1_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                disjunct.c = Constraint(expr=m.a==0)
            else:
                disjunct.c = Constraint(expr=m.a>=5)
        m.disjunct1 = Disjunct([0,1], rule=d1_rule)

        def d2_rule(disjunct, flag):
            if not flag:
                disjunct.c = Constraint(expr=m.a>=30)
        m.disjunct2 = Disjunct([0,1], rule=d2_rule)

        def disj1_rule(m):
            return [m.disjunct1[0], m.disjunct1[1]]
        m.disjunction1 = Disjunction(rule=disj1_rule)

        def disj2_rule(m):
            return [m.disjunct2[0], m.disjunct1[1]]
        m.disjunction2 = Disjunction(rule=disj2_rule)
        return m

    def test_disjunction1_xor(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        xor1 = m.component("_pyomo_gdp_relaxation_disjunction1_xor")
        self.assertIsInstance(xor1, Constraint)
        self.assertTrue(xor1.active)
        self.assertEqual(xor1.lower, 1)
        self.assertEqual(xor1.upper, 1)

        self.assertEqual(xor1.body._coef[0], 1)
        self.assertEqual(xor1.body._coef[1], 1)
        self.assertEqual(xor1.body._const, 0)
        self.assertIs(xor1.body._args[0], m.disjunct1[0].indicator_var)
        self.assertIs(xor1.body._args[1], m.disjunct1[1].indicator_var)

        self.assertEqual(len(xor1.body._args), 2)
        self.assertEqual(len(xor1.body._coef), 2)

    def test_disjunction2_xor(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        xor2 = m.component("_pyomo_gdp_relaxation_disjunction2_xor")
        self.assertIsInstance(xor2, Constraint)
        self.assertTrue(xor2.active)
        self.assertEqual(xor2.lower, 1)
        self.assertEqual(xor2.upper, 1)

        self.assertEqual(xor2.body._coef[0], 1)
        self.assertEqual(xor2.body._coef[1], 1)
        self.assertEqual(xor2.body._const, 0)
        self.assertIs(xor2.body._args[0], m.disjunct2[0].indicator_var)
        self.assertIs(xor2.body._args[1], m.disjunct1[1].indicator_var)

        self.assertEqual(len(xor2.body._args), 2)
        self.assertEqual(len(xor2.body._coef), 2)

    def test_constraints_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertFalse(m.disjunct1[0].c.active)
        self.assertFalse(m.disjunct1[1].c.active)
        self.assertFalse(m.disjunct2[0].c.active)
    
    def test_transformation_block_structure(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_relaxation")
        self.assertIsInstance(transBlock, Block)
        self.assertIsInstance(transBlock[0].component("c"), Constraint)
        self.assertIsInstance(transBlock[0].c, Constraint)
        self.assertIsInstance(transBlock[1].component("c"), Constraint)
        self.assertIsInstance(transBlock[1].c, Constraint)

        disj2 = transBlock[2]
        self.assertIsInstance(disj2.component("c"), Constraint)
        self.assertIsInstance(disj2.c, Constraint)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_relaxation")
        
        # TODO

    def test_disjunct_dicts(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_relaxation")
        disj1 = transBlock[0]
        dict1 = getattr(disj1, "_gdp_trans_info")
        self.assertIsInstance(dict1, dict)
        #self.assertEqual(dict1, {
        # TODO

# TODO, test m precedence stuffs...
