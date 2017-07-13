import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import expr_common, expr as EXPR

import random
import weakref
import sys

# DEBUG
from nose.tools import set_trace

# TODO: check lengths of containers (so in particular, the transformation
# block.
# TODO: Should mark the tests that are relying on the order of transforming
# the disjunctions...

class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        stdoutbuffer = sys.stdout.getvalue()
        starting_index = len(stdoutbuffer)
        modelcopy = TransformationFactory('gdp.bigm').create_using(model)
        modelcopy.pprint()
        modelcopy_output = sys.stdout.getvalue()[starting_index:]
        starting_index = len(modelcopy_output)
        TransformationFactory('gdp.bigm').apply_to(model)
        model.pprint()
        model_output = sys.stdout.getvalue()[starting_index:]
        self.assertMultiLineEqual(modelcopy_output, model_output)

class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)
    
    @staticmethod
    def makeModel():
        m = ConcreteModel()
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
        transBlock = m.component("_pyomo_gdp_bigm_relaxation")
        self.assertIsInstance(transBlock, Block)

       # check that we have the lbub set on the transformation block
        lbub = transBlock.component("lbub")
        self.assertIsInstance(lbub, Set)
        self.assertEqual(len(lbub), 2)
        self.assertEqual(lbub, ['lb', 'ub'])

        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        # it has the disjuncts on it
        self.assertIsInstance(
            disjBlock[0].component("d[0].c"),
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("d[1].c"),
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

    def test_xor_constraint_weakref(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)   
        infodict = m._gdp_transformation_info
        self.assertIsInstance(infodict, dict)
        self.assertEqual(len(infodict), 1)
        self.assertIs(
            infodict[m.disjunction.name](),
            m._gdp_bigm_relaxation_disjunction_xor)

    def test_disjunct_and_constraint_weakrefs(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)        
        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        oldblock = m.component("d")

        # we should have a dictionary on each _DisjunctData and similarly
        # on each _BlockData of the corresponding disjunct block on the
        # transformation block (we are also counting on the fact that the
        # disjuncts get relaxed in the same order every time. Which means
        # that in this case, the indices of the disjuncts correspond to the
        # indices of the transformation block.)
        for i in [0,1]:
            infodict = getattr(oldblock[i], "_gdp_transformation_info")
            self.assertIsInstance(infodict, dict)
            self.assertIs(infodict['bigm'](), disjBlock[i])
            self.assertEqual(len(infodict), 3)
            self.assertTrue(infodict['relaxed'])
            constraintdict = infodict['relaxedConstraints']
            self.assertIsInstance(constraintdict, dict)
            self.assertEqual(len(constraintdict), 1)
            # check constraint weakrefs
            self.assertIs(constraintdict[ComponentUID(oldblock[i].c)](),
                          disjBlock[i].component(oldblock[i].c.name))
        
            infodict2 = getattr(disjBlock[i], "_gdp_transformation_info")
            self.assertIsInstance(infodict2, dict)
            self.assertIs(infodict2['src'](), oldblock[i])
            self.assertEqual(len(infodict2), 1)

    def test_new_block_nameCollision(self):
        # make sure that if the model already has a block called
        # _pyomo_gdp_bigm_relaxation that we come up with a different name for
        # the transformation block (and put the relaxed disjuncts on it)
        m = self.makeModel()
        m._pyomo_gdp_bigm_relaxation = Block(Any)
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m.component("_pyomo_gdp_bigm_relaxation4")
        self.assertIsInstance(gdpblock, Block)

        disjBlock = gdpblock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        # both disjuncts on transformation block
        self.assertEqual(len(disjBlock), 2)
        # nothing got added to the block we collided with that's not ours
        self.assertEqual(len(m._pyomo_gdp_bigm_relaxation), 0)

        # disjBlock has the disjuncts on it
        self.assertIsInstance(
            disjBlock[0].component("d[0].c"), 
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("d[1].c"), 
            Constraint)

    def test_info_dict_nameCollision(self):
        # this is the one place we need to know the name. Make sure we yell
        # if it's taken.
        m = self.makeModel()
        # We have no way of knowing if there is already a dictionary called 
        # _gdp_transformation_info in the model. If that happens, it will just 
        # get used. We can, however, yell if there is an attribute of the wrong 
        # type with the same name.
        m.d[0]._gdp_transformation_info = Block()
        self.assertRaisesRegexp(
            GDP_Error, 
            "Disjunct d\[0\] contains an attribute named "
            "_gdp_transformation_info. The transformation requires that it can "
            "create this attribute!*", 
            TransformationFactory('gdp.bigm').apply_to,
            m)

    def test_indicator_vars(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        oldblock = m.component("d")
        # have indicator variables on original disjuncts
        self.assertIsInstance(oldblock[0].indicator_var, Var)
        self.assertTrue(oldblock[0].indicator_var.is_binary())
        self.assertIsInstance(oldblock[1].indicator_var, Var)
        self.assertTrue(oldblock[1].indicator_var.is_binary())

    def test_xor_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        # make sure we created the xor constraint and put it on the parent
        # block of the disjunction--in this case the model.
        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertIs(m.d[0].indicator_var, xor.body._args[0])
        self.assertIs(m.d[1].indicator_var, xor.body._args[1])
        self.assertEqual(1, xor.body._coef[0])
        self.assertEqual(1, xor.body._coef[1])
        self.assertEqual(xor.lower, 1)
        self.assertEqual(xor.upper, 1)

    def test_or_constraints(self):
        m = ConcreteModel()
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
        # ask for an or constraint instead of xor
        m.disjunction = Disjunction(rule=disj_rule, xor=False)
        TransformationFactory('gdp.bigm').apply_to(m)
        # check or constraint is an or (upper bound is None)
        orcons = m.component("_gdp_bigm_relaxation_disjunction_or")
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
        # old constraints still there, deactivated
        for i in [0,1]:
            oldc = oldblock[i].component("c")
            self.assertIsInstance(oldc, Constraint)
            self.assertFalse(oldc.active)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts

        oldc = m.d[0].component("c")
        # we have an indexed constraint called d[0].c (indexed by
        # ['lb', 'ub'] but we only had to use 'lb' since the original
        # constraint had no ub.  This test relies on the disjuncts
        # getting transformed in the same order every time because it
        # hard-codes which block of the relaxedDisjuncts corresponds
        # to which of the original disjuncts.
        newcons = disjBlock[0].component("d[0].c")
        self.assertIsInstance(newcons, Constraint)
        self.assertTrue(newcons.active)

        newc = newcons['lb']
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
        newc = disjBlock[1].component("d[1].c")
        self.assertIsInstance(newc, Constraint)
        # now we've used both indices since original constraint was equality
        newc_lo = newc['lb']
        newc_hi = newc['ub']

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

    # helper method to check the M values in all of the transformed
    # constraints (m, M) is the tuple for M.  This also relies on the
    # disjuncts being transformed in the same order every time.
    def checkMs(self, model, cons1lb, cons2lb, cons2ub):
        disjBlock = model._pyomo_gdp_bigm_relaxation.relaxedDisjuncts

        # first constraint
        self.assertEqual(disjBlock[0].component("d[0].c")['lb'].body._coef[1], 
                         cons1lb) 

        # second constraint
        newc = disjBlock[1].component("d[1].c")
        newc_lo = newc['lb']
        newc_hi = newc['ub']
        self.assertEqual(newc_lo.body._coef[1], cons2lb) 
        self.assertEqual(newc_hi.body._coef[1], cons2ub)

    def test_suffix_M_None(self):
        m = self.makeModel()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 20, 20, -20)

    def test_suffix_M_None_on_disjunctData(self):
        m = self.makeModel()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[None] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        # there should now be different values of m on d[0] and d[1]
        self.checkMs(m, 18, 20, -20)

    def test_suffix_M_simpleConstraint_on_disjunctData(self):
        m = self.makeModel()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[m.d[0].c] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 18, 20, -20)

    def test_arg_M_None(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM={None: 19})
        self.checkMs(m, 19, 19, -19)

    def test_arg_M_singleNum(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        
        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=19.2)
        self.checkMs(m, 19.2, 19.2, -19.2)

    def test_singleArg_M_tuple(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        
        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=(-18, 19.2))
        self.checkMs(m, 18, 18, -19.2)

    def test_singleArg_M_tuple_wrongLength(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        
        # give an arg
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \(-18, 19.2, 3\) for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m, 
            bigM=(-18, 19.2, 3))

    def test_singleArg_M_list(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        
        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=[-18, 19.2])
        self.checkMs(m, 18, 18, -19.2)

    def test_singleArg_M_list_wrongLength(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        
        # give an arg
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \[-18, 19.2, 3\] for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m, 
            bigM=[-18, 19.2, 3])

    def test_arg_M_simpleConstraint(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on constraints so we can be happy we overrode them
        m.BigM[m.d[0].c] = 200
        m.BigM[m.d[1].c] = 200

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            bigM={None: 19, 
                  ComponentUID(m.d[0].c): 18, 
                  ComponentUID(m.d[1].c): 17})
        self.checkMs(m, 18, 17, -17)

    def test_tuple_M_arg(self):
        m = self.makeModel()
        # give a tuple arg
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            bigM={None: (-20,19)})
        self.checkMs(m, 20, 20, -19)

    def test_tuple_M_suffix(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = (-18, 20)
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 18, 18, -20)

    def test_list_M_arg(self):
        m = self.makeModel()
        # give a tuple arg
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            bigM={None: [-20,19]})
        self.checkMs(m, 20, 20, -19)

    def test_list_M_suffix(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = [-18, 20]
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 18, 18, -20)

    def test_tuple_wrong_length_err(self):
        m = self.makeModel()
        M = (-20,19, 32)
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \(-20, 19, 32\) for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m, 
            bigM={None: M})

    def test_list_wrong_length_err(self):
        m = self.makeModel()
        M = [-20, 19, 34]
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \[-20, 19, 34\] for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM={None: M})

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)
        

class TwoTermIndexedDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)
        # These are the pairs of which disjunct indices map to which
        # blocks in the list of block on the transformation
        # block. This is needed in multiple tests, so I am storing it
        # here.
        self.pairs = [
            ( (0,1,'A'), 0 ),
            ( (1,1,'A'), 1 ),
            ( (0,1,'B'), 2 ),
            ( (1,1,'B'), 3 ),
            ( (0,2,'A'), 4 ),
            ( (1,2,'A'), 5 ),
            ( (0,2,'B'), 6 ),
            ( (1,2,'B'), 7 ),
        ]

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

        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
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
        transBlock = m.component("_pyomo_gdp_bigm_relaxation")
        self.assertIsInstance(transBlock, Block)
        
        # check that we have the lbub set on the transformation block
        lbub = transBlock.component("lbub")
        self.assertIsInstance(lbub, Set)
        self.assertEqual(len(lbub), 2)
        self.assertEqual(lbub, ['lb', 'ub'])
        
        # check the IndexedBlock of transformed disjuncts
        disjBlock = transBlock.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 8)
        
        # check that all 8 blocks have the right constraint on them.
        # this relies on the order in which they are transformed.
        for i,j in self.pairs:
            self.assertIsInstance(
                disjBlock[j].component(m.disjunct[i].c.name), 
                Constraint)

    def test_disjunct_and_constraint_weakrefs(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        oldblock = m.component("disjunct")
        
        # this test relies on the fact that the disjuncts are going to be 
        # relaxed in the same order every time, so they will correspond to 
        # these indices on the transformation block:
        for src, dest in self.pairs:
            infodict = getattr(disjBlock[dest], "_gdp_transformation_info")
            self.assertIsInstance(infodict, dict)
            srcDisjunct = oldblock[src]
            self.assertIs(infodict['src'](), srcDisjunct)
            self.assertEqual(len(infodict), 1)
            infodict2 = getattr(oldblock[src], "_gdp_transformation_info")
            self.assertIsInstance(infodict2, dict)
            relaxedDisj = disjBlock[dest]
            self.assertIs(infodict2['bigm'](), relaxedDisj)
            self.assertTrue(infodict2['relaxed'])
            consDict = infodict2['relaxedConstraints']
            self.assertIsInstance(consDict, dict)
            self.assertIs(
                consDict[ComponentUID(srcDisjunct.c)](),
                disjBlock[dest].component(srcDisjunct.c.name))
            self.assertTrue(len(consDict), 1)
            self.assertEqual(len(infodict2), 3)

    def test_deactivated_disjuncts(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        # all the disjuncts got transformed, so all should be deactivated
        for i in m.disjunct.index_set():
            self.assertFalse(m.disjunct[i].active)

    def test_deactivated_disjunction(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        # all the disjunctions got transformed, so they should be deactivated too
        self.assertFalse(m.disjunction.active)
        for i in m.disjunction.index_set():
            self.assertFalse(m.disjunction[i].active)

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class DisjOnBlock(unittest.TestCase, CommonTests):
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
       
        self.assertIsInstance(
            m.b.component('_gdp_bigm_relaxation_b.disjunction_xor'), 
            Constraint)

    def test_trans_block_created(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        # test that the transformation block go created on the model
        transBlock = m.component('_pyomo_gdp_bigm_relaxation')
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        # and that it didn't get created on the block
        self.assertFalse(hasattr(m.b, '_pyomo_gdp_bigm_relaxation'))

    def add_disj_not_on_block(self, m):
        def simpdisj_rule(disjunct):
            m = disjunct.model()
            disjunct.c = Constraint(expr=m.a >= 3)
        m.simpledisj = Disjunct(rule=simpdisj_rule)
        def simpledisj2_rule(disjunct):
            m = disjunct.model()
            disjunct.c = Constraint(expr=m.a <= 3.5)
        m.simpledisj2 = Disjunct(rule=simpledisj2_rule)
        m.disjunction2 = Disjunction(expr=[m.simpledisj, m.simpledisj2])
        return m

    def checkFirstDisjMs(self, model, disj1c1lb, disj1c1ub, disj1c2):
        c1 = model.b.disjunct[0]._gdp_transformation_info['bigm'](). \
             component("b.disjunct[0].c")
        self.assertEqual(c1['lb'].body._coef[1], disj1c1lb)
        self.assertEqual(c1['ub'].body._coef[1], disj1c1ub)
        self.assertEqual(
            (model.b.disjunct[1]._gdp_transformation_info['bigm']().
             component("b.disjunct[1].c")['ub'].body._coef[1]),
            disj1c2)

    def checkMs(self, model, disj1c1lb, disj1c1ub, disj1c2, disj2c1, disj2c2):
        self.checkFirstDisjMs(model, disj1c1lb, disj1c1ub, disj1c2)

        self.assertEqual(
            (model.simpledisj._gdp_transformation_info['bigm']().
             component("simpledisj.c")['lb'].body._coef[1]), 
            disj2c1)
        self.assertEqual(
            (model.simpledisj2._gdp_transformation_info['bigm']().
             component("simpledisj2.c")['ub'].body._coef[1]), 
            disj2c2)

    def test_suffix_M_onBlock(self):
        m = self.makeModel()
        # adding something that's not on the block so that I know that only 
        # the stuff on the block was changed
        m = self.add_disj_not_on_block(m)
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 34
        TransformationFactory('gdp.bigm').apply_to(m)

        # check m values
        self.checkMs(m, 34, -34, -34, 3, -1.5)

    def test_suffix_M_simple_disj(self):
        m = self.makeModel()
        m = self.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 20, -20, -20, 45, -20)

    def test_suffix_M_constraintKeyOnBlock(self):
        m = self.makeModel()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[m.b.disjunct[0].c] = 87
        m.b.BigM[None] = 64

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, 87, -87, -64)
        
    def test_suffix_M_constraintKeyOnModel(self):
        m = self.makeModel()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 64
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.b.disjunct[0].c] = 87

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, 87, -87, -64)
       
    def test_suffix_M_constraintKeyOnSimpleDisj(self):
        m = self.makeModel()
        m = self.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.simpledisj.BigM[m.simpledisj.c] = 87
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 20, -20, -20, 87, -20)

    def test_block_targets_inactive(self):
        m = self.makeModel()
        m = self.add_disj_not_on_block(m)
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b)])

        self.assertFalse(m.b.disjunct[0].active)
        self.assertFalse(m.b.disjunct[1].active)
        self.assertTrue(m.simpledisj.active)
        self.assertTrue(m.simpledisj2.active)

    def tests_block_only_targets_transformed(self):
        m = self.makeModel()
        m = self.add_disj_not_on_block(m)
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("b.disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b.disjunct[1].c"), 
                              Constraint)

        # this relies on the disjuncts being transformed in the same order every
        # time
        pairs = [
            (0,0),
            (1,1),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.b.disjunct[i])
            dict2 = getattr(m.b.disjunct[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class SimpleDisjIndexedConstraints(unittest.TestCase, CommonTests):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.s = Set(initialize=[1,2])
        m.a = Var(m.s)
        m.b = Block()
        def disj1_rule(disjunct):
            m = disjunct.model()
            def c_rule(d, s):
                return m.a[s] == 0
            disjunct.c = Constraint(m.s, rule=c_rule)
        m.b.simpledisj1 = Disjunct(rule=disj1_rule)
        def disj2_rule(disjunct):
            m = disjunct.model()
            def c_rule(d, s):
                return m.a[s] <= 3
            disjunct.c = Constraint(m.s, rule=c_rule)
        m.b.simpledisj2 = Disjunct(rule=disj2_rule)
        m.b.disjunction = Disjunction(expr=[m.b.simpledisj1, m.b.simpledisj2])

        return m

    def checkMs(self, m, disj1c1lb, disj1c1ub, disj1c2lb, disj1c2ub, disj2c1ub,
                disj2c2ub):
        cons = m.b.simpledisj1._gdp_transformation_info['bigm'](). \
        component("b.simpledisj1.c")
        self.assertEqual(cons[1,'lb'].body._coef[1], disj1c1lb)
        self.assertEqual(cons[1,'ub'].body._coef[1], disj1c1ub)
        self.assertEqual(cons[2,'lb'].body._coef[1], disj1c2lb)
        self.assertEqual(cons[2,'ub'].body._coef[1], disj1c2ub)

        cons2 = m.b.simpledisj2._gdp_transformation_info['bigm'](). \
        component("b.simpledisj2.c")
        self.assertEqual(cons2[1,'ub'].body._coef[1], disj2c1ub)
        self.assertEqual(cons2[2,'ub'].body._coef[1], disj2c2ub)

    def test_suffix_M_constraintData_on_block(self):
        m = self.makeModel()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj1.c[1]] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 15, -15, 30, -30, -30, -30)

    def test_suffix_M_indexedConstraint_on_block(self):
        m = self.makeModel()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj2.c] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 30, -30, 30, -30, -15, -15)

    def test_suffix_M_constraintData_on_simpleDisjunct(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c[2]] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 65, -65, 14, -13, -65, -65)

    def test_suffix_M_indexedConstraint_on_simpleDisjunct(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 14, -13, 14, -13, -65, -65)

    def test_unbounded_var_m_estimation_err(self):
        m = self.makeModel()
        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot estimate M for expressions with unbounded variables."
            "\n\t\(found while processing constraint b.simpledisj1.c\)",
            TransformationFactory('gdp.bigm').apply_to,
            m)

    def test_create_using(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 100
        self.diff_apply_to_and_create_using(m)


class MultiTermDisj(unittest.TestCase, CommonTests):
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
    
    def test_xor_constraint(self):
        # check that the xor constraint has all the indicator variables...
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
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

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class IndexedConstraintsInDisj(unittest.TestCase, CommonTests):
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
        # constraints should still be moved as indexed constraints, and we will 
        # just add ['lb', 'ub'] as another index (using both for equality and 
        # both bounds and the one that we need when we only have one bound)
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_bigm_relaxation")
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        
        cons1 = disjBlock[0].component("disjunct[0].c")
        self.assertIsInstance(cons1, Constraint)
        self.assertTrue(cons1.active)
        self.assertTrue(cons1[1,'lb'].active)
        self.assertTrue(cons1[2,'lb'].active)

        cons2 = disjBlock[1].component("disjunct[1].c")
        self.assertIsInstance(cons2, Constraint)
        self.assertTrue(cons2.active)
        self.assertTrue(cons2[1,'lb'].active)
        self.assertTrue(cons2[1,'ub'].active)
        self.assertTrue(cons2[2,'lb'].active)
        self.assertTrue(cons2[2,'ub'].active)

    def checkMs(self, model, c11lb, c12lb, c21lb, c21ub, c22lb, c22ub):
        cons1 = model.disjunct[0]._gdp_transformation_info['bigm'](). \
                component("disjunct[0].c")
        self.assertEqual(cons1[1,'lb'].body._coef[1], c11lb)
        self.assertEqual(cons1[2,'lb'].body._coef[1], c12lb)
        cons2 = model.disjunct[1]._gdp_transformation_info['bigm'](). \
                component("disjunct[1].c")
        self.assertEqual(cons2[1,'lb'].body._coef[1], c21lb)
        self.assertEqual(cons2[1,'ub'].body._coef[1], c21ub)
        self.assertEqual(cons2[2,'lb'].body._coef[1], c22lb)
        self.assertEqual(cons2[2,'ub'].body._coef[1], c22ub)

    def test_arg_M_constraintdata(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a componentdata so we can be happy we overrode it
        m.BigM[m.disjunct[0].c[1]] = 19
        
        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            bigM={None: 19, ComponentUID(m.disjunct[0].c[1]): 17,
                  ComponentUID(m.disjunct[0].c[2]): 18})
        
        # check that m values are what we expect
        self.checkMs(m, 17, 18, 19, -19, 19, -19)

    def test_arg_M_indexedConstraint(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a component so we can be happy we overrode it
        m.BigM[m.disjunct[0].c] = 19
        
        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            bigM={None: 19, ComponentUID(m.disjunct[0].c): 17})
        self.checkMs(m, 17, 17, 19, -19, 19, -19)

    def test_suffix_M_None_on_indexedConstraint(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 19, 19, 20, -20, 20, -20)

    def test_suffix_M_None_on_constraintdata(self):
        m = self.makeModel()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c[1]] = 19
        
        TransformationFactory('gdp.bigm').apply_to(m)
        
        # check that m values are what we expect
        self.checkMs(m, 19, 20, 20, -20, 20, -20)

    def test_suffix_M_indexedConstraint_on_disjData(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData 
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 19, 19, 20, -20, 20, -20)

    def test_suffix_M_constraintData_on_disjData(self):
        m = self.makeModel()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData 
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19
        m.BigM[m.disjunct[0].c[1]] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, 18, 19, 20, -20, 20, -20)

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class DisjunctInMultipleDisjunctions(unittest.TestCase, CommonTests):
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
            else:
                disjunct.c = Constraint(expr=m.a==100)
        m.disjunct2 = Disjunct([0,1], rule=d2_rule)

        def disj1_rule(m):
            return [m.disjunct1[0], m.disjunct1[1]]
        m.disjunction1 = Disjunction(rule=disj1_rule)

        def disj2_rule(m):
            return [m.disjunct2[0], m.disjunct1[1]]
        m.disjunction2 = Disjunction(rule=disj2_rule)
        return m

    def test_disjunction1_xor(self):
        # check the xor constraint for the first disjunction
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        xor1 = m.component("_gdp_bigm_relaxation_disjunction1_xor")
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
        # check the xor constraint from the second disjunction
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        xor2 = m.component("_gdp_bigm_relaxation_disjunction2_xor")
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
        # all the constraints that are on disjuncts we transformed should be
        # deactivated
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertFalse(m.disjunct1[0].c.active)
        self.assertFalse(m.disjunct1[1].c.active)
        self.assertFalse(m.disjunct2[0].c.active)

    def test_transformed_disjuncts_deactivated(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertFalse(m.disjunct1[0].active)
        self.assertFalse(m.disjunct1[1].active)
        self.assertFalse(m.disjunct2[0].active)

    def test_untransformed_disj_active(self):
        # We have an extra disjunct not in any of the disjunctions.
        # He doesn't get transformed, and so he should still be active
        # so the writers will scream. His constraint, also, is still active.
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        self.assertTrue(m.disjunct2[1].active)
        self.assertTrue(m.disjunct2[1].c.active)
    
    def test_transformation_block_structure(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m.component("_pyomo_gdp_bigm_relaxation")
        self.assertIsInstance(transBlock, Block)
        disjBlock = transBlock.component("relaxedDisjuncts")
        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 3)
        self.assertIsInstance(
            disjBlock[0].component("disjunct1[0].c"), Constraint)
        self.assertIsInstance(
            disjBlock[1].component("disjunct1[1].c"), Constraint)
        self.assertIsInstance(
            disjBlock[2].component("disjunct2[0].c"), Constraint)

    def test_info_dicts(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        # this is another test that relies on the fact that the disjuncts
        # are going to be transformed in the same order. These are the
        # pairings of disjunct indices and disjBlock indices:
        disj1pairs = [
            (0, 0),
            (1, 1),
        ]
        disj2pairs = [
            (0, 2)
        ]
        # check dictionaries in both disjuncts
        for k, disj in enumerate([m.disjunct1, m.disjunct2]):
            pairs = disj1pairs
            if k==1:
                pairs = disj2pairs
            for i, j in pairs:
                infodict = getattr(disjBlock[j], "_gdp_transformation_info")
                self.assertIsInstance(infodict, dict)
                self.assertEqual(len(infodict), 1)
                self.assertIs(infodict['src'](), disj[i])

                infodict2 = getattr(disj[i], "_gdp_transformation_info")
                self.assertIsInstance(infodict2, dict)
                self.assertEqual(len(infodict2), 3)
                self.assertIs(infodict2['bigm'](), disjBlock[j])
                self.assertTrue(infodict2['relaxed'])
                constraintdict = infodict2['relaxedConstraints']
                self.assertIsInstance(constraintdict, dict)
                self.assertEqual(len(constraintdict), 1)
                # check the constraint weakrefs
                self.assertIs(constraintdict[ComponentUID(disj[i].c)](),
                              disjBlock[j].component(disj[i].c.name))

    def test_xor_constraint_weakref(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        infodict = m._gdp_transformation_info
        self.assertIsInstance(infodict, dict)
        self.assertEqual(len(infodict), 2)
        self.assertIs(infodict[m.disjunction1.name](),
                      m._gdp_bigm_relaxation_disjunction1_xor)
        self.assertIs(infodict[m.disjunction2.name](),
                      m._gdp_bigm_relaxation_disjunction2_xor)

    def test_transformed_constraints(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        # we will gather the constraints and check the bounds here, then check
        # bodies below.
        # the real test is that disjunct1[1] only got transformed once
        disj11 = m.disjunct1[1]._gdp_transformation_info['bigm']()
        # check lb first
        cons11lb = disj11.component("disjunct1[1].c")['lb']
        self.assertEqual(cons11lb.lower, 0)
        self.assertIsNone(cons11lb.upper)

        # check ub
        cons11ub = disj11.component("disjunct1[1].c")['ub']
        self.assertEqual(cons11ub.upper, 0)
        self.assertIsNone(cons11ub.lower)

        # check disjunct1[0] for good measure
        disj10 = m.disjunct1[0]._gdp_transformation_info['bigm']()
        cons10 = disj10.component("disjunct1[0].c")['lb']
        self.assertEqual(cons10.lower, 5)
        self.assertIsNone(cons10.upper)

        # check disjunct2[0] for even better measure
        disj20 = m.disjunct2[0]._gdp_transformation_info['bigm']()
        cons20 = disj20.component("disjunct2[0].c")['lb']
        self.assertEqual(cons20.lower, 30)
        self.assertIsNone(cons20.upper)

        # these constraint bodies are all the same except for the indicator 
        # variables and the values of M. The mapping is below, and we check 
        # them in the loop.
        consinfo = [
            (cons11lb, 10, m.disjunct1[1].indicator_var),
            (cons11ub, -50, m.disjunct1[1].indicator_var),
            (cons10, 15, m.disjunct1[0].indicator_var),
            (cons20, 40, m.disjunct2[0].indicator_var),
        ]

        for cons, M, ind_var in consinfo:
            self.assertEqual(len(cons.body._args), 2)
            self.assertEqual(len(cons.body._coef), 2)
            self.assertIs(cons.body._args[0], m.a)
            self.assertEqual(cons.body._coef[0], 1)
            self.assertEqual(cons.body._coef[1], M)
            self.assertEqual(len(cons.body._args[1]._args), 1)
            self.assertEqual(len(cons.body._args[1]._coef), 1)
            self.assertIs(cons.body._args[1]._args[0], ind_var)
            self.assertEqual(cons.body._args[1]._coef[0], -1)
            self.assertEqual(cons.body._args[1]._const, 1)
            self.assertEqual(cons.body._const, 0)

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class TestTargets_SingleDisjunction(unittest.TestCase, CommonTests):
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
            else:
                disjunct.c = Constraint(expr=m.a==100)
        m.disjunct2 = Disjunct([0,1], rule=d2_rule)

        def disj1_rule(m):
            return [m.disjunct1[0], m.disjunct1[1]]
        m.disjunction1 = Disjunction(rule=disj1_rule)

        def disj2_rule(m):
            return [m.disjunct2[0], m.disjunct2[1]]
        m.disjunction2 = Disjunction(rule=disj2_rule)
        return m

    def test_only_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1)])

        self.assertFalse(m.disjunction1.active)
        # disjunction2 still active
        self.assertTrue(m.disjunction2.active)

        self.assertFalse(m.disjunct1[0].active)
        self.assertFalse(m.disjunct1[1].active)
        self.assertTrue(m.disjunct2[0].active)
        self.assertTrue(m.disjunct2[1].active)
        self.assertTrue(m.disjunct2.active)
        
    def test_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        # only two disjuncts relaxed
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("disjunct1[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[1].c"),
                              Constraint)

        pairs = [
            (0, 0),
            (1, 1)
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[i], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.disjunct1[j])
            dict2 = getattr(m.disjunct1[j], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[i])
            self.assertTrue(dict2['relaxed'])

        self.assertFalse(hasattr(m.disjunct2[0], "_gdp_transformation_info"))
        self.assertFalse(hasattr(m.disjunct2[1], "_gdp_transformation_info"))

    def test_target_not_a_component_err(self):
        decoy = ConcreteModel()
        decoy.block = Block()
        m = self.makeModel()
        self.assertRaisesRegexp(
            GDP_Error, 
            "Target %s is not a component on the instance!*" 
            % ComponentUID(decoy.block),
            TransformationFactory('gdp.bigm').apply_to,
            m, 
            targets=[ComponentUID(decoy.block)])

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):
    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.s = Set(initialize=[1,2])
        m.a = Var(m.s, bounds=(0, 70))
        def disjunct1_rule(disjunct, s, flag):
            m = disjunct.model()
            if not flag:
                disjunct.c = Constraint(expr=m.a[s] == 0)
            else:
                disjunct.c = Constraint(expr=m.a[s] >= 7)
        m.disjunct1 = Disjunct(m.s, [0,1], rule=disjunct1_rule)
        def disjunction1_rule(m, s):
            return [m.disjunct1[s, flag] for flag in [0,1]]
        m.disjunction1 = Disjunction(m.s, rule=disjunction1_rule)

        m.b = Block([0,1])
        m.b[0].x = Var(bounds=(-2, 2))
        def disjunct2_rule(disjunct, flag):
            if not flag:
                disjunct.c = Constraint(expr=m.b[0].x <= 0)
            else:
                disjunct.c = Constraint(expr=m.b[0].x >= 0)
        m.b[0].disjunct = Disjunct([0,1], rule=disjunct2_rule)
        m.b[0].disjunction = Disjunction(expr=[m.b[0].disjunct[0],
                                               m.b[0].disjunct[1]])
        m.b[1].y = Var(bounds=(-3, 3))
        def disjunct3_rule(disjunct, flag):
            if not flag:
                disjunct.c = Constraint(expr=m.b[1].y <= 0)
            else:
                disjunct.c = Constraint(expr=m.b[1].y >= 0)
        m.b[1].disjunct = Disjunct([0,1], rule=disjunct3_rule)
        m.b[1].disjunction = Disjunction(expr=[m.b[1].disjunct[0],
                                               m.b[1].disjunct[1]])
        return m

    def test_indexedDisj_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1)])

        self.assertFalse(m.disjunct1[1,0].active)
        self.assertFalse(m.disjunct1[1,1].active)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertFalse(m.disjunct1[2,1].active)

        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct[0].active)
        self.assertTrue(m.b[1].disjunct[1].active)

    def test_indexedDisj_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 4)
        self.assertIsInstance(disjBlock[0].component("disjunct1[1,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[1,1].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[2].component("disjunct1[2,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[3].component("disjunct1[2,1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. These are the mappings between the indices of the original
        # disjuncts and the indices on the indexed block on the transformation
        # block.
        pairs = [
            ((1,0), 0),
            ((1,1), 1),
            ((2,0), 2),
            ((2,1), 3),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.disjunct1[i])
            dict2 = getattr(m.disjunct1[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_disjData_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1[2])])

        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertFalse(m.disjunct1[2,1].active)

        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct[0].active)
        self.assertTrue(m.b[1].disjunct[1].active)

    def test_disjData_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.disjunction1[2])])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("disjunct1[2,0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("disjunct1[2,1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. These are the mappings between the indices of the original
        # disjuncts and the indices on the indexed block on the transformation
        # block.
        pairs = [
            ((2,0), 0),
            ((2,1), 1),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.disjunct1[i])
            dict2 = getattr(m.disjunct1[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_indexedBlock_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.b)])

        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)

        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertFalse(m.b[1].disjunct[0].active)
        self.assertFalse(m.b[1].disjunct[1].active)

    def test_indexedBlock_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.b)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 4)
        self.assertIsInstance(disjBlock[0].component("b[0].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b[0].disjunct[1].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[2].component("b[1].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[3].component("b[1].disjunct[1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. This dictionary maps the block index to the list of
        # pairs of (originalDisjunctIndex, transBlockIndex)
        pairs = {
            0:
            [
                (0,0),
                (1,1),
            ],
            1:
            [
                (0,2),
                (1,3),
            ]
        }
        for blocknum, lst in pairs.iteritems():
            original = m.b[blocknum].disjunct
            for i, j in lst:
                dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
                self.assertIsInstance(dict1, dict)
                self.assertIs(dict1['src'](), original[i])
                dict2 = getattr(original[i], "_gdp_transformation_info")
                self.assertIsInstance(dict2, dict)
                self.assertIs(dict2['bigm'](), disjBlock[j])
                self.assertTrue(dict2['relaxed'])
    
    def test_blockData_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.b[0])])

        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)

        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct[0].active)
        self.assertTrue(m.b[1].disjunct[1].active)

    def test_blockData_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m, 
            targets=[ComponentUID(m.b[0])])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(disjBlock[0].component("b[0].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b[0].disjunct[1].c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. This dictionary maps the block index to the list of
        # pairs of (originalDisjunctIndex, transBlockIndex)
        pairs = [
                (0,0),
                (1,1),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.b[0].disjunct[i])
            dict2 = getattr(m.b[0].disjunct[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


class DisjunctionInDisjunct(unittest.TestCase, CommonTests):
    def setUp(self):
        EXPR.set_expression_tree_format(expr_common.Mode.coopr3_trees)
        # set seed so we can test name collisions predictably
        random.seed(666)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    @staticmethod
    def makeModel():
        m = ConcreteModel()
        m.x = Var(bounds=(-9,9))
        m.z = Var(bounds=(0, 10))
        m.a = Var(bounds=(0, 23))
        def disjunct_rule(disjunct, flag):
            m = disjunct.model()
            if flag:
                def innerdisj_rule(disjunct, flag):
                    m = disjunct.model()
                    if flag:
                        disjunct.c = Constraint(expr=m.z>=5)
                    else:
                        disjunct.c = Constraint(expr=m.z==0)
                disjunct.innerdisjunct = Disjunct([0,1], rule=innerdisj_rule)
                disjunct.innerdisjunction = Disjunction(
                    expr=[disjunct.innerdisjunct[0], disjunct.innerdisjunct[1]])
                disjunct.c = Constraint(expr=m.a <= 2)
            else:
                disjunct.c = Constraint(expr=m.x==2)
        m.disjunct = Disjunct([0,1], rule=disjunct_rule)
        # I want a SimpleDisjunct with a disjunction in it too
        def simpledisj_rule(disjunct):
            m = disjunct.model()
            def inner_disj_rule(disjunct, flag):
                if flag:
                    disjunct.c = Constraint(expr=m.x >= 4)
                else:
                    disjunct.c = Constraint(expr=m.x <= 2)
            disjunct.innerdisjunct = Disjunct([0,1], rule=inner_disj_rule)
            disjunct.innerdisjunction = Disjunction(
                expr=[disjunct.innerdisjunct[0], disjunct.innerdisjunct[1]])
        m.simpledisjunct = Disjunct(rule=simpledisj_rule)
        m.disjunction = Disjunction(
            expr=[m.simpledisjunct, m.disjunct[0], m.disjunct[1]])
        return m

    def test_disjuncts_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        self.assertFalse(m.disjunction.active)
        self.assertFalse(m.simpledisjunct.active)
        self.assertFalse(m.disjunct[0].active)
        self.assertFalse(m.disjunct[1].active)

    def test_transformation_block_structure(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)
        
        transBlock = m._pyomo_gdp_bigm_relaxation
        self.assertIsInstance(transBlock, Block)

        # check that we have the lbub set on the transformation block
        lbub = transBlock.component("lbub")
        self.assertIsInstance(lbub, Set)
        self.assertEqual(len(lbub), 2)
        self.assertEqual(lbub, ['lb', 'ub'])

        disjBlock = transBlock.relaxedDisjuncts
        self.assertIsInstance(disjBlock, Block)
        # There are 7 total disjuncts to relax: 4 nested ones and 3 outer ones
        self.assertEqual(len(disjBlock), 7)
        pairs = [
            (0, ["disjunct[1].innerdisjunct[0].c"]),
            (1, ["disjunct[1].innerdisjunct[1].c"]),
            (2, ["simpledisjunct.innerdisjunct[0].c"]),
            (3, ["simpledisjunct.innerdisjunct[1].c"]),
            (4, ["simpledisjunct._gdp_bigm_relaxation_simpledisjunct."
                 "innerdisjunction_xor"]),
            (5, ["disjunct[0].c"]),
            (6, ["disjunct[1]._gdp_bigm_relaxation_disjunct[1]."
                 "innerdisjunction_xor",
                 "disjunct[1].c"]),
        ]
        # This test will also rely on the disjunctions being relaxed
        # in the same order every time.
        for i, j in pairs:
            for nm in j:
                self.assertIsInstance(
                    disjBlock[i].component(nm),
                    Constraint)

    # many of the transformed constraints look like this, so can call this
    # function to test them.
    def check_bigM_constraint(self, cons, variable, M, indicator_var):
        self.assertEqual(len(cons.body._args), 2)
        self.assertEqual(len(cons.body._coef), 2)
        self.assertIs(cons.body._args[0], variable)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertEqual(cons.body._coef[1], M)
        self.assertEqual(cons.body._const, 0)
        self.assertEqual(len(cons.body._args[1]._args), 1)
        self.assertEqual(len(cons.body._args[1]._coef), 1)
        self.assertEqual(cons.body._args[1]._const, 1)
        self.assertEqual(cons.body._args[1]._coef[0], -1)
        self.assertIs(cons.body._args[1]._args[0], indicator_var)

    def check_xor_relaxation(self, cons, indvar1, indvar2, indvar3, lb):
        self.assertEqual(len(cons.body._args), 3)
        self.assertEqual(len(cons.body._coef), 3)
        self.assertIs(cons.body._args[0], indvar1)
        self.assertIs(cons.body._args[1], indvar2)
        self.assertEqual(cons.body._coef[0], 1)
        self.assertEqual(cons.body._coef[1], 1)
        if not lb:
            self.assertEqual(cons.upper, 1)
            self.assertIsNone(cons.lower)
            self.assertEqual(cons.body._coef[2], 1)
            self.assertIs(cons.body._args[2], indvar3)
            self.assertEqual(cons.body._const, -1)
        else:
            self.assertEqual(cons.lower, 1)
            self.assertIsNone(cons.upper)
            self.assertEqual(cons.body._coef[2], 1)
            self.assertEqual(len(cons.body._args[2]._args), 1)
            self.assertEqual(len(cons.body._args[2]._coef), 1)
            self.assertIs(cons.body._args[2]._args[0], indvar3)
            self.assertEqual(cons.body._args[2]._const, 1)
            self.assertEqual(cons.body._args[2]._coef[0], -1)
            self.assertEqual(cons.body._const, 0)

    def test_transformed_constraints(self):
        # We'll check all the transformed constraints to make sure
        # that nothing was transformed twice. The real key is that the
        # xor constraints created by the innter disjunctions get
        # transformed by the outer ones.
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(m)

        cons1 = m.disjunct[1].innerdisjunct[0].\
                _gdp_transformation_info['bigm']().component(
                    m.disjunct[1].innerdisjunct[0].c.name)
        cons1lb = cons1['lb']
        self.assertEqual(cons1lb.lower, 0)
        self.assertIsNone(cons1lb.upper)
        self.assertIs(cons1lb.body, m.z)
        cons1ub = cons1['ub']
        self.assertIsNone(cons1ub.lower)
        self.assertEqual(cons1ub.upper, 0)
        self.check_bigM_constraint(cons1ub, m.z, -10, 
                                 m.disjunct[1].innerdisjunct[0].indicator_var)  
        
        cons2 = m.disjunct[1].innerdisjunct[1].\
                _gdp_transformation_info['bigm']().component(
                    m.disjunct[1].innerdisjunct[1].c.name)['lb']
        self.assertEqual(cons2.lower, 5)
        self.assertIsNone(cons2.upper)
        self.check_bigM_constraint(cons2, m.z, 5, 
                                   m.disjunct[1].innerdisjunct[1].indicator_var)

        cons3 = m.simpledisjunct.innerdisjunct[0].\
                _gdp_transformation_info['bigm']().component(
                    m.simpledisjunct.innerdisjunct[0].c.name)['ub']
        self.assertEqual(cons3.upper, 2)
        self.assertIsNone(cons3.lower)
        self.check_bigM_constraint(
            cons3, m.x, -7, 
            m.simpledisjunct.innerdisjunct[0].indicator_var)

        cons4 = m.simpledisjunct.innerdisjunct[1].\
                _gdp_transformation_info['bigm']().component(
                    m.simpledisjunct.innerdisjunct[1].c.name)['lb']
        self.assertEqual(cons4.lower, 4)
        self.assertIsNone(cons4.upper)
        self.check_bigM_constraint(
            cons4, m.x, 13, 
            m.simpledisjunct.innerdisjunct[1].indicator_var)

        # Here we check that the xor constraint from
        # simpledisjunct.innerdisjunction is transformed.
        cons5 = m.simpledisjunct._gdp_transformation_info['bigm']().component(
            "simpledisjunct._gdp_bigm_relaxation_simpledisjunct."
            "innerdisjunction_xor")
        cons5lb = cons5['lb']
        self.check_xor_relaxation(
            cons5lb, 
            m.simpledisjunct.innerdisjunct[0].indicator_var, 
            m.simpledisjunct.innerdisjunct[1].indicator_var, 
            m.simpledisjunct.indicator_var,
            lb=True)
        cons5ub = cons5['ub']
        self.check_xor_relaxation(
            cons5ub, 
            m.simpledisjunct.innerdisjunct[0].indicator_var, 
            m.simpledisjunct.innerdisjunct[1].indicator_var, 
            m.simpledisjunct.indicator_var,
            lb=False)
        
        cons6 = m.disjunct[0]._gdp_transformation_info['bigm']().component(
            "disjunct[0].c")
        cons6lb = cons6['lb']
        self.assertIsNone(cons6lb.upper)
        self.assertEqual(cons6lb.lower, 2)
        self.check_bigM_constraint(cons6lb, m.x, 11, m.disjunct[0].indicator_var)
        cons6ub = cons6['ub']
        self.assertIsNone(cons6ub.lower)
        self.assertEqual(cons6ub.upper, 2)
        self.check_bigM_constraint(cons6ub, m.x, -7, m.disjunct[0].indicator_var)
        
        # now we check that the xor constraint from
        # disjunct[1].innerdisjunction gets transformed alongside the
        # other constraint in disjunct[1].
        cons7 = m.disjunct[1]._gdp_transformation_info['bigm']().component(
            "disjunct[1]._gdp_bigm_relaxation_disjunct[1].innerdisjunction_xor")
        cons7lb = cons7['lb']
        self.check_xor_relaxation(
            cons7lb,
            m.disjunct[1].innerdisjunct[0].indicator_var,
            m.disjunct[1].innerdisjunct[1].indicator_var,
            m.disjunct[1].indicator_var,
            lb=True)
        cons7ub = cons7['ub']
        self.check_xor_relaxation(
            cons7ub,
            m.disjunct[1].innerdisjunct[0].indicator_var,
            m.disjunct[1].innerdisjunct[1].indicator_var,
            m.disjunct[1].indicator_var,
            lb=False)

        cons8 = m.disjunct[1]._gdp_transformation_info['bigm']().component(
            "disjunct[1].c")['ub']
        self.assertIsNone(cons8.lower)
        self.assertEqual(cons8.upper, 2)
        self.check_bigM_constraint(cons8, m.a, -21, m.disjunct[1].indicator_var)

    def test_disjunct_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.simpledisjunct)])
        
        self.assertTrue(m.disjunct[0].active)
        self.assertTrue(m.disjunct[1].active)
        self.assertTrue(m.disjunct[1].innerdisjunct[0].active)
        self.assertTrue(m.disjunct[1].innerdisjunct[1].active)
        
        self.assertTrue(m.simpledisjunct.active)
        self.assertFalse(m.simpledisjunct.innerdisjunct[0].active)
        self.assertFalse(m.simpledisjunct.innerdisjunct[1].active)

    def test_disjunct_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.simpledisjunct)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(
            disjBlock[0].component("simpledisjunct.innerdisjunct[0].c"), 
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("simpledisjunct.innerdisjunct[1].c"), 
            Constraint)
        
        # This also relies on the disjuncts being transformed in the same
        # order every time.
        pairs = [
            (0,0),
            (1,1),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.simpledisjunct.innerdisjunct[i])
            dict2 = getattr(
                m.simpledisjunct.innerdisjunct[i], 
                "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])
        
    def test_disjunctData_targets_inactive(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunct[1])])
        
        self.assertTrue(m.disjunct[0].active)
        self.assertTrue(m.disjunct[1].active)
        self.assertFalse(m.disjunct[1].innerdisjunct[0].active)
        self.assertFalse(m.disjunct[1].innerdisjunct[1].active)
        
        self.assertTrue(m.simpledisjunct.active)
        self.assertTrue(m.simpledisjunct.innerdisjunct[0].active)
        self.assertTrue(m.simpledisjunct.innerdisjunct[1].active)

    def test_disjunctData_only_targets_transformed(self):
        m = self.makeModel()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunct[1])])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(
            disjBlock[0].component("disjunct[1].innerdisjunct[0].c"), 
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("disjunct[1].innerdisjunct[1].c"), 
            Constraint)
        
        # This also relies on the disjuncts being transformed in the same
        # order every time.
        pairs = [
            (0,0),
            (1,1),
        ]
        for i, j in pairs:
            dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
            self.assertIsInstance(dict1, dict)
            self.assertIs(dict1['src'](), m.disjunct[1].innerdisjunct[i])
            dict2 = getattr(
                m.disjunct[1].innerdisjunct[i], 
                "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm'](), disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_create_using(self):
        m = self.makeModel()
        self.diff_apply_to_and_create_using(m)


# TODO: I need to test the crazy case where either I index by 1 and '1' or
# where I have a disjunction called m.b.d and one called d on m.b.
# to make sure that unique names are really doing what we want.
