#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import constraint
from pyomo.core.expr import current as EXPR
from pyomo.repn import generate_standard_repn

import pyomo.gdp.tests.models as models

import random
import sys

from nose.tools import set_trace
from six import iteritems, StringIO

def check_linear_coef(self, repn, var, coef):
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.linear_coefs[var_id], coef)


class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        modelcopy = TransformationFactory('gdp.bigm').create_using(model)
        modelcopy_buf = StringIO()
        modelcopy.pprint(ostream=modelcopy_buf)
        modelcopy_output = modelcopy_buf.getvalue()

        TransformationFactory('gdp.bigm').apply_to(model)
        model_buf = StringIO()
        model.pprint(ostream=model_buf)
        model_output = model_buf.getvalue()
        self.assertMultiLineEqual(modelcopy_output, model_output)

class TwoTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_new_block_created(self):
        m = models.makeTwoTermDisj()
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
            disjBlock[1].component("d[1].c1"),
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("d[1].c2"),
            Constraint)
        self.assertIsInstance(
            disjBlock[0].component("d[0].c"),
            Constraint)

    def test_disjunction_deactivated(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        oldblock = m.component("disjunction")
        self.assertIsInstance(oldblock, Disjunction)
        self.assertFalse(oldblock.active)

    def test_disjunctdatas_deactivated(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        oldblock = m.component("disjunction")
        self.assertFalse(oldblock.disjuncts[0].active)
        self.assertFalse(oldblock.disjuncts[1].active)

    def test_xor_constraint_mapping(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        infodict = m._gdp_transformation_info
        self.assertIsInstance(infodict, dict)
        self.assertEqual(len(infodict), 1)
        orDict = infodict['disjunction_or_constraint']
        self.assertIsInstance(orDict, ComponentMap)
        self.assertIs(
            orDict[m.disjunction],
            m._gdp_bigm_relaxation_disjunction_xor)

    def test_disjunct_and_constraint_maps(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        oldblock = m.component("d")

        # we should have a dictionary on each _DisjunctData and similarly
        # on each _BlockData of the corresponding disjunct block on the
        # transformation block (we are also counting on the fact that the
        # disjuncts get relaxed in the same order every time. Which means
        # that in this case, the indices of the disjuncts correspond to the
        # indices of the transformation block.)
        transinfodicts = [getattr(oldblock[i], "_gdp_transformation_info") \
                     for i in [0,1]]
        srcinfodicts = [getattr(disjBlock[i], "_gdp_transformation_info") \
                        for i in [0,1]]
        for i in [0,1]:
            infodict = transinfodicts[i]
            self.assertIsInstance(infodict, dict)
            self.assertIs(infodict['bigm']['relaxationBlock'], disjBlock[i])
            self.assertEqual(len(infodict), 2)
            self.assertEqual(sorted(infodict.keys()), ['bigm','relaxed'])
            self.assertTrue(infodict['relaxed'])

            infodict2 = srcinfodicts[i]
            self.assertIsInstance(infodict2, dict)
            self.assertIs(infodict2['src'], oldblock[i])
            self.assertEqual(len(infodict2), 2)

        # check the constraint mappings
        # original -> transformed
        constraintdict1 = transinfodicts[1]['bigm']['relaxedConstraints']
        self.assertIsInstance(constraintdict1, ComponentMap)
        self.assertEqual(len(constraintdict1), 2)
        # check constraint dict has right mapping
        self.assertIs(constraintdict1[oldblock[1].c1],
                      disjBlock[1].component(oldblock[1].c1.name))
        self.assertIs(constraintdict1[oldblock[1].c2],
                      disjBlock[1].component(oldblock[1].c2.name))
        # transformed -> original
        srcdict1 = srcinfodicts[1]['srcConstraints']
        self.assertIsInstance(srcdict1, ComponentMap)
        self.assertEqual(len(srcdict1), 2)
        self.assertIs(srcdict1[disjBlock[1].component("d[1].c1")],
                      oldblock[1].c1)
        self.assertIs(srcdict1[disjBlock[1].component("d[1].c2")],
                      oldblock[1].c2)

        constraintdict2 = transinfodicts[0]['bigm']['relaxedConstraints']
        self.assertIsInstance(constraintdict2, ComponentMap)
        self.assertEqual(len(constraintdict2), 1)
        # check constraint dict has right mapping
        self.assertIs(constraintdict2[oldblock[0].c],
                      disjBlock[0].component(oldblock[0].c.name))
        # transformed -> original
        srcdict2 = srcinfodicts[0]['srcConstraints']
        self.assertIsInstance(srcdict2, ComponentMap)
        self.assertEqual(len(srcdict2), 1)
        self.assertIs(srcdict2[disjBlock[0].component("d[0].c")],
                      oldblock[0].c)

    def test_new_block_nameCollision(self):
        # make sure that if the model already has a block called
        # _pyomo_gdp_bigm_relaxation that we come up with a different name for
        # the transformation block (and put the relaxed disjuncts on it)
        m = models.makeTwoTermDisj()
        m._pyomo_gdp_bigm_relaxation = Block(Any)
        TransformationFactory('gdp.bigm').apply_to(m)
        gdpblock = m.component("_pyomo_gdp_bigm_relaxation_4")
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
            disjBlock[1].component("d[1].c1"),
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("d[1].c2"),
            Constraint)

    def test_info_dict_nameCollision(self):
        # this is the one place we need to know the name. Make sure we yell
        # if it's taken.
        m = models.makeTwoTermDisj()
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
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        oldblock = m.component("d")
        # have indicator variables on original disjuncts and they are still
        # active.
        self.assertIsInstance(oldblock[0].indicator_var, Var)
        self.assertTrue(oldblock[0].indicator_var.active)
        self.assertTrue(oldblock[0].indicator_var.is_binary())
        self.assertIsInstance(oldblock[1].indicator_var, Var)
        self.assertTrue(oldblock[1].indicator_var.active)
        self.assertTrue(oldblock[1].indicator_var.is_binary())

    def test_xor_constraints(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        # make sure we created the xor constraint and put it on the parent
        # block of the disjunction--in this case the model.
        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertIs(m.d[0].indicator_var, xor.body.arg(0))
        self.assertIs(m.d[1].indicator_var, xor.body.arg(1))
        repn = generate_standard_repn(xor.body)
        check_linear_coef(self, repn, m.d[0].indicator_var, 1)
        check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(xor.lower, 1)
        self.assertEqual(xor.upper, 1)

    def test_or_constraints(self):
        m = models.makeTwoTermDisj()
        m.disjunction.xor = False
        TransformationFactory('gdp.bigm').apply_to(m)

        # check or constraint is an or (upper bound is None)
        orcons = m.component("_gdp_bigm_relaxation_disjunction_xor")
        self.assertIsInstance(orcons, Constraint)
        self.assertIs(m.d[0].indicator_var, orcons.body.arg(0))
        self.assertIs(m.d[1].indicator_var, orcons.body.arg(1))
        repn = generate_standard_repn(orcons.body)
        check_linear_coef(self, repn, m.d[0].indicator_var, 1)
        check_linear_coef(self, repn, m.d[1].indicator_var, 1)
        self.assertEqual(orcons.lower, 1)
        self.assertIsNone(orcons.upper)

    def test_deactivated_constraints(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        oldblock = m.component("d")
        # old constraints still there, deactivated
        oldc1 = oldblock[1].component("c1")
        self.assertIsInstance(oldc1, Constraint)
        self.assertFalse(oldc1.active)

        oldc2 = oldblock[1].component("c2")
        self.assertIsInstance(oldc2, Constraint)
        self.assertFalse(oldc2.active)

        oldc = oldblock[0].component("c")
        self.assertIsInstance(oldc, Constraint)
        self.assertFalse(oldc.active)

    def test_transformed_constraints(self):
        m = models.makeTwoTermDisj()
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -3, 2, 7, 2)

    def test_do_not_transform_userDeactivated_disjuncts(self):
        m = models.makeTwoTermDisj()
        m.d[0].deactivate()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        self.assertFalse(m.disjunction.active)
        self.assertFalse(m.d[1].active)

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        infodict = m.d[1]._gdp_transformation_info
        self.assertEqual(len(disjBlock), 1)
        self.assertIs(disjBlock[0], infodict['bigm']['relaxationBlock'])

    # helper method to check the M values in all of the transformed
    # constraints (m, M) is the tuple for M.  This also relies on the
    # disjuncts being transformed in the same order every time.
    def checkMs(self, model, cons1lb, cons2lb, cons2ub, cons3ub):
        disjBlock = model._pyomo_gdp_bigm_relaxation.relaxedDisjuncts

        # first constraint
        c = disjBlock[0].component("d[0].c")
        self.assertEqual(len(c), 1)
        self.assertTrue(c['lb'].active)
        repn = generate_standard_repn(c['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, model.a, 1)
        check_linear_coef(self, repn, model.d[0].indicator_var, cons1lb)
        self.assertEqual(repn.constant, -cons1lb)
        self.assertEqual(c['lb'].lower, model.d[0].c.lower)
        self.assertIsNone(c['lb'].upper)

        # second constraint
        c = disjBlock[1].component("d[1].c1")
        self.assertEqual(len(c), 2)
        self.assertTrue(c['lb'].active)
        repn = generate_standard_repn(c['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, model.a, 1)
        check_linear_coef(self, repn, model.d[1].indicator_var, cons2lb)
        self.assertEqual(repn.constant, -cons2lb)
        self.assertEqual(c['lb'].lower, model.d[1].c1.lower)
        self.assertIsNone(c['lb'].upper)
        self.assertTrue(c['ub'].active)
        repn = generate_standard_repn(c['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, model.a, 1)
        check_linear_coef(self, repn, model.d[1].indicator_var, cons2ub)
        self.assertEqual(repn.constant, -cons2ub)
        self.assertIsNone(c['ub'].lower)
        self.assertEqual(c['ub'].upper, model.d[1].c1.upper)

        # third constraint
        c = disjBlock[1].component("d[1].c2")
        self.assertEqual(len(c), 1)
        self.assertTrue(c['ub'].active)
        repn = generate_standard_repn(c['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, model.x, 1)
        check_linear_coef(self, repn, model.d[1].indicator_var, cons3ub)
        self.assertEqual(repn.constant, -cons3ub)
        self.assertIsNone(c['ub'].lower)
        self.assertEqual(c['ub'].upper, model.d[1].c2.upper)

    def test_suffix_M_None(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -20, -20, 20, 20)

    def test_suffix_M_None_on_disjunctData(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[None] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        # there should now be different values of m on d[0] and d[1]
        self.checkMs(m, -18, -20, 20, 20)

    def test_suffix_M_simpleConstraint_on_disjunctData(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # override for the first index:
        m.d[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.d[0].BigM[m.d[0].c] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -20, 20, 20)

    def test_arg_M_None(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM={None: 19})
        self.checkMs(m, -19, -19, 19, 19)

    def test_arg_M_singleNum(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=19.2)
        self.checkMs(m, -19.2, -19.2, 19.2, 19.2)

    def test_singleArg_M_tuple(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=(-18, 19.2))
        self.checkMs(m, -18, -18, 19.2, 19.2)

    def test_singleArg_M_tuple_wrongLength(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \([^)]*\) for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM=(-18, 19.2, 3))

    def test_singleArg_M_list(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(m, bigM=[-18, 19.2])
        self.checkMs(m, -18, -18, 19.2, 19.2)

    def test_singleArg_M_list_wrongLength(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        # give an arg
        self.assertRaisesRegexp(
            GDP_Error,
            "Big-M \[[^\]]*\] for constraint d\[0\].c is not of "
            "length two. Expected either a single value or "
            "tuple or list of length two for M.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            bigM=[-18, 19.2, 3])

    def test_arg_M_simpleConstraint(self):
        m = models.makeTwoTermDisj()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on constraints so we can be happy we overrode them
        m.BigM[m.d[0].c] = 200
        m.BigM[m.d[1].c1] = 200
        m.BigM[m.d[1].c2] = 200

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m,
            bigM={None: 19,
                  ComponentUID(m.d[0].c): 18,
                  ComponentUID(m.d[1].c1): 17,
                  ComponentUID(m.d[1].c2): 16})
        self.checkMs(m, -18, -17, 17, 16)

    def test_tuple_M_arg(self):
        m = models.makeTwoTermDisj()
        # give a tuple arg
        TransformationFactory('gdp.bigm').apply_to(
            m,
            bigM={None: (-20,19)})
        self.checkMs(m, -20, -20, 19, 19)

    def test_tuple_M_suffix(self):
        m = models.makeTwoTermDisj()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = (-18, 20)
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -18, 20, 20)

    def test_list_M_arg(self):
        m = models.makeTwoTermDisj()
        # give a tuple arg
        TransformationFactory('gdp.bigm').apply_to(
            m,
            bigM={None: [-20,19]})
        self.checkMs(m, -20, -20, 19, 19)

    def test_list_M_suffix(self):
        m = models.makeTwoTermDisj()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = [-18, 20]
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -18, 20, 20)

    def test_tuple_wrong_length_err(self):
        m = models.makeTwoTermDisj()
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
        m = models.makeTwoTermDisj()
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
        m = models.makeTwoTermDisj()
        self.diff_apply_to_and_create_using(m)

    def test_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1,2,3]
        m.x = Var(m.I, bounds=(0,10))
        def c_rule(b,i):
            m = b.model()
            return m.x[i] >= i
        def d_rule(d,j):
            m = d.model()
            d.c = Constraint(m.I[:j], rule=c_rule)
        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m._pyomo_gdp_bigm_relaxation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 2)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 0)

        # Each relaxed disjunct should have 0 vars, and i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 0)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 0)
            self.assertEqual(
                len(list(relaxed.component_objects(Constraint))), 1)
            self.assertEqual(
                len(list(relaxed.component_data_objects(Constraint))), i)
            self.assertEqual(len(relaxed.component('d[%s].c'%i)), i)

    def test_virtual_indexed_constraints_in_disjunct(self):
        m = ConcreteModel()
        m.I = [1,2,3]
        m.x = Var(m.I, bounds=(0,10))
        def d_rule(d,j):
            m = d.model()
            d.c = Constraint(Any)
            for k in range(j):
                d.c[k+1] = m.x[k+1] >= k+1
        m.d = Disjunct(m.I, rule=d_rule)
        m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])

        TransformationFactory('gdp.bigm').apply_to(m)
        transBlock = m._pyomo_gdp_bigm_relaxation

        # 2 blocks: the original Disjunct and the transformation block
        self.assertEqual(
            len(list(m.component_objects(Block, descend_into=False))), 2)
        self.assertEqual(
            len(list(m.component_objects(Disjunct))), 0)

        # Each relaxed disjunct should have 0 vars, and i "d[i].c"
        # Constraints
        for i in [1,2,3]:
            relaxed = transBlock.relaxedDisjuncts[i-1]
            self.assertEqual(len(list(relaxed.component_objects(Var))), 0)
            self.assertEqual(len(list(relaxed.component_data_objects(Var))), 0)
            self.assertEqual(
                len(list(relaxed.component_objects(Constraint))), 1)
            self.assertEqual(
                len(list(relaxed.component_data_objects(Constraint))), i)
            self.assertEqual(len(relaxed.component('d[%s].c'%i)), i)


class TwoTermIndexedDisj(unittest.TestCase, CommonTests):
    def setUp(self):
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

    def test_xor_constraints(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        for i in m.disjunction.index_set():
            repn = generate_standard_repn(xor[i].body)
            self.assertEqual(repn.constant, 0)
            self.assertTrue(repn.is_linear())
            self.assertEqual(len(repn.linear_vars), 2)
            check_linear_coef(
                self, repn, m.disjunction[i].disjuncts[0].indicator_var, 1)
            check_linear_coef(
                self, repn, m.disjunction[i].disjuncts[1].indicator_var, 1)
            self.assertEqual(xor[i].lower, 1)
            self.assertEqual(xor[i].upper, 1)

    def test_deactivated_constraints(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m)

        for i in m.disjunct.index_set():
            self.assertFalse(m.disjunct[i].c.active)

    def test_transformed_block_structure(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
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

    def test_disjunct_and_constraint_maps(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
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
            self.assertIs(infodict['src'], srcDisjunct)
            srcConsMap = infodict['srcConstraints']
            self.assertIsInstance(srcConsMap, ComponentMap)
            self.assertEqual(len(srcConsMap), 1)
            self.assertIs(
                srcConsMap[disjBlock[dest].component(srcDisjunct.c.name)],
                srcDisjunct.c)
            self.assertEqual(len(infodict), 2)
            infodict2 = getattr(oldblock[src], "_gdp_transformation_info")
            self.assertIsInstance(infodict2, dict)
            relaxedDisj = disjBlock[dest]
            self.assertIs(infodict2['bigm']['relaxationBlock'], relaxedDisj)
            self.assertTrue(infodict2['relaxed'])
            consMap = infodict2['bigm']['relaxedConstraints']
            self.assertIsInstance(consMap, ComponentMap)
            self.assertIs(
                consMap[srcDisjunct.c],
                disjBlock[dest].component(srcDisjunct.c.name))
            self.assertTrue(len(consMap), 1)
            self.assertEqual(len(infodict2), 2)
            self.assertEqual(sorted(infodict2.keys()), ['bigm','relaxed'])

    def test_deactivated_disjuncts(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))
        # all the disjuncts got transformed, so all should be deactivated
        for i in m.disjunct.index_set():
            self.assertFalse(m.disjunct[i].active)
        self.assertFalse(m.disjunct.active)

    def test_deactivated_disjunction(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        # all the disjunctions got transformed, so they should be
        # deactivated too
        for i in m.disjunction.index_set():
            self.assertFalse(m.disjunction[i].active)
        self.assertFalse(m.disjunction.active)

    def test_create_using(self):
        m = models.makeTwoTermMultiIndexedDisjunction()
        self.diff_apply_to_and_create_using(m)


class DisjOnBlock(unittest.TestCase, CommonTests):
    # when the disjunction is on a block, we want the xor constraint
    # on its parent block, but the transformation block still on the
    # model.

    def test_xor_constraint_added(self):
        m = models.makeTwoTermDisjOnBlock()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertIsInstance(
            m.b.component('_gdp_bigm_relaxation_b.disjunction_xor'),
            Constraint)

    def test_trans_block_created(self):
        m = models.makeTwoTermDisjOnBlock()
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
        c1 = model.b.disjunct[0]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("b.disjunct[0].c")
        self.assertEqual(len(c1), 2)
        repn = generate_standard_repn(c1['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1lb)
        check_linear_coef(
            self, repn, model.b.disjunct[0].indicator_var, disj1c1lb)
        repn = generate_standard_repn(c1['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1ub)
        check_linear_coef(
            self, repn, model.b.disjunct[0].indicator_var, disj1c1ub)

        c2 = model.b.disjunct[1]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("b.disjunct[1].c")
        self.assertEqual(len(c2), 1)
        repn = generate_standard_repn(c2['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2)
        check_linear_coef(
            self, repn, model.b.disjunct[1].indicator_var, disj1c2)

    def checkMs(self, model, disj1c1lb, disj1c1ub, disj1c2, disj2c1, disj2c2):
        self.checkFirstDisjMs(model, disj1c1lb, disj1c1ub, disj1c2)

        c = model.simpledisj._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("simpledisj.c")
        self.assertEqual(len(c), 1)
        repn = generate_standard_repn(c['lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c1)
        check_linear_coef(
            self, repn, model.simpledisj.indicator_var, disj2c1)

        c = model.simpledisj2._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("simpledisj2.c")
        self.assertEqual(len(c), 1)
        repn = generate_standard_repn(c['ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c2)
        check_linear_coef(
            self, repn, model.simpledisj2.indicator_var, disj2c2)

    def test_suffix_M_onBlock(self):
        m = models.makeTwoTermDisjOnBlock()
        # adding something that's not on the block so that I know that only
        # the stuff on the block was changed
        m = self.add_disj_not_on_block(m)
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 34
        TransformationFactory('gdp.bigm').apply_to(m)

        # check m values
        self.checkMs(m, -34, 34, 34, -3, 1.5)

    def test_suffix_M_simple_disj(self):
        m = models.makeTwoTermDisjOnBlock()
        m = self.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -20, 20, 20, -45, 20)

    def test_suffix_M_constraintKeyOnBlock(self):
        m = models.makeTwoTermDisjOnBlock()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[m.b.disjunct[0].c] = 87
        m.b.BigM[None] = 64

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, -87, 87, 64)

    def test_suffix_M_constraintKeyOnModel(self):
        m = models.makeTwoTermDisjOnBlock()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 64
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.b.disjunct[0].c] = 87

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkFirstDisjMs(m, -87, 87, 64)

    def test_suffix_M_constraintKeyOnSimpleDisj(self):
        m = models.makeTwoTermDisjOnBlock()
        m = self.add_disj_not_on_block(m)
        m.simpledisj.BigM = Suffix(direction=Suffix.LOCAL)
        m.simpledisj.BigM[None] = 45
        m.simpledisj.BigM[m.simpledisj.c] = 87
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -20, 20, 20, -87, 20)

    def test_block_targets_inactive(self):
        m = models.makeTwoTermDisjOnBlock()
        m = self.add_disj_not_on_block(m)
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b)])

        self.assertFalse(m.b.disjunct[0].active)
        self.assertFalse(m.b.disjunct[1].active)
        self.assertFalse(m.b.disjunct.active)
        self.assertTrue(m.simpledisj.active)
        self.assertTrue(m.simpledisj2.active)

    def tests_block_only_targets_transformed(self):
        m = models.makeTwoTermDisjOnBlock()
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
            self.assertIs(dict1['src'], m.b.disjunct[i])
            dict2 = getattr(m.b.disjunct[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_create_using(self):
        m = models.makeTwoTermDisjOnBlock()
        self.diff_apply_to_and_create_using(m)


class SimpleDisjIndexedConstraints(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_do_not_transform_deactivated_constraintDatas(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 30
        m.b.simpledisj1.c[1].deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)

        transformedDisj = m.b.simpledisj1._gdp_transformation_info['bigm'][
            'relaxationBlock']
        transformedConstraints = m.b.simpledisj1._gdp_transformation_info[
            'bigm']['relaxedConstraints']
        self.assertEqual(len(transformedConstraints), 1)
        indexedCons = transformedConstraints[m.b.simpledisj1.c]
        self.assertEqual(len(indexedCons), 2)
        self.assertIsInstance(indexedCons[2, 'lb'],
                              constraint._GeneralConstraintData)
        self.assertIsInstance(indexedCons[2, 'ub'],
                              constraint._GeneralConstraintData)

    def checkMs(self, m, disj1c1lb, disj1c1ub, disj1c2lb, disj1c2ub, disj2c1ub,
                disj2c2ub):
        c = m.b.simpledisj1._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("b.simpledisj1.c")
        self.assertEqual(len(c), 4)
        repn = generate_standard_repn(c[1, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1lb)
        check_linear_coef(
            self, repn, m.b.simpledisj1.indicator_var, disj1c1lb)
        repn = generate_standard_repn(c[1, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c1ub)
        check_linear_coef(
            self, repn, m.b.simpledisj1.indicator_var, disj1c1ub)
        repn = generate_standard_repn(c[2, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2lb)
        check_linear_coef(
            self, repn, m.b.simpledisj1.indicator_var, disj1c2lb)
        repn = generate_standard_repn(c[2, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj1c2ub)
        check_linear_coef(
            self, repn, m.b.simpledisj1.indicator_var, disj1c2ub)


        c = m.b.simpledisj2._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("b.simpledisj2.c")
        self.assertEqual(len(c), 2)
        repn = generate_standard_repn(c[1, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c1ub)
        check_linear_coef(
            self, repn, m.b.simpledisj2.indicator_var, disj2c1ub)
        repn = generate_standard_repn(c[2, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -disj2c2ub)
        check_linear_coef(
            self, repn, m.b.simpledisj2.indicator_var, disj2c2ub)

    def test_suffix_M_constraintData_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj1.c[1]] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -15, 15, -30, 30, 30, 30)

    def test_suffix_M_indexedConstraint_on_block(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.b.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.BigM[None] = 30
        m.b.BigM[m.b.simpledisj2.c] = 15

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -30, 30, -30, 30, 15, 15)

    def test_suffix_M_constraintData_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c[2]] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -65, 65, -14, 13, 65, 65)

    def test_suffix_M_indexedConstraint_on_simpleDisjunct(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 65
        m.b.simpledisj1.BigM = Suffix(direction=Suffix.LOCAL)
        m.b.simpledisj1.BigM[m.b.simpledisj1.c] = (-14, 13)

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -14, 13, -14, 13, 65, 65)

    def test_unbounded_var_m_estimation_err(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        self.assertRaisesRegexp(
            GDP_Error,
            "Cannot estimate M for expressions with unbounded variables."
            "\n\t\(found unbounded var a\[1\] while processing constraint "
            "b.simpledisj1.c\)",
            TransformationFactory('gdp.bigm').apply_to,
            m)

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 100
        self.diff_apply_to_and_create_using(m)


class MultiTermDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_xor_constraint(self):
        # check that the xor constraint has all the indicator variables...
        m = models.makeThreeTermIndexedDisj()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor = m.component("_gdp_bigm_relaxation_disjunction_xor")
        self.assertIsInstance(xor, Constraint)
        self.assertEqual(xor[1].lower, 1)
        self.assertEqual(xor[1].upper, 1)
        self.assertEqual(xor[2].lower, 1)
        self.assertEqual(xor[2].upper, 1)

        repn = generate_standard_repn(xor[1].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        for i in range(3):
            check_linear_coef(self, repn, m.disjunct[i,1].indicator_var, 1)

        repn = generate_standard_repn(xor[2].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 3)
        for i in range(3):
            check_linear_coef(self, repn, m.disjunct[i,2].indicator_var, 1)

    def test_create_using(self):
        m = models.makeThreeTermIndexedDisj()
        self.diff_apply_to_and_create_using(m)


class IndexedConstraintsInDisj(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_transformed_constraints_on_block(self):
        # constraints should still be moved as indexed constraints, and we will
        # just add ['lb', 'ub'] as another index (using both for equality and
        # both bounds and the one that we need when we only have one bound)
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
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
        c = model.disjunct[0]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("disjunct[0].c")
        self.assertEqual(len(c), 2)
        repn = generate_standard_repn(c[1, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c11lb)
        check_linear_coef(self, repn, model.disjunct[0].indicator_var, c11lb)
        repn = generate_standard_repn(c[2, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c12lb)
        check_linear_coef(self, repn, model.disjunct[0].indicator_var, c12lb)

        c = model.disjunct[1]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("disjunct[1].c")
        self.assertEqual(len(c), 4)
        repn = generate_standard_repn(c[1, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21lb)
        check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21lb)
        repn = generate_standard_repn(c[1, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c21ub)
        check_linear_coef(self, repn, model.disjunct[1].indicator_var, c21ub)
        repn = generate_standard_repn(c[2, 'lb'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22lb)
        check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22lb)
        repn = generate_standard_repn(c[2, 'ub'].body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, -c22ub)
        check_linear_coef(self, repn, model.disjunct[1].indicator_var, c22ub)

    def test_arg_M_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
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
        self.checkMs(m, -17, -18, -19, 19, -19, 19)

    def test_arg_M_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a component so we can be happy we overrode it
        m.BigM[m.disjunct[0].c] = 19

        # give an arg
        TransformationFactory('gdp.bigm').apply_to(
            m,
            bigM={None: 19, ComponentUID(m.disjunct[0].c): 17})
        self.checkMs(m, -17, -17, -19, 19, -19, 19)

    def test_suffix_M_None_on_indexedConstraint(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c] = 19
        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_None_on_constraintdata(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        # specify a suffix on None so we can be happy we overrode it.
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        m.BigM[m.disjunct[0].c[1]] = 19

        TransformationFactory('gdp.bigm').apply_to(m)

        # check that m values are what we expect
        self.checkMs(m, -19, -20, -20, 20, -20, 20)

    def test_suffix_M_indexedConstraint_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -19, -19, -20, 20, -20, 20)

    def test_suffix_M_constraintData_on_disjData(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        m.BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[None] = 20
        # specify a suffix on a disjunctData
        m.disjunct[0].BigM = Suffix(direction=Suffix.LOCAL)
        m.BigM[m.disjunct[0].c] = 19
        m.BigM[m.disjunct[0].c[1]] = 18

        TransformationFactory('gdp.bigm').apply_to(m)
        self.checkMs(m, -18, -19, -20, 20, -20, 20)

    def test_create_using(self):
        m = models.makeTwoTermDisj_IndexedConstraints_BoundedVars()
        self.diff_apply_to_and_create_using(m)


class DisjunctInMultipleDisjunctions(unittest.TestCase, CommonTests):
    def test_disjunction1_xor(self):
        # check the xor constraint for the first disjunction
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor1 = m.component("_gdp_bigm_relaxation_disjunction1_xor")
        self.assertIsInstance(xor1, Constraint)
        self.assertTrue(xor1.active)
        self.assertEqual(xor1.lower, 1)
        self.assertEqual(xor1.upper, 1)

        repn = generate_standard_repn(xor1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, m.disjunct1[0].indicator_var, 1)
        check_linear_coef(self, repn, m.disjunct1[1].indicator_var, 1)

    def test_disjunction2_xor(self):
        # check the xor constraint from the second disjunction
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        xor2 = m.component("_gdp_bigm_relaxation_disjunction2_xor")
        self.assertIsInstance(xor2, Constraint)
        self.assertTrue(xor2.active)
        self.assertEqual(xor2.lower, 1)
        self.assertEqual(xor2.upper, 1)

        repn = generate_standard_repn(xor2.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, m.disjunct2[0].indicator_var, 1)
        check_linear_coef(self, repn, m.disjunct1[1].indicator_var, 1)

    def test_constraints_deactivated(self):
        # all the constraints that are on disjuncts we transformed should be
        # deactivated
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        self.assertFalse(m.disjunct1[0].c.active)
        self.assertFalse(m.disjunct1[1].c.active)
        self.assertFalse(m.disjunct2[0].c.active)

    def test_transformed_disjuncts_deactivated(self):
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        self.assertFalse(m.disjunct1[0].active)
        self.assertFalse(m.disjunct1[1].active)
        self.assertFalse(m.disjunct2[0].active)
        self.assertFalse(m.disjunct1.active)

    def test_untransformed_disj_active(self):
        # We have an extra disjunct not in any of the disjunctions.
        # He doesn't get transformed, and so he should still be active
        # so the writers will scream. His constraint, also, is still active.
        m = models.makeDisjunctInMultipleDisjunctions_no_deactivate()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(
            m.disjunction1, m.disjunction2))

        self.assertTrue(m.disjunct2[1].active)
        self.assertTrue(m.disjunct2[1].c.active)
        # and it means his container is active
        self.assertTrue(m.disjunct2.active)

    def test_untransformed_disj_error(self):
        # If we try to transform the whole model without deactivating the extra
        # disjunct, the reclassify transformation should complain.
        m = models.makeDisjunctInMultipleDisjunctions_no_deactivate()
        with self.assertRaises(GDP_Error):
            TransformationFactory('gdp.bigm').apply_to(m)

    def test_transformation_block_structure(self):
        m = models.makeDisjunctInMultipleDisjunctions()
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
        m = models.makeDisjunctInMultipleDisjunctions()
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
                srcConsDict = infodict['srcConstraints']
                self.assertIsInstance(srcConsDict, ComponentMap)
                self.assertEqual(len(srcConsDict), 1)
                self.assertIs(
                    srcConsDict[disjBlock[j].component(disj[i].c.name)],
                    disj[i].c)
                self.assertEqual(len(infodict), 2)
                self.assertIs(infodict['src'], disj[i])

                infodict2 = getattr(disj[i], "_gdp_transformation_info")
                self.assertIsInstance(infodict2, dict)
                self.assertEqual(len(infodict2), 2)
                self.assertEqual( sorted(infodict2.keys()),
                                  ['bigm','relaxed'] )
                self.assertIs(
                    infodict2['bigm']['relaxationBlock'], disjBlock[j])
                self.assertTrue(infodict2['relaxed'])
                constraintdict = infodict2['bigm']['relaxedConstraints']
                self.assertIsInstance(constraintdict, ComponentMap)
                self.assertEqual(len(constraintdict), 1)
                # check the constraint mappings
                self.assertIs(constraintdict[disj[i].c],
                              disjBlock[j].component(disj[i].c.name))

    def test_xor_constraint_map(self):
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        infodict = m._gdp_transformation_info
        self.assertIsInstance(infodict, dict)
        self.assertEqual(len(infodict), 1)
        orDict = infodict['disjunction_or_constraint']
        self.assertIsInstance(orDict, ComponentMap)
        self.assertIs(orDict[m.disjunction1],
                      m._gdp_bigm_relaxation_disjunction1_xor)
        self.assertIs(orDict[m.disjunction2],
                      m._gdp_bigm_relaxation_disjunction2_xor)

    def test_transformed_constraints(self):
        m = models.makeDisjunctInMultipleDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        # we will gather the constraints and check the bounds here, then check
        # bodies below.
        # the real test is that disjunct1[1] only got transformed once
        disj11 = m.disjunct1[1]._gdp_transformation_info['bigm'][
            'relaxationBlock']
        # check lb first
        cons11lb = disj11.component("disjunct1[1].c")['lb']
        self.assertEqual(cons11lb.lower, 0)
        self.assertIsNone(cons11lb.upper)

        # check ub
        cons11ub = disj11.component("disjunct1[1].c")['ub']
        self.assertEqual(cons11ub.upper, 0)
        self.assertIsNone(cons11ub.lower)

        # check disjunct1[0] for good measure
        disj10 = m.disjunct1[0]._gdp_transformation_info['bigm'][
            'relaxationBlock']
        cons10 = disj10.component("disjunct1[0].c")['lb']
        self.assertEqual(cons10.lower, 5)
        self.assertIsNone(cons10.upper)

        # check disjunct2[0] for even better measure
        disj20 = m.disjunct2[0]._gdp_transformation_info['bigm'][
            'relaxationBlock']
        cons20 = disj20.component("disjunct2[0].c")['lb']
        self.assertEqual(cons20.lower, 30)
        self.assertIsNone(cons20.upper)

        # these constraint bodies are all the same except for the indicator
        # variables and the values of M. The mapping is below, and we check
        # them in the loop.
        consinfo = [
            (cons11lb, -10, m.disjunct1[1].indicator_var),
            (cons11ub, 50, m.disjunct1[1].indicator_var),
            (cons10, -15, m.disjunct1[0].indicator_var),
            (cons20, -40, m.disjunct2[0].indicator_var),
        ]

        for cons, M, ind_var in consinfo:
            repn = generate_standard_repn(cons.body)
            self.assertTrue(repn.is_linear())
            self.assertEqual(len(repn.linear_vars), 2)
            self.assertEqual(repn.constant, -M)
            check_linear_coef(self, repn, m.a, 1)
            check_linear_coef(self, repn, ind_var, M)

    def test_create_using(self):
        m = models.makeDisjunctInMultipleDisjunctions()
        self.diff_apply_to_and_create_using(m)


class TestTargets_SingleDisjunction(unittest.TestCase, CommonTests):
    def test_only_targets_inactive(self):
        m = models.makeDisjunctInMultipleDisjunctions_no_deactivate()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunction1)])

        self.assertFalse(m.disjunction1.active)
        # disjunction2 still active
        self.assertTrue(m.disjunction2.active)

        self.assertFalse(m.disjunct1[0].active)
        self.assertFalse(m.disjunct1[1].active)
        self.assertFalse(m.disjunct1.active)
        self.assertTrue(m.disjunct2[0].active)
        self.assertTrue(m.disjunct2[1].active)
        self.assertTrue(m.disjunct2.active)

    def test_only_targets_transformed(self):
        m = models.makeDisjunctInMultipleDisjunctions()
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
            self.assertIs(dict1['src'], m.disjunct1[j])
            dict2 = getattr(m.disjunct1[j], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[i])
            self.assertTrue(dict2['relaxed'])

        self.assertFalse(hasattr(m.disjunct2[0], "_gdp_transformation_info"))
        self.assertFalse(hasattr(m.disjunct2[1], "_gdp_transformation_info"))

    def test_target_not_a_component_err(self):
        decoy = ConcreteModel()
        decoy.block = Block()
        m = models.makeDisjunctInMultipleDisjunctions()
        self.assertRaisesRegexp(
            GDP_Error,
            "Target %s is not a component on the instance!*"
            % ComponentUID(decoy.block),
            TransformationFactory('gdp.bigm').apply_to,
            m,
            targets=[ComponentUID(decoy.block)])

    def test_create_using(self):
        m = models.makeDisjunctInMultipleDisjunctions()
        self.diff_apply_to_and_create_using(m)


class TestTargets_IndexedDisjunction(unittest.TestCase, CommonTests):
    def test_indexedDisj_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunction1)])

        self.assertFalse(m.disjunction1.active)
        self.assertFalse(m.disjunction1[1].active)
        self.assertFalse(m.disjunction1[2].active)

        self.assertFalse(m.disjunct1[1,0].active)
        self.assertFalse(m.disjunct1[1,1].active)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertFalse(m.disjunct1[2,1].active)
        self.assertFalse(m.disjunct1.active)

        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertTrue(m.b[1].disjunct1.active)

    def test_indexedDisj_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
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
            self.assertIs(dict1['src'], m.disjunct1[i])
            dict2 = getattr(m.disjunct1[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_warn_for_untransformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        def innerdisj_rule(d, flag):
            m = d.model()
            if flag:
                d.c = Constraint(expr=m.a[1] <= 2)
            else:
                d.c = Constraint(expr=m.a[1] >= 65)
        m.disjunct1[1,1].innerdisjunct = Disjunct([0,1], rule=innerdisj_rule)
        m.disjunct1[1,1].innerdisjunction = Disjunction([0],
            rule=lambda a,i: [m.disjunct1[1,1].innerdisjunct[0],
                              m.disjunct1[1,1].innerdisjunct[1]])
        # This test relies on the order that the component objects of
        # the disjunct get considered. In this case, the disjunct
        # causes the error, but in another world, it could be the
        # disjunction, which is also active.
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.bigm').create_using,
            m,
            targets=[ComponentUID(m.disjunction1[1])])
        #
        # we will make that disjunction come first now...
        #
        tmp = m.disjunct1[1,1].innerdisjunct
        m.disjunct1[1,1].del_component(tmp)
        m.disjunct1[1,1].add_component('innerdisjunct', tmp)
        self.assertRaisesRegexp(
            GDP_Error,
            "Found untransformed disjunction disjunct1\[1,1\]."
            "innerdisjunction\[0\] in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.bigm').create_using,
            m,
            targets=[ComponentUID(m.disjunction1[1])])
        # Deactivating the disjunction will allow us to get past it back
        # to the Disjunct (after we realize there are no active
        # DisjunctionData within the active Disjunction)
        m.disjunct1[1,1].innerdisjunction[0].deactivate()
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
            "in disjunct disjunct1\[1,1\]!.*",
            TransformationFactory('gdp.bigm').create_using,
            m,
            targets=[ComponentUID(m.disjunction1[1])])

    def test_disjData_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunction1[2])])

        self.assertFalse(m.disjunction1[2].active)

        self.assertTrue(m.disjunct1.active)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertFalse(m.disjunct1[2,0].active)
        self.assertFalse(m.disjunct1[2,1].active)

        self.assertTrue(m.b[0].disjunct.active)
        self.assertTrue(m.b[0].disjunct[0].active)
        self.assertTrue(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertTrue(m.b[1].disjunct1.active)

    def test_disjData_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
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
            self.assertIs(dict1['src'], m.disjunct1[i])
            dict2 = getattr(m.disjunct1[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_indexedBlock_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b)])

        self.assertTrue(m.disjunct1.active)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)

        self.assertFalse(m.b[0].disjunct.active)
        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertFalse(m.b[1].disjunct0.active)
        self.assertFalse(m.b[1].disjunct1.active)

    def test_indexedBlock_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 4)
        self.assertIsInstance(disjBlock[0].component("b[0].disjunct[0].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[1].component("b[0].disjunct[1].c"),
                              Constraint)
        self.assertIsInstance(disjBlock[2].component("b[1].disjunct0.c"),
                              Constraint)
        self.assertIsInstance(disjBlock[3].component("b[1].disjunct1.c"),
                              Constraint)

        # This relies on the disjunctions being transformed in the same order
        # every time. This dictionary maps the block index to the list of
        # pairs of (originalDisjunctIndex, transBlockIndex)
        pairs = {
            0:
            [
                ('disjunct',0,0),
                ('disjunct',1,1),
            ],
            1:
            [
                ('disjunct0',None,2),
                ('disjunct1',None,3),
            ]
        }
        for blocknum, lst in iteritems(pairs):
            for comp, i, j in lst:
                original = m.b[blocknum].component(comp)
                dict1 = getattr(disjBlock[j], "_gdp_transformation_info")
                self.assertIsInstance(dict1, dict)
                self.assertIs(dict1['src'], original[i])
                dict2 = getattr(original[i], "_gdp_transformation_info")
                self.assertIsInstance(dict2, dict)
                self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
                self.assertTrue(dict2['relaxed'])

    def checkb0TargetsInactive(self, m):
        self.assertTrue(m.disjunct1.active)
        self.assertTrue(m.disjunct1[1,0].active)
        self.assertTrue(m.disjunct1[1,1].active)
        self.assertTrue(m.disjunct1[2,0].active)
        self.assertTrue(m.disjunct1[2,1].active)

        self.assertFalse(m.b[0].disjunct.active)
        self.assertFalse(m.b[0].disjunct[0].active)
        self.assertFalse(m.b[0].disjunct[1].active)
        self.assertTrue(m.b[1].disjunct0.active)
        self.assertTrue(m.b[1].disjunct1.active)

    def checkb0TargetsTransformed(self, m):
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
            self.assertIs(dict1['src'], m.b[0].disjunct[i])
            dict2 = getattr(m.b[0].disjunct[i], "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_blockData_targets_inactive(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b[0])])

        self.checkb0TargetsInactive(m)

    def test_blockData_only_targets_transformed(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b[0])])
        self.checkb0TargetsTransformed(m)

    def test_do_not_transform_deactivated_targets(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        m.b[1].deactivate()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.b[0]), ComponentUID(m.b[1])])

        self.checkb0TargetsInactive(m)
        self.checkb0TargetsTransformed(m)

    def test_create_using(self):
        m = models.makeDisjunctionsOnIndexedBlock()
        self.diff_apply_to_and_create_using(m)


class DisjunctionInDisjunct(unittest.TestCase, CommonTests):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_disjuncts_inactive(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m, targets=(m,))

        self.assertFalse(m.disjunction.active)
        self.assertFalse(m.simpledisjunct.active)
        self.assertFalse(m.disjunct[0].active)
        self.assertFalse(m.disjunct[1].active)
        self.assertFalse(m.disjunct.active)

    def test_transformation_block_structure(self):
        m = models.makeNestedDisjunctions()
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
            (2, ["simpledisjunct.innerdisjunct0.c"]),
            (3, ["simpledisjunct.innerdisjunct1.c"]),
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
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, -M)
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, variable, 1)
        check_linear_coef(self, repn, indicator_var, M)

    def check_xor_relaxation(self, cons, indvar1, indvar2, indvar3, lb):
        repn = generate_standard_repn(cons.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        check_linear_coef(self, repn, indvar1, 1)
        check_linear_coef(self, repn, indvar2, 1)
        if not lb:
            self.assertEqual(cons.upper, 1)
            self.assertIsNone(cons.lower)
            self.assertEqual(repn.constant, -1)
            check_linear_coef(self, repn, indvar3, 1)
        else:
            self.assertEqual(cons.lower, 1)
            self.assertIsNone(cons.upper)
            self.assertEqual(repn.constant, 1)
            check_linear_coef(self, repn, indvar3, -1)

    def test_transformed_constraints(self):
        # We'll check all the transformed constraints to make sure
        # that nothing was transformed twice. The real key is that the
        # xor constraints created by the inner disjunctions get
        # transformed by the outer ones.
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(m)

        cons1 = m.disjunct[1].innerdisjunct[0].\
                _gdp_transformation_info['bigm']['relaxationBlock'].component(
                    m.disjunct[1].innerdisjunct[0].c.name)
        cons1lb = cons1['lb']
        self.assertEqual(cons1lb.lower, 0)
        self.assertIsNone(cons1lb.upper)
        self.assertIs(cons1lb.body, m.z)
        cons1ub = cons1['ub']
        self.assertIsNone(cons1ub.lower)
        self.assertEqual(cons1ub.upper, 0)
        self.check_bigM_constraint(cons1ub, m.z, 10,
                                 m.disjunct[1].innerdisjunct[0].indicator_var)

        cons2 = m.disjunct[1].innerdisjunct[1].\
                _gdp_transformation_info['bigm']['relaxationBlock'].component(
                    m.disjunct[1].innerdisjunct[1].c.name)['lb']
        self.assertEqual(cons2.lower, 5)
        self.assertIsNone(cons2.upper)
        self.check_bigM_constraint(cons2, m.z, -5,
                                   m.disjunct[1].innerdisjunct[1].indicator_var)

        cons3 = m.simpledisjunct.innerdisjunct0.\
                _gdp_transformation_info['bigm']['relaxationBlock'].component(
                    m.simpledisjunct.innerdisjunct0.c.name)['ub']
        self.assertEqual(cons3.upper, 2)
        self.assertIsNone(cons3.lower)
        self.check_bigM_constraint(
            cons3, m.x, 7,
            m.simpledisjunct.innerdisjunct0.indicator_var)

        cons4 = m.simpledisjunct.innerdisjunct1.\
                _gdp_transformation_info['bigm']['relaxationBlock'].component(
                    m.simpledisjunct.innerdisjunct1.c.name)['lb']
        self.assertEqual(cons4.lower, 4)
        self.assertIsNone(cons4.upper)
        self.check_bigM_constraint(
            cons4, m.x, -13,
            m.simpledisjunct.innerdisjunct1.indicator_var)

        # Here we check that the xor constraint from
        # simpledisjunct.innerdisjunction is transformed.
        cons5 = m.simpledisjunct._gdp_transformation_info['bigm'][
            'relaxationBlock'].component(
            "simpledisjunct._gdp_bigm_relaxation_simpledisjunct."
            "innerdisjunction_xor")
        cons5lb = cons5['lb']
        self.check_xor_relaxation(
            cons5lb,
            m.simpledisjunct.innerdisjunct0.indicator_var,
            m.simpledisjunct.innerdisjunct1.indicator_var,
            m.simpledisjunct.indicator_var,
            lb=True)
        cons5ub = cons5['ub']
        self.check_xor_relaxation(
            cons5ub,
            m.simpledisjunct.innerdisjunct0.indicator_var,
            m.simpledisjunct.innerdisjunct1.indicator_var,
            m.simpledisjunct.indicator_var,
            lb=False)

        cons6 = m.disjunct[0]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("disjunct[0].c")
        cons6lb = cons6['lb']
        self.assertIsNone(cons6lb.upper)
        self.assertEqual(cons6lb.lower, 2)
        self.check_bigM_constraint(cons6lb, m.x, -11, m.disjunct[0].indicator_var)
        cons6ub = cons6['ub']
        self.assertIsNone(cons6ub.lower)
        self.assertEqual(cons6ub.upper, 2)
        self.check_bigM_constraint(cons6ub, m.x, 7, m.disjunct[0].indicator_var)

        # now we check that the xor constraint from
        # disjunct[1].innerdisjunction gets transformed alongside the
        # other constraint in disjunct[1].
        cons7 = m.disjunct[1]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component(
            "disjunct[1]._gdp_bigm_relaxation_disjunct[1].innerdisjunction_xor"
            )
        cons7lb = cons7[0,'lb']
        self.check_xor_relaxation(
            cons7lb,
            m.disjunct[1].innerdisjunct[0].indicator_var,
            m.disjunct[1].innerdisjunct[1].indicator_var,
            m.disjunct[1].indicator_var,
            lb=True)
        cons7ub = cons7[0,'ub']
        self.check_xor_relaxation(
            cons7ub,
            m.disjunct[1].innerdisjunct[0].indicator_var,
            m.disjunct[1].innerdisjunct[1].indicator_var,
            m.disjunct[1].indicator_var,
            lb=False)

        cons8 = m.disjunct[1]._gdp_transformation_info['bigm'][
            'relaxationBlock'].component("disjunct[1].c")['ub']
        self.assertIsNone(cons8.lower)
        self.assertEqual(cons8.upper, 2)
        self.check_bigM_constraint(cons8, m.a, 21, m.disjunct[1].indicator_var)

    def test_disjunct_targets_inactive(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.simpledisjunct)])

        self.assertTrue(m.disjunct.active)
        self.assertTrue(m.disjunct[0].active)
        self.assertTrue(m.disjunct[1].active)
        self.assertTrue(m.disjunct[1].innerdisjunct.active)
        self.assertTrue(m.disjunct[1].innerdisjunct[0].active)
        self.assertTrue(m.disjunct[1].innerdisjunct[1].active)

        # We basically just treated simpledisjunct as a block. It
        # itself has not been transformed and should not be
        # deactivated. We just transformed everything in it.
        self.assertTrue(m.simpledisjunct.active)
        self.assertFalse(m.simpledisjunct.innerdisjunct0.active)
        self.assertFalse(m.simpledisjunct.innerdisjunct1.active)

    def test_disjunct_only_targets_transformed(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.simpledisjunct)])

        disjBlock = m._pyomo_gdp_bigm_relaxation.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 2)
        self.assertIsInstance(
            disjBlock[0].component("simpledisjunct.innerdisjunct0.c"),
            Constraint)
        self.assertIsInstance(
            disjBlock[1].component("simpledisjunct.innerdisjunct1.c"),
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
            self.assertIs(dict1['src'],
                          m.simpledisjunct.component('innerdisjunct%d'%i))
            dict2 = getattr(
                m.simpledisjunct.component('innerdisjunct%d'%i),
                "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_disjunctData_targets_inactive(self):
        m = models.makeNestedDisjunctions()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunct[1])])

        self.assertTrue(m.disjunct[0].active)
        self.assertTrue(m.disjunct[1].active)
        self.assertTrue(m.disjunct.active)
        self.assertFalse(m.disjunct[1].innerdisjunct[0].active)
        self.assertFalse(m.disjunct[1].innerdisjunct[1].active)
        self.assertFalse(m.disjunct[1].innerdisjunct.active)

        self.assertTrue(m.simpledisjunct.active)
        self.assertTrue(m.simpledisjunct.innerdisjunct0.active)
        self.assertTrue(m.simpledisjunct.innerdisjunct1.active)

    def test_disjunctData_only_targets_transformed(self):
        m = models.makeNestedDisjunctions()
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
            self.assertIs(dict1['src'], m.disjunct[1].innerdisjunct[i])
            dict2 = getattr(
                m.disjunct[1].innerdisjunct[i],
                "_gdp_transformation_info")
            self.assertIsInstance(dict2, dict)
            self.assertIs(dict2['bigm']['relaxationBlock'], disjBlock[j])
            self.assertTrue(dict2['relaxed'])

    def test_disjunction_target_err(self):
        m = models.makeNestedDisjunctions()
        self.assertRaisesRegexp(
            GDP_Error,
            "Found active disjunct simpledisjunct.innerdisjunct0 in "
            "disjunct simpledisjunct!.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,

            targets=[ComponentUID(m.disjunction)])

    def test_create_using(self):
        m = models.makeNestedDisjunctions()
        self.diff_apply_to_and_create_using(m)


class IndexedDisjunction(unittest.TestCase):
    # this tests that if the targets are a subset of the
    # _DisjunctDatas in an IndexedDisjunction that the xor constraint
    # created on the parent block will still be indexed as expected.
    def test_xor_constraint(self):
        m = models.makeTwoTermIndexedDisjunction_BoundedVars()
        TransformationFactory('gdp.bigm').apply_to(
            m,
            targets=[ComponentUID(m.disjunction[1]),
                     ComponentUID(m.disjunction[3])])

        xorC = m._gdp_transformation_info['disjunction_or_constraint'][
            m.disjunction]
        self.assertIsInstance(xorC, Constraint)
        self.assertEqual(len(xorC), 2)

        # check the constraints
        for i in [1,3]:
            self.assertEqual(xorC[i].lower, 1)
            self.assertEqual(xorC[i].upper, 1)
            repn = generate_standard_repn(xorC[i].body)
            self.assertTrue(repn.is_linear())
            self.assertEqual(repn.constant, 0)
            check_linear_coef(self, repn, m.disjunct[i, 0].indicator_var, 1)
            check_linear_coef(self, repn, m.disjunct[i, 1].indicator_var, 1)

    def test_partial_deactivate_indexed_disjunction(self):
        """Test for partial deactivation of a indexed disjunction."""
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        @m.Disjunction([0, 1])
        def disj(m, i):
            if i == 0:
                return [m.x >= 1, m.x >= 2]
            else:
                return [m.x >= 3, m.x >= 4]

        m.disj[0].disjuncts[0].indicator_var.fix(1)
        m.disj[0].disjuncts[1].indicator_var.fix(1)
        m.disj[0].deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)
        self.assertEqual(
            len(m._gdp_bigm_relaxation_disj_xor), 1,
            "There should only be one XOR constraint generated. Found %s." %
            len(m._gdp_bigm_relaxation_disj_xor))


class BlocksOnDisjuncts(unittest.TestCase):
    def setUp(self):
        # set seed so we can test name collisions predictably
        random.seed(666)

    def test_transformed_constraint_nameConflicts(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m._pyomo_gdp_bigm_relaxation
        disjBlock = transBlock.relaxedDisjuncts

        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        self.assertEqual(len(disjBlock[0].component_map()), 1)
        self.assertEqual(len(disjBlock[1].component_map()), 4)
        self.assertIsInstance(disjBlock[0].component("evil[0].c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].b.c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].bb[1].c"),
                              Constraint)
        self.assertIsInstance(
            disjBlock[1].component("evil[1].b.c_4"), Constraint)
        self.assertIsInstance(
            disjBlock[1].component("evil[1].b.anotherblock.c"),
                                                     Constraint)

    def test_do_not_transform_deactivated_constraint(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        m.evil[1].b.anotherblock.c.deactivate()

        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m._pyomo_gdp_bigm_relaxation
        disjBlock = transBlock.relaxedDisjuncts

        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        self.assertEqual(len(disjBlock[0].component_map()), 1)
        self.assertEqual(len(disjBlock[1].component_map()), 3)
        self.assertIsInstance(disjBlock[0].component("evil[0].c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].b.c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].bb[1].c"),
                              Constraint)
        self.assertIsInstance(
            disjBlock[1].component("evil[1].b.c_4"), Constraint)

    def test_do_not_transform_deactivated_block(self):
        m = models.makeTwoTermDisj_BlockOnDisj()
        m.evil[1].b.anotherblock.deactivate()

        TransformationFactory('gdp.bigm').apply_to(m)

        transBlock = m._pyomo_gdp_bigm_relaxation
        disjBlock = transBlock.relaxedDisjuncts

        self.assertIsInstance(disjBlock, Block)
        self.assertEqual(len(disjBlock), 2)
        self.assertEqual(len(disjBlock[0].component_map()), 1)
        self.assertEqual(len(disjBlock[1].component_map()), 3)
        self.assertIsInstance(disjBlock[0].component("evil[0].c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].b.c"), Constraint)
        self.assertIsInstance(disjBlock[1].component("evil[1].bb[1].c"),
                              Constraint)
        self.assertIsInstance(
            disjBlock[1].component("evil[1].b.c_4"), Constraint)


class InnerDisjunctionSharedDisjuncts(unittest.TestCase):
    def test_activeInnerDisjunction_err(self):
        m = models.makeDuplicatedNestedDisjunction()
        self.assertRaisesRegexp(
            GDP_Error,
            "Found untransformed disjunction "
            "outerdisjunct\[1\].duplicateddisjunction in disjunct "
            "outerdisjunct\[1\]! The disjunction must be transformed before "
            "the disjunct. If you are using targets, put the disjunction "
            "before the disjunct in the list.*",
            TransformationFactory('gdp.bigm').apply_to,
            m,
            targets=[ComponentUID(m.outerdisjunct[1].innerdisjunction),
                     ComponentUID(m.disjunction)])


if __name__ == '__main__':
    unittest.main()
