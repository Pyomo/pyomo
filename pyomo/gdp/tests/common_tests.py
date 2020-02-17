from pyomo.environ import *
from pyomo.gdp import *
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
from six import StringIO
import random

from nose.tools import set_trace

# utitility functions

def check_linear_coef(self, repn, var, coef):
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.linear_coefs[var_id], coef)

def diff_apply_to_and_create_using(self, model, transformation):
    modelcopy = TransformationFactory(transformation).create_using(model)
    modelcopy_buf = StringIO()
    modelcopy.pprint(ostream=modelcopy_buf)
    modelcopy_output = modelcopy_buf.getvalue()
    
    # reset the seed for the apply_to call.
    random.seed(666)
    TransformationFactory(transformation).apply_to(model)
    model_buf = StringIO()
    model.pprint(ostream=model_buf)
    model_output = model_buf.getvalue()
    self.assertMultiLineEqual(modelcopy_output, model_output)

def check_relaxation_block(self, m, name, numdisjuncts):
    transBlock = m.component(name)
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), numdisjuncts)

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

def checkb0TargetsTransformed(self, m, transformation):
    trans = TransformationFactory('gdp.%s' % transformation)
    disjBlock = m.b[0].component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
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
        self.assertIs(m.b[0].disjunct[i].transformation_block(),
                      disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), 
                      m.b[0].disjunct[i])

# active status checks

def check_user_deactivated_disjuncts(self, transformation):
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    bigm = TransformationFactory('gdp.%s' % transformation) 
    bigm.apply_to(m, targets=(m,))

    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.d[1].active)

    rBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    disjBlock = rBlock.relaxedDisjuncts
    self.assertIs(disjBlock[0], m.d[1].transformation_block())
    self.assertIs(bigm.get_src_disjunct(disjBlock[0]), m.d[1])

def check_do_not_transform_userDeactivated_indexedDisjunction(self,
                                                              transformation):
    m = models.makeTwoTermIndexedDisjunction()
    # If you truly want to transform nothing, deactivate everything
    m.disjunction.deactivate()
    for idx in m.disjunct:
        m.disjunct[idx].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    # no transformation block, nothing transformed
    self.assertIsNone(m.component("_pyomo_gdp_%s_transformation" 
                                  % transformation))
    for idx in m.disjunct:
        self.assertIsNone(m.disjunct[idx].transformation_block)
    for idx in m.disjunction:
        self.assertIsNone(m.disjunction[idx].algebraic_constraint)

def check_disjunction_deactivated(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    oldblock = m.component("disjunction")
    self.assertIsInstance(oldblock, Disjunction)
    self.assertFalse(oldblock.active)

def check_disjunctDatas_deactivated(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    oldblock = m.component("disjunction")
    self.assertFalse(oldblock.disjuncts[0].active)
    self.assertFalse(oldblock.disjuncts[1].active)

def check_deactivated_constraints(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
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

def check_do_not_transform_twice_if_disjunction_reactivated(self,
                                                            transformation):
    m = models.makeTwoTermDisj()
    # this is a hack, but just diff the pprint from this and from calling
    # the transformation again.
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    first_buf = StringIO()
    m.pprint(ostream=first_buf)
    first_output = first_buf.getvalue()

    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    second_buf = StringIO()
    m.pprint(ostream=second_buf)
    second_output = second_buf.getvalue()

    self.assertMultiLineEqual(first_output, second_output)

    # this is a stupid thing to do, but we should still know not to
    # retransform because active status is now *not* the source of truth.
    m.disjunction.activate()

    # This is kind of the wrong error, but I'll live with it: at least we
    # get an error.
    self.assertRaisesRegexp(
        GDP_Error,
        "The disjunct d\[0\] has been transformed, but a disjunction "
        "it appears in has not. Putting the same disjunct in "
        "multiple disjunctions is not supported.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_constraints_deactivated_indexedDisjunction(self, transformation):
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    
    for i in m.disjunct.index_set():
        self.assertFalse(m.disjunct[i].c.active)


# transformation block

def check_transformation_block_name_collision(self, transformation):
    # make sure that if the model already has a block called
    # _pyomo_gdp_bigm_relaxation that we come up with a different name for the
    # transformation block (and put the relaxed disjuncts on it)
    m = models.makeTwoTermDisj()
    # add block with the name we are about to try to use
    m.add_component("_pyomo_gdp_%s_relaxation" % transformation, Block(Any))
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    # check that we got a uniquely named block
    transBlock = m.component("_pyomo_gdp_%s_relaxation_4" % transformation)
    self.assertIsInstance(transBlock, Block)

    # check that the relaxed disjuncts really are here.
    disjBlock = transBlock.relaxedDisjuncts
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    self.assertIsInstance(disjBlock[0].component("d[0].c"), Constraint)
    self.assertIsInstance(disjBlock[1].component("d[1].c1"), Constraint)
    self.assertIsInstance(disjBlock[1].component("d[1].c2"), Constraint)

    # we didn't add to the block that wasn't ours
    self.assertEqual(len(m.component("_pyomo_gdp_%s_relaxation" %
                                     transformation)), 0)

# XOR constraints

def check_indicator_vars(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    oldblock = m.component("d")
    # have indicator variables on original disjuncts and they are still
    # active.
    self.assertIsInstance(oldblock[0].indicator_var, Var)
    self.assertTrue(oldblock[0].indicator_var.active)
    self.assertTrue(oldblock[0].indicator_var.is_binary())
    self.assertIsInstance(oldblock[1].indicator_var, Var)
    self.assertTrue(oldblock[1].indicator_var.active)
    self.assertTrue(oldblock[1].indicator_var.is_binary())

def check_xor_constraint(self, transformation):
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    # make sure we created the xor constraint and put it on the relaxation
    # block
    rBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    xor = rBlock.component("disjunction_xor")
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(len(xor), 1)
    self.assertIs(m.d[0].indicator_var, xor.body.arg(0))
    self.assertIs(m.d[1].indicator_var, xor.body.arg(1))
    repn = generate_standard_repn(xor.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    check_linear_coef(self, repn, m.d[0].indicator_var, 1)
    check_linear_coef(self, repn, m.d[1].indicator_var, 1)
    self.assertEqual(xor.lower, 1)
    self.assertEqual(xor.upper, 1)

def check_indexed_xor_constraints(self, transformation):
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    xor = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
          component("disjunction_xor")
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

# mappings

def check_xor_constraint_mapping(self, transformation):
    m = models.makeTwoTermDisj()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    self.assertIs( trans.get_src_disjunction(transBlock.disjunction_xor),
                   m.disjunction)
    self.assertIs( m.disjunction.algebraic_constraint(),
                   transBlock.disjunction_xor)


def check_xor_constraint_mapping_two_disjunctions(self, transformation):
    m = models.makeDisjunctionOfDisjunctDatas()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    transBlock2 = m.component("_pyomo_gdp_%s_relaxation_4" % transformation)
    self.assertIs( trans.get_src_disjunction(transBlock.disjunction_xor),
                   m.disjunction)
    self.assertIs( trans.get_src_disjunction(transBlock2.disjunction2_xor),
                   m.disjunction2)

    self.assertIs( m.disjunction.algebraic_constraint(),
                   transBlock.disjunction_xor)
    self.assertIs( m.disjunction2.algebraic_constraint(),
                   transBlock2.disjunction2_xor)

def check_disjunct_mapping(self, transformation):
    m = models.makeTwoTermDisj_Nonlinear()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    disjBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts

    # the disjuncts will always be transformed in the same order,
    # and d[0] goes first, so we can check in a loop.
    for i in [0,1]:
        self.assertIs(disjBlock[i]._srcDisjunct(), m.d[i])
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.d[i])

# targets

def check_only_targets_inactive(self, transformation):
    m = models.makeTwoSimpleDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.disjunction1])

    self.assertFalse(m.disjunction1.active)
    self.assertIsNotNone(m.disjunction1._algebraic_constraint)
    # disjunction2 still active
    self.assertTrue(m.disjunction2.active)
    self.assertIsNone(m.disjunction2._algebraic_constraint)

    self.assertFalse(m.disjunct1[0].active)
    self.assertFalse(m.disjunct1[1].active)
    self.assertFalse(m.disjunct1.active)
    self.assertTrue(m.disjunct2[0].active)
    self.assertTrue(m.disjunct2[1].active)
    self.assertTrue(m.disjunct2.active)

def check_only_targets_get_transformed(self, transformation):
    m = models.makeTwoSimpleDisjunctions()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1])

    disjBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
    # only two disjuncts relaxed
    self.assertEqual(len(disjBlock), 2)
    # Note that in chull, these aren't the only components that get created, but
    # they are a proxy for which disjuncts got relaxed, which is what we want to
    # check.
    self.assertIsInstance(disjBlock[0].component("disjunct1[0].c"),
                          Constraint)
    self.assertIsInstance(disjBlock[1].component("disjunct1[1].c"),
                          Constraint)

    pairs = [
        (0, 0),
        (1, 1)
    ]
    for i, j in pairs:
        self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block())
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.disjunct1[j])

    self.assertIsNone(m.disjunct2[0].transformation_block)
    self.assertIsNone(m.disjunct2[1].transformation_block)

def check_target_not_a_component_error(self, transformation):
    decoy = ConcreteModel()
    decoy.block = Block()
    m = models.makeTwoSimpleDisjunctions()
    self.assertRaisesRegexp(
        GDP_Error,
        "Target block is not a component on instance unknown!",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[decoy.block])

# [ESJ 08/22/2019] This is a test for when targets can no longer be CUIDs
# def check_targets_cannot_be_cuids(self, transformation):
#     m = models.makeTwoTermDisj()
#     self.assertRaisesRegexp(
#         ValueError,
#         "invalid value for configuration 'targets':\n"
#         "\tFailed casting \[disjunction\]\n"
#         "\tto target_list\n"
#         "\tError: Expected Component or list of Components."
#         "\n\tRecieved %s" % type(ComponentUID(m.disjunction)),
#         TransformationFactory('gdp.%s' % transformation).apply_to,
#         m,
#         targets=[ComponentUID(m.disjunction)])

# test that cuid targets still work for now. This and the next test should
# go away when the above comes in.
def check_cuid_targets_still_work_for_now(self, transformation):
    m = models.makeTwoSimpleDisjunctions()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[ComponentUID(m.disjunction1)])

    disjBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
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
        self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block())
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.disjunct1[j])

    self.assertIsNone(m.disjunct2[0].transformation_block)
    self.assertIsNone(m.disjunct2[1].transformation_block)

def check_cuid_target_error_still_works_for_now(self, transformation):
    m = models.makeTwoSimpleDisjunctions()
    m2 = ConcreteModel()
    m2.oops = Block()
    self.assertRaisesRegexp(
        GDP_Error,
        "Target %s is not a component on the instance!" % 
        ComponentUID(m2.oops),
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=ComponentUID(m2.oops))

def check_indexedDisj_targets_inactive(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.disjunction1])

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

def check_indexedDisj_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1])

    disjBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
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
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.disjunct1[i])
        self.assertIs(disjBlock[j], m.disjunct1[i].transformation_block())

def check_warn_for_untransformed(self, transformation):
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
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]])
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
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]])
    # Deactivating the disjunction will allow us to get past it back
    # to the Disjunct (after we realize there are no active
    # DisjunctionData within the active Disjunction)
    m.disjunct1[1,1].innerdisjunction[0].deactivate()
    self.assertRaisesRegexp(
        GDP_Error,
        "Found active disjunct disjunct1\[1,1\].innerdisjunct\[0\] "
        "in disjunct disjunct1\[1,1\]!.*",
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]])

def check_disjData_targets_inactive(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.disjunction1[2]])

    self.assertIsNotNone(m.disjunction1[2]._algebraic_constraint)
    self.assertFalse(m.disjunction1[2].active)

    self.assertTrue(m.disjunct1.active)
    self.assertIsNotNone(m.disjunction1._algebraic_constraint)
    self.assertTrue(m.disjunct1[1,0].active)
    self.assertIsNone(m.disjunct1[1,0]._transformation_block)
    self.assertTrue(m.disjunct1[1,1].active)
    self.assertIsNone(m.disjunct1[1,1]._transformation_block)
    self.assertFalse(m.disjunct1[2,0].active)
    self.assertIsNotNone(m.disjunct1[2,0]._transformation_block)
    self.assertFalse(m.disjunct1[2,1].active)
    self.assertIsNotNone(m.disjunct1[2,1]._transformation_block)

    self.assertTrue(m.b[0].disjunct.active)
    self.assertTrue(m.b[0].disjunct[0].active)
    self.assertIsNone(m.b[0].disjunct[0]._transformation_block)
    self.assertTrue(m.b[0].disjunct[1].active)
    self.assertIsNone(m.b[0].disjunct[1]._transformation_block)
    self.assertTrue(m.b[1].disjunct0.active)
    self.assertIsNone(m.b[1].disjunct0._transformation_block)
    self.assertTrue(m.b[1].disjunct1.active)
    self.assertIsNone(m.b[1].disjunct1._transformation_block)

def check_disjData_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1[2]])

    disjBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
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
        self.assertIs(m.disjunct1[i].transformation_block(), disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.disjunct1[i])

def check_indexedBlock_targets_inactive(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b])

    self.assertTrue(m.disjunct1.active)
    self.assertTrue(m.disjunct1[1,0].active)
    self.assertTrue(m.disjunct1[1,1].active)
    self.assertTrue(m.disjunct1[2,0].active)
    self.assertTrue(m.disjunct1[2,1].active)
    self.assertIsNone(m.disjunct1[1,0].transformation_block)
    self.assertIsNone(m.disjunct1[1,1].transformation_block)
    self.assertIsNone(m.disjunct1[2,0].transformation_block)
    self.assertIsNone(m.disjunct1[2,1].transformation_block)

    self.assertFalse(m.b[0].disjunct.active)
    self.assertFalse(m.b[0].disjunct[0].active)
    self.assertFalse(m.b[0].disjunct[1].active)
    self.assertFalse(m.b[1].disjunct0.active)
    self.assertFalse(m.b[1].disjunct1.active)

def check_indexedBlock_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.b])

    disjBlock1 = m.b[0].component("_pyomo_gdp_%s_relaxation" % transformation).\
                 relaxedDisjuncts
    self.assertEqual(len(disjBlock1), 2)
    self.assertIsInstance(disjBlock1[0].component("b[0].disjunct[0].c"),
                          Constraint)
    self.assertIsInstance(disjBlock1[1].component("b[0].disjunct[1].c"),
                          Constraint)
    disjBlock2 = m.b[1].component("_pyomo_gdp_%s_relaxation" % transformation).\
                 relaxedDisjuncts
    self.assertEqual(len(disjBlock2), 2)
    self.assertIsInstance(disjBlock2[0].component("b[1].disjunct0.c"),
                          Constraint)
    self.assertIsInstance(disjBlock2[1].component("b[1].disjunct1.c"),
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
            ('disjunct0',None,0),
            ('disjunct1',None,1),
        ]
    }

    for blocknum, lst in iteritems(pairs):
        for comp, i, j in lst:
            original = m.b[blocknum].component(comp)
            if blocknum == 0:
                disjBlock = disjBlock1
            if blocknum == 1:
                disjBlock = disjBlock2
            self.assertIs(original[i].transformation_block(), disjBlock[j])
            self.assertIs(trans.get_src_disjunct(disjBlock[j]), original[i])

def check_blockData_targets_inactive(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0]])

    checkb0TargetsInactive(self, m)

def check_blockData_only_targets_transformed(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0]])
    checkb0TargetsTransformed(self, m, transformation)

def check_do_not_transform_deactivated_targets(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    m.b[1].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0], m.b[1]])

    checkb0TargetsInactive(self, m)
    checkb0TargetsTransformed(self, m, transformation)

def check_disjunction_data_target(self, transformation):
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[2]])

    # we got a transformation block on the model
    transBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component("disjunction_xor"),
                          Constraint)
    self.assertIsInstance(transBlock.disjunction_xor[2],
                          constraint._GeneralConstraintData)
    self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 3)

    # suppose we transform the next one separately
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[1]])
    # we added to the same XOR constraint before
    self.assertIsInstance(transBlock.disjunction_xor[1], 
                          constraint._GeneralConstraintData)
    # we used the same transformation block, so we have more relaxed
    # disjuncts
    self.assertEqual(len(transBlock.relaxedDisjuncts), 6)

def check_disjunction_data_target_any_index(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))
    m.disjunct3 = Disjunct(Any)
    m.disjunct4 = Disjunct(Any)
    m.disjunction2=Disjunction(Any)
    for i in range(2):
        m.disjunct3[i].cons = Constraint(expr=m.x == 2)
        m.disjunct4[i].cons = Constraint(expr=m.x <= 3)
        m.disjunction2[i] = [m.disjunct3[i], m.disjunct4[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(
            m, targets=[m.disjunction2[i]]) 

        if i == 0:
            check_relaxation_block(self, m, "_pyomo_gdp_%s_relaxation" %
                                   transformation, 2)
        if i == 2:
            check_relaxation_block(self, m, "_pyomo_gdp_%s_relaxation" %
                                   transformation, 4)

# disjunction generation tests

def check_iteratively_adding_to_indexed_disjunction_on_block(self,
                                                             transformation):
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var(bounds=(-100, 100))
    m.b.firstTerm = Disjunct([1,2])
    m.b.firstTerm[1].cons = Constraint(expr=m.b.x == 0)
    m.b.firstTerm[2].cons = Constraint(expr=m.b.x == 2)
    m.b.secondTerm = Disjunct([1,2])
    m.b.secondTerm[1].cons = Constraint(expr=m.b.x >= 2)
    m.b.secondTerm[2].cons = Constraint(expr=m.b.x >= 3)
    m.b.disjunctionList = Disjunction(Any)

    m.b.obj = Objective(expr=m.b.x)

    for i in range(1,3):
        m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(m,
                                                                  targets=[m.b])
        m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(m,
                                                                  targets=[m.b])

        if i == 1:
            check_relaxation_block(self, m.b, "_pyomo_gdp_%s_relaxation" %
                                   transformation, 2)
        if i == 2:
            check_relaxation_block(self, m.b, "_pyomo_gdp_%s_relaxation" %
                                   transformation, 4)

def check_simple_disjunction_of_disjunct_datas(self, transformation):
    # This is actually a reasonable use case if you are generating
    # disjunctions with the same structure. So you might have Disjuncts
    # indexed by Any and disjunctions indexed by Any and be adding a
    # disjunction of two of the DisjunctDatas in every iteration.
    m = models.makeDisjunctionOfDisjunctDatas()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    self.check_trans_block_disjunctions_of_disjunct_datas(m)
    transBlock = m.component("_pyomo_gdp_%s_relaxation" % transformation)
    self.assertIsInstance( transBlock.component("disjunction_xor"),
                           Constraint)
    transBlock2 = m.component("_pyomo_gdp_%s_relaxation_4" % transformation)
    self.assertIsInstance( transBlock2.component("disjunction2_xor"),
                           Constraint)

# these tests have different checks for what ends up on the model, but they have
# the same structure
def check_iteratively_adding_disjunctions_transform_container(self,
                                                              transformation):
    # If you are iteratively adding Disjunctions to an IndexedDisjunction,
    # then if you are lazy about what you transform, you might shoot
    # yourself in the foot because if the whole IndexedDisjunction gets
    # deactivated by the first transformation, the new DisjunctionDatas
    # don't get transformed. Interestingly, this isn't what happens. We
    # deactivate the container and then still transform what's inside. I
    # don't think we should deactivate the container at all, maybe?
    model = ConcreteModel()
    model.x = Var(bounds=(-100, 100))
    model.disjunctionList = Disjunction(Any)
    model.obj = Objective(expr=model.x)
    for i in range(2):
        firstTermName = "firstTerm[%s]" % i
        model.add_component(firstTermName, Disjunct())
        model.component(firstTermName).cons = Constraint(
            expr=model.x == 2*i)
        secondTermName = "secondTerm[%s]" % i
        model.add_component(secondTermName, Disjunct())
        model.component(secondTermName).cons = Constraint(
            expr=model.x >= i + 2)
        model.disjunctionList[i] = [model.component(firstTermName),
                                    model.component(secondTermName)]

        # we're lazy and we just transform the disjunctionList (and in
        # theory we are transforming at every iteration because we are
        # solving at every iteration)
        TransformationFactory('gdp.%s' % transformation).apply_to(
            model, targets=[model.disjunctionList])
        if i == 0:
            self.check_first_iteration(model)

        if i == 1:
            self.check_second_iteration(model)

# transforming blocks

# If you transform a block as if it is a model, the transformation should
# only modify the block you passed it, else when you solve the block, you
# are missing the disjunction you thought was on there.
def check_transformation_simple_block(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b)

    # transformation block not on m
    self.assertIsNone(m.component("_pyomo_gdp_%s_relaxation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(m.b.component("_pyomo_gdp_%s_relaxation" %
                                        transformation), Block)

def check_transform_block_data(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b[0])

    self.assertIsNone(m.component("_pyomo_gdp_%s_relaxation" % transformation))

    self.assertIsInstance(m.b[0].component("_pyomo_gdp_%s_relaxation" %
                                           transformation), Block)

def check_simple_block_target(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])

    # transformation block not on m
    self.assertIsNone(m.component("_pyomo_gdp_%s_relaxation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(m.b.component("_pyomo_gdp_%s_relaxation" %
                                        transformation), Block)

def check_block_data_target(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m,
                                                              targets=[m.b[0]])

    self.assertIsNone(m.component("_pyomo_gdp_%s_relaxation" % transformation))

    self.assertIsInstance(m.b[0].component("_pyomo_gdp_%s_relaxation" %
                                           transformation), Block)

def check_indexed_block_target(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])

    # We expect the transformation block on each of the BlockDatas. Because
    # it is always going on the parent block of the disjunction.

    self.assertIsNone(m.component("_pyomo_gdp_%s_relaxation" % transformation))

    for i in [0,1]:
        self.assertIsInstance( m.b[i].component("_pyomo_gdp_%s_relaxation" %
                                                transformation), Block)

def check_block_targets_inactive(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b])

    self.assertFalse(m.b.disjunct[0].active)
    self.assertFalse(m.b.disjunct[1].active)
    self.assertFalse(m.b.disjunct.active)
    self.assertTrue(m.simpledisj.active)
    self.assertTrue(m.simpledisj2.active)

def check_block_only_targets_transformed(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.b])

    disjBlock = m.b.component("_pyomo_gdp_%s_relaxation" % transformation).\
                relaxedDisjuncts
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
        self.assertIs(m.b.disjunct[i].transformation_block(), disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.b.disjunct[i])

# common error messages

def check_transform_empty_disjunction(self, transformation):
    m = ConcreteModel()
    m.empty = Disjunction(expr=[])

    self.assertRaisesRegexp(
        GDP_Error,
        "Disjunction empty is empty. This is likely indicative of a "
        "modeling error.*",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_deactivated_disjunct_nonzero_indicator_var(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(0,8))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].indicator_var.fix(1)

    self.assertRaisesRegexp(
        GDP_Error,
        "The disjunct disjunction_disjuncts\[0\] is deactivated, but the "
        "indicator_var is fixed to 1. This makes no sense.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_deactivated_disjunct_unfixed_indicator_var(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(0,8))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].indicator_var.fixed = False

    self.assertRaisesRegexp(
        GDP_Error,
        "The disjunct disjunction_disjuncts\[0\] is deactivated, but the "
        "indicator_var is not fixed and the disjunct does not "
        "appear to have been relaxed. This makes no sense. "
        "\(If the intent is to deactivate the disjunct, fix its "
        "indicator_var to 0.\)",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_retrieving_nondisjunctive_components(self, transformation):
    m = models.makeTwoTermDisj()
    m.b = Block()
    m.b.global_cons = Constraint(expr=m.a + m.x >= 8)
    m.another_global_cons = Constraint(expr=m.a + m.x <= 11)

    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint b.global_cons is not on a disjunct and so was not "
        "transformed",
        trans.get_transformed_constraint,
        m.b.global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint b.global_cons is not a transformed constraint",
        trans.get_src_constraint,
        m.b.global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint another_global_cons is not a transformed constraint",
        trans.get_src_constraint,
        m.another_global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Block b doesn't appear to be a transformation block for a "
        "disjunct. No source disjunct found.",
        trans.get_src_disjunct,
        m.b)

    self.assertRaisesRegexp(
        GDP_Error,
        "It appears that another_global_cons is not an XOR or OR"
        " constraint resulting from transforming a Disjunction.",
        trans.get_src_disjunction,
        m.another_global_cons)

def check_silly_target(self, transformation):
    m = models.makeTwoTermDisj()
    self.assertRaisesRegexp(
        GDP_Error,
        "Target d\[1\].c1 was not a Block, Disjunct, or Disjunction. "
        "It was of type "
        "<class 'pyomo.core.base.constraint.SimpleConstraint'> and "
        "can't be transformed.",
        TransformationFactory('gdp.chull').apply_to,
        m,
        targets=[m.d[1].c1])

def check_ask_for_transformed_constraint_from_untransformed_disjunct(
        self, transformation):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=m.disjunction[1])

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint disjunct\[2,b\].cons_b is on a disjunct which has "
        "not been transformed",
        trans.get_transformed_constraint,
        m.disjunct[2, 'b'].cons_b)

# This is really neurotic, but test that we will create an infeasible XOR
# constraint. We have to because in the case of nested disjunctions, our model
# is not necessarily infeasible because of this. It just might make a Disjunct
# infeasible.
def setup_infeasible_xor_because_all_disjuncts_deactivated(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(0,8))
    m.y = Var(bounds=(0,7))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])
    m.disjunction_disjuncts[0].nestedDisjunction = Disjunction(
        expr=[m.y == 6, m.y <= 1])
    # Note that this fixes the indicator variables to 0, but since the
    # disjunction is still active, the XOR constraint will be created. So we
    # will have to land in the second disjunct of m.disjunction
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[1].deactivate()
    # This should create a 0 = 1 XOR constraint, actually...
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, 
        targets=m.disjunction.disjuncts[0].nestedDisjunction)

    # check that our XOR is the bad thing it should be.
    transBlock = m.disjunction.disjuncts[0].component(
        "_pyomo_gdp_%s_relaxation" % transformation)
    xor = transBlock.component(
        "disjunction_disjuncts[0].nestedDisjunction_xor")
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(value(xor.lower), 1)
    self.assertEqual(value(xor.upper), 1)
    repn = generate_standard_repn(xor.body)
    for v in repn.linear_vars:
        self.assertTrue(v.is_fixed())
        self.assertEqual(value(v), 0)

    # make sure when we transform the outer thing, all is well
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    return m
