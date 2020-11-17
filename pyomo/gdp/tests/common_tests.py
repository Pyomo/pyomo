#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


from pyomo.environ import TransformationFactory, ConcreteModel, Constraint, Var, Objective, Block, Any, RangeSet, Expression, value
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, ComponentUID
from pyomo.repn import generate_standard_repn
import pyomo.gdp.tests.models as models
from six import StringIO, iteritems
import random

# utitility functions

def check_linear_coef(self, repn, var, coef):
    # utility used to check a variable-coefficient pair in a standard_repn
    var_id = None
    for i,v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.linear_coefs[var_id], coef)

def diff_apply_to_and_create_using(self, model, transformation):
    # compares the pprint from the transformed model after using both apply_to
    # and create_using to make sure the two do the same thing
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
    # utility for checking the transformation block (this method is generic to
    # bigm and hull though there is more on the hull transformation block, and
    # the lbub set differs between the two
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
    disjBlock = m.b[0].component("_pyomo_gdp_%s_reformulation" % transformation).\
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
    # check that we do not transform a deactivated DisjunctData
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m, targets=(m,))

    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.d[1].active)

    rBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    disjBlock = rBlock.relaxedDisjuncts
    self.assertEqual(len(disjBlock), 1)
    self.assertIs(disjBlock[0], m.d[1].transformation_block())
    self.assertIs(transform.get_src_disjunct(disjBlock[0]), m.d[1])

def check_improperly_deactivated_disjuncts(self, transformation):
    # check that if a Disjunct is deactivated but its indicator variable is not
    # fixed to 0, we express our confusion.
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    self.assertEqual(value(m.d[0].indicator_var), 0)
    self.assertTrue(m.d[0].indicator_var.is_fixed())
    m.d[0].indicator_var.fix(1)
    self.assertRaisesRegexp(
        GDP_Error,
        "The disjunct 'd\[0\]' is deactivated, but the "
        "indicator_var is fixed to 1. This makes no sense.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_indexed_disjunction_not_transformed(self, m, transformation):
    # no transformation block, nothing transformed
    self.assertIsNone(m.component("_pyomo_gdp_%s_transformation"
                                  % transformation))
    for idx in m.disjunct:
        self.assertIsNone(m.disjunct[idx].transformation_block)
    for idx in m.disjunction:
        self.assertIsNone(m.disjunction[idx].algebraic_constraint)

def check_do_not_transform_userDeactivated_indexedDisjunction(self,
                                                              transformation):
    # check that we do not transform a deactivated disjunction
    m = models.makeTwoTermIndexedDisjunction()
    # If you truly want to transform nothing, deactivate everything
    m.disjunction.deactivate()
    for idx in m.disjunct:
        m.disjunct[idx].deactivate()
    directly = TransformationFactory('gdp.%s' % transformation).create_using(m)
    check_indexed_disjunction_not_transformed(self, directly, transformation)

    targets = TransformationFactory('gdp.%s' % transformation).create_using(
        m, targets=(m.disjunction))
    check_indexed_disjunction_not_transformed(self, targets, transformation)

def check_disjunction_deactivated(self, transformation):
    # check that we deactivate disjunctions after we transform them
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    oldblock = m.component("disjunction")
    self.assertIsInstance(oldblock, Disjunction)
    self.assertFalse(oldblock.active)

def check_disjunctDatas_deactivated(self, transformation):
    # check that we deactivate disjuncts after we transform them
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    oldblock = m.component("disjunction")
    self.assertFalse(oldblock.disjuncts[0].active)
    self.assertFalse(oldblock.disjuncts[1].active)

def check_deactivated_constraints(self, transformation):
    # test that we deactivate constraints after we transform them
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

def check_deactivated_disjuncts(self, transformation):
    # another test that we deactivated transformed Disjuncts, but this one
    # includes a SimpleDisjunct as well
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))
    # all the disjuncts got transformed, so all should be deactivated
    for i in m.disjunct.index_set():
        self.assertFalse(m.disjunct[i].active)
    self.assertFalse(m.disjunct.active)

def check_deactivated_disjunctions(self, transformation):
    # another test that we deactivated transformed Disjunctions, but including a
    # SimpleDisjunction
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    # all the disjunctions got transformed, so they should be
    # deactivated too
    for i in m.disjunction.index_set():
        self.assertFalse(m.disjunction[i].active)
    self.assertFalse(m.disjunction.active)

def check_do_not_transform_twice_if_disjunction_reactivated(self,
                                                            transformation):
    # test that if an already-transformed disjunction is reactivated, we will
    # not retransform it in a subsequent call to the transformation.
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
        "The disjunct 'd\[0\]' has been transformed, but a disjunction "
        "it appears in has not. Putting the same disjunct in "
        "multiple disjunctions is not supported.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

def check_constraints_deactivated_indexedDisjunction(self, transformation):
    # check that we deactivate transformed constraints
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    for i in m.disjunct.index_set():
        self.assertFalse(m.disjunct[i].c.active)

def check_partial_deactivate_indexed_disjunction(self, transformation):
    """Test for partial deactivation of an indexed disjunction."""
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
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertEqual(
        len(transBlock.disj_xor), 1,
        "There should only be one XOR constraint generated. Found %s." %
        len(transBlock.disj_xor))

# transformation block

def check_transformation_block_name_collision(self, transformation):
    # make sure that if the model already has a block called
    # _pyomo_gdp_*_relaxation that we come up with a different name for the
    # transformation block (and put the relaxed disjuncts on it)
    m = models.makeTwoTermDisj()
    # add block with the name we are about to try to use
    m.add_component("_pyomo_gdp_%s_reformulation" % transformation, Block(Any))
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    # check that we got a uniquely named block
    transBlock = m.component("_pyomo_gdp_%s_reformulation_4" % transformation)
    self.assertIsInstance(transBlock, Block)

    # check that the relaxed disjuncts really are here.
    disjBlock = transBlock.relaxedDisjuncts
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    self.assertIsInstance(disjBlock[0].component("d[0].c"), Constraint)
    self.assertIsInstance(disjBlock[1].component("d[1].c1"), Constraint)
    self.assertIsInstance(disjBlock[1].component("d[1].c2"), Constraint)

    # we didn't add to the block that wasn't ours
    self.assertEqual(len(m.component("_pyomo_gdp_%s_reformulation" %
                                     transformation)), 0)

# XOR constraints

def check_indicator_vars(self, transformation):
    # particularly paranoid test checking that the indicator_vars are intact
    # after transformation
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
    # verify xor constraint for a SimpleDisjunction
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    # make sure we created the xor constraint and put it on the relaxation
    # block
    rBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
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
    # verify xor constraint for an IndexedDisjunction
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    xor = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
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

def check_indexed_xor_constraints_with_targets(self, transformation):
    # check that when we use targets to specfy some DisjunctionDatas in an
    # IndexedDisjunction, the xor constraint is indexed correctly
    m = models.makeTwoTermIndexedDisjunction_BoundedVars()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.disjunction[1],
                 m.disjunction[3]])

    xorC = m.disjunction[1].algebraic_constraint().parent_component()
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

def check_three_term_xor_constraint(self, transformation):
    # check that the xor constraint has all the indicator variables from a
    # three-term disjunction
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    xor = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
          component("disjunction_xor")
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


# mappings

def check_xor_constraint_mapping(self, transformation):
    # test that we correctly map between disjunctions and XOR constraints
    m = models.makeTwoTermDisj()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIs( trans.get_src_disjunction(transBlock.disjunction_xor),
                   m.disjunction)
    self.assertIs( m.disjunction.algebraic_constraint(),
                   transBlock.disjunction_xor)


def check_xor_constraint_mapping_two_disjunctions(self, transformation):
    # test that we correctly map between disjunctions and xor constraints when
    # we have multiple SimpleDisjunctions (probably redundant with the above)
    m = models.makeDisjunctionOfDisjunctDatas()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    transBlock2 = m.component("_pyomo_gdp_%s_reformulation_4" % transformation)
    self.assertIs( trans.get_src_disjunction(transBlock.disjunction_xor),
                   m.disjunction)
    self.assertIs( trans.get_src_disjunction(transBlock2.disjunction2_xor),
                   m.disjunction2)

    self.assertIs( m.disjunction.algebraic_constraint(),
                   transBlock.disjunction_xor)
    self.assertIs( m.disjunction2.algebraic_constraint(),
                   transBlock2.disjunction2_xor)

def check_disjunct_mapping(self, transformation):
    # check that we correctly map between Disjuncts and their transformation
    # blocks
    m = models.makeTwoTermDisj_Nonlinear()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    disjBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
                relaxedDisjuncts

    # the disjuncts will always be transformed in the same order,
    # and d[0] goes first, so we can check in a loop.
    for i in [0,1]:
        self.assertIs(disjBlock[i]._srcDisjunct(), m.d[i])
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.d[i])

# targets

def check_only_targets_inactive(self, transformation):
    # test that we only transform targets (by checking active status)
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
    # test that we only transform targets (by checking the actual components)
    m = models.makeTwoSimpleDisjunctions()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1])

    disjBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
                relaxedDisjuncts
    # only two disjuncts relaxed
    self.assertEqual(len(disjBlock), 2)
    # Note that in hull, these aren't the only components that get created, but
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

def check_targets_with_container_as_arg(self, transformation):
    # check that we can give a Disjunction as the argument to the transformation
    # and use targets to specify a DisjunctionData to transform
    m = models.makeTwoTermIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m.disjunction,
        targets=(m.disjunction[2]))
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIsNone(m.disjunction[1].algebraic_constraint)
    self.assertIsNone(m.disjunction[3].algebraic_constraint)
    self.assertIs(m.disjunction[2].algebraic_constraint(),
                  transBlock.disjunction_xor[2])
    self.assertIs(m.disjunction._algebraic_constraint(),
                  transBlock.disjunction_xor)

def check_target_not_a_component_error(self, transformation):
    # test error message for crazy targets
    decoy = ConcreteModel()
    decoy.block = Block()
    m = models.makeTwoSimpleDisjunctions()
    self.assertRaisesRegexp(
        GDP_Error,
        "Target 'block' is not a component on instance 'unknown'!",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[decoy.block])

def check_targets_cannot_be_cuids(self, transformation):
    # check that we scream if targets are cuids
    m = models.makeTwoTermDisj()
    self.assertRaisesRegexp(
        ValueError,
        "invalid value for configuration 'targets':\n"
        "\tFailed casting \[disjunction\]\n"
        "\tto target_list\n"
        "\tError: Expected Component or list of Components."
        "\n\tRecieved %s" % type(ComponentUID(m.disjunction)),
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[ComponentUID(m.disjunction)])

def check_indexedDisj_targets_inactive(self, transformation):
    # check that targets are deactivated (when target is IndexedDisjunction)
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
    # check that only the targets are transformed (with IndexedDisjunction as
    # target)
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1])

    disjBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
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
    # Check that we complain if we find an untransformed Disjunct inside of
    # another Disjunct we are transforming
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
        "Found active disjunct 'disjunct1\[1,1\].innerdisjunct\[0\]' "
        "in disjunct 'disjunct1\[1,1\]'!.*",
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
        "Found untransformed disjunction 'disjunct1\[1,1\]."
        "innerdisjunction\[0\]' in disjunct 'disjunct1\[1,1\]'!.*",
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]])
    # Deactivating the disjunction will allow us to get past it back
    # to the Disjunct (after we realize there are no active
    # DisjunctionData within the active Disjunction)
    m.disjunct1[1,1].innerdisjunction[0].deactivate()
    self.assertRaisesRegexp(
        GDP_Error,
        "Found active disjunct 'disjunct1\[1,1\].innerdisjunct\[0\]' "
        "in disjunct 'disjunct1\[1,1\]'!.*",
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]])

def check_disjData_targets_inactive(self, transformation):
    # check targets deactivated with DisjunctionData is the target
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
    # check that targets are transformed when DisjunctionData is the target
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.disjunction1[2]])

    disjBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation).\
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
    # check that targets are deactivated when target is an IndexedBlock
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
    # check that targets are transformed when target is an IndexedBlock
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(
        m,
        targets=[m.b])

    disjBlock1 = m.b[0].component("_pyomo_gdp_%s_reformulation" % transformation).\
                 relaxedDisjuncts
    self.assertEqual(len(disjBlock1), 2)
    self.assertIsInstance(disjBlock1[0].component("b[0].disjunct[0].c"),
                          Constraint)
    self.assertIsInstance(disjBlock1[1].component("b[0].disjunct[1].c"),
                          Constraint)
    disjBlock2 = m.b[1].component("_pyomo_gdp_%s_reformulation" % transformation).\
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
    # test that BlockData target is deactivated
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0]])

    checkb0TargetsInactive(self, m)

def check_blockData_only_targets_transformed(self, transformation):
    # test that BlockData target is transformed
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0]])
    checkb0TargetsTransformed(self, m, transformation)

def check_do_not_transform_deactivated_targets(self, transformation):
    # test that if a deactivated component is given as a target, we don't
    # transform it. (This is actually an important test because it is the only
    # reason to check active status at the beginning of many of the methods in
    # the transformation like _transform_disjunct and _transform_disjunction. In
    # the absence of targets, those checks wouldn't be necessary.)
    m = models.makeDisjunctionsOnIndexedBlock()
    m.b[1].deactivate()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.b[0], m.b[1]])

    checkb0TargetsInactive(self, m)
    checkb0TargetsTransformed(self, m, transformation)

def check_disjunction_data_target(self, transformation):
    # test that if we transform DisjunctionDatas one at a time, we get what we
    # expect in terms of using the same transformation block and the indexing of
    # the xor constraint.
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[2]])

    # we got a transformation block on the model
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
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
    # check the same as the above, but that it still works when the Disjunction
    # is indexed by Any.
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
            check_relaxation_block(self, m, "_pyomo_gdp_%s_reformulation" %
                                   transformation, 2)
        if i == 2:
            check_relaxation_block(self, m, "_pyomo_gdp_%s_reformulation" %
                                   transformation, 4)

# tests that we treat disjunctions on blocks correctly (the main issue here is
# that if you were to solve that block post-transformation that you would have
# the whole transformed model)

def check_xor_constraint_added(self, transformation):
    # test we put the xor on the transformation block
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    self.assertIsInstance(
        m.b.component("_pyomo_gdp_%s_reformulation" % transformation).\
        component('b.disjunction_xor'), Constraint)

def check_trans_block_created(self, transformation):
    # check we put the transformation block on the parent block of the
    # disjunction
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    # test that the transformation block go created on the model
    transBlock = m.b.component('_pyomo_gdp_%s_reformulation' % transformation)
    self.assertIsInstance(transBlock, Block)
    disjBlock = transBlock.component("relaxedDisjuncts")
    self.assertIsInstance(disjBlock, Block)
    self.assertEqual(len(disjBlock), 2)
    # and that it didn't get created on the model
    self.assertIsNone(m.component('_pyomo_gdp_%s_reformulation' % transformation))


# disjunction generation tests: These all suppose that you are doing some sort
# of column and constraint generation algorithm, but you are in fact generating
# Disjunctions and retransforming the model after each addition.

def check_iteratively_adding_to_indexed_disjunction_on_block(self,
                                                             transformation):
    # check that we can iteratively add to an IndexedDisjunction and transform
    # the block it lives on
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
            check_relaxation_block(self, m.b, "_pyomo_gdp_%s_reformulation" %
                                   transformation, 2)
        if i == 2:
            check_relaxation_block(self, m.b, "_pyomo_gdp_%s_reformulation" %
                                   transformation, 4)

def check_simple_disjunction_of_disjunct_datas(self, transformation):
    # This is actually a reasonable use case if you are generating
    # disjunctions with the same structure. So you might have Disjuncts
    # indexed by Any and disjunctions indexed by Any and be adding a
    # disjunction of two of the DisjunctDatas in every iteration.
    m = models.makeDisjunctionOfDisjunctDatas()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    self.check_trans_block_disjunctions_of_disjunct_datas(m)
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIsInstance( transBlock.component("disjunction_xor"),
                           Constraint)
    transBlock2 = m.component("_pyomo_gdp_%s_reformulation_4" % transformation)
    self.assertIsInstance( transBlock2.component("disjunction2_xor"),
                           Constraint)

# these tests have different checks for what ends up on the model between bigm
# and hull, but they have the same structure
def check_iteratively_adding_disjunctions_transform_container(self,
                                                              transformation):
    # Check that we can play the same game with iteratively adding Disjunctions,
    # but this time just specify the IndexedDisjunction as the argument. Note
    # that the success of this depends on our rebellion regarding the active
    # status of containers.
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

def check_disjunction_and_disjuncts_indexed_by_any(self, transformation):
    # check that we can play the same game when the Disjuncts also are indexed
    # by Any
    model = ConcreteModel()
    model.x = Var(bounds=(-100, 100))

    model.firstTerm = Disjunct(Any)
    model.secondTerm = Disjunct(Any)
    model.disjunctionList = Disjunction(Any)

    model.obj = Objective(expr=model.x)

    for i in range(2):
        model.firstTerm[i].cons = Constraint(expr=model.x == 2*i)
        model.secondTerm[i].cons = Constraint(expr=model.x >= i + 2)
        model.disjunctionList[i] = [model.firstTerm[i], model.secondTerm[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(model)

        if i == 0:
            self.check_first_iteration(model)

        if i == 1:
            self.check_second_iteration(model)

def check_iteratively_adding_disjunctions_transform_model(self, transformation):
    # Same as above, but transforming whole model in every iteration
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

        # we're lazy and we just transform the model (and in
        # theory we are transforming at every iteration because we are
        # solving at every iteration)
        TransformationFactory('gdp.%s' % transformation).apply_to(model)
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
    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(m.b.component("_pyomo_gdp_%s_reformulation" %
                                        transformation), Block)

def check_transform_block_data(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b[0])

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    self.assertIsInstance(m.b[0].component("_pyomo_gdp_%s_reformulation" %
                                           transformation), Block)

def check_simple_block_target(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])

    # transformation block not on m
    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(m.b.component("_pyomo_gdp_%s_reformulation" %
                                        transformation), Block)

def check_block_data_target(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m,
                                                              targets=[m.b[0]])

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    self.assertIsInstance(m.b[0].component("_pyomo_gdp_%s_reformulation" %
                                           transformation), Block)

def check_indexed_block_target(self, transformation):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])

    # We expect the transformation block on each of the BlockDatas. Because
    # it is always going on the parent block of the disjunction.

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    for i in [0,1]:
        self.assertIsInstance( m.b[i].component("_pyomo_gdp_%s_reformulation" %
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

    disjBlock = m.b.component("_pyomo_gdp_%s_reformulation" % transformation).\
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
        "Disjunction 'empty' is empty. This is likely indicative of a "
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
        "The disjunct 'disjunction_disjuncts\[0\]' is deactivated, but the "
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
        "The disjunct 'disjunction_disjuncts\[0\]' is deactivated, but the "
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
        "Constraint 'b.global_cons' is not on a disjunct and so was not "
        "transformed",
        trans.get_transformed_constraints,
        m.b.global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint 'b.global_cons' is not a transformed constraint",
        trans.get_src_constraint,
        m.b.global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint 'another_global_cons' is not a transformed constraint",
        trans.get_src_constraint,
        m.another_global_cons)

    self.assertRaisesRegexp(
        GDP_Error,
        "Block 'b' doesn't appear to be a transformation block for a "
        "disjunct. No source disjunct found.",
        trans.get_src_disjunct,
        m.b)

    self.assertRaisesRegexp(
        GDP_Error,
        "It appears that 'another_global_cons' is not an XOR or OR"
        " constraint resulting from transforming a Disjunction.",
        trans.get_src_disjunction,
        m.another_global_cons)

def check_silly_target(self, transformation):
    m = models.makeTwoTermDisj()
    self.assertRaisesRegexp(
        GDP_Error,
        "Target 'd\[1\].c1' was not a Block, Disjunct, or Disjunction. "
        "It was of type "
        "<class 'pyomo.core.base.constraint.SimpleConstraint'> and "
        "can't be transformed.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[m.d[1].c1])

def check_ask_for_transformed_constraint_from_untransformed_disjunct(
        self, transformation):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=m.disjunction[1])

    self.assertRaisesRegexp(
        GDP_Error,
        "Constraint 'disjunct\[2,b\].cons_b' is on a disjunct which has "
        "not been transformed",
        trans.get_transformed_constraints,
        m.disjunct[2, 'b'].cons_b)

def check_error_for_same_disjunct_in_multiple_disjunctions(self, transformation):
    m = models.makeDisjunctInMultipleDisjunctions()
    self.assertRaisesRegexp(
        GDP_Error,
        "The disjunct 'disjunct1\[1\]' has been transformed, "
        "but a disjunction it appears in has not. Putting the same "
        "disjunct in multiple disjunctions is not supported.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m)

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
        "_pyomo_gdp_%s_reformulation" % transformation)
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

def check_disjunction_target_err(self, transformation):
    m = models.makeNestedDisjunctions()
    self.assertRaisesRegexp(
        GDP_Error,
        "Found active disjunct 'simpledisjunct.innerdisjunct0' in "
        "disjunct 'simpledisjunct'!.*",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[m.disjunction])

def check_activeInnerDisjunction_err(self, transformation):
    m = models.makeDuplicatedNestedDisjunction()
    self.assertRaisesRegexp(
        GDP_Error,
        "Found untransformed disjunction "
        "'outerdisjunct\[1\].duplicateddisjunction' in disjunct "
        "'outerdisjunct\[1\]'! The disjunction must be transformed before "
        "the disjunct. If you are using targets, put the disjunction "
        "before the disjunct in the list.*",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[m.outerdisjunct[1].innerdisjunction,
                 m.disjunction])


# nested disjunctions: hull and bigm have very different handling for nested
# disjunctions, but these tests check *that* everything is transformed, not how

def check_disjuncts_inactive_nested(self, transformation):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,))

    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.simpledisjunct.active)
    self.assertFalse(m.disjunct[0].active)
    self.assertFalse(m.disjunct[1].active)
    self.assertFalse(m.disjunct.active)

def check_deactivated_disjunct_leaves_nested_disjunct_active(self,
                                                             transformation):
    m = models.makeNestedDisjunctions_FlatDisjuncts()
    m.d1.deactivate()
    # Specifying 'targets' prevents the HACK_GDP_Disjunct_Reclassifier
    # transformation of Disjuncts to Blocks
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m])

    self.assertFalse(m.d1.active)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertEqual(m.d1.indicator_var.value, 0)

    self.assertFalse(m.d2.active)
    self.assertFalse(m.d2.indicator_var.fixed)

    self.assertTrue(m.d3.active)
    self.assertFalse(m.d3.indicator_var.fixed)

    self.assertTrue(m.d4.active)
    self.assertFalse(m.d4.indicator_var.fixed)

    m = models.makeNestedDisjunctions_NestedDisjuncts()
    m.d1.deactivate()
    # Specifying 'targets' prevents the HACK_GDP_Disjunct_Reclassifier
    # transformation of Disjuncts to Blocks
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m])

    self.assertFalse(m.d1.active)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertEqual(m.d1.indicator_var.value, 0)

    self.assertFalse(m.d2.active)
    self.assertFalse(m.d2.indicator_var.fixed)

    self.assertTrue(m.d1.d3.active)
    self.assertFalse(m.d1.d3.indicator_var.fixed)

    self.assertTrue(m.d1.d4.active)
    self.assertFalse(m.d1.d4.indicator_var.fixed)

def check_mappings_between_disjunctions_and_xors(self, transformation):
    m = models.makeNestedDisjunctions()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)

    disjunctionPairs = [
        (m.disjunction, transBlock.disjunction_xor),
        (m.disjunct[1].innerdisjunction[0],
         m.disjunct[1].component("_pyomo_gdp_%s_reformulation" 
                                 % transformation).\
         component("disjunct[1].innerdisjunction_xor")[0]),
        (m.simpledisjunct.innerdisjunction,
         m.simpledisjunct.component(
             "_pyomo_gdp_%s_reformulation" % transformation).component(
                 "simpledisjunct.innerdisjunction_xor"))
     ]

    # check disjunction mappings
    for disjunction, xor in disjunctionPairs:
        self.assertIs(disjunction.algebraic_constraint(), xor)
        self.assertIs(transform.get_src_disjunction(xor), disjunction)

def check_disjunct_targets_inactive(self, transformation):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.simpledisjunct])

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

def check_disjunct_only_targets_transformed(self, transformation):
    m = models.makeNestedDisjunctions()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(
        m,
        targets=[m.simpledisjunct])

    disjBlock = m.simpledisjunct.component("_pyomo_gdp_%s_reformulation" %
                                           transformation).relaxedDisjuncts
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
        self.assertIs(m.simpledisjunct.component('innerdisjunct%d'%i),
                      transform.get_src_disjunct(disjBlock[j]))
        self.assertIs(disjBlock[j],
                      m.simpledisjunct.component(
                          'innerdisjunct%d'%i).transformation_block())

def check_disjunctData_targets_inactive(self, transformation):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m,
        targets=[m.disjunct[1]])

    self.assertTrue(m.disjunct[0].active)
    self.assertTrue(m.disjunct[1].active)
    self.assertTrue(m.disjunct.active)
    self.assertFalse(m.disjunct[1].innerdisjunct[0].active)
    self.assertFalse(m.disjunct[1].innerdisjunct[1].active)
    self.assertFalse(m.disjunct[1].innerdisjunct.active)

    self.assertTrue(m.simpledisjunct.active)
    self.assertTrue(m.simpledisjunct.innerdisjunct0.active)
    self.assertTrue(m.simpledisjunct.innerdisjunct1.active)

def check_disjunctData_only_targets_transformed(self, transformation):
    m = models.makeNestedDisjunctions()
    # This is so convoluted, but you can treat a disjunct like a block:
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(
        m,
        targets=[m.disjunct[1]])

    disjBlock = m.disjunct[1].component("_pyomo_gdp_%s_reformulation" %
                                        transformation).relaxedDisjuncts
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
        self.assertIs(transform.get_src_disjunct(disjBlock[j]),
                      m.disjunct[1].innerdisjunct[i])
        self.assertIs(m.disjunct[1].innerdisjunct[i].transformation_block(),
                      disjBlock[j])

# checks for handling of benign types that could be on disjuncts we're
# transforming

def check_RangeSet(self, transformation):
    m = models.makeDisjunctWithRangeSet()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    self.assertIsInstance(m.d1.s, RangeSet)

def check_Expression(self, transformation):
    m = models.makeDisjunctWithExpression()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    self.assertIsInstance(m.d1.e, Expression)
