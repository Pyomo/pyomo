#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle
from pyomo.common.dependencies import dill

from pyomo.environ import (
    TransformationFactory,
    ConcreteModel,
    Constraint,
    Var,
    Objective,
    Block,
    Any,
    RangeSet,
    Expression,
    value,
    BooleanVar,
    SolverFactory,
    TerminationCondition,
)
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.base import constraint, ComponentUID
from pyomo.core.base.block import BlockData
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
from io import StringIO
import random

import pyomo.opt

linear_solvers = pyomo.opt.check_available_solvers('glpk', 'cbc', 'gurobi', 'cplex')
nonlinear_solvers = pyomo.opt.check_available_solvers('ipopt')

# utility functions


def check_linear_coef(self, repn, var, coef):
    # Map logical variables to their Boolean counterparts
    if isinstance(var, BooleanVar):
        var = var.get_associated_binary()

    # utility used to check a variable-coefficient pair in a standard_repn
    var_id = None
    for i, v in enumerate(repn.linear_vars):
        if v is var:
            var_id = i
    self.assertIsNotNone(var_id)
    self.assertAlmostEqual(repn.linear_coefs[var_id], coef)


def check_quadratic_coef(self, repn, v1, v2, coef):
    if isinstance(v1, BooleanVar):
        v1 = v1.get_associated_binary()
    if isinstance(v2, BooleanVar):
        v2 = v2.get_associated_binary()

    v1id = id(v1)
    v2id = id(v2)

    qcoef_map = dict()
    for (_v1, _v2), _coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
        qcoef_map[id(_v1), id(_v2)] = _coef
        qcoef_map[id(_v2), id(_v1)] = _coef

    self.assertIn((v1id, v2id), qcoef_map)
    self.assertAlmostEqual(qcoef_map[v1id, v2id], coef)


def check_squared_term_coef(self, repn, var, coef):
    var_id = None
    for i, (v1, v2) in enumerate(repn.quadratic_vars):
        if v1 is var and v2 is var:
            var_id = i
            break
    self.assertIsNotNone(var_id)
    self.assertEqual(repn.quadratic_coefs[var_id], coef)


def diff_apply_to_and_create_using(self, model, transformation, **kwargs):
    # compares the pprint from the transformed model after using both apply_to
    # and create_using to make sure the two do the same thing
    modelcopy = TransformationFactory(transformation).create_using(model, **kwargs)
    modelcopy_buf = StringIO()
    modelcopy.pprint(ostream=modelcopy_buf)
    modelcopy_output = modelcopy_buf.getvalue()

    # reset the seed for the apply_to call.
    random.seed(666)
    TransformationFactory(transformation).apply_to(model, **kwargs)
    model_buf = StringIO()
    model.pprint(ostream=model_buf)
    model_output = model_buf.getvalue()
    self.assertMultiLineEqual(modelcopy_output, model_output)


def check_obj_in_active_tree(self, obj, root=None):
    # Utility for checking that transformed components are indeed on the new
    # model, but without relying on private names to locate them. Basically,
    # I've been known to change where transformed components go, and I don't
    # want to test that. But I would like to check that they went somewhere that
    # the writers can find them.
    self.assertTrue(obj.active)
    parent = obj.parent_component()
    self.assertTrue(parent.active)
    blk = parent.parent_block()
    while blk is not root:
        self.assertTrue(blk.active)
        blk = blk.parent_block()


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
    self.assertTrue(m.disjunct1[1, 0].active)
    self.assertTrue(m.disjunct1[1, 1].active)
    self.assertTrue(m.disjunct1[2, 0].active)
    self.assertTrue(m.disjunct1[2, 1].active)

    self.assertFalse(m.b[0].disjunct.active)
    self.assertFalse(m.b[0].disjunct[0].active)
    self.assertFalse(m.b[0].disjunct[1].active)
    self.assertTrue(m.b[1].disjunct0.active)
    self.assertTrue(m.b[1].disjunct1.active)


def checkb0TargetsTransformed(self, m, transformation):
    trans = TransformationFactory('gdp.%s' % transformation)
    disjBlock = (
        m.b[0]
        .component("_pyomo_gdp_%s_reformulation" % transformation)
        .relaxedDisjuncts
    )
    self.assertEqual(len(disjBlock), 2)
    self.assertIs(
        trans.get_transformed_constraints(m.b[0].disjunct[0].c)[0].parent_block(),
        disjBlock[0],
    )
    self.assertIs(
        trans.get_transformed_constraints(m.b[0].disjunct[1].c)[0].parent_block(),
        disjBlock[1],
    )

    # This relies on the disjunctions being transformed in the same order
    # every time. This dictionary maps the block index to the list of
    # pairs of (originalDisjunctIndex, transBlockIndex)
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(m.b[0].disjunct[i].transformation_block, disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.b[0].disjunct[i])


# active status checks


def check_user_deactivated_disjuncts(
    self, transformation, check_trans_block=True, **kwargs
):
    # check that we do not transform a deactivated DisjunctData
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m, targets=(m,), **kwargs)

    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.d[1].active)

    if check_trans_block:
        rBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
        disjBlock = rBlock.relaxedDisjuncts
        self.assertEqual(len(disjBlock), 1)
        self.assertIs(disjBlock[0], m.d[1].transformation_block)
        self.assertIs(transform.get_src_disjunct(disjBlock[0]), m.d[1])


def check_improperly_deactivated_disjuncts(self, transformation, **kwargs):
    # check that if a Disjunct is deactivated but its indicator variable is not
    # fixed to 0, we express our confusion.
    m = models.makeTwoTermDisj()
    m.d[0].deactivate()
    self.assertEqual(value(m.d[0].indicator_var), False)
    self.assertTrue(m.d[0].indicator_var.is_fixed())
    m.d[0].indicator_var.fix(True)
    self.assertRaisesRegex(
        GDP_Error,
        r"The disjunct 'd\[0\]' is deactivated, but the "
        r"indicator_var is fixed to True. This makes no sense.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_indexed_disjunction_not_transformed(self, m, transformation):
    # no transformation block, nothing transformed
    self.assertIsNone(m.component("_pyomo_gdp_%s_transformation" % transformation))
    for idx in m.disjunct:
        self.assertIsNone(m.disjunct[idx].transformation_block)
    for idx in m.disjunction:
        self.assertIsNone(m.disjunction[idx].algebraic_constraint)


def check_do_not_transform_userDeactivated_indexedDisjunction(
    self, transformation, **kwargs
):
    # check that we do not transform a deactivated disjunction
    m = models.makeTwoTermIndexedDisjunction()
    # If you truly want to transform nothing, deactivate everything
    m.disjunction.deactivate()
    for idx in m.disjunct:
        m.disjunct[idx].deactivate()
    directly = TransformationFactory('gdp.%s' % transformation).create_using(
        m, **kwargs
    )
    check_indexed_disjunction_not_transformed(self, directly, transformation)

    targets = TransformationFactory('gdp.%s' % transformation).create_using(
        m, targets=(m.disjunction), **kwargs
    )
    check_indexed_disjunction_not_transformed(self, targets, transformation)


def check_disjunction_deactivated(self, transformation, **kwargs):
    # check that we deactivate disjunctions after we transform them
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)

    oldblock = m.component("disjunction")
    self.assertIsInstance(oldblock, Disjunction)
    self.assertFalse(oldblock.active)


def check_disjunctDatas_deactivated(self, transformation, **kwargs):
    # check that we deactivate disjuncts after we transform them
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)

    oldblock = m.component("disjunction")
    self.assertFalse(oldblock.disjuncts[0].active)
    self.assertFalse(oldblock.disjuncts[1].active)


def check_deactivated_constraints(self, transformation, **kwargs):
    # test that we deactivate constraints after we transform them
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwargs)
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


def check_deactivated_disjuncts(self, transformation, **kwargs):
    # another test that we deactivated transformed Disjuncts, but this one
    # includes a SimpleDisjunct as well
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)
    # all the disjuncts got transformed, so all should be deactivated
    for i in m.disjunct.index_set():
        self.assertFalse(m.disjunct[i].active)
    self.assertFalse(m.disjunct.active)


def check_deactivated_disjunctions(self, transformation, **kwargs):
    # another test that we deactivated transformed Disjunctions, but including a
    # SimpleDisjunction
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)

    # all the disjunctions got transformed, so they should be
    # deactivated too
    for i in m.disjunction.index_set():
        self.assertFalse(m.disjunction[i].active)
    self.assertFalse(m.disjunction.active)


def check_do_not_transform_twice_if_disjunction_reactivated(self, transformation):
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
    self.assertRaisesRegex(
        GDP_Error,
        r"The disjunct 'd\[0\]' has been transformed, but 'disjunction', "
        r"a disjunction it appears in, has not. Putting the same disjunct in "
        r"multiple disjunctions is not supported.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
    )


def check_constraints_deactivated_indexedDisjunction(self, transformation, **kwargs):
    # check that we deactivate transformed constraints
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwargs)

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
        len(transBlock.disj_xor),
        1,
        "There should only be one XOR constraint generated. Found %s."
        % len(transBlock.disj_xor),
    )


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
    self.assertIs(m.d[0].transformation_block, disjBlock[0])
    self.assertIs(m.d[1].transformation_block, disjBlock[1])

    # we didn't add to the block that wasn't ours
    self.assertEqual(
        len(m.component("_pyomo_gdp_%s_reformulation" % transformation)), 0
    )


# XOR constraints


def check_indicator_vars(self, transformation):
    # particularly paranoid test checking that the indicator_vars are intact
    # after transformation
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    oldblock = m.component("d")
    # have indicator variables on original disjuncts and they are still
    # active.
    _binary0 = oldblock[0].binary_indicator_var
    self.assertIsInstance(_binary0, Var)
    self.assertTrue(_binary0.active)
    self.assertTrue(_binary0.is_binary())
    _binary1 = oldblock[1].binary_indicator_var
    self.assertIsInstance(_binary1, Var)
    self.assertTrue(_binary1.active)
    self.assertTrue(_binary1.is_binary())


def check_two_term_disjunction_xor(self, xor, disj1, disj2):
    self.assertIsInstance(xor, Constraint)
    self.assertEqual(len(xor), 1)
    assertExpressionsEqual(
        self,
        xor.body,
        EXPR.LinearExpression([disj1.binary_indicator_var, disj2.binary_indicator_var]),
    )
    self.assertEqual(xor.lower, 1)
    self.assertEqual(xor.upper, 1)


def check_xor_constraint(self, transformation):
    # verify xor constraint for a SimpleDisjunction
    m = models.makeTwoTermDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    # make sure we created the xor constraint and put it on the relaxation
    # block
    rBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    xor = rBlock.component("disjunction_xor")
    check_two_term_disjunction_xor(self, xor, m.d[0], m.d[1])


def check_indexed_xor_constraints(self, transformation):
    # verify xor constraint for an IndexedDisjunction
    m = models.makeTwoTermMultiIndexedDisjunction()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    xor = m.component("_pyomo_gdp_%s_reformulation" % transformation).component(
        "disjunction_xor"
    )
    self.assertIsInstance(xor, Constraint)
    for i in m.disjunction.index_set():
        repn = generate_standard_repn(xor[i].body)
        self.assertEqual(repn.constant, 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, m.disjunction[i].disjuncts[0].indicator_var, 1)
        check_linear_coef(self, repn, m.disjunction[i].disjuncts[1].indicator_var, 1)
        self.assertEqual(xor[i].lower, 1)
        self.assertEqual(xor[i].upper, 1)


def check_indexed_xor_constraints_with_targets(self, transformation):
    # check that when we use targets to specify some DisjunctionDatas in an
    # IndexedDisjunction, the xor constraint is indexed correctly
    m = models.makeTwoTermIndexedDisjunction_BoundedVars()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[1], m.disjunction[3]]
    )

    xorC = m.disjunction[1].algebraic_constraint.parent_component()
    self.assertIsInstance(xorC, Constraint)
    self.assertEqual(len(xorC), 2)

    # check the constraints
    for i in [1, 3]:
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

    xor = m.component("_pyomo_gdp_%s_reformulation" % transformation).component(
        "disjunction_xor"
    )
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
        check_linear_coef(self, repn, m.disjunct[i, 1].indicator_var, 1)

    repn = generate_standard_repn(xor[2].body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 3)
    for i in range(3):
        check_linear_coef(self, repn, m.disjunct[i, 2].indicator_var, 1)


# mappings


def check_xor_constraint_mapping(self, transformation):
    # test that we correctly map between disjunctions and XOR constraints
    m = models.makeTwoTermDisj()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIs(trans.get_src_disjunction(transBlock.disjunction_xor), m.disjunction)
    self.assertIs(m.disjunction.algebraic_constraint, transBlock.disjunction_xor)


def check_xor_constraint_mapping_two_disjunctions(self, transformation):
    # test that we correctly map between disjunctions and xor constraints when
    # we have multiple SimpleDisjunctions (probably redundant with the above)
    m = models.makeDisjunctionOfDisjunctDatas()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIs(trans.get_src_disjunction(transBlock.disjunction_xor), m.disjunction)

    self.assertIs(m.disjunction.algebraic_constraint, transBlock.disjunction_xor)
    self.assertIs(m.disjunction2.algebraic_constraint, transBlock.disjunction2_xor)


def check_disjunct_mapping(self, transformation):
    # check that we correctly map between Disjuncts and their transformation
    # blocks
    m = models.makeTwoTermDisj_Nonlinear()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    disjBlock = m.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts

    # the disjuncts will always be transformed in the same order,
    # and d[0] goes first, so we can check in a loop.
    for i in [0, 1]:
        self.assertIs(disjBlock[i]._src_disjunct(), m.d[i])
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.d[i])


# targets


def check_only_targets_inactive(self, transformation, **kwargs):
    # test that we only transform targets (by checking active status)
    m = models.makeTwoSimpleDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction1], **kwargs
    )

    self.assertFalse(m.disjunction1.active)
    # disjunction2 still active
    self.assertTrue(m.disjunction2.active)

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
    trans.apply_to(m, targets=[m.disjunction1])

    disjBlock = m.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts
    # only two disjuncts relaxed
    self.assertEqual(len(disjBlock), 2)

    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(disjBlock[i], m.disjunct1[j].transformation_block)
        self.assertIs(trans.get_src_disjunct(disjBlock[i]), m.disjunct1[j])

    self.assertIsNone(m.disjunct2[0].transformation_block)
    self.assertIsNone(m.disjunct2[1].transformation_block)


def check_target_not_a_component_error(self, transformation, **kwargs):
    # test error message for crazy targets
    decoy = ConcreteModel()
    decoy.block = Block()
    m = models.makeTwoSimpleDisjunctions()
    self.assertRaisesRegex(
        GDP_Error,
        "Target 'block' is not a component on instance 'unknown'!",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[decoy.block],
        **kwargs,
    )


def check_targets_cannot_be_cuids(self, transformation):
    # check that we scream if targets are cuids
    m = models.makeTwoTermDisj()
    self.assertRaisesRegex(
        ValueError,
        r"invalid value for configuration 'targets':\n"
        r"\tFailed casting \[disjunction\]\n"
        r"\tto target_list\n"
        r"\tError: Expected Component or list of Components."
        r"\n\tReceived %s" % type(ComponentUID(m.disjunction)),
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[ComponentUID(m.disjunction)],
    )


def check_indexedDisj_targets_inactive(self, transformation, **kwargs):
    # check that targets are deactivated (when target is IndexedDisjunction)
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction1], **kwargs
    )

    self.assertFalse(m.disjunction1.active)
    self.assertFalse(m.disjunction1[1].active)
    self.assertFalse(m.disjunction1[2].active)

    self.assertFalse(m.disjunct1[1, 0].active)
    self.assertFalse(m.disjunct1[1, 1].active)
    self.assertFalse(m.disjunct1[2, 0].active)
    self.assertFalse(m.disjunct1[2, 1].active)
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
    trans.apply_to(m, targets=[m.disjunction1])

    disjBlock = m.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 4)
    if transformation == 'bigm':
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[1, 0].c)[0].parent_block(),
            disjBlock[0],
        )
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[1, 1].c)[0].parent_block(),
            disjBlock[1],
        )
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0].parent_block(),
            disjBlock[2],
        )
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 1].c)[0].parent_block(),
            disjBlock[3],
        )
    elif transformation == 'hull':
        # In the disaggregated var bounds
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[1, 0].c)[0]
            .parent_block()
            .parent_block(),
            disjBlock[0],
        )
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[1, 1].c)[0].parent_block(),
            disjBlock[1],
        )
        # In the disaggregated var bounds
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0]
            .parent_block()
            .parent_block(),
            disjBlock[2],
        )
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 1].c)[0].parent_block(),
            disjBlock[3],
        )

    # This relies on the disjunctions being transformed in the same order
    # every time. These are the mappings between the indices of the original
    # disjuncts and the indices on the indexed block on the transformation
    # block.
    pairs = [((1, 0), 0), ((1, 1), 1), ((2, 0), 2), ((2, 1), 3)]

    for i, j in pairs:
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.disjunct1[i])
        self.assertIs(disjBlock[j], m.disjunct1[i].transformation_block)


def check_warn_for_untransformed(self, transformation, **kwargs):
    # Check that we complain if we find an untransformed Disjunct inside of
    # another Disjunct we are transforming
    m = models.makeDisjunctionsOnIndexedBlock()

    def innerdisj_rule(d, flag):
        m = d.model()
        if flag:
            d.c = Constraint(expr=m.a[1] <= 2)
        else:
            d.c = Constraint(expr=m.a[1] >= 65)

    m.disjunct1[1, 1].innerdisjunct = Disjunct([0, 1], rule=innerdisj_rule)
    m.disjunct1[1, 1].innerdisjunction = Disjunction(
        [0],
        rule=lambda a, i: [
            m.disjunct1[1, 1].innerdisjunct[0],
            m.disjunct1[1, 1].innerdisjunct[1],
        ],
    )
    # if the disjunction doesn't drive the transformation of the Disjuncts, we
    # get the error
    m.disjunct1[1, 1].innerdisjunction.deactivate()
    # This test relies on the order that the component objects of
    # the disjunct get considered. In this case, the disjunct
    # causes the error, but in another world, it could be the
    # disjunction, which is also active.
    self.assertRaisesRegex(
        GDP_Error,
        r"Found active disjunct 'disjunct1\[1,1\].innerdisjunct\[0\]' "
        r"in disjunct 'disjunct1\[1,1\]'!.*",
        TransformationFactory('gdp.%s' % transformation).create_using,
        m,
        targets=[m.disjunction1[1]],
        **kwargs,
    )
    m.disjunct1[1, 1].innerdisjunction.activate()


def check_disjData_targets_inactive(self, transformation, **kwargs):
    # check targets deactivated with DisjunctionData is the target
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction1[2]], **kwargs
    )

    self.assertFalse(m.disjunction1[2].active)

    self.assertTrue(m.disjunct1.active)
    self.assertTrue(m.disjunct1[1, 0].active)
    self.assertIsNone(m.disjunct1[1, 0]._transformation_block)
    self.assertTrue(m.disjunct1[1, 1].active)
    self.assertIsNone(m.disjunct1[1, 1]._transformation_block)
    self.assertFalse(m.disjunct1[2, 0].active)
    self.assertIsNotNone(m.disjunct1[2, 0]._transformation_block)
    self.assertFalse(m.disjunct1[2, 1].active)
    self.assertIsNotNone(m.disjunct1[2, 1]._transformation_block)

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
    trans.apply_to(m, targets=[m.disjunction1[2]])

    disjBlock = m.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    if transformation == 'bigm':
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0].parent_block(),
            disjBlock[0],
        )
    elif transformation == 'hull':
        self.assertIs(
            trans.get_transformed_constraints(m.disjunct1[2, 0].c)[0]
            .parent_block()
            .parent_block(),
            disjBlock[0],
        )
    self.assertIs(
        trans.get_transformed_constraints(m.disjunct1[2, 1].c)[0].parent_block(),
        disjBlock[1],
    )

    # This relies on the disjunctions being transformed in the same order
    # every time. These are the mappings between the indices of the original
    # disjuncts and the indices on the indexed block on the transformation
    # block.
    pairs = [((2, 0), 0), ((2, 1), 1)]
    for i, j in pairs:
        self.assertIs(m.disjunct1[i].transformation_block, disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.disjunct1[i])


def check_indexedBlock_targets_inactive(self, transformation, **kwargs):
    # check that targets are deactivated when target is an IndexedBlock
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b], **kwargs
    )

    self.assertTrue(m.disjunct1.active)
    self.assertTrue(m.disjunct1[1, 0].active)
    self.assertTrue(m.disjunct1[1, 1].active)
    self.assertTrue(m.disjunct1[2, 0].active)
    self.assertTrue(m.disjunct1[2, 1].active)
    self.assertIsNone(m.disjunct1[1, 0].transformation_block)
    self.assertIsNone(m.disjunct1[1, 1].transformation_block)
    self.assertIsNone(m.disjunct1[2, 0].transformation_block)
    self.assertIsNone(m.disjunct1[2, 1].transformation_block)

    self.assertFalse(m.b[0].disjunct.active)
    self.assertFalse(m.b[0].disjunct[0].active)
    self.assertFalse(m.b[0].disjunct[1].active)
    self.assertFalse(m.b[1].disjunct0.active)
    self.assertFalse(m.b[1].disjunct1.active)


def check_indexedBlock_only_targets_transformed(self, transformation):
    # check that targets are transformed when target is an IndexedBlock
    m = models.makeDisjunctionsOnIndexedBlock()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.b])

    disjBlock1 = (
        m.b[0]
        .component("_pyomo_gdp_%s_reformulation" % transformation)
        .relaxedDisjuncts
    )
    self.assertEqual(len(disjBlock1), 2)
    self.assertIs(
        trans.get_transformed_constraints(m.b[0].disjunct[0].c)[0].parent_block(),
        disjBlock1[0],
    )
    self.assertIs(
        trans.get_transformed_constraints(m.b[0].disjunct[1].c)[0].parent_block(),
        disjBlock1[1],
    )

    disjBlock2 = (
        m.b[1]
        .component("_pyomo_gdp_%s_reformulation" % transformation)
        .relaxedDisjuncts
    )
    self.assertEqual(len(disjBlock2), 2)
    self.assertIs(
        trans.get_transformed_constraints(m.b[1].disjunct0.c)[0].parent_block(),
        disjBlock2[0],
    )
    self.assertIs(
        trans.get_transformed_constraints(m.b[1].disjunct1.c)[0].parent_block(),
        disjBlock2[1],
    )

    # This relies on the disjunctions being transformed in the same order
    # every time. This dictionary maps the block index to the list of
    # pairs of (originalDisjunctIndex, transBlockIndex)
    pairs = {
        0: [('disjunct', 0, 0), ('disjunct', 1, 1)],
        1: [('disjunct0', None, 0), ('disjunct1', None, 1)],
    }

    for blocknum, lst in pairs.items():
        for comp, i, j in lst:
            original = m.b[blocknum].component(comp)
            if blocknum == 0:
                disjBlock = disjBlock1
            if blocknum == 1:
                disjBlock = disjBlock2
            self.assertIs(original[i].transformation_block, disjBlock[j])
            self.assertIs(trans.get_src_disjunct(disjBlock[j]), original[i])


def check_blockData_targets_inactive(self, transformation, **kwargs):
    # test that BlockData target is deactivated
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b[0]], **kwargs
    )

    checkb0TargetsInactive(self, m)


def check_blockData_only_targets_transformed(self, transformation):
    # test that BlockData target is transformed
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b[0]])
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
        m, targets=[m.b[0], m.b[1]]
    )

    checkb0TargetsInactive(self, m)
    checkb0TargetsTransformed(self, m, transformation)


def check_disjunction_data_target(self, transformation):
    # test that if we transform DisjunctionDatas one at a time, we get what we
    # expect in terms of using the same transformation block and the indexing of
    # the xor constraint.
    m = models.makeThreeTermIndexedDisj()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[2]]
    )

    # we got a transformation block on the model
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIsInstance(transBlock, Block)
    self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)
    self.assertIsInstance(transBlock.disjunction_xor[2], constraint.ConstraintData)
    self.assertIsInstance(transBlock.component("relaxedDisjuncts"), Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 3)

    # suppose we transform the next one separately
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunction[1]]
    )
    self.assertIsInstance(
        m.disjunction[1].algebraic_constraint, constraint.ConstraintData
    )
    transBlock = m.component("_pyomo_gdp_%s_reformulation_4" % transformation)
    self.assertIsInstance(transBlock, Block)
    self.assertEqual(len(transBlock.relaxedDisjuncts), 3)


def check_disjunction_data_target_any_index(self, transformation):
    # check the same as the above, but that it still works when the Disjunction
    # is indexed by Any.
    m = ConcreteModel()
    m.x = Var(bounds=(-100, 100))
    m.disjunct3 = Disjunct(Any)
    m.disjunct4 = Disjunct(Any)
    m.disjunction2 = Disjunction(Any)
    for i in range(2):
        m.disjunct3[i].cons = Constraint(expr=m.x == 2)
        m.disjunct4[i].cons = Constraint(expr=m.x <= 3)
        m.disjunction2[i] = [m.disjunct3[i], m.disjunct4[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(
            m, targets=[m.disjunction2[i]]
        )

        if i == 0:
            check_relaxation_block(
                self, m, "_pyomo_gdp_%s_reformulation" % transformation, 2
            )
        if i == 2:
            check_relaxation_block(
                self, m, "_pyomo_gdp_%s_reformulation" % transformation, 4
            )


# tests that we treat disjunctions on blocks correctly (the main issue here is
# that if you were to solve that block post-transformation that you would have
# the whole transformed model)


def check_xor_constraint_added(self, transformation):
    # test we put the xor on the transformation block
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    self.assertIsInstance(
        m.b.component("_pyomo_gdp_%s_reformulation" % transformation).component(
            m.b.disjunction.algebraic_constraint.local_name
        ),
        Constraint,
    )


def check_trans_block_created(self, transformation):
    # check we put the transformation block on the parent block of the
    # disjunction
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    # test that the transformation block got created on the block
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


def check_iteratively_adding_to_indexed_disjunction_on_block(self, transformation):
    # check that we can iteratively add to an IndexedDisjunction and transform
    # the block it lives on
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var(bounds=(-100, 100))
    m.b.firstTerm = Disjunct([1, 2])
    m.b.firstTerm[1].cons = Constraint(expr=m.b.x == 0)
    m.b.firstTerm[2].cons = Constraint(expr=m.b.x == 2)
    m.b.secondTerm = Disjunct([1, 2])
    m.b.secondTerm[1].cons = Constraint(expr=m.b.x >= 2)
    m.b.secondTerm[2].cons = Constraint(expr=m.b.x >= 3)
    m.b.disjunctionList = Disjunction(Any)

    m.b.obj = Objective(expr=m.b.x)

    for i in range(1, 3):
        m.b.disjunctionList[i] = [m.b.firstTerm[i], m.b.secondTerm[i]]

        TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.b])

        if i == 1:
            check_relaxation_block(
                self, m.b, "_pyomo_gdp_%s_reformulation" % transformation, 2
            )
        if i == 2:
            check_relaxation_block(
                self, m.b, "_pyomo_gdp_%s_reformulation_4" % transformation, 2
            )


def check_simple_disjunction_of_disjunct_datas(self, transformation):
    # This is actually a reasonable use case if you are generating
    # disjunctions with the same structure. So you might have Disjuncts
    # indexed by Any and disjunctions indexed by Any and be adding a
    # disjunction of two of the DisjunctDatas in every iteration.
    m = models.makeDisjunctionOfDisjunctDatas()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)

    self.check_trans_block_disjunctions_of_disjunct_datas(m)
    transBlock = m.component("_pyomo_gdp_%s_reformulation" % transformation)
    self.assertIsInstance(transBlock.component("disjunction_xor"), Constraint)
    self.assertIsInstance(transBlock.component("disjunction2_xor"), Constraint)


# these tests have different checks for what ends up on the model between bigm
# and hull, but they have the same structure
def check_iteratively_adding_disjunctions_transform_container(self, transformation):
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
        model.component(firstTermName).cons = Constraint(expr=model.x == 2 * i)
        secondTermName = "secondTerm[%s]" % i
        model.add_component(secondTermName, Disjunct())
        model.component(secondTermName).cons = Constraint(expr=model.x >= i + 2)
        model.disjunctionList[i] = [
            model.component(firstTermName),
            model.component(secondTermName),
        ]

        # we're lazy and we just transform the disjunctionList (and in
        # theory we are transforming at every iteration because we are
        # solving at every iteration)
        TransformationFactory('gdp.%s' % transformation).apply_to(
            model, targets=[model.disjunctionList]
        )
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
        model.firstTerm[i].cons = Constraint(expr=model.x == 2 * i)
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
        model.component(firstTermName).cons = Constraint(expr=model.x == 2 * i)
        secondTermName = "secondTerm[%s]" % i
        model.add_component(secondTermName, Disjunct())
        model.component(secondTermName).cons = Constraint(expr=model.x >= i + 2)
        model.disjunctionList[i] = [
            model.component(firstTermName),
            model.component(secondTermName),
        ]

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
def check_transformation_simple_block(self, transformation, **kwargs):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b, **kwargs)

    # transformation block not on m
    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(
        m.b.component("_pyomo_gdp_%s_reformulation" % transformation), Block
    )


def check_transform_block_data(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(m.b[0], **kwargs)

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    self.assertIsInstance(
        m.b[0].component("_pyomo_gdp_%s_reformulation" % transformation), Block
    )


def check_simple_block_target(self, transformation, **kwargs):
    m = models.makeTwoTermDisjOnBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b], **kwargs
    )

    # transformation block not on m
    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    # transformation block on m.b
    self.assertIsInstance(
        m.b.component("_pyomo_gdp_%s_reformulation" % transformation), Block
    )


def check_block_data_target(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b[0]], **kwargs
    )

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    self.assertIsInstance(
        m.b[0].component("_pyomo_gdp_%s_reformulation" % transformation), Block
    )


def check_indexed_block_target(self, transformation, **kwargs):
    m = models.makeDisjunctionsOnIndexedBlock()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b], **kwargs
    )

    # We expect the transformation block on each of the BlockDatas. Because
    # it is always going on the parent block of the disjunction.

    self.assertIsNone(m.component("_pyomo_gdp_%s_reformulation" % transformation))

    for i in [0, 1]:
        self.assertIsInstance(
            m.b[i].component("_pyomo_gdp_%s_reformulation" % transformation), Block
        )


def check_block_targets_inactive(self, transformation, **kwargs):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.b], **kwargs
    )

    self.assertFalse(m.b.disjunct[0].active)
    self.assertFalse(m.b.disjunct[1].active)
    self.assertFalse(m.b.disjunct.active)
    self.assertTrue(m.simpledisj.active)
    self.assertTrue(m.simpledisj2.active)


def check_block_only_targets_transformed(self, transformation):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=[m.b])

    disjBlock = m.b.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    if transformation == 'bigm':
        self.assertIs(
            disjBlock[0],
            trans.get_transformed_constraints(m.b.disjunct[0].c)[0].parent_block(),
        )
    elif transformation == 'hull':
        # this constraint is on the bounds of the disaggregated var
        self.assertIs(
            disjBlock[0],
            trans.get_transformed_constraints(m.b.disjunct[0].c)[0]
            .parent_block()
            .parent_block(),
        )
    self.assertIs(
        disjBlock[1],
        trans.get_transformed_constraints(m.b.disjunct[1].c)[0].parent_block(),
    )

    # this relies on the disjuncts being transformed in the same order every
    # time
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(m.b.disjunct[i].transformation_block, disjBlock[j])
        self.assertIs(trans.get_src_disjunct(disjBlock[j]), m.b.disjunct[i])


# common error messages


def check_transform_empty_disjunction(self, transformation, **kwargs):
    m = ConcreteModel()
    m.empty = Disjunction(expr=[])

    self.assertRaisesRegex(
        GDP_Error,
        "Disjunction 'empty' is empty. This is likely indicative of a "
        "modeling error.*",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_deactivated_disjunct_nonzero_indicator_var(self, transformation, **kwargs):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].indicator_var.fix(1)

    self.assertRaisesRegex(
        GDP_Error,
        r"The disjunct 'disjunction_disjuncts\[0\]' is deactivated, but the "
        r"indicator_var is fixed to True. This makes no sense.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_deactivated_disjunct_unfixed_indicator_var(self, transformation, **kwargs):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])

    m.disjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].indicator_var.fixed = False

    self.assertRaisesRegex(
        GDP_Error,
        r"The disjunct 'disjunction_disjuncts\[0\]' is deactivated, but the "
        r"indicator_var is not fixed and the disjunct does not "
        r"appear to have been transformed. This makes no sense. "
        r"\(If the intent is to deactivate the disjunct, fix its "
        r"indicator_var to False.\)",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_retrieving_nondisjunctive_components(self, transformation):
    m = models.makeTwoTermDisj()
    m.b = Block()
    m.b.global_cons = Constraint(expr=m.a + m.x >= 8)
    m.another_global_cons = Constraint(expr=m.a + m.x <= 11)

    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    self.assertRaisesRegex(
        GDP_Error,
        "Constraint 'b.global_cons' is not on a disjunct and so was not transformed",
        trans.get_transformed_constraints,
        m.b.global_cons,
    )

    self.assertRaisesRegex(
        GDP_Error,
        "Constraint 'b.global_cons' is not a transformed constraint",
        trans.get_src_constraint,
        m.b.global_cons,
    )

    self.assertRaisesRegex(
        GDP_Error,
        "Constraint 'another_global_cons' is not a transformed constraint",
        trans.get_src_constraint,
        m.another_global_cons,
    )

    self.assertRaisesRegex(
        GDP_Error,
        "Block 'b' doesn't appear to be a transformation block for a "
        "disjunct. No source disjunct found.",
        trans.get_src_disjunct,
        m.b,
    )

    self.assertRaisesRegex(
        GDP_Error,
        "It appears that 'another_global_cons' is not an XOR or OR"
        " constraint resulting from transforming a Disjunction.",
        trans.get_src_disjunction,
        m.another_global_cons,
    )


def check_silly_target(self, transformation, **kwargs):
    m = models.makeTwoTermDisj()
    self.assertRaisesRegex(
        GDP_Error,
        r"Target 'd\[1\].c1' was not a Block, Disjunct, or Disjunction. "
        r"It was of type "
        r"<class 'pyomo.core.base.constraint.ScalarConstraint'> and "
        r"can't be transformed.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[m.d[1].c1],
        **kwargs,
    )


def check_ask_for_transformed_constraint_from_untransformed_disjunct(
    self, transformation
):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m, targets=m.disjunction[1])

    self.assertRaisesRegex(
        GDP_Error,
        r"Constraint 'disjunct\[2,b\].cons_b' is on a disjunct which has "
        r"not been transformed",
        trans.get_transformed_constraints,
        m.disjunct[2, 'b'].cons_b,
    )


def check_error_for_same_disjunct_in_multiple_disjunctions(
    self, transformation, **kwargs
):
    m = models.makeDisjunctInMultipleDisjunctions()
    self.assertRaisesRegex(
        GDP_Error,
        r"The disjunct 'disjunct1\[1\]' has been transformed, "
        r"but 'disjunction2', a disjunction it appears in, has not. "
        r"Putting the same disjunct in multiple disjunctions is not supported.",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_cannot_call_transformation_on_disjunction(self, transformation, **kwargs):
    m = models.makeTwoTermIndexedDisjunction()
    trans = TransformationFactory('gdp.%s' % transformation)
    self.assertRaisesRegex(
        GDP_Error,
        r"Transformation called on disjunction of type "
        r"<class 'pyomo.gdp.disjunct.Disjunction'>. 'instance' "
        r"must be a ConcreteModel, Block, or Disjunct \(in "
        r"the case of nested disjunctions\).",
        trans.apply_to,
        m.disjunction,
        targets=m.disjunction[1],
        **kwargs,
    )


# This is really neurotic, but test that we will create an infeasible XOR
# constraint. We have to because in the case of nested disjunctions, our model
# is not necessarily infeasible because of this. It just might make a Disjunct
# infeasible.
def setup_infeasible_xor_because_all_disjuncts_deactivated(self, transformation):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.y = Var(bounds=(0, 7))
    m.disjunction = Disjunction(expr=[m.x == 0, m.x >= 4])
    m.disjunction_disjuncts[0].nestedDisjunction = Disjunction(
        expr=[m.y == 6, m.y <= 1]
    )
    # Note that this fixes the indicator variables to 0, but since the
    # disjunction is still active, the XOR constraint will be created. So we
    # will have to land in the second disjunct of m.disjunction
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[0].deactivate()
    m.disjunction.disjuncts[0].nestedDisjunction.disjuncts[1].deactivate()
    # This should create a 0 = 1 XOR constraint, actually...
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=m.disjunction.disjuncts[0].nestedDisjunction
    )

    # check that our XOR is the bad thing it should be.
    xor = m.disjunction_disjuncts[0].nestedDisjunction.algebraic_constraint
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


def check_disjunction_target_err(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    # deactivate the disjunction that would transform the nested Disjuncts so
    # that we see it is possible to get the error.
    m.simpledisjunct.innerdisjunction.deactivate()
    self.assertRaisesRegex(
        GDP_Error,
        "Found active disjunct 'simpledisjunct.innerdisjunct0' in "
        "disjunct 'simpledisjunct'!.*",
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        targets=[m.disjunction],
        **kwargs,
    )


# nested disjunctions: hull and bigm have very different handling for nested
# disjunctions, but these tests check *that* everything is transformed, not how


def check_disjuncts_inactive_nested(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=(m,), **kwargs)

    self.assertFalse(m.disjunction.active)
    self.assertFalse(m.simpledisjunct.active)
    self.assertFalse(m.disjunct[0].active)
    self.assertFalse(m.disjunct[1].active)
    self.assertFalse(m.disjunct.active)


def check_deactivated_disjunct_leaves_nested_disjunct_active(
    self, transformation, **kwargs
):
    m = models.makeNestedDisjunctions_FlatDisjuncts()
    m.d1.deactivate()
    # Specifying 'targets' prevents the HACK_GDP_Disjunct_Reclassifier
    # transformation of Disjuncts to Blocks
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m], **kwargs)

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
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m], **kwargs)

    self.assertFalse(m.d1.active)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertEqual(m.d1.indicator_var.value, 0)

    self.assertFalse(m.d2.active)
    self.assertFalse(m.d2.indicator_var.fixed)

    self.assertTrue(m.d1.d3.active)
    self.assertFalse(m.d1.d3.indicator_var.fixed)

    self.assertTrue(m.d1.d4.active)
    self.assertFalse(m.d1.d4.indicator_var.fixed)


def check_disjunct_targets_inactive(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.simpledisjunct], **kwargs
    )

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
    transform.apply_to(m, targets=[m.simpledisjunct])

    disjBlock = m.simpledisjunct.component(
        "_pyomo_gdp_%s_reformulation" % transformation
    ).relaxedDisjuncts
    self.assertEqual(len(disjBlock), 2)
    self.assertIs(
        transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)[
            0
        ].parent_block(),
        disjBlock[0],
    )
    self.assertIs(
        transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct0.c)[
            0
        ].parent_block(),
        disjBlock[0],
    )
    self.assertIs(
        transform.get_transformed_constraints(m.simpledisjunct.innerdisjunct1.c)[
            0
        ].parent_block(),
        disjBlock[1],
    )

    # This also relies on the disjuncts being transformed in the same
    # order every time.
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(
            m.simpledisjunct.component('innerdisjunct%d' % i),
            transform.get_src_disjunct(disjBlock[j]),
        )
        self.assertIs(
            disjBlock[j],
            m.simpledisjunct.component('innerdisjunct%d' % i).transformation_block,
        )


def check_disjunctData_targets_inactive(self, transformation, **kwargs):
    m = models.makeNestedDisjunctions()
    TransformationFactory('gdp.%s' % transformation).apply_to(
        m, targets=[m.disjunct[1]], **kwargs
    )

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
    transform.apply_to(m, targets=[m.disjunct[1]])

    disjBlock = (
        m.disjunct[1]
        .component("_pyomo_gdp_%s_reformulation" % transformation)
        .relaxedDisjuncts
    )
    self.assertEqual(len(disjBlock), 2)
    if transformation == 'bigm':
        self.assertIs(
            transform.get_transformed_constraints(m.disjunct[1].innerdisjunct[0].c)[
                0
            ].parent_block(),
            disjBlock[0],
        )
    elif transformation == 'hull':
        # This constraint is on Block deeper because it is in the bounds of a
        # disaggregated var
        self.assertIs(
            transform.get_transformed_constraints(m.disjunct[1].innerdisjunct[0].c)[0]
            .parent_block()
            .parent_block(),
            disjBlock[0],
        )
    self.assertIs(
        transform.get_transformed_constraints(m.disjunct[1].innerdisjunct[1].c)[
            0
        ].parent_block(),
        disjBlock[1],
    )

    # This also relies on the disjuncts being transformed in the same
    # order every time.
    pairs = [(0, 0), (1, 1)]
    for i, j in pairs:
        self.assertIs(
            transform.get_src_disjunct(disjBlock[j]), m.disjunct[1].innerdisjunct[i]
        )
        self.assertIs(m.disjunct[1].innerdisjunct[i].transformation_block, disjBlock[j])


def check_all_components_transformed(self, m):
    # checks that all the disjunctive components claim to be transformed in the
    # makeNestedDisjunctions_NestedDisjuncts model.
    self.assertIsInstance(m.disj.algebraic_constraint, Constraint)
    self.assertIsInstance(m.d1.disj2.algebraic_constraint, Constraint)
    self.assertIsInstance(m.d1.transformation_block, BlockData)
    self.assertIsInstance(m.d2.transformation_block, BlockData)
    self.assertIsInstance(m.d1.d3.transformation_block, BlockData)
    self.assertIsInstance(m.d1.d4.transformation_block, BlockData)


def check_transformation_blocks_nestedDisjunctions(self, m, transformation):
    disjunctionTransBlock = m.disj.algebraic_constraint.parent_block()
    transBlocks = disjunctionTransBlock.relaxedDisjuncts
    if transformation == 'bigm':
        self.assertEqual(len(transBlocks), 4)
        self.assertIs(transBlocks[0], m.d1.d3.transformation_block)
        self.assertIs(transBlocks[1], m.d1.d4.transformation_block)
        self.assertIs(transBlocks[2], m.d1.transformation_block)
        self.assertIs(transBlocks[3], m.d2.transformation_block)
    if transformation == 'hull':
        # This is a much more comprehensive test that doesn't depend on
        # transformation Block structure, so just reuse it:
        hull = TransformationFactory('gdp.hull')
        d3 = hull.get_disaggregated_var(m.d1.d3.binary_indicator_var, m.d1)
        d4 = hull.get_disaggregated_var(m.d1.d4.binary_indicator_var, m.d1)
        self.check_transformed_model_nestedDisjuncts(m, d3, d4)

        # Check the 4 constraints that are unique to the case where we didn't
        # declare d1.d3 and d1.d4 as local
        d32 = hull.get_disaggregated_var(m.d1.d3.binary_indicator_var, m.d2)
        d42 = hull.get_disaggregated_var(m.d1.d4.binary_indicator_var, m.d2)
        # check the additional disaggregated indicator var bound constraints
        cons = hull.get_var_bounds_constraint(d32)
        self.assertEqual(len(cons), 1)
        check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        # Note that this comes out as d32 <= 1 - d1.ind_var because it's the
        # "extra" disaggregated var that gets created when it need to be
        # disaggregated for d1, but it's not used in d2
        assertExpressionsEqual(
            self, cons_expr, d32 + m.d1.binary_indicator_var - 1 <= 0.0
        )

        cons = hull.get_var_bounds_constraint(d42)
        self.assertEqual(len(cons), 1)
        check_obj_in_active_tree(self, cons['ub'])
        cons_expr = self.simplify_leq_cons(cons['ub'])
        # Note that this comes out as d42 <= 1 - d1.ind_var because it's the
        # "extra" disaggregated var that gets created when it need to be
        # disaggregated for d1, but it's not used in d2
        assertExpressionsEqual(
            self, cons_expr, d42 + m.d1.binary_indicator_var - 1 <= 0.0
        )
        # check the aggregation constraints for the disaggregated indicator vars
        cons = hull.get_disaggregation_constraint(m.d1.d3.binary_indicator_var, m.disj)
        check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(
            self, cons_expr, m.d1.d3.binary_indicator_var - d32 - d3 == 0.0
        )
        cons = hull.get_disaggregation_constraint(m.d1.d4.binary_indicator_var, m.disj)
        check_obj_in_active_tree(self, cons)
        cons_expr = self.simplify_cons(cons)
        assertExpressionsEqual(
            self, cons_expr, m.d1.d4.binary_indicator_var - d42 - d4 == 0.0
        )

        num_cons = len(
            list(m.component_data_objects(Constraint, active=True, descend_into=Block))
        )
        # 30 total constraints in transformed model minus 10 trivial bounds
        # (lower bounds of 0) gives us 20 constraints total:
        self.assertEqual(num_cons, 20)
        # (And this is 4 more than we test in
        # self.check_transformed_model_nestedDisjuncts, so that's comforting
        # too.)


def check_nested_disjunction_target(self, transformation):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    transform = TransformationFactory('gdp.%s' % transformation)
    transform.apply_to(m, targets=[m.disj])

    # the bug that inspired this test throws an error while doing the
    # transformation, so we'll just do a quick check that all the GDP
    # components think they are transformed.
    check_all_components_transformed(self, m)
    check_transformation_blocks_nestedDisjunctions(self, m, transformation)


def check_target_appears_twice(self, transformation):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    # Because of the way we preprocess targets, the result here will be that
    # m.d1 appears twice in the list of targets. However, this is fine because
    # the transformation will not try to retransform anything that has already
    # been transformed.
    m1 = TransformationFactory('gdp.%s' % transformation).create_using(
        m, targets=[m.d1, m.disj]
    )

    check_all_components_transformed(self, m1)
    # check we have correct number of transformation blocks
    check_transformation_blocks_nestedDisjunctions(self, m1, transformation)

    # Now check the same thing, but if the already-transformed disjunct appears
    # after its disjunction.
    TransformationFactory('gdp.%s' % transformation).apply_to(m, targets=[m.disj, m.d1])
    check_all_components_transformed(self, m)
    check_transformation_blocks_nestedDisjunctions(self, m, transformation)


def check_unique_reference_to_nested_indicator_var(self, transformation):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    # find the references to the nested indicator var
    num_references_d3 = 0
    num_references_d4 = 0
    for v in m.component_data_objects(Var, active=True, descend_into=Block):
        if v is m.d1.d3.binary_indicator_var:
            num_references_d3 += 1
        if v is m.d1.d4.binary_indicator_var:
            num_references_d4 += 1
    self.assertEqual(num_references_d3, 1)
    self.assertEqual(num_references_d4, 1)


# checks for handling of benign types that could be on disjuncts we're
# transforming


def check_RangeSet(self, transformation, **kwargs):
    m = models.makeDisjunctWithRangeSet()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwargs)
    self.assertIsInstance(m.d1.s, RangeSet)


def check_Expression(self, transformation, **kwargs):
    m = models.makeDisjunctWithExpression()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwargs)
    self.assertIsInstance(m.d1.e, Expression)


def check_untransformed_network_raises_GDPError(self, transformation, **kwargs):
    m = models.makeNetworkDisjunction()
    self.assertRaisesRegex(
        GDP_Error,
        "No %s transformation handler registered for modeling "
        "components of type <class 'pyomo.network.arc.Arc'>. If "
        "your disjuncts contain non-GDP Pyomo components that require "
        "transformation, please transform them first." % transformation,
        TransformationFactory('gdp.%s' % transformation).apply_to,
        m,
        **kwargs,
    )


def check_trivial_constraints(self, solver, transformation, **kwds):
    m = models.makeTrivialGDP()
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwds)
    results = SolverFactory(solver).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertTrue(m.numeric.disjuncts[0].indicator_var.value)
    self.assertFalse(m.numeric.disjuncts[1].indicator_var.value)
    self.assertTrue(m.logical.disjuncts[0].indicator_var.value)
    self.assertFalse(m.logical.disjuncts[1].indicator_var.value)

    m.numeric.disjuncts[0].indicator_var.fix(False)
    results = SolverFactory(solver).solve(m)
    self.assertEqual(
        results.solver.termination_condition, TerminationCondition.infeasible
    )
    m.numeric.disjuncts[0].indicator_var.unfix()

    m.logical.disjuncts[0].indicator_var.fix(False)
    results = SolverFactory(solver).solve(m)
    self.assertEqual(
        results.solver.termination_condition, TerminationCondition.infeasible
    )
    m.logical.disjuncts[0].indicator_var.unfix()


def check_network_disjuncts(self, minimize, transformation, **kwds):
    m = models.makeExpandedNetworkDisjunction(minimize=minimize)
    TransformationFactory('gdp.%s' % transformation).apply_to(m, **kwds)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    if minimize:
        self.assertAlmostEqual(value(m.dest.x), 0.42)
    else:
        self.assertAlmostEqual(value(m.dest.x), 0.84)


def check_solution_obeys_logical_constraints(self, transformation, m):
    # m is expected to either by models.makeLogicalConstraintsOnDisjuncts or
    # models.makeBooleanVarsOnDisjuncts
    trans = TransformationFactory('gdp.%s' % transformation)

    # no logical constraints
    m.p.deactivate()
    m.bwahaha.deactivate()
    no_logic = trans.create_using(m)

    results = SolverFactory(linear_solvers[0]).solve(no_logic)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(no_logic.x), 2.5)

    # with logical constraints
    m.p.activate()
    m.bwahaha.activate()
    trans.apply_to(m)
    results = SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(m.x), 8)


# test pickling transformed models


def check_pprint_equal(self, m, unpickle):
    # This is almost the same as in the diff_apply_to_and_create_using test but
    # we don't have to transform in the middle or mess with seeds.
    m_buf = StringIO()
    m.pprint(ostream=m_buf)
    m_output = m_buf.getvalue()

    unpickle_buf = StringIO()
    unpickle.pprint(ostream=unpickle_buf)
    unpickle_output = unpickle_buf.getvalue()
    self.assertMultiLineEqual(m_output, unpickle_output)


def check_transformed_model_pickles(self, transformation):
    # Do a model where we'll have to call logical_to_disjunctive too.
    m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    # pickle and unpickle the transformed model
    unpickle = pickle.loads(pickle.dumps(m))

    check_pprint_equal(self, m, unpickle)


def check_transformed_model_pickles_with_dill(self, transformation):
    m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
    trans = TransformationFactory('gdp.%s' % transformation)
    trans.apply_to(m)

    # pickle and unpickle the transformed model
    unpickle = dill.loads(dill.dumps(m))

    check_pprint_equal(self, m, unpickle)


def check_nested_disjuncts_in_flat_gdp(self, transformation):
    m = models.make_non_nested_model_declaring_Disjuncts_on_each_other()
    TransformationFactory('gdp.%s' % transformation).apply_to(m)
    SolverFactory('gurobi').solve(m)
    self.assertAlmostEqual(value(m.obj), 1020)

    # check the Boolean solution
    for t in m.T:
        self.assertTrue(value(m.disj1[t].indicator_var))
        self.assertTrue(value(m.disj1[t].sub1.indicator_var))


def check_do_not_assume_nested_indicators_local(self, transformation):
    m = models.why_indicator_vars_are_not_always_local()
    TransformationFactory(transformation).apply_to(m)

    results = SolverFactory('gurobi').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(value(m.obj), 9)
    self.assertAlmostEqual(value(m.x), 9)
    self.assertTrue(value(m.Y2.indicator_var))
    self.assertFalse(value(m.Y1.indicator_var))
    self.assertTrue(value(m.Z1.indicator_var))
    self.assertTrue(value(m.Z1.indicator_var))
