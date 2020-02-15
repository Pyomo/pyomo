from pyomo.environ import *
from pyomo.gdp import *
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
