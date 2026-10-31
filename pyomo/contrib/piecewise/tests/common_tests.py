# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import value, Expression
from pyomo.gdp import Disjunct, Disjunction


def check_trans_block_structure(test, block):
    # One (indexed) disjunct
    test.assertEqual(len(block.component_map(Disjunct)), 1)
    # One disjunction
    test.assertEqual(len(block.component_map(Disjunction)), 1)
    # The 'z' var (that we will substitute in for the function being
    # approximated) is here:
    test.assertEqual(len(block.component_map(Var)), 1)
    test.assertIsInstance(block.substitute_var, Var)


def check_log_x_model_soln(test, m):
    test.assertAlmostEqual(value(m.x), 4)
    test.assertAlmostEqual(value(m.x1), 1)
    test.assertAlmostEqual(value(m.x2), 1)
    test.assertAlmostEqual(value(m.obj), m.f2(4))


def check_transformation_do_not_descend(test, transformation, m=None):
    if m is None:
        m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m)

    test.check_pw_log(m)
    test.check_pw_paraboloid(m)


def check_transformation_PiecewiseLinearFunction_targets(test, transformation, m=None):
    if m is None:
        m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, targets=[m.pw_log])

    test.check_pw_log(m)

    # And check that the paraboloid was *not* transformed.
    test.assertIsNone(m.pw_paraboloid.get_transformation_var(m.paraboloid_expr))


def check_descend_into_expressions(test, transformation, m=None):
    if m is None:
        m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, descend_into_expressions=True)

    # Everything should be transformed
    test.check_pw_log(m)
    test.check_pw_paraboloid(m)


def check_descend_into_expressions_constraint_target(test, transformation, m=None):
    if m is None:
        m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, descend_into_expressions=True, targets=[m.indexed_c])

    test.check_pw_paraboloid(m)
    # And check that the log was *not* transformed.
    test.assertIsNone(m.pw_log.get_transformation_var(m.log_expr))


def check_descend_into_expressions_objective_target(test, transformation, m=None):
    if m is None:
        m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m, descend_into_expressions=True, targets=[m.obj])

    test.check_pw_log(m)
    # And check that the paraboloid was *not* transformed.
    test.assertIsNone(m.pw_paraboloid.get_transformation_var(m.paraboloid_expr))


def check_single_segment_no_disjunction(test, transformation, m=None):
    if m is None:
        m = models.make_single_segment_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m)

    # A single segment doesn't require making a discrete choice, so this
    # should not have created any GDP components.
    test.assertEqual(len(list(m.component_data_objects(Disjunct))), 0)
    test.assertEqual(len(list(m.component_data_objects(Disjunction))), 0)
    # Nor should it have introduced any new Vars (such as a substitute_var
    # or an indicator_var)--m.x should be the only Var in the model.
    model_vars = list(m.component_data_objects(Var))
    test.assertEqual(len(model_vars), 1)
    test.assertIs(model_vars[0], m.x)

    z = m.pw_log.get_transformation_var(m.log_expr)
    test.assertIsInstance(z, Expression)
    assertExpressionsEqual(test, z.expr, m.f1(m.x), places=7)
    test.assertIs(m.log_expr.expr, z)
