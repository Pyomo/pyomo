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

import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
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
