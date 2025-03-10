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

from pyomo.core import Var
from pyomo.core.base import Constraint
from pyomo.core.expr.compare import assertExpressionsEqual

# This file contains check methods shared between GDP inner representation-based
# transformations. Currently, those are the inner_representation_gdp and
# nested_inner_repn_gdp transformations, since each have disjuncts with the
# same structure.


# Check one disjunct from the log model for proper contents
def check_log_disjunct(test, d, pts, f, substitute_var, x):
    test.assertEqual(len(d.component_map(Constraint)), 3)
    # lambdas and indicator_var
    test.assertEqual(len(d.component_map(Var)), 2)
    test.assertIsInstance(d.lambdas, Var)
    test.assertEqual(len(d.lambdas), 2)
    for lamb in d.lambdas.values():
        test.assertEqual(lamb.lb, 0)
        test.assertEqual(lamb.ub, 1)
    test.assertIsInstance(d.convex_combo, Constraint)
    assertExpressionsEqual(test, d.convex_combo.expr, d.lambdas[0] + d.lambdas[1] == 1)
    test.assertIsInstance(d.set_substitute, Constraint)
    assertExpressionsEqual(
        test, d.set_substitute.expr, substitute_var == f(x), places=7
    )
    test.assertIsInstance(d.linear_combo, Constraint)
    test.assertEqual(len(d.linear_combo), 1)
    assertExpressionsEqual(
        test, d.linear_combo[0].expr, x == pts[0] * d.lambdas[0] + pts[1] * d.lambdas[1]
    )


# Check one disjunct from the paraboloid model for proper contents.
def check_paraboloid_disjunct(test, d, pts, f, substitute_var, x1, x2):
    test.assertEqual(len(d.component_map(Constraint)), 3)
    # lambdas and indicator_var
    test.assertEqual(len(d.component_map(Var)), 2)
    test.assertIsInstance(d.lambdas, Var)
    test.assertEqual(len(d.lambdas), 3)
    for lamb in d.lambdas.values():
        test.assertEqual(lamb.lb, 0)
        test.assertEqual(lamb.ub, 1)
    test.assertIsInstance(d.convex_combo, Constraint)
    assertExpressionsEqual(
        test, d.convex_combo.expr, d.lambdas[0] + d.lambdas[1] + d.lambdas[2] == 1
    )
    test.assertIsInstance(d.set_substitute, Constraint)
    assertExpressionsEqual(
        test, d.set_substitute.expr, substitute_var == f(x1, x2), places=7
    )
    test.assertIsInstance(d.linear_combo, Constraint)
    test.assertEqual(len(d.linear_combo), 2)
    assertExpressionsEqual(
        test,
        d.linear_combo[0].expr,
        x1
        == pts[0][0] * d.lambdas[0]
        + pts[1][0] * d.lambdas[1]
        + pts[2][0] * d.lambdas[2],
    )
    assertExpressionsEqual(
        test,
        d.linear_combo[1].expr,
        x2
        == pts[0][1] * d.lambdas[0]
        + pts[1][1] * d.lambdas[1]
        + pts[2][1] * d.lambdas[2],
    )
