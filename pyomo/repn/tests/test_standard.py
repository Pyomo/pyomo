#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the standard expressions
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

from pyomo.core.base.expr import Expr_if
from pyomo.core.base import expr_common, expr as EXPR
from pyomo.repn import *
from pyomo.environ import *
from pyomo.core.base.numvalue import native_numeric_types

from six import iteritems
from six.moves import range

class frozendict(dict):
    __slots__ = ('_hash',)
    def __hash__(self):
        rval = getattr(self, '_hash', None)
        if rval is None:
            rval = self._hash = hash(frozenset(iteritems(self)))
        return rval


# A utility to facilitate comparison of tuples where we don't care about ordering
def linear_repn_to_dict(repn):
    result = {}
    for i in repn._linear_vars:
        result[id(repn._linear_vars[i])] = repn._linear_coefs[i]
    if not (type(repn._constant) in native_numeric_types and repn._constant == 0):
        result[None] = repn._constant
    return result


class TestSimple(unittest.TestCase):

    def setUp(self):
        # This class tests the Pyomo 5.x expression trees
        EXPR.set_expression_tree_format(expr_common.Mode.pyomo5_trees)

    def tearDown(self):
        EXPR.set_expression_tree_format(expr_common._default_mode)

    def test_number(self):
        # 1.0
        m = AbstractModel()
        m.a = Var()
        e = 1.0

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        
    def test_var(self):
        # a
        m = AbstractModel()
        m.a = Var()
        e = m.a

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        
    def test_param(self):
        # p
        m = AbstractModel()
        m.p = Param()
        e = m.p

        with self.assertRaises(ValueError):
            rep = generate_standard_repn(e)
        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None : m.p }
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_simplesum(self):
        # a + b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
 
        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_constsum(self):
        # a + 5
        m = AbstractModel()
        m.a = Var()
        e = m.a + 5
 
        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        # 5 + a
        m = AbstractModel()
        m.a = Var()
        e = 5 + m.a
 
        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_nestedSum(self):
        #
        # Check the structure of nested sums
        #
        expectedType = EXPR._SumExpression

        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #           +
        #          / \
        #         +   5
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       + 
        #      / \ 
        #     5   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = 5 + e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #           +
        #          / \
        #         +   c
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + m.c

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       + 
        #      / \ 
        #     c   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = m.c + e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            +
        #          /   \
        #         +     +
        #        / \   / \
        #       a   b c   d
        e1 = m.a + m.b
        e2 = m.c + m.d
        e = e1 + e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 4)
        self.assertTrue(len(rep._linear_coefs) == 4)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1, id(m.d) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_sumOf_nestedTrivialProduct(self):
        #
        # Check sums with nested products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       +
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 + m.b

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       +
        #      / \
        #     b   *
        #        / \
        #       a   5
        e = m.b + e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            +
        #          /   \
        #         *     +
        #        / \   / \
        #       a   5 b   c
        e2 = m.b + m.c
        e = e1 + e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            +
        #          /   \
        #         +     *
        #        / \   / \
        #       b   c a   5
        e2 = m.b + m.c
        e = e2 + e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            +
        #          /   \
        #         *     *
        #        / \   / \
        #       a   5 b   5
        e2 = m.b * 5
        e = e2 + e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_negation(self):
        #    -
        #     \
        #      a
        m = AbstractModel()
        m.a = Var()
        e = - m.a

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : -1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_simpleDiff(self):
        #    -
        #   / \
        #  a   b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a - m.b

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : -1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_constDiff(self):
        #    -
        #   / \
        #  a   5
        m = AbstractModel()
        m.a = Var()
        e = m.a - 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:-5, id(m.a) : 1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #    -
        #   / \
        #  5   a
        e = 5 - m.a

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : -1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_nestedDiff(self):
        #
        # Check the structure of nested differences
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       -
        #      / \
        #     -   5
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:-5, id(m.a):1, id(m.b):-1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       -
        #      / \
        #     5   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = 5 - e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:5, id(m.a):-1, id(m.b):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       -
        #      / \
        #     -   c
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - m.c

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):1, id(m.b):-1, id(m.c):-1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       -
        #      / \
        #     c   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = m.c - e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):-1, id(m.b):1, id(m.c):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            -
        #          /   \
        #         -     -
        #        / \   / \
        #       a   b c   d
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e1 - e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 4)
        self.assertTrue(len(rep._linear_coefs) == 4)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):1, id(m.b):-1, id(m.c):-1, id(m.d):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            -
        #          /   \
        #         -     -
        #        / \   / \
        #       c   d a   b
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e2 - e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 4)
        self.assertTrue(len(rep._linear_coefs) == 4)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):-1, id(m.b):1, id(m.c):1, id(m.d):-1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_sumOf_nestedTrivialProduct2(self):
        #
        # Check the structure of sum of products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       -
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 - m.b

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):-1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       -
        #      / \
        #     b   *
        #        / \
        #       a   5
        e1 = m.a * 5
        e = m.b - e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):-5, id(m.b):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            -
        #          /   \
        #         *     -
        #        / \   / \
        #       a   5 b   c
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e1 - e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):-1, id(m.c):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            -
        #          /   \
        #         -     *
        #        / \   / \
        #       b   c a   5
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e2 - e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 3)
        self.assertTrue(len(rep._linear_coefs) == 3)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):-5, id(m.b):1, id(m.c):-1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_simpleProduct(self):
        #    *
        #   / \
        #  a   p
        m = AbstractModel()
        m.a = Var()
        m.p = Param(default=2)
        e = m.a * m.p

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):2 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_simpleProduct(self):
        #    *
        #   / \
        #  a   5
        m = AbstractModel()
        m.a = Var()
        e = m.a * 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #    *
        #   / \
        #  5   a
        e = 5 * m.a

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_nestedProduct(self):
        #       *
        #      / \
        #     *   5
        #    / \
        #   a   b
        m = ConcreteModel()
        m.a = Var()
        m.b = Param(default=2)
        m.c = Param(default=3)
        m.d = Param(default=7)

        e1 = m.a * m.b
        e = e1 * 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):10 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       *
        #      / \
        #     5   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = 5 * e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):10 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       a   b c   d
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):42 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_nestedProduct2(self):
        #
        # Check the structure of nested products
        #
        m = ConcreteModel()
        m.a = Param(default=2)
        m.b = Param(default=3)
        m.c = Param(default=5)
        m.d = Var()
        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         +     +
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c + e1
        e3 = e1 + m.d
        e = e2 * e3

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:50, id(m.d):10 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c * e1
        e3 = e1 * m.d
        e = e2 * e3

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.d):125 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_division(self):
        #
        #           /
        #          / \
        #         +   2
        #        / \
        #       a   b
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        m.y = Var(initialize=2.0)
        m.y.fixed = True

        e = (m.a + m.b)/2.0

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):0.5, id(m.b):0.5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #           /
        #          / \
        #         +   y
        #        / \
        #       a   b
        e = (m.a + m.b)/m.y

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):0.5, id(m.b):0.5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #            /
        #          /   \
        #         +     +
        #        / \   / \
        #       a   b y   2
        e = (m.a + m.b)/(m.y+2)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):0.25, id(m.b):0.25 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_weighted_sum1(self):
        #       *
        #      / \
        #     +   5
        #    / \
        #   a   b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        e1 = m.a + m.b
        e = e1 * 5

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       *
        #      / \
        #     5   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = 5 * e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):5 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       *
        #      / \
        #     5   *
        #        / \
        #       2   +
        #          / \
        #         a   b
        e1 = m.a + m.b
        e = 5 * 2* e1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):10, id(m.b):10 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       5(a+2(a+b))
        e = 5*(m.a+2*(m.a+m.b))

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 2)
        self.assertTrue(len(rep._linear_coefs) == 2)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):15, id(m.b):10 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_pow(self):
        #       ^
        #      / \
        #     a   0
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        m.p = Param()
        m.q = Param(default=1)

        e = m.a ** 0

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       ^
        #      / \
        #     a   1
        e = m.a ** 1

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       ^
        #      / \
        #     a   2
        e = m.a ** 2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertFalse(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))

        #       ^
        #      / \
        #     a   p
        e = m.a ** m.p

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertFalse(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))

        #       ^
        #      / \
        #     a   q
        e = m.a ** m.q

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_fabs(self):
        #      fabs
        #      / 
        #     a   
        m = ConcreteModel()
        m.a = Var()
        m.q = Param(default=-1)

        e = fabs(m.a)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertFalse(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))

        #      fabs
        #      / 
        #     q   
        e = fabs(m.q)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        ##
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_cos(self):
        #      cos
        #      / 
        #     a   
        m = ConcreteModel()
        m.a = Var()
        m.q = Param(default=0)

        e = cos(m.a)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertFalse(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))

        #      cos
        #      / 
        #     q   
        e = cos(m.q)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { None:1.0 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        ##
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

    def test_ExprIf(self):
        #       ExprIf
        #      /  |   \
        #   True  a    b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.q = Param(default=1)

        e = EXPR.Expr_if(IF=True, THEN=m.a, ELSE=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       ExprIf
        #      /  |   \
        #  False  a    b
        e = EXPR.Expr_if(IF=False, THEN=m.a, ELSE=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 1)
        self.assertTrue(len(rep._linear_coefs) == 1)
        self.assertTrue(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 0)
        baseline = { id(m.b):1 }
        self.assertEqual(baseline, linear_repn_to_dict(rep))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, linear_repn_to_dict(rep))

        #       ExprIf
        #      /  |   \
        #  bool  a    b
        e = EXPR.Expr_if(IF=m.q, THEN=m.a, ELSE=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep._linear_vars) == 0)
        self.assertTrue(len(rep._linear_coefs) == 0)
        self.assertFalse(rep._nonlinear_expr is None)
        self.assertTrue(len(rep._nonlinear_vars) == 2)
        baseline = set([ id(m.a), id(m.b) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))
        #
        e_ = EXPR.compress_expression(e)
        rep = generate_standard_repn(e_)
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep._nonlinear_expr,include_potentially_variable=True)))


if __name__ == "__main__":
    unittest.main()
