# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""
Walker unit tests: verify that XpressExpressionWalker produces correct xp expressions.

Mutable parameter tracking is now done by generate_standard_repn in
xpress_persistent.py, not by the walker.  These tests verify only xp_expr
building correctness across LP/MIP/QP/NLP expression patterns.
"""

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from parameterized import parameterized

from pyomo.common.dependencies import attempt_import
from pyomo.contrib.solver.common.util import IncompatibleModelError
from pyomo.core.expr import sin, ceil
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.contrib.solver.solvers.xpress.xpress_base import _before_linear
from pyomo.core.expr import MinExpression, MaxExpression
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.environ import Expr_if
from pyomo.contrib.solver.solvers.xpress.xpress_base import _EXIT_HANDLERS

xp, xpress_available = attempt_import('xpress', catch_exceptions=(Exception,))

try:
    from pyomo.contrib.solver.solvers.xpress.xpress_base import XpressExpressionWalker
except Exception:
    pass

if not xpress_available:
    raise unittest.SkipTest('Xpress not available')


def _make_walker():
    """Return (model, walker, xv, yv, zv) with x/y/z vars registered."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    prob = xp.problem()
    xv = prob.addVariable(name='x')
    yv = prob.addVariable(name='y')
    zv = prob.addVariable(name='z')
    var_map = {id(m.x): xv, id(m.y): yv, id(m.z): zv}
    walker = XpressExpressionWalker(var_map, prob)
    return m, walker, xv, yv, zv


def _walk_and_assert(
    test_case,
    walker,
    expr,
    expected_type,
    expected_lin=[],
    expected_quad=[],
    expected_nlp_vars=[],
    expected_nlp_scalars=[],
    expected_nlp_operators=[],
):
    """Walk a Pyomo expression and assert structural properties of the result.

    expected_type: xp.expression (linear/mixed), xp.linterm (single scaled var),
                   xp.quadterm (pure quad), xp.nonlin (nonlinear).
    expected_lin:  [] = assert no linear terms; [(pyo_var, coef), ...] = exact.
    expected_quad: [] = assert no quadratic terms; [(v1, v2, coef), ...] = exact.
                   Quadratic lookup is symmetric (x*y == y*x).

    NLP formula checks:
    expected_nlp_vars:      list of Pyomo VarData -- all must appear as COL tokens.
    expected_nlp_scalars:   list of float -- all must appear as CON tokens.
    expected_nlp_operators: list of str -- each must appear at least once as an
                            IFUN or OP token. Supported names (case-insensitive):
                            'sin','cos','tan','asin','acos','atan',
                            'exp','ln','log10','sqrt','abs','min','max',
                            'uminus','mul','div','plus','minus','pow'.

    Returns the xp expression produced by the walker.
    """
    result = walker.walk_expression(expr)

    tc = test_case
    var_map = walker.var_map

    tc.assertIsInstance(result, expected_type)

    # Test linear part
    lin_vars, lin_coefs = result.extractLinear()
    lin_by_id = {id(v): c for v, c in zip(lin_vars, lin_coefs)}
    tc.assertEqual(len(lin_by_id), len(expected_lin))
    for pyo_var, expected_coef in expected_lin:
        xp_var = var_map[id(pyo_var)]
        tc.assertIn(id(xp_var), lin_by_id, f"{pyo_var.name} missing from linear terms")
        tc.assertAlmostEqual(lin_by_id[id(xp_var)], expected_coef, places=12)

    # Test quadratic part
    def qkey(v1, v2):
        id1, id2 = id(v1), id(v2)
        return (id1, id2) if id1 < id2 else (id2, id1)

    q_v1s, q_v2s, q_coefs = result.extractQuadratic()
    quad_by_ids = {qkey(v1, v2): c for v1, v2, c in zip(q_v1s, q_v2s, q_coefs)}
    tc.assertEqual(len(q_coefs), len(expected_quad))
    for pyo_v1, pyo_v2, expected_coef in expected_quad:
        key = qkey(var_map[id(pyo_v1)], var_map[id(pyo_v2)])
        tc.assertIn(
            key,
            quad_by_ids,
            f"({pyo_v1.name}, {pyo_v2.name}) missing from quadratic terms",
        )
        tc.assertAlmostEqual(quad_by_ids[key], expected_coef, places=12)

    # Test constraint is a valid Xpress constraint
    test_con = xp.constraint(body=result, lb=-1e20, ub=1e20)
    n_before = walker.prob.attributes.rows
    walker.prob.addConstraint(test_con)
    tc.assertEqual(walker.prob.attributes.rows, n_before + 1)
    tc.assertEqual(test_con, walker.prob.getConstraint(n_before))

    # NLP formula checks -- only when at least one expected_nlp_* is provided.
    tok_types, tok_values = walker.prob.nlpGetFormula(test_con, 0)

    # Token type constants (from xprs.h)
    _TOK_CON = 1  # numeric constant
    _TOK_COL = 10  # variable column index
    _TOK_IFUN = 12  # intrinsic function
    _TOK_OP = 31  # arithmetic operator
    # Intrinsic function codes
    _IFUN = {
        'log10': 14,
        'ln': 15,
        'exp': 16,
        'abs': 17,
        'sqrt': 18,
        'sin': 27,
        'cos': 28,
        'tan': 291,
        'asin': 30,
        'arcsin': 30,
        'acos': 31,
        'arccos': 31,
        'atan': 32,
        'arctan': 32,
        'min': 33,
        'max': 34,
    }
    # Arithmetic operator codes
    _OP = {'uminus': 1, 'pow': 2, 'mul': 3, 'div': 4, 'plus': 5, 'minus': 6}

    col_indices = {int(v) for t, v in zip(tok_types, tok_values) if int(t) == _TOK_COL}
    scalars = [v for t, v in zip(tok_types, tok_values) if int(t) == _TOK_CON]
    ifun_codes = {int(v) for t, v in zip(tok_types, tok_values) if int(t) == _TOK_IFUN}
    op_codes = {int(v) for t, v in zip(tok_types, tok_values) if int(t) == _TOK_OP}

    for pyo_var in expected_nlp_vars:
        xp_var = var_map[id(pyo_var)]
        tc.assertIn(
            xp_var.index,
            col_indices,
            f"Variable {pyo_var.name} not found in NLP formula",
        )

    tc.assertEqual(len(scalars), len(expected_nlp_scalars))
    for s in expected_nlp_scalars:
        tc.assertTrue(
            any(abs(v - s) < 1e-10 for v in scalars),
            f"Scalar {s} not found in NLP formula scalars {scalars}",
        )

    for op_name in expected_nlp_operators:
        key = op_name.lower()
        if key in _IFUN:
            tc.assertIn(
                _IFUN[key],
                ifun_codes,
                f"NLP function '{op_name}' (code {_IFUN[key]}) not found in formula",
            )
        elif key in _OP:
            tc.assertIn(
                _OP[key],
                op_codes,
                f"Operator '{op_name}' (code {_OP[key]}) not found in formula",
            )
        else:
            raise ValueError(f"Unknown NLP operator name '{op_name}'")

    walker.prob.delConstraint(test_con)
    tc.assertEqual(walker.prob.attributes.rows, n_before)
    return result


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressWalkerLinear(unittest.TestCase):

    def test_linear_float_coef(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            3.0 * m.x + 2.0 * m.y,
            expected_type=xp.expression,
            expected_lin=[(m.x, 3.0), (m.y, 2.0)],
        )

    def test_linear_mutable_coef(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=5.0)
        _walk_and_assert(
            self,
            w,
            m.p * m.x + m.y,
            expected_type=xp.expression,
            expected_lin=[(m.x, 5.0), (m.y, 1.0)],
        )

    def test_linear_zero_coef(self):
        # 0*x is folded to constant 0 by _before_monomial; only y remains as a term.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            0 * m.x + m.y,
            expected_type=xp.expression,
            expected_lin=[(m.y, 1.0)],
        )

    def test_fixed_var_kept_as_column(self):
        # x fixed at 2.0 must still appear as an xp.var column, not folded to 6.0.
        m, w, xv, yv, _ = _make_walker()
        m.x.fix(2.0)
        _walk_and_assert(
            self,
            w,
            3.0 * m.x + m.y,
            expected_type=xp.expression,
            expected_lin=[(m.x, 3.0), (m.y, 1.0)],
        )

    def test_sum_with_constant(self):
        # The scalar 5.0 is embedded as a constant; extractLinear does not return it.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            3.0 * m.x + 5.0,
            expected_type=xp.expression,
            expected_lin=[(m.x, 3.0)],
        )

    def test_zero_coef_monomial_is_constant(self):
        # 0*y -> constant 0 (not a linear term); sin(x) is the only meaningful part.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            sin(m.x) + 0 * m.y,
            expected_type=xp.nonlin,
            expected_lin=[],
            expected_quad=[],
        )

    def test_mutable_constant_body_const(self):
        # x + p where p=3.0 (mutable scalar). The constant is embedded; only x linear.
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=3.0)
        _walk_and_assert(
            self, w, m.x + m.p, expected_type=xp.expression, expected_lin=[(m.x, 1.0)]
        )

    def test_negation(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self, w, -m.x, expected_type=xp.linterm, expected_lin=[(m.x, -1.0)]
        )

    def test_division_by_constant(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self, w, m.x / 2.0, expected_type=xp.linterm, expected_lin=[(m.x, 0.5)]
        )

    def test_linear_two_vars_no_coef(self):
        # _before_linear branch (a): bare VarData with implicit coefficient 1.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            m.x + m.y,
            expected_type=xp.expression,
            expected_lin=[(m.x, 1.0), (m.y, 1.0)],
        )

    def test_all_constant_expr(self):
        m, w, xv, yv, _ = _make_walker()
        result = w.walk_expression(3.0 + 2.0)
        self.assertAlmostEqual(result, 5.0)

    def test_linear_fast_path(self):
        """_before_linear handles LinearExpression directly without full walker descent."""

        m, w, xv, yv, _ = _make_walker()
        expr = m.x + m.y
        self.assertIsInstance(expr, LinearExpression)
        _, result = _before_linear(w, expr)
        self.assertIsNotNone(result)

    def test_before_linear_dispatcher_contract(self):
        """_before_linear must return (False, result) -- False prevents descent into children."""

        m, w, xv, yv, _ = _make_walker()
        expr = 3.0 * m.x + 2.0 * m.y
        self.assertIsInstance(expr, LinearExpression)
        should_descend, result = _before_linear(w, expr)
        self.assertFalse(should_descend)
        self.assertIsNotNone(result)


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressWalkerQuadratic(unittest.TestCase):

    def test_power_squared(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            m.x**2,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.x, 1.0)],
        )

    def test_product_two_vars(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            m.x * m.y,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.y, 1.0)],
        )

    def test_mutable_quad_coef(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=3.0)
        _walk_and_assert(
            self,
            w,
            m.p * m.x * m.y,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.y, 3.0)],
        )

    def test_bilinear_p_times_paren_xy(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=3.0)
        _walk_and_assert(
            self,
            w,
            m.p * (m.x * m.y),
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.y, 3.0)],
        )

    def test_monomial_times_var(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        _walk_and_assert(
            self,
            w,
            (m.p * m.x) * m.y,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.y, 2.0)],
        )

    def test_monomial_times_same_var(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=4.0)
        _walk_and_assert(
            self,
            w,
            (m.p * m.x) * m.x,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.x, 4.0)],
        )

    def test_mutable_coef_var_squared(self):
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=3.0)
        _walk_and_assert(
            self,
            w,
            m.p * m.x**2,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.x, 3.0)],
        )

    def test_non_mutable_coef_var_squared(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            3 * m.x**2,
            expected_type=xp.quadterm,
            expected_lin=[],
            expected_quad=[(m.x, m.x, 3.0)],
        )


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressWalkerNonlinear(unittest.TestCase):

    def test_division_by_variable(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            m.x / m.y,
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y],
            expected_nlp_operators=['div'],
        )

    def test_power_cubic_is_nl(self):
        # Xpress stores x*y*z as a pure NL term; col list is empty in the formula.
        m, w, xv, yv, zv = _make_walker()
        _walk_and_assert(self, w, m.x * m.y * m.z, expected_type=xp.nonlin)

    @parameterized.expand(
        [
            # (name, fn, expected_nlp_scalars, primary_operator)
            # sin/cos map directly; hyperbolic decompose via exp so scalars appear.
            # sinh=0.5*(exp(x)-exp(-x)): scalar 0.5
            # cosh=0.5*(exp(x)+exp(-x)): scalar 0.5
            # tanh=(exp(x)-exp(-x))/(exp(x)+exp(-x)): no scalars
            # asinh=ln(x+sqrt(x^2+1)): scalar 1.0
            # acosh=ln(x+sqrt(x^2-1)): scalar -1.0
            # atanh=0.5*ln((1+x)/(1-x)): scalars [1.0, 1.0, 0.5]
            ('sin', pyo.sin, [], ['sin']),
            ('cos', pyo.cos, [], ['cos']),
            ('sinh', pyo.sinh, [0.5], ['exp']),
            ('cosh', pyo.cosh, [0.5], ['exp']),
            ('tanh', pyo.tanh, [], ['exp', 'div']),
            ('asinh', pyo.asinh, [1.0], ['ln', 'sqrt']),
            ('acosh', pyo.acosh, [-1.0], ['ln', 'sqrt']),
            ('atanh', pyo.atanh, [1.0, 1.0, 0.5], ['ln']),
        ]
    )
    def test_nl_trig_hyperbolic_plus_linear(self, fn_name, fn, nlp_scalars, nlp_ops):
        # fn(x) + y: NL formula contains x and the primary operator.
        # y is in the linear part (extractLinear), not in the NLP formula.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            fn(m.x) + m.y,
            expected_type=xp.nonlin,
            expected_lin=[(m.y, 1.0)],
            expected_quad=[],
            expected_nlp_vars=[m.x],
            expected_nlp_scalars=nlp_scalars,
            expected_nlp_operators=nlp_ops,
        )

    def test_nl_mutable_outside_nl(self):
        # p*sin(x): p=2.0 is folded as a CON token in the NLP formula.
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        _walk_and_assert(
            self,
            w,
            m.p * sin(m.x),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x],
            expected_nlp_scalars=[2.0],
            expected_nlp_operators=['sin'],
        )

    def test_nl_mutable_linear_coexist(self):
        # sin(x) + p*y: sin(x) is in the NLP formula; p*y is linear.
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        _walk_and_assert(
            self,
            w,
            sin(m.x) + m.p * m.y,
            expected_type=xp.nonlin,
            expected_lin=[(m.y, 2.0)],
            expected_nlp_vars=[m.x],
            expected_nlp_scalars=[],
            expected_nlp_operators=['sin'],
        )

    def test_nl_mutable_sin_arg(self):
        # sin(p*x): p=2.0 is a CON token inside the argument.
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        _walk_and_assert(
            self,
            w,
            sin(m.p * m.x),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x],
            expected_nlp_scalars=[2.0],
            expected_nlp_operators=['sin'],
        )

    def test_min_expression(self):

        m, w, xv, yv, zv = _make_walker()
        _walk_and_assert(
            self,
            w,
            MinExpression([m.x, m.y, m.z]),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y, m.z],
            expected_nlp_scalars=[],
            expected_nlp_operators=['min'],
        )

    def test_abs_expression(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            abs(m.x + m.y),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y],
            expected_nlp_scalars=[],
            expected_nlp_operators=['abs'],
        )

    def test_max_expression(self):

        m, w, xv, yv, zv = _make_walker()
        _walk_and_assert(
            self,
            w,
            MaxExpression([m.x, m.y, m.z]),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y, m.z],
            expected_nlp_scalars=[],
            expected_nlp_operators=['max'],
        )

    def test_max_all_constants(self):

        _, w, _, _, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            MaxExpression([NumericConstant(1.0), NumericConstant(2.0)]),
            expected_type=xp.nonlin,
            expected_nlp_scalars=[1.0, 2.0],
            expected_nlp_operators=['max'],
        )

    def test_mutable_param_in_max(self):

        m, w, xv, _, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=3.0)
        _walk_and_assert(
            self,
            w,
            MaxExpression([m.x, m.p]),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x],
            expected_nlp_scalars=[3.0],
            expected_nlp_operators=['max'],
        )

    def test_sum_divided_by_constant(self):
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            (m.x + m.y) / 2.0,
            expected_type=xp.expression,
            expected_lin=[(m.x, 0.5), (m.y, 0.5)],
        )

    def test_nl_times_nl(self):
        # sin(x)*sin(y): product of two NL expressions. Both variables and both
        # sin operators appear in the row formula.
        m, w, xv, yv, _ = _make_walker()
        _walk_and_assert(
            self,
            w,
            sin(m.x) * sin(m.y),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y],
            expected_nlp_operators=['sin'],
        )

    def test_mutable_param_in_nl_product(self):
        # sin(p*x)*sin(y): mutable param inside the NL argument of a product of two
        # NL expressions. Exercises the ProductExpression handler with a mutable param.
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        _walk_and_assert(
            self,
            w,
            pyo.sin(m.p * m.x) * pyo.sin(m.y),
            expected_type=xp.nonlin,
            expected_nlp_vars=[m.x, m.y],
            expected_nlp_scalars=[2.0],
            expected_nlp_operators=['sin'],
        )

    def test_before_npv_evaluation_error_arithmetic(self):
        """NPV sub-expression m.q/m.r with r=0 raises ZeroDivisionError at walk time.
        _before_npv calls value() on the sub-expression; the error propagates out."""
        m, w, xv, yv, _ = _make_walker()
        m.q = pyo.Param(mutable=True, initialize=1.0)
        m.r = pyo.Param(mutable=True, initialize=0.0)
        with self.assertRaises(ZeroDivisionError):
            w.walk_expression(sin(m.x) + m.q / m.r)

    def test_exit_unary_const_domain_error(self):
        """sqrt(-1) in an NPV sub-expression raises ValueError from value() -- loud,
        not silent nan. _before_npv has no try-except so domain errors propagate."""

        m, w, xv, _, _ = _make_walker()
        m.s = pyo.Param(mutable=True, initialize=1.0)
        expr = sin(m.x) + pyo.sqrt(m.s)
        m.s.set_value(-1.0)
        with self.assertRaises(ValueError):
            w.walk_expression(expr)

    def test_npv_evaluation_error(self):
        """NPV sub-expression log(m.p) with p<0 raises ValueError at walk time.
        _before_npv calls value() on the sub-expression; the error propagates out."""
        m, w, xv, _, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=-1.0)
        expr = pyo.log(m.p) + m.x
        with self.assertRaises(ValueError):
            w.walk_expression(expr)


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressWalkerCache(unittest.TestCase):

    def test_named_expr_cache_populated(self):
        """_exit_named_expression must insert into subexpression_cache on first walk."""
        m, w, xv, yv, _ = _make_walker()
        m.e = pyo.Expression(expr=m.x + m.y)
        self.assertEqual(len(w.subexpression_cache), 0)
        _walk_and_assert(
            self,
            w,
            m.e + 1.0,
            expected_type=xp.expression,
            expected_lin=[(m.x, 1.0), (m.y, 1.0)],
        )
        self.assertEqual(len(w.subexpression_cache), 1)

    def test_named_expr_cache_hit(self):
        """Named expression walked once; cache grows only on first walk."""
        m, w, xv, yv, _ = _make_walker()
        m.e = pyo.Expression(expr=m.x + m.y)
        self.assertEqual(len(w.subexpression_cache), 0)
        _walk_and_assert(
            self,
            w,
            m.e + 1.0,
            expected_type=xp.expression,
            expected_lin=[(m.x, 1.0), (m.y, 1.0)],
        )
        self.assertEqual(len(w.subexpression_cache), 1)
        _walk_and_assert(
            self,
            w,
            m.e + 2.0,
            expected_type=xp.expression,
            expected_lin=[(m.x, 1.0), (m.y, 1.0)],
        )
        self.assertEqual(len(w.subexpression_cache), 1)  # Hit

    def test_named_expr_produces_valid_xp_expr(self):
        """Named expression with mutable coef: valid xp expression on cache miss."""
        m, w, xv, yv, _ = _make_walker()
        m.p = pyo.Param(mutable=True, initialize=2.0)
        m.e = pyo.Expression(expr=m.p * m.x)
        _walk_and_assert(
            self,
            w,
            m.e + m.y,
            expected_type=xp.expression,
            expected_lin=[(m.x, 2.0), (m.y, 1.0)],
        )


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressWalkerErrors(unittest.TestCase):

    def test_unsupported_function_ceil_raises(self):
        """ceil is not supported by Xpress -- must raise IncompatibleModelError."""
        m, w, xv, yv, _ = _make_walker()
        with self.assertRaises(IncompatibleModelError) as ctx:
            w.walk_expression(ceil(m.x))
        self.assertIn('ceil', str(ctx.exception))

    def test_unsupported_function_floor_raises(self):
        """floor is not supported by Xpress -- must raise IncompatibleModelError."""
        m, w, xv, yv, _ = _make_walker()
        with self.assertRaises(IncompatibleModelError) as ctx:
            w.walk_expression(pyo.floor(m.x))
        self.assertIn('floor', str(ctx.exception))

    def test_expr_if_raises(self):

        m, w, xv, yv, _ = _make_walker()
        expr = Expr_if(IF=m.x > 0, THEN=m.x, ELSE=-m.x)
        with self.assertRaises(IncompatibleModelError):
            w.walk_expression(expr)

    def test_unregistered_expression_type_raises(self):
        """_ExitHandlerMap.__missing__ raises IncompatibleModelError when the
        expression type's MRO has no registered exit handler."""

        class _CustomUnregisteredExpr:
            """Synthetic class that is not part of the Pyomo expression hierarchy."""

        with self.assertRaises(IncompatibleModelError):
            _EXIT_HANDLERS[_CustomUnregisteredExpr]


if __name__ == '__main__':
    unittest.main()
