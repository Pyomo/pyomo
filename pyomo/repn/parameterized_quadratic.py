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

from pyomo.common.numeric_types import native_numeric_types
from pyomo.core.expr.numeric_expr import (
    DivisionExpression,
    Expr_ifExpression,
    mutable_expression,
    PowExpression,
    ProductExpression,
)
from pyomo.repn.linear import (
    ExitNodeDispatcher,
    initialize_exit_node_dispatcher,
    _handle_division_ANY_constant,
    _handle_expr_if_const,
    _handle_pow_ANY_constant,
    _handle_product_ANY_constant,
    _handle_product_constant_ANY,
)
from pyomo.repn.parameterized_linear import (
    define_exit_node_handlers as _param_linear_def_exit_node_handlers,
    ParameterizedLinearRepnVisitor,
    to_expression,
    _handle_division_ANY_pseudo_constant,
    _merge_dict,
)
from pyomo.repn.quadratic import QuadraticRepn, _mul_linear_linear
from pyomo.repn.util import ExprType


_FIXED = ExprType.FIXED
_CONSTANT = ExprType.CONSTANT
_LINEAR = ExprType.LINEAR
_GENERAL = ExprType.GENERAL
_QUADRATIC = ExprType.QUADRATIC


class ParameterizedQuadraticRepn(QuadraticRepn):
    def __str__(self):
        return (
            "ParameterizedQuadraticRepn("
            f"mult={self.multiplier}, "
            f"const={self.constant}, "
            f"linear={self.linear}, "
            f"quadratic={self.quadratic}, "
            f"nonlinear={self.nonlinear})"
        )

    def __repr__(self):
        return str(self)

    def walker_exitNode(self):
        if self.nonlinear is not None:
            return _GENERAL, self
        elif self.quadratic:
            return _QUADRATIC, self
        elif self.linear:
            return _LINEAR, self
        elif self.constant.__class__ in native_numeric_types:
            return _CONSTANT, self.multiplier * self.constant
        else:
            return _FIXED, self.multiplier * self.constant

    def to_expression(self, visitor):
        var_map = visitor.var_map
        if self.nonlinear is not None:
            # We want to start with the nonlinear term (and use
            # assignment) in case the term is a non-numeric node (like a
            # relational expression)
            ans = self.nonlinear
        else:
            ans = 0
        if self.quadratic:
            with mutable_expression() as e:
                for (x1, x2), coef in self.quadratic.items():
                    if x1 == x2:
                        e += coef * var_map[x1] ** 2
                    else:
                        e += coef * (var_map[x1] * var_map[x2])
            ans += e
        if self.linear:
            var_map = visitor.var_map
            with mutable_expression() as e:
                for vid, coef in self.linear.items():
                    if not is_zero(coef):
                        e += coef * var_map[vid]
            if e.nargs() > 1:
                ans += e
            elif e.nargs() == 1:
                ans += e.arg(0)
        if not is_zero(self.constant):
            ans += self.constant
        if not is_equal_to(self.multiplier, 1):
            ans *= self.multiplier
        return ans

    def append(self, other):
        """Append a child result from acceptChildResult

        Notes
        -----
        This method assumes that the operator was "+". It is implemented
        so that we can directly use a ParameterizedLinearRepn() as a `data` object in
        the expression walker (thereby allowing us to use the default
        implementation of acceptChildResult [which calls
        `data.append()`] and avoid the function call for a custom
        callback).

        """
        _type, other = other
        if _type is _CONSTANT or _type is _FIXED:
            self.constant += other
            return

        mult = other.multiplier
        try:
            _mult = bool(mult)
            if not _mult:
                return
            if mult == 1:
                _mult = False
        except:
            _mult = True

        const = other.constant
        try:
            _const = bool(const)
        except:
            _const = True

        if _mult:
            if _const:
                self.constant += mult * const
            if other.linear:
                _merge_dict(self.linear, mult, other.linear)
            if other.quadratic:
                if not self.quadratic:
                    self.quadratic = {}
                _merge_dict(self.quadratic, mult, other.quadratic)
            if other.nonlinear is not None:
                nl = mult * other.nonlinear
                if self.nonlinear is None:
                    self.nonlinear = nl
                else:
                    self.nonlinear += nl
        else:
            if _const:
                self.constant += const
            if other.linear:
                _merge_dict(self.linear, 1, other.linear)
            if other.quadratic:
                if not self.quadratic:
                    self.quadratic = {}
                _merge_dict(self.quadratic, 1, other.quadratic)
            if other.nonlinear is not None:
                nl = other.nonlinear
                if self.nonlinear is None:
                    self.nonlinear = nl
                else:
                    self.nonlinear += nl


def is_zero(obj):
    """Return true if expression/constant is zero, False otherwise."""
    return obj.__class__ in native_numeric_types and not obj


def is_zero_product(e1, e2):
    """
    Return True if e1 is zero and e2 is not known to be an indeterminate
    (e.g., NaN, inf), or vice versa, False otherwise.
    """
    return (is_zero(e1) and e2 == e2) or (e1 == e1 and is_zero(e2))


def is_equal_to(obj, val):
    return obj.__class__ in native_numeric_types and obj == val


def _handle_product_linear_linear(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    # Quadratic first, because we will update linear in a minute
    arg1.quadratic = _mul_linear_linear(visitor, arg1.linear, arg2.linear)
    # Linear second, as this relies on knowing the original constants
    if is_zero(arg2.constant):
        arg1.linear = {}
    elif not is_equal_to(arg2.constant, 1):
        c = arg2.constant
        for vid, coef in arg1.linear.items():
            arg1.linear[vid] = c * coef
    if not is_zero(arg1.constant):
        # TODO: what if a linear coefficient is indeterminate (nan/inf)?
        #       might that also affect nonlinear product handler?
        _merge_dict(arg1.linear, arg1.constant, arg2.linear)

    # Finally, the constant and multipliers
    if is_zero_product(arg1.constant, arg2.constant):
        arg1.constant = 0
    else:
        arg1.constant *= arg2.constant

    arg1.multiplier *= arg2.multiplier
    return _QUADRATIC, arg1


def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return _GENERAL, ans

    # multiplying (A1 + B1x + C1x^2 + D1(x)) * (A2 + B2x + C2x^2 + D2x))
    _, x1 = arg1
    _, x2 = arg2
    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1

    # constant term [A1A2]
    if is_zero_product(x1.constant, x2.constant):
        ans.constant = 0
    else:
        ans.constant = x1.constant * x2.constant

    # linear & quadratic terms
    if not is_zero(x2.constant):
        # [B1A2], [C1A2]
        x2_c = x2.constant
        if is_equal_to(x2_c, 1):
            ans.linear = dict(x1.linear)
            if x1.quadratic:
                ans.quadratic = dict(x1.quadratic)
        else:
            ans.linear = {vid: x2_c * coef for vid, coef in x1.linear.items()}
            if x1.quadratic:
                ans.quadratic = {k: x2_c * coef for k, coef in x1.quadratic.items()}
    if not is_zero(x1.constant):
        # [A1B2]
        _merge_dict(ans.linear, x1.constant, x2.linear)
        # [A1C2]
        if x2.quadratic:
            if ans.quadratic:
                _merge_dict(ans.quadratic, x1.constant, x2.quadratic)
            elif is_equal_to(x1.constant, 1):
                ans.quadratic = dict(x2.quadratic)
            else:
                c = x1.constant
                ans.quadratic = {k: c * coef for k, coef in x2.quadratic.items()}
    # [B1B2]
    if x1.linear and x2.linear:
        quad = _mul_linear_linear(visitor, x1.linear, x2.linear)
        if ans.quadratic:
            _merge_dict(ans.quadratic, 1, quad)
        else:
            ans.quadratic = quad

    # nonlinear portion
    # [D1A2] + [D1B2] + [D1C2] + [D1D2]
    ans.nonlinear = 0
    if x1.nonlinear is not None:
        ans.nonlinear += x1.nonlinear * x2.to_expression(visitor)
    x1.nonlinear = None
    x2.constant = 0
    x1_c = x1.constant
    x1.constant = 0
    x1_lin = x1.linear
    x1.linear = {}
    # [C1B2] + [C1C2] + [C1D2]
    if x1.quadratic:
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
        x1.quadratic = None
    x2.linear = {}
    # [B1C2] + [B1D2]
    if x1_lin and (x2.nonlinear is not None or x2.quadratic):
        x1.linear = x1_lin
        ans.nonlinear += x1.to_expression(visitor) * x2.to_expression(visitor)
    # [A1D2]
    if not is_zero(x1_c) and x2.nonlinear is not None:
        # TODO: what if nonlinear contains nan?
        ans.nonlinear += x1_c * x2.nonlinear
    return _GENERAL, ans


def define_exit_node_handlers(exit_node_handlers=None):
    if exit_node_handlers is None:
        exit_node_handlers = {}
    _param_linear_def_exit_node_handlers(exit_node_handlers)

    exit_node_handlers[ProductExpression].update(
        {
            None: _handle_product_nonlinear,
            (_CONSTANT, _QUADRATIC): _handle_product_constant_ANY,
            (_QUADRATIC, _CONSTANT): _handle_product_ANY_constant,
            # Replace handler from the linear walker
            (_LINEAR, _LINEAR): _handle_product_linear_linear,
            (_QUADRATIC, _FIXED): _handle_product_ANY_constant,
            (_FIXED, _QUADRATIC): _handle_product_constant_ANY,
        }
    )
    exit_node_handlers[DivisionExpression].update(
        {
            (_QUADRATIC, _CONSTANT): _handle_division_ANY_constant,
            (_QUADRATIC, _FIXED): _handle_division_ANY_pseudo_constant,
        }
    )
    exit_node_handlers[PowExpression].update(
        {(_QUADRATIC, _CONSTANT): _handle_pow_ANY_constant}
    )
    exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, i, _QUADRATIC): _handle_expr_if_const
            for i in (_CONSTANT, _LINEAR, _QUADRATIC, _GENERAL)
        }
    )
    exit_node_handlers[Expr_ifExpression].update(
        {
            (_CONSTANT, _QUADRATIC, i): _handle_expr_if_const
            for i in (_CONSTANT, _LINEAR, _GENERAL)
        }
    )
    return exit_node_handlers


class ParameterizedQuadraticRepnVisitor(ParameterizedLinearRepnVisitor):
    Result = ParameterizedQuadraticRepn
    exit_node_dispatcher = ExitNodeDispatcher(
        initialize_exit_node_dispatcher(define_exit_node_handlers())
    )
    max_exponential_expansion = 2
    expand_nonlinear_products = True

    def _factor_multiplier_into_quadratic_terms(self, ans, mult):
        linear = ans.linear
        zeros = []
        for vid, coef in linear.items():
            if not is_zero(coef):
                linear[vid] = mult * coef
            else:
                zeros.append(vid)
        for vid in zeros:
            del linear[vid]

        quadratic = ans.quadratic
        if quadratic is not None:
            quad_zeros = []
            for vid_pair, coef in ans.quadratic.items():
                if not is_zero(coef):
                    ans.quadratic[vid_pair] = mult * coef
                else:
                    quad_zeros.append(vid_pair)
            for vid_pair in quad_zeros:
                del quadratic[vid_pair]

        if ans.nonlinear is not None:
            ans.nonlinear *= mult
        if not is_zero(ans.constant):
            ans.constant *= mult
        ans.multiplier = 1

    def finalizeResult(self, result):
        ans = result[1]
        if ans.__class__ is self.Result:
            mult = ans.multiplier
            if mult.__class__ not in native_numeric_types:
                # mult is an expression--we should push it back into the other terms
                self._factor_multiplier_into_quadratic_terms(ans, mult)
                return ans
            if mult == 1:
                linear_zeros = [
                    (vid, coef) for vid, coef in ans.linear.items() if is_zero(coef)
                ]
                for vid, coef in linear_zeros:
                    del ans.linear[vid]

                if ans.quadratic:
                    quadratic_zeros = [
                        (vidpair, coef)
                        for vidpair, coef in ans.quadratic.items()
                        if is_zero(coef)
                    ]
                    for vidpair, coef in quadratic_zeros:
                        del ans.quadratic[vidpair]
            elif not mult:
                # the multiplier has cleared out the entire expression.
                # check if this is suppressing a NaN because we can't
                # clear everything out if it is
                has_nan_coefficient = (
                    ans.constant != ans.constant
                    or any(lcoeff != lcoeff for lcoeff in ans.linear.values())
                    or (
                        ans.quadratic is not None
                        and any(qcoeff != qcoeff for qcoeff in ans.quadratic.values())
                    )
                )
                if has_nan_coefficient:
                    # There's a nan in here, so we distribute the 0
                    self._factor_multiplier_into_quadratic_terms(ans, mult)
                    return ans
                return self.Result()
            else:
                # mult not in {0, 1}: factor it into the constant,
                # linear coefficients, quadratic coefficients,
                # and nonlinear term
                self._factor_multiplier_into_quadratic_terms(ans, mult)
            return ans

        ans = self.Result()
        assert result[0] in (_CONSTANT, _FIXED)
        ans.constant = result[1]
        return ans
