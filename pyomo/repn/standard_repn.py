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


import sys
import logging
import itertools

from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap

import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import ObjectiveData, ScalarObjective
from pyomo.core.base import Expression
from pyomo.core.base.expression import (
    ScalarExpression,
    NamedExpressionData,
    ExpressionData,
)
from pyomo.core.base.var import ScalarVar, Var, VarData, value
from pyomo.core.base.param import ScalarParam, ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective

from io import StringIO

logger = logging.getLogger('pyomo.core')


#
# This checks if the first argument is a numeric value.  If not
# then this is a Pyomo constant expression, and we can only check if its
# close to 'b' when it is constant.
#
def isclose_const(a, b, rel_tol=1e-9, abs_tol=0.0):
    if not a.__class__ in native_numeric_types:
        if a.is_constant():
            a = value(a)
        else:
            return False
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class StandardRepn(object):
    """
    This class defines a standard/common representation for Pyomo expressions
    that provides an efficient interface for writing all models.

    TODO: define what "efficient" means to us.
    """

    __slots__ = (
        'constant',  # The constant term
        'linear_coefs',  # Linear coefficients
        'linear_vars',  # Linear variables
        'quadratic_coefs',  # Quadratic coefficients
        'quadratic_vars',  # Quadratic variables
        'nonlinear_expr',  # Nonlinear expression
        'nonlinear_vars',
    )  # Variables that appear in the nonlinear expression

    def __init__(self):
        self.constant = 0
        self.linear_vars = tuple()
        self.linear_coefs = tuple()
        self.quadratic_vars = tuple()
        self.quadratic_coefs = tuple()
        self.nonlinear_expr = None
        self.nonlinear_vars = tuple()

    def __getstate__(self):
        """
        This method is required because this class uses slots.
        """
        return (
            self.constant,
            self.linear_coefs,
            self.linear_vars,
            self.quadratic_coefs,
            self.quadratic_vars,
            self.nonlinear_expr,
            self.nonlinear_vars,
        )

    def __setstate__(self, state):
        """
        This method is required because this class uses slots.
        """
        (
            self.constant,
            self.linear_coefs,
            self.linear_vars,
            self.quadratic_coefs,
            self.quadratic_vars,
            self.nonlinear_expr,
            self.nonlinear_vars,
        ) = state

    #
    # Generate a string representation of the expression
    #
    def __str__(self):  # pragma: nocover
        output = StringIO()
        output.write("\n")
        output.write("constant:       " + str(self.constant) + "\n")
        output.write(
            "linear vars:    " + str([v_.name for v_ in self.linear_vars]) + "\n"
        )
        output.write(
            "linear var ids: " + str([id(v_) for v_ in self.linear_vars]) + "\n"
        )
        output.write("linear coef:    " + str(list(self.linear_coefs)) + "\n")
        output.write(
            "quadratic vars:    "
            + str([(v_[0].name, v_[1].name) for v_ in self.quadratic_vars])
            + "\n"
        )
        output.write(
            "quadratic var ids: "
            + str([(id(v_[0]), id(v_[1])) for v_ in self.quadratic_vars])
            + "\n"
        )
        output.write("quadratic coef:    " + str(list(self.quadratic_coefs)) + "\n")
        if self.nonlinear_expr is None:
            output.write("nonlinear expr: None\n")
        else:
            output.write("nonlinear expr:\n")
            try:
                output.write(self.nonlinear_expr.to_string())
                output.write("\n")
            except AttributeError:
                output.write(str(self.nonlinear_expr))
                output.write("\n")
        output.write(
            "nonlinear vars: " + str([v_.name for v_ in self.nonlinear_vars]) + "\n"
        )
        output.write("\n")

        ret_str = output.getvalue()
        output.close()
        return ret_str

    def is_fixed(self):
        if (
            len(self.linear_vars) == 0
            and len(self.nonlinear_vars) == 0
            and len(self.quadratic_vars) == 0
        ):
            return True
        return False

    def polynomial_degree(self):
        if not self.nonlinear_expr is None:
            return None
        if len(self.quadratic_coefs) > 0:
            return 2
        if len(self.linear_coefs) > 0:
            return 1
        return 0

    def is_constant(self):
        return (
            self.nonlinear_expr is None
            and len(self.quadratic_coefs) == 0
            and len(self.linear_coefs) == 0
        )

    def is_linear(self):
        return self.nonlinear_expr is None and len(self.quadratic_coefs) == 0

    def is_quadratic(self):
        return len(self.quadratic_coefs) > 0 and self.nonlinear_expr is None

    def is_nonlinear(self):
        return not (self.nonlinear_expr is None and len(self.quadratic_coefs) == 0)

    def to_expression(self, sort=True):
        #
        # When an standard representation is created, the ordering of the
        # linear and quadratic terms may not be preserved.  Hence, the
        # sorting option ensures that an expression generated from a
        # standard representation has a consistent order.
        #
        # TODO: Should this replace non-mutable parameters with constants?
        #
        expr = self.constant

        lvars = [(i, v) for i, v in enumerate(self.linear_vars)]
        if sort:
            lvars = sorted(lvars, key=lambda x: str(x[1]))
        for i, v in lvars:
            c = self.linear_coefs[i]
            if c.__class__ in native_numeric_types:
                if not c:
                    pass
                if isclose_const(c, 1.0):
                    expr += v
                elif isclose_const(c, -1.0):
                    expr -= v
                elif c < 0.0:
                    expr -= -c * v
                else:
                    expr += c * v
            else:
                expr += c * v

        qvars = [(i, v) for i, v in enumerate(self.quadratic_vars)]
        if sort:
            qvars = sorted(qvars, key=lambda x: (str(x[1][0]), str(x[1][1])))
        for i, v in qvars:
            if id(v[0]) == id(v[1]):
                term = v[0] ** 2
            else:
                term = v[0] * v[1]
            c = self.quadratic_coefs[i]
            if c.__class__ in native_numeric_types:
                if isclose_const(c, 1.0):
                    expr += term
                elif isclose_const(c, -1.0):
                    expr -= term
                else:
                    expr += c * term
            else:
                expr += c * term

        if self.nonlinear_expr is not None:
            if expr.__class__ in native_numeric_types and expr == 0:
                # Some "NL" expressions do not support addition
                # (e.g. relational expressions)
                return self.nonlinear_expr
            expr += self.nonlinear_expr
        return expr


"""

Note:  This function separates linear terms from nonlinear terms.
Along the way, fixed variable and mutable parameter values *may* be
replaced with constants.  However, that is not guaranteed.  Thus,
the nonlinear expression may contain subexpressions whose value is
constant.  This was done to avoid additional work when a subexpression
is clearly nonlinear.  However, this requires that standard
representations be temporary.  They should be used to interface
to a solver and then be deleted.

"""


# @profile
def generate_standard_repn(
    expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None
):
    #
    # Use a custom Results object
    #
    global Results
    if quadratic:
        Results = ResultsWithQuadratics
    else:
        Results = ResultsWithoutQuadratics

    if True:
        #
        # Setup
        #
        if idMap is None:
            idMap = {}
        idMap.setdefault(None, {})
        if repn is None:
            repn = StandardRepn()
        #
        # The expression is a number or a non-variable constant
        # expression.
        #
        if expr.__class__ in native_numeric_types or not expr.is_potentially_variable():
            if compute_values:
                repn.constant = EXPR.evaluate_expression(expr)
            else:
                repn.constant = expr
            return repn
        #
        # The expression is a variable
        #
        elif expr.is_variable_type():
            if expr.fixed:
                if compute_values:
                    repn.constant = value(expr)
                else:
                    repn.constant = expr
                return repn
            repn.linear_coefs = (1,)
            repn.linear_vars = (expr,)
            return repn
        #
        # The expression is linear
        #
        elif expr.__class__ is EXPR.LinearExpression:
            linear_coefs = {}
            linear_vars = {}
            C_ = 0
            if compute_values:
                for arg in expr.args:
                    if arg.__class__ is EXPR.MonomialTermExpression:
                        c, v = arg.args
                        if c.__class__ not in native_numeric_types:
                            c = EXPR.evaluate_expression(c)
                        if v.fixed:
                            C_ += c * v.value
                            continue
                        id_ = id(v)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += c
                        else:
                            linear_coefs[id_] = c
                            linear_vars[id_] = v
                    elif arg.__class__ in native_numeric_types:
                        C_ += arg
                    elif arg.is_variable_type():
                        if arg.fixed:
                            C_ += arg.value
                            continue
                        id_ = id(arg)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += 1
                        else:
                            linear_coefs[id_] = 1
                            linear_vars[id_] = arg
                    else:
                        C_ += EXPR.evaluate_expression(arg)
            else:  # compute_values == False
                for arg in expr.args:
                    if arg.__class__ is EXPR.MonomialTermExpression:
                        c, v = arg.args
                        if v.fixed:
                            C_ += c * v
                            continue
                        id_ = id(v)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += c
                        else:
                            linear_coefs[id_] = c
                            linear_vars[id_] = v
                    elif arg.__class__ in native_numeric_types:
                        C_ += arg
                    elif arg.is_variable_type():
                        if arg.fixed:
                            C_ += arg
                            continue
                        id_ = id(arg)
                        if id_ in linear_coefs:
                            linear_coefs[id_] += 1
                        else:
                            linear_coefs[id_] = 1
                            linear_vars[id_] = arg
                    else:
                        C_ += arg

            vars_ = []
            coef_ = []
            for id_, coef in linear_coefs.items():
                if coef.__class__ in native_numeric_types and not coef:
                    continue
                if id_ not in idMap[None]:
                    key = len(idMap) - 1
                    idMap[None][id_] = key
                    idMap[key] = linear_vars[id_]
                else:
                    key = idMap[None][id_]
                vars_.append(idMap[key])
                coef_.append(coef)

            repn.linear_vars = tuple(vars_)
            repn.linear_coefs = tuple(coef_)
            repn.constant = C_
            return repn

        #
        # Unknown expression object
        #
        elif not expr.is_expression_type():  # pragma: nocover
            raise ValueError("Unexpected expression type: " + str(expr))

        #
        # WEH - Checking the polynomial degree didn't
        #       turn out to be a win.  But I'm leaving this
        #       in as a comment for now, since we're not
        #       done tuning this code.
        #
        # degree = expr.polynomial_degree()
        # if degree == 1:
        #    return _generate_linear_standard_repn(expr,
        #                        idMap=idMap,
        #                        compute_values=compute_values,
        #                        verbose=verbose,
        #                        repn=repn)
        # else:
        return _generate_standard_repn(
            expr,
            idMap=idMap,
            compute_values=compute_values,
            verbose=verbose,
            quadratic=quadratic,
            repn=repn,
        )


##-----------------------------------------------------------------------
##
## Logic for _generate_standard_repn
##
##-----------------------------------------------------------------------


class ResultsWithQuadratics(object):
    __slot__ = ('const', 'nonl', 'linear', 'quadratic')

    def __init__(self, constant=0, nonl=0, linear=None, quadratic=None):
        self.constant = constant
        self.nonl = nonl
        self.linear = {}
        # if linear is None:
        #    self.linear = {}
        # else:
        #    self.linear = linear
        self.quadratic = {}
        # if quadratic is None:
        #    self.quadratic = {}
        # else:
        #    self.quadratic = quadratic

    def __str__(self):  # pragma: nocover
        return "Const:\t%s\nLinear:\t%s\nQuadratic:\t%s\nNonlinear:\t%s" % (
            str(self.constant),
            str(self.linear),
            str(self.quadratic),
            str(self.nonl),
        )


class ResultsWithoutQuadratics(object):
    __slot__ = ('const', 'nonl', 'linear')

    def __init__(self, constant=0, nonl=0, linear=None):
        self.constant = constant
        self.nonl = nonl
        self.linear = {}
        # if linear is None:
        #    self.linear = {}
        # else:
        #    self.linear = linear

    def __str__(self):  # pragma: nocover
        return "Const:\t%s\nLinear:\t%s\nNonlinear:\t%s" % (
            str(self.constant),
            str(self.linear),
            str(self.nonl),
        )


Results = ResultsWithQuadratics


# @profile
def _collect_sum(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    nonl = []
    varkeys = idMap[None]

    for e_ in itertools.islice(exp._args_, exp.nargs()):
        if e_.__class__ is EXPR.MonomialTermExpression:
            lhs, v = e_.args
            if lhs.__class__ in native_numeric_types:
                if not lhs:
                    continue
            elif compute_values:
                lhs = value(lhs)
                if not lhs:
                    continue
            if v.fixed:
                if compute_values:
                    ans.constant += multiplier * lhs * value(v)
                else:
                    ans.constant += multiplier * lhs * v
            else:
                id_ = id(v)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = v
                if key in ans.linear:
                    ans.linear[key] += multiplier * lhs
                else:
                    ans.linear[key] = multiplier * lhs
        elif e_.__class__ in native_numeric_types:
            ans.constant += multiplier * e_
        elif e_.is_variable_type():
            if e_.fixed:
                if compute_values:
                    ans.constant += multiplier * e_.value
                else:
                    ans.constant += multiplier * e_
            else:
                id_ = id(e_)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = e_
                if key in ans.linear:
                    ans.linear[key] += multiplier
                else:
                    ans.linear[key] = multiplier
        elif not e_.is_potentially_variable():
            if compute_values:
                ans.constant += multiplier * value(e_)
            else:
                ans.constant += multiplier * e_
        else:
            res_ = _collect_standard_repn(
                e_, multiplier, idMap, compute_values, verbose, quadratic
            )
            #
            # Add returned from recursion
            #
            ans.constant += res_.constant
            if not (res_.nonl.__class__ in native_numeric_types and res_.nonl == 0):
                nonl.append(res_.nonl)
            for i, v in res_.linear.items():
                ans.linear[i] = ans.linear.get(i, 0) + v
            if quadratic:
                for i in res_.quadratic:
                    ans.quadratic[i] = ans.quadratic.get(i, 0) + res_.quadratic[i]

    if len(nonl) > 0:
        if len(nonl) == 1:
            ans.nonl = nonl[0]
        else:
            ans.nonl = EXPR.SumExpression(nonl)
    zero_coef = [
        k
        for k, coef in ans.linear.items()
        if coef.__class__ in native_numeric_types and not coef
    ]
    for k in zero_coef:
        ans.linear.pop(k)
    return ans


# @profile
def _collect_term(exp, multiplier, idMap, compute_values, verbose, quadratic):
    #
    # LHS is a numeric value
    #
    if exp._args_[0].__class__ in native_numeric_types:
        if exp._args_[0] == 0:  # TODO: coverage?
            return Results()
        return _collect_standard_repn(
            exp._args_[1],
            multiplier * exp._args_[0],
            idMap,
            compute_values,
            verbose,
            quadratic,
        )
    #
    # LHS is a non-variable expression
    #
    else:
        if compute_values:
            val = value(exp._args_[0])
            if val == 0:
                return Results()
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * val,
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
        else:
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * exp._args_[0],
                idMap,
                compute_values,
                verbose,
                quadratic,
            )


def _collect_prod(exp, multiplier, idMap, compute_values, verbose, quadratic):
    #
    # LHS is a numeric value
    #
    if exp._args_[0].__class__ in native_numeric_types:
        if exp._args_[0] == 0:  # TODO: coverage?
            return Results()
        return _collect_standard_repn(
            exp._args_[1],
            multiplier * exp._args_[0],
            idMap,
            compute_values,
            verbose,
            quadratic,
        )
    #
    # RHS is a numeric value
    #
    if exp._args_[1].__class__ in native_numeric_types:
        if exp._args_[1] == 0:  # TODO: coverage?
            return Results()
        return _collect_standard_repn(
            exp._args_[0],
            multiplier * exp._args_[1],
            idMap,
            compute_values,
            verbose,
            quadratic,
        )
    #
    # LHS is a non-variable expression
    #
    elif not exp._args_[0].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[0])
            if val == 0:
                return Results()
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * val,
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
        else:
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * exp._args_[0],
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
    #
    # RHS is a non-variable expression
    #
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[1])
            if val == 0:
                return Results()
            return _collect_standard_repn(
                exp._args_[0],
                multiplier * val,
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
        else:
            return _collect_standard_repn(
                exp._args_[0],
                multiplier * exp._args_[1],
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
    #
    # Both the LHS and RHS are potentially variable ...
    #
    # Collect LHS
    #
    lhs = _collect_standard_repn(
        exp._args_[0], 1, idMap, compute_values, verbose, quadratic
    )
    lhs_nonl_None = lhs.nonl.__class__ in native_numeric_types and not lhs.nonl
    #
    # LHS is potentially variable, but it turns out to be a constant
    # because the variables were fixed.
    #
    if (
        lhs_nonl_None
        and len(lhs.linear) == 0
        and (not quadratic or len(lhs.quadratic) == 0)
    ):
        if lhs.constant.__class__ in native_numeric_types and lhs.constant == 0:
            return Results()
        if compute_values:
            val = value(lhs.constant)
            if val == 0:  # TODO: coverage?
                return Results()
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * val,
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
        else:
            return _collect_standard_repn(
                exp._args_[1],
                multiplier * lhs.constant,
                idMap,
                compute_values,
                verbose,
                quadratic,
            )
    #
    # Collect RHS
    #
    rhs = _collect_standard_repn(
        exp._args_[1], 1, idMap, compute_values, verbose, quadratic
    )
    rhs_nonl_None = rhs.nonl.__class__ in native_numeric_types and not rhs.nonl
    #
    # If RHS is zero, then return an empty results
    #
    if (
        rhs_nonl_None
        and len(rhs.linear) == 0
        and (not quadratic or len(rhs.quadratic) == 0)
        and rhs.constant.__class__ in native_numeric_types
        and rhs.constant == 0
    ):
        return Results()
    #
    # If either the LHS or RHS are nonlinear, then simply return the nonlinear expression
    #
    if not lhs_nonl_None or not rhs_nonl_None:
        return Results(nonl=multiplier * exp)

    # If the resulting expression has a polynomial degree greater than 2
    # (1 if quadratic is False), then simply return this as a general
    # nonlinear expression
    #
    if max(1 if lhs.linear else 0, 2 if quadratic and lhs.quadratic else 0) + max(
        1 if rhs.linear else 0, 2 if quadratic and rhs.quadratic else 0
    ) > (2 if quadratic else 1):
        return Results(nonl=multiplier * exp)

    ans = Results()
    ans.constant = multiplier * lhs.constant * rhs.constant
    if not (lhs.constant.__class__ in native_numeric_types and lhs.constant == 0):
        for key, coef in rhs.linear.items():
            ans.linear[key] = multiplier * coef * lhs.constant
    if not (rhs.constant.__class__ in native_numeric_types and rhs.constant == 0):
        for key, coef in lhs.linear.items():
            if key in ans.linear:
                ans.linear[key] += multiplier * coef * rhs.constant
            else:
                ans.linear[key] = multiplier * coef * rhs.constant

    if quadratic:
        if not (lhs.constant.__class__ in native_numeric_types and lhs.constant == 0):
            for key, coef in rhs.quadratic.items():
                ans.quadratic[key] = multiplier * coef * lhs.constant
        if not (rhs.constant.__class__ in native_numeric_types and rhs.constant == 0):
            for key, coef in lhs.quadratic.items():
                if key in ans.quadratic:
                    ans.quadratic[key] += multiplier * coef * rhs.constant
                else:
                    ans.quadratic[key] = multiplier * coef * rhs.constant
        for lkey, lcoef in lhs.linear.items():
            for rkey, rcoef in rhs.linear.items():
                ndx = (lkey, rkey) if lkey <= rkey else (rkey, lkey)
                if ndx in ans.quadratic:
                    ans.quadratic[ndx] += multiplier * lcoef * rcoef
                else:
                    ans.quadratic[ndx] = multiplier * lcoef * rcoef
        # TODO - Use quicksum here?
        el_linear = multiplier * sum(
            coef * idMap[key]
            for key, coef in lhs.linear.items()
            if coef.__class__ not in native_numeric_types or coef
        )
        er_linear = multiplier * sum(
            coef * idMap[key]
            for key, coef in rhs.linear.items()
            if coef.__class__ not in native_numeric_types or coef
        )
        el_quadratic = multiplier * sum(
            coef * idMap[key[0]] * idMap[key[1]]
            for key, coef in lhs.quadratic.items()
            if coef.__class__ not in native_numeric_types or coef
        )
        er_quadratic = multiplier * sum(
            coef * idMap[key[0]] * idMap[key[1]]
            for key, coef in rhs.quadratic.items()
            if coef.__class__ not in native_numeric_types or coef
        )
        if (el_linear.__class__ not in native_numeric_types or el_linear) and (
            er_quadratic.__class__ not in native_numeric_types or er_quadratic
        ):
            ans.nonl += el_linear * er_quadratic
        if (er_linear.__class__ not in native_numeric_types or er_linear) and (
            el_quadratic.__class__ not in native_numeric_types or el_quadratic
        ):
            ans.nonl += er_linear * el_quadratic

    return ans


# @profile
def _collect_var(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()

    if exp.fixed:
        if compute_values:
            ans.constant += multiplier * value(exp)
        else:
            ans.constant += multiplier * exp
    else:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        ans.linear[key] = multiplier

    return ans


def _collect_pow(exp, multiplier, idMap, compute_values, verbose, quadratic):
    #
    # Exponent is a numeric value
    #
    if exp._args_[1].__class__ in native_numeric_types:
        exponent = exp._args_[1]
    #
    # Exponent is not potentially variable
    #
    # Compute its value if compute_values==True
    #
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            exponent = value(exp._args_[1])
        else:
            exponent = exp._args_[1]
    #
    # Otherwise collect a standard repn
    #
    else:
        res = _collect_standard_repn(
            exp._args_[1], 1, idMap, compute_values, verbose, quadratic
        )
        #
        # If the expression is variable, then return a nonlinear expression
        #
        if (
            not (res.nonl.__class__ in native_numeric_types and res.nonl == 0)
            or len(res.linear) > 0
            or (quadratic and len(res.quadratic) > 0)
        ):
            return Results(nonl=multiplier * exp)
        exponent = res.constant

    if exponent.__class__ in native_numeric_types:
        #
        # #**0 = 1
        #
        if exponent == 0:
            return Results(constant=multiplier)
        #
        # #**1 = #
        #
        # Return the standard repn for arg(0)
        #
        elif exponent == 1:
            return _collect_standard_repn(
                exp._args_[0], multiplier, idMap, compute_values, verbose, quadratic
            )
        #
        # Ignore #**2 unless quadratic==True
        #
        elif exponent == 2 and quadratic:
            res = _collect_standard_repn(
                exp._args_[0], 1, idMap, compute_values, verbose, quadratic
            )
            #
            # If arg(0) is nonlinear, then this is a nonlinear repn
            #
            if (
                not (res.nonl.__class__ in native_numeric_types and res.nonl == 0)
                or len(res.quadratic) > 0
            ):
                return Results(nonl=multiplier * exp)
            #
            # If computing values and no linear terms, then the return a constant repn
            #
            elif compute_values and len(res.linear) == 0:
                return Results(constant=multiplier * res.constant**exponent)
            #
            # If the base is linear, then we compute the quadratic expression for it.
            #
            else:
                ans = Results()
                has_constant = (
                    res.constant.__class__ not in native_numeric_types
                    or res.constant != 0
                )
                if has_constant:
                    ans.constant = multiplier * res.constant * res.constant

                # this is reversed since we want to pop off the end for efficiency
                # and the quadratic terms have a convention that the indexing tuple
                # of key1, key2 is such that key1 <= key2
                keys = sorted(res.linear.keys(), reverse=True)
                while len(keys) > 0:
                    key1 = keys.pop()
                    coef1 = res.linear[key1]
                    if has_constant:
                        ans.linear[key1] = 2 * multiplier * coef1 * res.constant
                    ans.quadratic[key1, key1] = multiplier * coef1 * coef1
                    for key2 in keys:
                        coef2 = res.linear[key2]
                        ans.quadratic[key1, key2] = 2 * multiplier * coef1 * coef2
                return ans

    #
    # If args(0) is a numeric value or it is fixed, then we have a constant value
    #
    if exp._args_[0].__class__ in native_numeric_types or exp._args_[0].is_fixed():
        if compute_values:
            return Results(constant=multiplier * value(exp._args_[0]) ** exponent)
        else:
            return Results(constant=multiplier * exp)
    #
    # Return a nonlinear expression here
    #
    return Results(nonl=multiplier * exp)


def _collect_division(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if (
        exp._args_[1].__class__ in native_numeric_types
        or not exp._args_[1].is_potentially_variable()
    ):  # TODO: coverage?
        # Denominator is trivially constant
        if compute_values:
            denom = 1.0 * value(exp._args_[1])
        else:
            denom = 1.0 * exp._args_[1]
    else:
        res = _collect_standard_repn(
            exp._args_[1], 1, idMap, compute_values, verbose, quadratic
        )
        if (
            not (res.nonl.__class__ in native_numeric_types and res.nonl == 0)
            or len(res.linear) > 0
            or (quadratic and len(res.quadratic) > 0)
        ):
            # Denominator is variable, give up: this is nonlinear
            return Results(nonl=multiplier * exp)
        else:
            # Denominaor ended up evaluating to a constant
            denom = 1.0 * res.constant
    if denom.__class__ in native_numeric_types and denom == 0:
        raise ZeroDivisionError

    if (
        exp._args_[0].__class__ in native_numeric_types
        or not exp._args_[0].is_potentially_variable()
    ):
        num = exp._args_[0]
        if compute_values:
            num = value(num)
        return Results(constant=multiplier * num / denom)

    return _collect_standard_repn(
        exp._args_[0], multiplier / denom, idMap, compute_values, verbose, quadratic
    )


def _collect_branching_expr(exp, multiplier, idMap, compute_values, verbose, quadratic):
    _if, _then, _else = exp.args
    if _if.__class__ in native_types:
        if_val = _if
    elif not _if.is_potentially_variable():
        if compute_values:
            if_val = value(_if)
        else:
            return Results(nonl=multiplier * exp)
    else:
        res = _collect_standard_repn(_if, 1, idMap, compute_values, verbose, quadratic)
        if (
            not (res.nonl.__class__ in native_numeric_types and res.nonl == 0)
            or len(res.linear) > 0
            or (quadratic and len(res.quadratic) > 0)
        ):
            return Results(nonl=multiplier * exp)
        elif res.constant.__class__ in native_numeric_types:
            if_val = res.constant
        else:
            return Results(constant=multiplier * exp)
    if if_val:
        if _then.__class__ in native_numeric_types:
            return Results(constant=multiplier * _then)
        return _collect_standard_repn(
            _then, multiplier, idMap, compute_values, verbose, quadratic
        )
    else:
        if _else.__class__ in native_numeric_types:
            return Results(constant=multiplier * _else)
        return _collect_standard_repn(
            _else, multiplier, idMap, compute_values, verbose, quadratic
        )


def _collect_nonl(exp, multiplier, idMap, compute_values, verbose, quadratic):
    res = _collect_standard_repn(
        exp._args_[0], 1, idMap, compute_values, verbose, quadratic
    )
    if (
        not (res.nonl.__class__ in native_numeric_types and res.nonl == 0)
        or len(res.linear) > 0
        or (quadratic and len(res.quadratic) > 0)
    ):
        return Results(nonl=multiplier * exp)
    if compute_values:
        return Results(constant=multiplier * exp._apply_operation([res.constant]))
    else:
        return Results(constant=multiplier * exp)


def _collect_negation(exp, multiplier, idMap, compute_values, verbose, quadratic):
    return _collect_standard_repn(
        exp._args_[0], -1 * multiplier, idMap, compute_values, verbose, quadratic
    )


#
# TODO - Verify if code is used
#
def _collect_const(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if compute_values:
        return Results(constant=multiplier * value(exp))
    else:
        return Results(constant=multiplier * exp)


def _collect_identity(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[0].__class__ in native_numeric_types:
        return Results(constant=multiplier * exp._args_[0])
    if not exp._args_[0].is_potentially_variable():
        if compute_values:
            return Results(constant=multiplier * value(exp._args_[0]))
        else:
            return Results(constant=multiplier * exp._args_[0])
    return _collect_standard_repn(
        exp.expr, multiplier, idMap, compute_values, verbose, quadratic
    )


def _collect_linear(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    if compute_values:
        ans.constant = multiplier * value(exp.constant)
    else:
        ans.constant = multiplier * exp.constant

    linear = {}
    linear_vars = {}
    for c, v in zip(exp.linear_coefs, exp.linear_vars):
        if v.fixed:
            if compute_values:
                ans.constant += multiplier * value(c) * value(v)
            else:
                ans.constant += multiplier * c * v
        else:
            key = id(v)
            if compute_values:
                if key in linear:
                    linear[key] += multiplier * value(c)
                else:
                    linear[key] = multiplier * value(c)
                    linear_vars[key] = v
            else:
                if key in linear:
                    linear[key] += multiplier * c
                else:
                    linear[key] = multiplier * c
                    linear_vars[key] = v
    for id_, coef in linear.items():
        if coef.__class__ in native_numeric_types and not coef:
            continue
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = linear_vars[id_]
        ans.linear[key] = coef
    return ans


def _collect_comparison(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if multiplier != 1:
        # this *will* generate an exception with the new relational expressions
        exp = multiplier * exp
    return Results(nonl=exp)


def _collect_external_fn(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if compute_values and exp.is_fixed():
        return Results(constant=multiplier * value(exp))
    return Results(nonl=multiplier * exp)


_repn_collectors = {
    EXPR.SumExpression: _collect_sum,
    EXPR.ProductExpression: _collect_prod,
    EXPR.MonomialTermExpression: _collect_term,
    EXPR.PowExpression: _collect_pow,
    EXPR.DivisionExpression: _collect_division,
    EXPR.Expr_ifExpression: _collect_branching_expr,
    EXPR.UnaryFunctionExpression: _collect_nonl,
    EXPR.AbsExpression: _collect_nonl,
    EXPR.NegationExpression: _collect_negation,
    EXPR.LinearExpression: _collect_linear,
    EXPR.InequalityExpression: _collect_comparison,
    EXPR.RangedExpression: _collect_comparison,
    EXPR.EqualityExpression: _collect_comparison,
    EXPR.ExternalFunctionExpression: _collect_external_fn,
    # ConnectorData          : _collect_linear_connector,
    # ScalarConnector         : _collect_linear_connector,
    ParamData: _collect_const,
    ScalarParam: _collect_const,
    # param.Param             : _collect_linear_const,
    # parameter               : _collect_linear_const,
    NumericConstant: _collect_const,
    VarData: _collect_var,
    ScalarVar: _collect_var,
    Var: _collect_var,
    variable: _collect_var,
    IVariable: _collect_var,
    ExpressionData: _collect_identity,
    ScalarExpression: _collect_identity,
    expression: _collect_identity,
    noclone: _collect_identity,
    NamedExpressionData: _collect_identity,
    Expression: _collect_identity,
    ObjectiveData: _collect_identity,
    ScalarObjective: _collect_identity,
    objective: _collect_identity,
}


def _collect_standard_repn(exp, multiplier, idMap, compute_values, verbose, quadratic):
    fn = _repn_collectors.get(exp.__class__, None)
    if fn is not None:
        return fn(exp, multiplier, idMap, compute_values, verbose, quadratic)
    #
    # Catch any known numeric constants
    #
    if exp.__class__ in native_numeric_types or not exp.is_potentially_variable():
        return _collect_const(
            exp, multiplier, idMap, compute_values, verbose, quadratic
        )
    #
    # These are types that might be extended using duck typing.
    #
    try:
        if exp.is_variable_type():
            fn = _collect_var
        if exp.is_named_expression_type():
            fn = _collect_identity
    except AttributeError:  # TODO: coverage?
        pass
    if fn is not None:
        _repn_collectors[exp.__class__] = fn
        return fn(exp, multiplier, idMap, compute_values, verbose, quadratic)
    raise ValueError(
        "Unexpected expression (type %s)" % type(exp).__name__
    )  # TODO: coverage?


def _generate_standard_repn(
    expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None
):
    if expr.__class__ is EXPR.SumExpression:
        #
        # This is the common case, so start collecting the sum
        #
        ans = _collect_sum(expr, 1, idMap, compute_values, verbose, quadratic)
    else:
        #
        # Call generic recursive logic
        #
        ans = _collect_standard_repn(expr, 1, idMap, compute_values, verbose, quadratic)
    #
    # Create the final object here from 'ans'
    #
    repn.constant = ans.constant
    #
    # Create a list (tuple) of the variables and coefficients
    #
    v = []
    c = []
    for key, val in ans.linear.items():
        if val.__class__ in native_numeric_types:
            if not val:
                continue
        elif val.is_constant():  # TODO: coverage?
            if value(val) == 0:
                continue
        v.append(idMap[key])
        c.append(val)
    repn.linear_vars = tuple(v)
    repn.linear_coefs = tuple(c)

    if quadratic:
        repn.quadratic_vars = []
        repn.quadratic_coefs = []
        for key in ans.quadratic:
            val = ans.quadratic[key]
            if val.__class__ in native_numeric_types:
                if val == 0:  # TODO: coverage?
                    continue
            elif val.is_constant():  # TODO: coverage?
                if value(val) == 0:
                    continue
            repn.quadratic_vars.append((idMap[key[0]], idMap[key[1]]))
            repn.quadratic_coefs.append(val)
        repn.quadratic_vars = tuple(repn.quadratic_vars)
        repn.quadratic_coefs = tuple(repn.quadratic_coefs)
        v = []
        c = []
        for key in ans.quadratic:
            v.append((idMap[key[0]], idMap[key[1]]))
            c.append(ans.quadratic[key])
        repn.quadratic_vars = tuple(v)
        repn.quadratic_coefs = tuple(c)

    if ans.nonl is not None and not isclose_const(ans.nonl, 0):
        repn.nonlinear_expr = ans.nonl
        repn.nonlinear_vars = []
        for v_ in EXPR.identify_variables(repn.nonlinear_expr, include_fixed=False):
            repn.nonlinear_vars.append(v_)
            #
            # Update idMap in case we skipped nonlinear sub-expressions
            #
            # Q: Should we skip nonlinear sub-expressions?
            #
            id_ = id(v_)
            if id_ in idMap[None]:
                key = idMap[None][id_]
            else:
                key = len(idMap) - 1
                idMap[None][id_] = key
                idMap[key] = v_
        repn.nonlinear_vars = tuple(repn.nonlinear_vars)

    return repn


"""
WEH - This code assumes the expression is linear and fills in a dictionary.
      This avoids creating temporary Results objects, but in practice that's
      not a big win.  Hence, this is deprecated.

##-----------------------------------------------------------------------
##
## Logic for _generate_linear_standard_repn
##
##-----------------------------------------------------------------------

def _linear_collect_sum(exp, multiplier, idMap, compute_values, verbose, coef):
    varkeys = idMap[None]

    for e_ in itertools.islice(exp._args_, exp.nargs()):
        if e_.__class__ in native_numeric_types:
            coef[None] += multiplier*e_

        elif e_.is_variable_type():
            if e_.fixed:
                if compute_values:
                    coef[None] += multiplier*e_.value
                else:
                    coef[None] += multiplier*e_
            else:
                id_ = id(e_)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = e_
                if key in coef:
                    coef[key] += multiplier
                else:
                    coef[key] = multiplier

        elif not e_.is_potentially_variable():
            if compute_values:
                coef[None] += multiplier * value(e_)
            else:
                coef[None] += multiplier * e_

        elif e_.__class__ is EXPR.NegationExpression:
            arg = e_._args_[0]
            if arg.is_variable_type():
                if arg.fixed:
                    if compute_values:
                        coef[None] -= multiplier*arg.value
                    else:
                        coef[None] -= multiplier*arg
                else:
                    id_ = id(arg)
                    if id_ in varkeys:
                        key = varkeys[id_]
                    else:
                        key = len(idMap) - 1
                        varkeys[id_] = key
                        idMap[key] = arg
                    if key in coef:
                        coef[key] -= multiplier
                    else:
                        coef[key] = -1 * multiplier
            else:
                _collect_linear_standard_repn(arg, -1*multiplier, idMap, compute_values, verbose, coef)

        elif e_.__class__ is EXPR.MonomialTermExpression:
            if compute_values:
                lhs = value(e_._args_[0])
            else:
                lhs = e_._args_[0]
            if e_._args_[1].fixed:
                if compute_values:
                    coef[None] += multiplier*lhs*value(e_._args_[1])
                else:
                    coef[None] += multiplier*lhs*e_._args_[1]
            else:
                id_ = id(e_._args_[1])
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = e_._args_[1]
                if key in coef:
                    coef[key] += multiplier*lhs
                else:
                    coef[key] = multiplier*lhs

        else:
            _collect_linear_standard_repn(e_, multiplier, idMap, compute_values, verbose, coef)

def _linear_collect_linear(exp, multiplier, idMap, compute_values, verbose, coef):
    varkeys = idMap[None]

    if compute_values:
        coef[None] += multiplier*value(exp.constant)
    else:
        coef[None] += multiplier*exp.constant

    for c,v in zip(exp.linear_coefs, exp.linear_vars):
        if v.fixed:
            if compute_values:
                coef[None] += multiplier*v.value
            else:
                coef[None] += multiplier*v
        else:
            id_ = id(v)
            if id_ in varkeys:
                key = varkeys[id_]
            else:
                key = len(idMap) - 1
                varkeys[id_] = key
                idMap[key] = v
            if compute_values:
                if key in coef:
                    coef[key] += multiplier*value(c)
                else:
                    coef[key] = multiplier*value(c)
            else:
                if key in coef:
                    coef[key] += multiplier*c
                else:
                    coef[key] = multiplier*c

def _linear_collect_term(exp, multiplier, idMap, compute_values, verbose, coef):
    #
    # LHS is a numeric value
    #
    if exp._args_[0].__class__ in native_numeric_types:
        if isclose_default(exp._args_[0],0):
            return
        _collect_linear_standard_repn(exp._args_[1], multiplier * exp._args_[0], idMap,
                                  compute_values, verbose, coef)
    #
    # LHS is a non-variable expression
    #
    else:
        if compute_values:
            val = value(exp._args_[0])
            if isclose_default(val,0):
                return
            _collect_linear_standard_repn(exp._args_[1], multiplier * val, idMap,
                                  compute_values, verbose, coef)
        else:
            _collect_linear_standard_repn(exp._args_[1], multiplier*exp._args_[0], idMap,
                                  compute_values, verbose, coef)

def _linear_collect_prod(exp, multiplier, idMap, compute_values, verbose, coef):
    #
    # LHS is a numeric value
    #
    if exp._args_[0].__class__ in native_numeric_types:
        if isclose_default(exp._args_[0],0):
            return
        _collect_linear_standard_repn(exp._args_[1], multiplier * exp._args_[0], idMap,
                                  compute_values, verbose, coef)
    #
    # LHS is a non-variable expression
    #
    elif not exp._args_[0].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[0])
            if isclose_default(val,0):
                return
            _collect_linear_standard_repn(exp._args_[1], multiplier * val, idMap,
                                  compute_values, verbose, coef)
        else:
            _collect_linear_standard_repn(exp._args_[1], multiplier*exp._args_[0], idMap,
                                  compute_values, verbose, coef)
    #
    # The LHS should never be variable
    #
    elif exp._args_[0].is_fixed():
        if compute_values:
            val = value(exp._args_[0])
            if isclose_default(val,0):
                return
            _collect_linear_standard_repn(exp._args_[1], multiplier * val, idMap,
                                  compute_values, verbose, coef)
        else:
            _collect_linear_standard_repn(exp._args_[1], multiplier*exp._args_[0], idMap,
                                  compute_values, verbose, coef)
    else:
        if compute_values:
            val = value(exp._args_[1])
            if isclose_default(val,0):
                return
            _collect_linear_standard_repn(exp._args_[0], multiplier * val, idMap,
                                  compute_values, verbose, coef)
        else:
            _collect_linear_standard_repn(exp._args_[0], multiplier*exp._args_[1], idMap,
                                  compute_values, verbose, coef)

def _linear_collect_var(exp, multiplier, idMap, compute_values, verbose, coef):
    if exp.fixed:
        if compute_values:
            coef[None] += multiplier*value(exp)
        else:
            coef[None] += multiplier*exp
    else:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        if key in coef:
            coef[key] += multiplier
        else:
            coef[key] = multiplier

def _linear_collect_negation(exp, multiplier, idMap, compute_values, verbose, coef):
    _collect_linear_standard_repn(exp._args_[0], -1*multiplier, idMap, compute_values, verbose, coef)

def _linear_collect_identity(exp, multiplier, idMap, compute_values, verbose, coef):
    arg = exp._args_[0]
    if arg.__class__ in native_numeric_types:
        coef[None] += arg
    elif not arg.is_potentially_variable():
        if compute_values:
            coef[None] += value(arg)
        else:
            coef[None] += arg
    else:
        _collect_linear_standard_repn(exp.expr, multiplier, idMap, compute_values, verbose, coef)

def _linear_collect_branching_expr(exp, multiplier, idMap, compute_values, verbose, coef):
    if exp._if.__class__ in native_numeric_types:
        if_val = exp._if
    else:
        # If this value is not constant, then the expression is nonlinear.
        if_val = value(exp._if)
    if if_val:
        _collect_linear_standard_repn(exp._then, multiplier, idMap, compute_values, verbose, coef)
    else:
        _collect_linear_standard_repn(exp._else, multiplier, idMap, compute_values, verbose, coef)

def _linear_collect_pow(exp, multiplier, idMap, compute_values, verbose, coef):
    if exp._args_[1].__class__ in native_numeric_types:
        exponent = exp._args_[1]
    else:
        # If this value is not constant, then the expression is nonlinear.
        exponent = value(exp._args_[1])

    if exponent == 0:
        coef[None] += multiplier
    else: #exponent == 1
        _collect_linear_standard_repn(exp._args_[0], multiplier, idMap, compute_values, verbose, coef)


_linear_repn_collectors = {
    EXPR.SumExpression                          : _linear_collect_sum,
    EXPR.ProductExpression                      : _linear_collect_prod,
    EXPR.MonomialTermExpression                 : _linear_collect_term,
    EXPR.PowExpression                          : _linear_collect_pow,
    #EXPR.DivisionExpression                     : _linear_collect_division,
    EXPR.Expr_ifExpression                      : _linear_collect_branching_expr,
    #EXPR.UnaryFunctionExpression                : _linear_collect_nonl,
    #EXPR.AbsExpression                          : _linear_collect_nonl,
    EXPR.NegationExpression                     : _linear_collect_negation,
    EXPR.LinearExpression                       : _linear_collect_linear,
    #EXPR.InequalityExpression                   : _linear_collect_comparison,
    #EXPR.RangedExpression                       : _linear_collect_comparison,
    #EXPR.EqualityExpression                     : _linear_collect_comparison,
    #EXPR.ExternalFunctionExpression             : _linear_collect_external_fn,
    ##EXPR.LinearSumExpression               : _collect_linear_sum,
    ##ConnectorData          : _collect_linear_connector,
    ##ScalarConnector         : _collect_linear_connector,
    ##param.ParamData        : _collect_linear_const,
    ##param.ScalarParam       : _collect_linear_const,
    ##param.Param             : _collect_linear_const,
    ##parameter               : _collect_linear_const,
    VarData                             : _linear_collect_var,
    ScalarVar                                   : _linear_collect_var,
    Var                                         : _linear_collect_var,
    variable                                    : _linear_collect_var,
    IVariable                                   : _linear_collect_var,
    ExpressionData                      : _linear_collect_identity,
    ScalarExpression                            : _linear_collect_identity,
    expression                                  : _linear_collect_identity,
    noclone                                     : _linear_collect_identity,
    NamedExpressionData                             : _linear_collect_identity,
    Expression                                  : _linear_collect_identity,
    ObjectiveData                       : _linear_collect_identity,
    ScalarObjective                             : _linear_collect_identity,
    objective                                   : _linear_collect_identity,
    }


def _collect_linear_standard_repn(exp, multiplier, idMap, compute_values, verbose, coefs):
    fn = _linear_repn_collectors.get(exp.__class__, None)
    if fn is not None:
        return fn(exp, multiplier, idMap, compute_values, verbose, coefs)
    #
    # These are types that might be extended using duck typing.
    #
    try:
        if exp.is_variable_type():
            fn = _linear_collect_var
        if exp.is_named_expression_type():
            fn = _linear_collect_identity
    except AttributeError:
        pass
    if fn is not None:
        return fn(exp, multiplier, idMap, compute_values, verbose, coefs)
    raise ValueError( "Unexpected expression (type %s)" % type(exp).__name__)

def _generate_linear_standard_repn(expr, idMap=None, compute_values=True, verbose=False, repn=None):
    coef = {None:0}
    #
    # Call recursive logic
    #
    ans = _collect_linear_standard_repn(expr, 1, idMap, compute_values, verbose, coef)
    #
    # Create the final object here from 'ans'
    #
    repn.constant = coef[None]
    del coef[None]
    #
    # Create a list (tuple) of the variables and coefficients
    #
    # If we compute the values of constants, then we can skip terms with zero
    # coefficients
    #
    if compute_values:
        keys = list(key for key in coef if not isclose(coef[key],0))
    else:
        keys = list(coef.keys())
    repn.linear_vars = tuple(idMap[key] for key in keys)
    repn.linear_coefs = tuple(coef[key] for key in keys)

    return repn
"""


##-----------------------------------------------------------------------
##
## Functions to preprocess blocks
##
##-----------------------------------------------------------------------


def preprocess_block_objectives(block, idMap=None):
    # Get/Create the ComponentMap for the repn
    if not hasattr(block, '_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for objective_data in block.component_data_objects(
        Objective, active=True, descend_into=False
    ):
        if objective_data.expr is None:
            raise ValueError(
                "No expression has been defined for objective %s"
                % (objective_data.name)
            )

        try:
            repn = generate_standard_repn(objective_data.expr, idMap=idMap)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error(
                "exception generating a standard representation for objective %s: %s"
                % (objective_data.name, str(err))
            )
            raise

        block_repn[objective_data] = repn


def preprocess_block_constraints(block, idMap=None):
    # Get/Create the ComponentMap for the repn
    if not hasattr(block, '_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for constraint in block.component_objects(
        Constraint, active=True, descend_into=False
    ):
        preprocess_constraint(block, constraint, idMap=idMap, block_repn=block_repn)


def preprocess_constraint(block, constraint, idMap=None, block_repn=None):
    from pyomo.repn.beta.matrix import MatrixConstraint

    if isinstance(constraint, MatrixConstraint):
        return

    # Get/Create the ComponentMap for the repn
    if not hasattr(block, '_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for index, constraint_data in constraint.items():
        if not constraint_data.active:
            continue

        if constraint_data.body is None:
            raise ValueError(
                "No expression has been defined for the body "
                "of constraint %s" % (constraint_data.name)
            )

        try:
            repn = generate_standard_repn(constraint_data.body, idMap=idMap)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error(
                "exception generating a standard representation for "
                "constraint %s: %s" % (constraint_data.name, str(err))
            )
            raise

        block_repn[constraint_data] = repn


def preprocess_constraint_data(block, constraint_data, idMap=None, block_repn=None):
    # Get/Create the ComponentMap for the repn
    if not hasattr(block, '_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    if constraint_data.body is None:
        raise ValueError(
            "No expression has been defined for the body "
            "of constraint %s" % (constraint_data.name)
        )

    try:
        repn = generate_standard_repn(constraint_data.body, idMap=idMap)
    except Exception:
        err = sys.exc_info()[1]
        logging.getLogger('pyomo.core').error(
            "exception generating a standard representation for "
            "constraint %s: %s" % (constraint_data.name, str(err))
        )
        raise

    block_repn[constraint_data] = repn
