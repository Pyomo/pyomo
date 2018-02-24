#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

__all__ = ['StandardRepn', 'generate_standard_repn']


import sys
import logging
import math
import itertools

from pyomo.core.base import (Constraint,
                             Objective,
                             ComponentMap)

import pyomo.util
from pyutilib.misc import Bunch
from pyutilib.math.util import isclose as isclose_default

from pyomo.core.expr import current as EXPR
from pyomo.core.base.objective import (_GeneralObjectiveData,
                                       SimpleObjective)
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import SimpleExpression, _GeneralExpressionData
from pyomo.core.base.var import (SimpleVar,
                                 Var,
                                 _GeneralVarData,
                                 _VarData,
                                 value)
from pyomo.core.base.param import _ParamData
from pyomo.core.base.numvalue import (NumericConstant,
                                      native_numeric_types,
                                      is_fixed)
from pyomo.core.kernel.component_expression import IIdentityExpression, expression, noclone
from pyomo.core.kernel.component_variable import IVariable
from pyomo.core.kernel.component_objective import objective

import six
from six import iteritems
from six import itervalues, iteritems, StringIO
from six.moves import xrange, zip
try:
    basestring
except:
    basestring = str

logger = logging.getLogger('pyomo.core')

using_py3 = six.PY3

from pyomo.core.base import _VarData, _GeneralVarData, SimpleVar
from pyomo.core.kernel.component_variable import IVariable, variable


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
    # Copied from pyutilib.math after here
    diff = math.fabs(a-b)
    if diff <= rel_tol*max(math.fabs(a),math.fabs(b)):
        return True
    if diff <= abs_tol:
        return True
    return False

#
# The global isclose() function used below.  This is either isclose_default
# (defined in pyutilib) or isclose_const
#
isclose = isclose_default

#
# A context manager that makes sure we set/reset the
# isclose() function.
#
class isclose_context(object):

    def __init__(self, compute_values):
        self.compute_values = compute_values

    def __enter__(self):
        if not self.compute_values:
            global isclose
            isclose = isclose_const

    def __exit__(self, *args):
        global isclose
        isclose = isclose_default


class StandardRepn(object):
    """
    This class defines a standard/common representation for Pyomo expressions
    that provides an efficient interface for writing all models.

    TODO: define what "efficient" means to us.
    """

    __slots__ = ('constant',          # The constant term
                 'linear_coefs',      # Linear coefficients
                 'linear_vars',       # Linear variables
                 'quadratic_coefs',   # Quadratic coefficients
                 'quadratic_vars',    # Quadratic variables
                 'nonlinear_expr',    # Nonlinear expression
                 'nonlinear_vars')    # Variables that appear in the nonlinear expression

    def __init__(self, expr=None):
        self.constant = 0
        self.linear_vars = tuple()
        self.linear_coefs = tuple()
        self.quadratic_vars = tuple()
        self.quadratic_coefs = tuple()
        self.nonlinear_expr = None
        self.nonlinear_vars = tuple()
        if not expr is None:
            generate_standard_repn(expr, repn=self)

    def __getstate__(self):
        """
        This method is required because this class uses slots.
        """
        return  (self.constant,
                 self.linear_coefs,
                 self.linear_vars,
                 self.quadratic_coefs,
                 self.quadratic_vars,
                 self.nonlinear_expr,
                 self.nonlinear_vars)

    def __setstate__(self, state):
        """
        This method is required because this class uses slots.
        """
        self.constant, \
        self.linear_coefs, \
        self.linear_vars, \
        self.quadratic_coefs, \
        self.quadratic_vars, \
        self.nonlinear_expr, \
        self.nonlinear_vars = state

    #
    # Generate a string representation of the expression
    #
    def __str__(self):
        output = StringIO()
        output.write("\n")
        output.write("constant:       "+str(self.constant)+"\n")
        output.write("linear vars:    "+str([v_.name for v_ in self.linear_vars])+"\n")
        output.write("linear var ids: "+str([id(v_) for v_ in self.linear_vars])+"\n")
        output.write("linear coef:    "+str(list(self.linear_coefs))+"\n")
        output.write("quadratic vars:    "+str([(v_[0].name,v_[1].name) for v_ in self.quadratic_vars])+"\n")
        output.write("quadratic var ids: "+str([(id(v_[0]), id(v_[1])) for v_ in self.quadratic_vars])+"\n")
        output.write("quadratic coef:    "+str(list(self.quadratic_coefs))+"\n")
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
        output.write("nonlinear vars: "+str([v_.name for v_ in self.nonlinear_vars])+"\n")
        output.write("\n")

        ret_str = output.getvalue()
        output.close()
        return ret_str

    def is_fixed(self):
        if len(self.linear_vars) == 0 and len(self.nonlinear_vars) == 0 and len(self.quadratic_vars) == 0:
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
        return self.nonlinear_expr is None and len(self.quadratic_coefs) == 0 and len(self.linear_coefs) == 0

    def is_linear(self):
        return self.nonlinear_expr is None and len(self.quadratic_coefs) == 0

    def is_quadratic(self):
        return len(self.quadratic_coefs) > 0 and self.nonlinear_expr is None

    def is_nonlinear(self):
        return not (self.nonlinear_expr is None and len(self.quadratic_coefs) == 0)

    def to_expression(self):
        #
        # TODO: Should this replace non-mutable parameters with constants?
        #
        expr = self.constant
        for i,v in enumerate(self.linear_vars):
            if self.linear_coefs[i].__class__ in native_numeric_types:
                val = self.linear_coefs[i]
                if isclose_const(val, 1.0):
                    expr += self.linear_vars[i]
                elif isclose_const(val, -1.0):
                    expr -= self.linear_vars[i]
                elif val < 0.0:
                    expr -= - self.linear_coefs[i]*self.linear_vars[i]
                else:
                    expr += self.linear_coefs[i]*self.linear_vars[i]
            else:
                expr += self.linear_coefs[i]*self.linear_vars[i]
        for i,v in enumerate(self.quadratic_vars):
            if id(self.quadratic_vars[i][0]) == id(self.quadratic_vars[i][1]):
                term = self.quadratic_vars[i][0]**2
            else:
                term = self.quadratic_vars[i][0]*self.quadratic_vars[i][1]
            if self.quadratic_coefs[i].__class__ in native_numeric_types:
                val = self.quadratic_coefs[i]
                if isclose_const(val, 1.0):
                    expr += term
                elif isclose_const(val, -1.0):
                    expr -= term
                else:
                    expr += self.quadratic_coefs[i]*term
            else:
                expr += self.quadratic_coefs[i]*term
        if not self.nonlinear_expr is None:
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
#@profile
def generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None):
    #
    # Use a custom isclose function
    #
    with isclose_context(compute_values):
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
            if compute_values:
                C_ = EXPR.evaluate_expression(expr.constant)
            else:
                C_ = expr.constant
            if compute_values:
                v_ = []
                c_ = []
                for c,v in zip(expr.linear_coefs, expr.linear_vars):
                    if v.fixed:
                        if c.__class__ in native_numeric_types:
                            C_ += c*v.value
                        elif c.is_expression_type():
                            C_ += EXPR.evaluate_expression(c)*v.value
                        else:
                            C_ += value(c)*v.value
                    else:
                        if c.__class__ in native_numeric_types:
                            c_.append( c )
                        elif c.is_expression_type():
                            c_.append( EXPR.evaluate_expression(c) )
                        else:
                            c_.append( value(c) )
                        v_.append( v )
                repn.linear_coefs = tuple(c_)
                repn.linear_vars = tuple(v_)
            else:
                linear_coefs = {}
                for c,v in zip(expr.linear_coefs, expr.linear_vars):
                    if v.fixed:
                        C_ += c*v
                    else:
                        id_ = id(v)
                        if not id_ in idMap[None]:
                            key = len(idMap) - 1
                            idMap[None][id_] = key
                            idMap[key] = v
                        else:
                            key = idMap[None][id_]
                        if key in linear_coefs:
                            linear_coefs[key] += c
                        else:
                            linear_coefs[key] = c
                keys = list(linear_coefs.keys())
                repn.linear_vars = tuple(idMap[key] for key in keys)
                repn.linear_coefs = tuple(linear_coefs[key] for key in keys)
            repn.constant = C_
            return repn

        #
        # Unknown expression object
        #
        elif not expr.is_expression_type():
            raise ValueError("Unexpected expression type: "+str(expr))

        return _generate_standard_repn(expr, 
                                idMap=idMap,
                                compute_values=compute_values,
                                verbose=verbose,
                                quadratic=quadratic,
                                repn=repn)

class Results(object):
    __slot__ = ('const', 'nonl', 'linear', 'quadratic')

    def __init__(self, constant=0, nonl=0, linear=None, quadratic=None):
        self.constant = constant
        self.nonl = nonl
        if linear is None:
            self.linear = {}
        else:
            self.linear = linear
        if quadratic is None:
            self.quadratic = {}
        else:
            self.quadratic = quadratic

    def __str__(self):
        return "Const:\t%f\nLinear:\t%s\nQuadratic:\t%s\nNonlinear:\t%s" % (self.constant, str(self.linear), str(self.quadratic), str(self.nonl))


#@profile
def _collect_sum(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    varkeys = idMap[None]

    for e_ in itertools.islice(exp._args_, exp.nargs()):
        if e_.__class__ in native_numeric_types:
            ans.constant += multiplier*e_
        elif e_.is_variable_type():
            if e_.fixed:
                if compute_values:
                    ans.constant += multiplier*e_.value
                else:
                    ans.constant += multiplier*e_
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
        elif e_.__class__ is EXPR.ProductExpression and e_._args_[1].is_variable_type() and (e_._args_[0].__class__ in native_numeric_types or not e_._args_[0].is_potentially_variable()):
            if compute_values:
                lhs = value(e_._args_[0])
            else:
                lhs = e_._args_[0]
            if e_._args_[1].fixed:
                if compute_values:
                    ans.constant += multiplier*lhs*value(e_._args_[1])
                else:
                    ans.constant += multiplier*lhs*e_._args_[1]
            else:
                id_ = id(e_._args_[1])
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = e_._args_[1]
                if key in ans.linear:
                    ans.linear[key] += multiplier*lhs
                else:
                    ans.linear[key] = multiplier*lhs
        else:
            res_ = _collect_standard_repn(e_, multiplier, idMap, 
                                      compute_values, verbose, quadratic)
            #
            # Add returned from recursion
            #
            ans.constant += res_.constant
            ans.nonl += res_.nonl
            for i in res_.linear:
                ans.linear[i] = ans.linear.get(i,0) + res_.linear[i]
            if quadratic:
                for i in res_.quadratic:
                    ans.quadratic[i] = ans.quadratic.get(i, 0) + res_.quadratic[i]

    return ans

#@profile
def _collect_prod(exp, multiplier, idMap, compute_values, verbose, quadratic):
    #
    # LHS is a numeric value
    #
    if exp._args_[0].__class__ in native_numeric_types:
        if isclose_default(exp._args_[0],0):
            return Results()
        return _collect_standard_repn(exp._args_[1], multiplier * exp._args_[0], idMap, 
                                  compute_values, verbose, quadratic)
    #
    # LHS is a non-variable expression
    #
    elif not exp._args_[0].is_potentially_variable():
        if compute_values:
            val = value(exp._args_[0])
            if isclose_default(val,0):
                return Results()
            return _collect_standard_repn(exp._args_[1], multiplier * val, idMap, 
                                  compute_values, verbose, quadratic)
        else:
            return _collect_standard_repn(exp._args_[1], multiplier*exp._args_[0], idMap, 
                                  compute_values, verbose, quadratic)

    lhs = _collect_standard_repn(exp._args_[0], 1, idMap, 
                                  compute_values, verbose, quadratic)
    lhs_nonl_None = isclose_const(lhs.nonl,0)

    if lhs_nonl_None and len(lhs.linear) == 0 and len(lhs.quadratic) == 0:
        if isclose(lhs.constant,0):
            return Results()
        if compute_values:
            val = value(lhs.constant)
            if isclose_default(val,0):
                return Results()
            return _collect_standard_repn(exp._args_[1], multiplier*val, idMap, 
                                  compute_values, verbose, quadratic)
        else:
            return _collect_standard_repn(exp._args_[1], multiplier*lhs.constant, idMap, 
                                  compute_values, verbose, quadratic)

    if exp._args_[1].__class__ in native_numeric_types:
        rhs = Results(constant=exp._args_[1])
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            rhs = Results(constant=value(exp._args_[1]))
        else:
            rhs = Results(constant=exp._args_[1])
    else:
        rhs = _collect_standard_repn(exp._args_[1], 1, idMap, 
                                  compute_values, verbose, quadratic)
    rhs_nonl_None = isclose_const(rhs.nonl,0)
    if rhs_nonl_None and len(rhs.linear) == 0 and len(rhs.quadratic) == 0 and isclose(rhs.constant,0):
        return Results()

    if not lhs_nonl_None or not rhs_nonl_None:
        return Results(nonl=multiplier*exp)
    if not quadratic and len(lhs.linear) > 0 and len(rhs.linear) > 0:
        # NOTE: We treat a product of linear terms as nonlinear unless quadratic==2
        return Results(nonl=multiplier*exp)

    ans = Results()
    ans.constant = multiplier*lhs.constant * rhs.constant
    if not isclose(lhs.constant,0):
        for key, coef in six.iteritems(rhs.linear):
            ans.linear[key] = multiplier*coef*lhs.constant
    if not isclose(rhs.constant,0):
        for key, coef in six.iteritems(lhs.linear):
            if key in ans.linear:
                ans.linear[key] += multiplier*coef*rhs.constant
            else:
                ans.linear[key] = multiplier*coef*rhs.constant

    if quadratic:
        if not isclose(lhs.constant,0):
            for key, coef in six.iteritems(rhs.quadratic):
                ans.quadratic[key] = multiplier*coef*lhs.constant
        if not isclose(rhs.constant,0):
            for key, coef in six.iteritems(lhs.quadratic):
                if key in ans.quadratic:
                    ans.quadratic[key] += multiplier*coef*rhs.constant
                else:
                    ans.quadratic[key] = multiplier*coef*rhs.constant
        for lkey, lcoef in six.iteritems(lhs.linear):
            for rkey, rcoef in six.iteritems(rhs.linear):
                if lkey <= rkey:
                    ans.quadratic[lkey,rkey] = multiplier*lcoef*rcoef
                else:
                    ans.quadratic[rkey,lkey] = multiplier*lcoef*rcoef
        el_linear = multiplier*sum(coef*idMap[key] for key, coef in six.iteritems(lhs.linear))
        er_linear = multiplier*sum(coef*idMap[key] for key, coef in six.iteritems(rhs.linear))
        el_quadratic = multiplier*sum(coef*idMap[key[0]]*idMap[key[1]] for key, coef in six.iteritems(lhs.quadratic))
        er_quadratic = multiplier*sum(coef*idMap[key[0]]*idMap[key[1]] for key, coef in six.iteritems(rhs.quadratic))
        ans.nonl += el_linear*er_quadratic + el_quadratic*er_linear
    elif len(lhs.linear) + len(rhs.linear) > 1:
        el_linear = multiplier*sum(coef*idMap[key] for key, coef in six.iteritems(lhs.linear))
        er_linear = multiplier*sum(coef*idMap[key] for key, coef in six.iteritems(rhs.linear))
        ans.nonl += el_linear*er_linear

    return ans

#@profile
def _collect_var(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()

    if exp.fixed:
        if compute_values:
            ans.constant += multiplier*value(exp)
        else:
            ans.constant += multiplier*exp
    else:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        if key in ans.linear:
            ans.linear[key] += multiplier
        else:
            ans.linear[key] = multiplier

    return ans

def _collect_pow(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[1].__class__ in native_numeric_types:
            exponent = exp._args_[1]
    elif not exp._args_[1].is_potentially_variable():
        if compute_values:
            exponent = value(exp._args_[1])
        else:
            exponent = exp._args_[1]
    else:
        res = _collect_standard_repn(exp._args_[1], 1, idMap, compute_values, verbose, quadratic)
        if not isclose_const(res.nonl,0) or len(res.linear) > 0 or len(res.quadratic) > 0:
            # The exponent is variable, so this is a nonlinear expression
            return Results(nonl=multiplier*exp)
        exponent = res.constant

    if exponent == 0:
        return Results(constant=multiplier)
    elif exponent == 1:
        return _collect_standard_repn(exp._args_[0], multiplier, idMap, compute_values, verbose, quadratic)
    # If the exponent is >= 2, then this is a nonlinear expression
    if exponent == 2 and quadratic:
        # NOTE: We treat a product of linear terms as nonlinear unless quadratic==2
        res =_collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
        if not isclose_const(res.nonl,0) or len(res.quadratic) > 0:
            return Results(nonl=multiplier*exp)
        ans = Results()
        if not isclose(res.constant,0):
            ans.constant = multiplier*res.constant*res.constant
            for key, coef in six.iteritems(res.linear):
                ans.linear[key] = 2*multiplier*coef*res.constant
        for key, coef in six.iteritems(res.linear):
            ans.quadratic[key,key] = multiplier*coef
        return ans
        
    return Results(nonl=multiplier*exp)

def _collect_reciprocal(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[0].__class__ in native_numeric_types or not exp._args_[0].is_potentially_variable():
        if compute_values:
            denom = 1.0 * value(exp._args_[0])
        else:
            denom = 1.0 * exp._args_[0]
    else:
        res =_collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
        if not isclose_const(res.nonl,0) or len(res.linear) > 0 or len(res.quadratic) > 0:
            return Results(nonl=multiplier*exp)
        else:
            denom = 1.0*res.constant
    if isclose_const(denom, 0):
        raise ZeroDivisionError()
    return Results(constant=multiplier/denom)
   
def _collect_branching_expr(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._if.__class__ in native_numeric_types:
        if_val = exp._if
    elif not exp._if.is_potentially_variable():
        if compute_values:
            if_val = value(exp._if)
        else:
            return Results(nonl=multiplier*exp)
    else:
        res = _collect_standard_repn(exp._if, 1, idMap, compute_values, verbose, quadratic)
        if not isclose_const(res.nonl,0) or len(res.linear) > 0 or len(res.quadratic) > 0:
            return Results(nonl=multiplier*exp)
        else:
            if_val = res.constant
    if if_val:
        return _collect_standard_repn(exp._then, multiplier, idMap, compute_values, verbose, quadratic)
    else:
        return _collect_standard_repn(exp._else, multiplier, idMap, compute_values, verbose, quadratic)

def _collect_nonl(exp, multiplier, idMap, compute_values, verbose, quadratic):
    res = _collect_standard_repn(exp._args_[0], 1, idMap, compute_values, verbose, quadratic)
    if not isclose_const(res.nonl,0) or len(res.linear) > 0 or len(res.quadratic) > 0:
        return Results(nonl=multiplier*exp)
    return Results(constant=multiplier*exp._apply_operation([res.constant]))

def _collect_negation(exp, multiplier, idMap, compute_values, verbose, quadratic):
    return _collect_standard_repn(exp._args_[0], -1*multiplier, idMap, compute_values, verbose, quadratic)

def _collect_identity(exp, multiplier, idMap, compute_values, verbose, quadratic):
    if exp._args_[0].__class__ in native_numeric_types:
        return Results(constant=exp._args_[0])
    if not exp._args_[0].is_potentially_variable():
        if compute_values:
            return Results(constant=value(exp._args_[0]))
        else:
            return Results(constant=exp._args_[0])
    return _collect_standard_repn(exp.expr, multiplier, idMap, compute_values, verbose, quadratic)

def _collect_linear(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    if compute_values:
        ans.constant = multiplier*value(exp.constant)
    else:
        ans.constant = multiplier*exp.constant

    for c,v in zip(exp.linear_coefs, exp.linear_vars):
        if v.fixed:
            if compute_values:
                ans.constant += multiplier*v.value
            else:
                ans.constant += multiplier*v
        else:
            id_ = id(v)
            if id_ in idMap[None]:
                key = idMap[None][id_]
            else:
                key = len(idMap) - 1
                idMap[None][id_] = key
                idMap[key] = v
            if compute_values:
                if key in ans.linear:
                    ans.linear[key] += multiplier*value(c)
                else:
                    ans.linear[key] = multiplier*value(c)
            else:
                if key in ans.linear:
                    ans.linear[key] += multiplier*c
                else:
                    ans.linear[key] = multiplier*c
    return ans

def _collect_comparison(exp, multiplier, idMap, compute_values, verbose, quadratic):
    return Results(nonl=multiplier*exp)
    
def _collect_external_fn(exp, multiplier, idMap, compute_values, verbose, quadratic):
    return Results(nonl=multiplier*exp)
    
def _collect_linear_sum(exp, multiplier, idMap, compute_values, verbose, quadratic):
    ans = Results()
    varkeys = idMap[None]

    for e_ in itertools.islice(exp._args_, exp.nargs()):
        c,v = e_
        if not v is None:
            if v.fixed:
                if compute_values:
                    ans.constant += multiplier*c*v.value
                else:
                    ans.constant += multiplier*c*v
            else:
                id_ = id(v)
                if id_ in varkeys:
                    key = varkeys[id_]
                else:
                    key = len(idMap) - 1
                    varkeys[id_] = key
                    idMap[key] = v
                if key in ans.linear:
                    ans.linear[key] += multiplier*c
                else:
                    ans.linear[key] = multiplier*c
        elif c.__class__ in native_numeric_types:
            ans.constant += multiplier*c
        else:       # not c.is_potentially_variable()
            if compute_values:
                ans.constant += multiplier * value(c)
            else:
                ans.constant += multiplier * c

    return ans


_repn_collectors = {
    EXPR.ViewSumExpression                      : _collect_sum,
    EXPR.ProductExpression                      : _collect_prod,
    EXPR.PowExpression                          : _collect_pow,
    EXPR.ReciprocalExpression                   : _collect_reciprocal,
    EXPR.Expr_if                                : _collect_branching_expr,
    EXPR.UnaryFunctionExpression                : _collect_nonl,
    EXPR.AbsExpression                          : _collect_nonl,
    EXPR.NegationExpression                     : _collect_negation,
    EXPR.LinearExpression                       : _collect_linear,
    EXPR.InequalityExpression                   : _collect_comparison,
    EXPR.RangedExpression                       : _collect_comparison,
    EXPR.EqualityExpression                     : _collect_comparison,
    EXPR.ExternalFunctionExpression             : _collect_external_fn,
    #EXPR.LinearViewSumExpression               : _collect_linear_sum,
    #_ConnectorData          : _collect_linear_connector,
    #SimpleConnector         : _collect_linear_connector,
    #param._ParamData        : _collect_linear_const,
    #param.SimpleParam       : _collect_linear_const,
    #param.Param             : _collect_linear_const,
    #parameter               : _collect_linear_const,
    _GeneralVarData                             : _collect_var,
    SimpleVar                                   : _collect_var,
    Var                                         : _collect_var,
    variable                                    : _collect_var,
    IVariable                                   : _collect_var,
    _GeneralExpressionData                      : _collect_identity,
    SimpleExpression                            : _collect_identity,
    expression                                  : _collect_identity,
    noclone                                     : _collect_identity,
    _ExpressionData                             : _collect_identity,
    Expression                                  : _collect_identity,
    _GeneralObjectiveData                       : _collect_identity,
    SimpleObjective                             : _collect_identity,
    objective                                   : _collect_identity,
    }


def _collect_standard_repn(exp, multiplier, idMap, 
                                      compute_values, verbose, quadratic):
    try:
        return _repn_collectors[exp.__class__](exp, multiplier, idMap, 
                                          compute_values, verbose, quadratic)
    except KeyError:
        #
        # These are types that might be extended using duck typing.
        #
        if exp.is_variable_type():
            return _collect_var(exp, multiplier, idMap, compute_values, verbose, quadratic)
        if exp.is_named_expression_type():
            return _collect_identity(exp, multiplier, idMap, compute_values, verbose, quadratic)
        raise ValueError( "Unexpected expression (type %s)" % type(exp).__name__)


def _generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None):
    #
    # Call recursive logic
    #
    ans = _collect_standard_repn(expr, 1, idMap, compute_values, verbose, quadratic)
    #
    # Create the final object here from 'ans'
    #
    repn.constant = ans.constant
    #
    # Create a list (tuple) of the variables and coefficients
    #
    # If we compute the values of constants, then we can skip terms with zero
    # coefficients
    #
    if compute_values:
        keys = list(key for key in ans.linear if not isclose(ans.linear[key],0))
    else:
        keys = list(ans.linear.keys())
    repn.linear_vars = tuple(idMap[key] for key in keys)
    repn.linear_coefs = tuple(ans.linear[key] for key in keys)

    if quadratic:
        keys = list(key for key in ans.quadratic if not isclose(ans.quadratic[key],0))
        repn.quadratic_vars = tuple((idMap[key[0]],idMap[key[1]]) for key in keys)
        repn.quadratic_coefs = tuple(ans.quadratic[key] for key in keys)

    if not isclose_const(ans.nonl,0):
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


##
##
## Define the compute_standard_repn function
##
##


def preprocess_block_objectives(block, idMap=None):

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for objective_data in block.component_data_objects(Objective,
                                                       active=True,
                                                       descend_into=False):

        if objective_data.expr is None:
            raise ValueError("No expression has been defined for objective %s"
                             % (objective_data.name))

        try:
            repn = generate_standard_repn(objective_data.expr, idMap=idMap)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error\
                ( "exception generating a standard representation for objective %s: %s" \
                      % (objective_data.name, str(err)) )
            raise

        block_repn[objective_data] = repn

def preprocess_block_constraints(block, idMap=None):

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for constraint in block.component_objects(Constraint,
                                              active=True,
                                              descend_into=False):

        preprocess_constraint(block,
                              constraint,
                              idMap=idMap,
                              block_repn=block_repn)

def preprocess_constraint(block,
                      constraint,
                      idMap=None,
                      block_repn=None):

    from pyomo.repn.beta.matrix import MatrixConstraint
    if isinstance(constraint, MatrixConstraint):
        return

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    for index, constraint_data in iteritems(constraint):

        if not constraint_data.active:
            continue

        if constraint_data.body is None:
            raise ValueError(
                "No expression has been defined for the body "
                "of constraint %s" % (constraint_data.name))

        try:
            repn = generate_standard_repn(constraint_data.body,
                                           idMap=idMap)
        except Exception:
            err = sys.exc_info()[1]
            logging.getLogger('pyomo.core').error(
                "exception generating a standard representation for "
                "constraint %s: %s"
                % (constraint_data.name, str(err)))
            raise

        block_repn[constraint_data] = repn

def preprocess_constraint_data(block,
                           constraint_data,
                           idMap=None,
                           block_repn=None):

    # Get/Create the ComponentMap for the repn
    if not hasattr(block,'_repn'):
        block._repn = ComponentMap()
    block_repn = block._repn

    if constraint_data.body is None:
        raise ValueError(
            "No expression has been defined for the body "
            "of constraint %s" % (constraint_data.name))

    try:
        repn = generate_standard_repn(constraint_data.body,
                                       idMap=idMap)
    except Exception:
        err = sys.exc_info()[1]
        logging.getLogger('pyomo.core').error(
            "exception generating a standard representation for "
            "constraint %s: %s"
            % (constraint_data.name, str(err)))
        raise

    block_repn[constraint_data] = repn

