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

__all__ = ['StandardRepn', 'generate_standard_repn', 'compute_standard_repn']


import sys
import logging
import math
import itertools

from pyomo.core.base import (Constraint,
                             Objective,
                             ComponentMap)

import pyomo.util
from pyutilib.misc import Bunch
from pyutilib.math.util import isclose

from pyomo.core.base import expr as EXPR
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.var import (SimpleVar,
                                 Var,
                                 _GeneralVarData,
                                 _VarData,
                                 value)
from pyomo.core.base.param import _ParamData
from pyomo.core.base.numvalue import (NumericConstant,
                                      native_numeric_types,
                                      is_fixed)
from pyomo.core.util import Sum
from pyomo.core.kernel.component_expression import IIdentityExpression
from pyomo.core.kernel.component_variable import IVariable

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
pyomo5_variable_types = set([_VarData, _GeneralVarData, IVariable, variable, SimpleVar])


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
                self.nonlinear_expr.to_string(ostream=output)
                output.write("\n")
            except AttributeError:
                output.write(str([(i,str(e)) for i,e in self.nonlinear_expr])+"\n")
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
            val = value(self.linear_coefs[i])
            if isclose(val, 1.0):
                expr += self.linear_vars[i]
            elif isclose(val, -1.0):
                expr -= self.linear_vars[i]
            elif val < 0.0:
                expr -= - self.linear_coefs[i]*self.linear_vars[i]
            else:
                expr += self.linear_coefs[i]*self.linear_vars[i]
        expr += Sum(self.quadratic_coefs[i]*self.quadratic_vars[i][0]*self.quadratic_vars[i][0] for i,v in enumerate(self.quadratic_vars))
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
    # Disable implicit cloning while creating a standard representation.
    # We allow the representation to be entangled with the original expression.
    #
    with EXPR.ignore_entangled_expressions:
        #
        # Setup
        #
        if idMap is None:
            idMap = {}
        idMap.setdefault(None, {})
        if repn is None:
            repn = StandardRepn()
        #
        # Eliminate top-level negations
        #
        _multiplier = 1
        while expr.__class__ is EXPR._NegationExpression:
            #
            # Replace a negation sub-expression
            #
            _multiplier *= -1
            expr = expr._args[0]
        #
        # The expression is a number or a non-variable constant
        # expression.
        #
        if expr.__class__ in native_numeric_types or not expr._potentially_variable():
            if compute_values:
                repn.constant = _multiplier*EXPR.evaluate_expression(expr)
            else:
                repn.constant = _multiplier*expr
            return repn
        #
        # The expression is a variable
        #
        elif isinstance(expr, (_VarData, IVariable)):
            if expr.fixed:
                if compute_values:
                    repn.constant = _multiplier*value(expr)
                else:
                    repn.constant = _multiplier*expr
                return repn
            repn.linear_coefs = (_multiplier,)
            repn.linear_vars = (expr,)
            return repn

        #
        # The expression is linear
        #
        elif expr.__class__ is EXPR._StaticLinearExpression:
            if compute_values:
                C_ = EXPR.evaluate_expression(expr.constant)
            else:
                C_ = expr.constant
            v_ = []
            c_ = []
            if compute_values:
                for c,v in zip(expr.linear_coefs, expr.linear_vars):
                    if v.fixed:
                        if c.__class__ in native_numeric_types:
                            C_ += c*v.value
                        elif c.is_expression():
                            C_ += EXPR.evaluate_expression(c)*v.value
                        else:
                            C_ += value(c)*v.value
                    else:
                        if c.__class__ in native_numeric_types:
                            c_.append( c )
                        elif c.is_expression():
                            c_.append( EXPR.evaluate_expression(c) )
                        else:
                            c_.append( value(c) )
                    v_.append( v )
            else:
                for c,v in zip(expr.linear_coefs, expr.linear_vars):
                    if v.fixed:
                        C_ += c*v
                    else:
                        c_.append( c )
                    v_.append( v )
            repn.constant = C_
            repn.linear_coefs = tuple(c_)
            repn.linear_vars = tuple(v_)
            for v in repn.linear_vars:
                id_ = id(v)
                if not id_ in idMap[None]:
                    key = len(idMap) - 1
                    idMap[None][id_] = key
                    idMap[key] = v
            return repn

        #
        # If the expression is a sum, then we try to 
        # extract a linear representation.
        #
        elif expr.__class__ is EXPR._ViewSumExpression:
            repn.linear_coefs = []
            repn.linear_vars = []
            linear=True
            for e_ in itertools.islice(expr._args, expr.nargs()):
                if e_.__class__ in native_numeric_types:
                    repn.constant += e_
                elif not e_._potentially_variable():
                    repn.constant += value(e_)
                elif e_.__class__ in pyomo5_variable_types:
                    if e_.fixed:
                        if compute_values:
                            repn.constant += e_.value
                        else:
                            repn.constant += e_
                    else:
                        repn.linear_coefs.append(1)
                        repn.linear_vars.append(e_)
                elif e_.__class__ is EXPR._ProductExpression:
                    if e_._args[1].__class__ in pyomo5_variable_types:
                        if e_._args[1].fixed:
                            if e_._args[0].__class__ in native_numeric_types:
                                repn.constant += e._args[0] * e._args[1].value
                            elif not e_._args[0]._potentially_variable():
                                repn.constant += e._args[0] * value(e._args[1])
                            elif e_._args[0].__class__ in pyomo5_variable_types:
                                if e_._args[0].fixed:
                                    repn.constant += e._args[0].value * e._args[1].value
                                else:
                                    v = e_._args[1].value
                                    if not isclose(v,0.0):
                                        repn.linear_coefs.append(v)
                                        repn.linear_vars.append(e_._args[0])
                            else:
                                linear=False
                                break
                        elif e_._args[0].__class__ in native_numeric_types:
                            if not isclose(e_._args[0],0.0):
                                repn.linear_coefs.append(e_._args[0])
                                repn.linear_vars.append(e_._args[1])
                        elif not e_._args[0]._potentially_variable() or \
                             (e_._args[0].__class__ in pyomo5_variable_types and e_._args[0].fixed):
                            v = value(e_._args[0])
                            if not isclose(v, 0.0):
                                repn.linear_coefs.append(v)
                                repn.linear_vars.append(e_._args[1])
                        else:
                            linear=False
                            break
                    else:
                        linear=False
                        break
                else:
                    linear=False
                    break
            if linear:
                repn.linear_vars = tuple(repn.linear_vars)
                repn.linear_coefs = tuple(repn.linear_coefs)
                for v in repn.linear_vars:
                    id_ = id(v)
                    if not id_ in idMap[None]:
                        key = len(idMap) - 1
                        idMap[None][id_] = key
                        idMap[key] = v
                return repn

        #
        # Unknown expression object
        #
        elif not expr.is_expression():
            raise ValueError("Unexpected expression type: "+str(expr))

        return nonrecursive_generate_standard_repn(expr, 
                                idMap=idMap,
                                compute_values=compute_values,
                                verbose=verbose,
                                quadratic=quadratic,
                                repn=repn,
                                _multiplier=_multiplier)


def nonrecursive_generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, quadratic=True, repn=None, _multiplier=None):
    ##
    ## Recurse through the expression tree, collecting variables and linear terms, etc
    ##
    linear = True
    #
    # The stack starts with the current expression
    #
    _stack = [ (expr, expr._args, 0, expr.nargs(), False, [])]
    #
    # Iterate until the stack is empty
    #
    # Note: 1 is faster than True for Python 2.x
    #
    while 1:
        #
        # Get the top of the stack
        #   _obj        Current expression object
        #   _argList    The arguments for this expression objet
        #   _idx        The current argument being considered
        #   _len        The number of arguments
        #
        # Note: expressions pushed onto the stack are guaranteed to 
        # be potentially variable.
        #
        _obj, _argList, _idx, _len, _compute_value, _result = _stack.pop()
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)

        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj.to_string())
                print(_argList)
                print(_idx)
                print(_len)
                print(_compute_value)
                print(_result)

            ##
            ## Process context based on _obj type
            ##

            # No special processing for *Sum* objects

            # No special processing for _ProductExpression

            if _obj.__class__ is EXPR._PowExpression:
                if _idx == 0:
                    #
                    # Evaluate the RHS (_args[1]) first, and compute its value
                    #
                    _argList = (_argList[1], _argList[0])
                    _compute_value = True
                elif _idx == 1:
                    _compute_value = False
                    if -999 in _result[0]:
                        #
                        # If the RHS (_args[1]) is variable, then
                        # treat the entire subexpression as a nonlinear expression
                        #
                        _result = [{None:_obj}]
                        linear = False
                        break
                    else:
                        val = _result[0][0]
                        if val == 0:
                            #
                            # If the exponent is zero, then the value of this expression is 1
                            #
                            _result = [{0:1}]
                            break
                        elif val == 1:
                            #
                            # If the exponent is one, then simply return 
                            # the value of the LHS (_args[0])
                            #
                            _result = []
                        elif val == 2 and quadratic:
                            #
                            # If the exponent is two, then set the value of the exponent and continue
                            # processing the value of the LHS (_args[0])
                            #
                            _result = [{0:2}]
                        else:
                            #
                            # Otherwise, we treat this as a nonlinear expression
                            #
                            _result = [{None:_obj}]
                            linear = False
                            break

            elif _obj.__class__ is EXPR.Expr_if:
                if _idx == 0:
                    #
                    # Compute the value of the condition argument
                    #
                    _compute_value = True
                elif _idx == 1:
                    _compute_value = False
                    if -999 in _result[0]:
                        #
                        # If the condition argument is variable, then
                        # treat the entire subexpression as a nonlinear expression
                        #
                        _result = [{None:_obj}]
                        linear = False
                        break
                    else:
                        val = _result[0][0]
                        _idx = 0
                        _len = 1
                        _result = []
                        if val:
                            _argList = [_argList[1]]
                        else:
                            _argList = [_argList[2]]
            
            ##
            ## Process the next current _obj object
            ##

            _sub = _argList[_idx]
            _idx += 1

            if _sub.__class__ in native_numeric_types:
                #
                # Store a native object
                #
                _result.append( {0:_sub} )

            elif _compute_value:
                val = EXPR.evaluate_expression(_sub, only_fixed_vars=True, exception=False)
                if val is None:
                    _result = [{-999: "Error evaluating expression: %s" % str(_sub)}] 
                else:
                    _result.append( {0:val} )

            elif (_sub.__class__ is _GeneralVarData) or isinstance(_sub, (_VarData, IVariable)):
                #
                # Process a single variable
                #
                if not _sub.fixed:
                    #
                    # Store a variable 
                    #
                    id_ = id(_sub)
                    if id_ in idMap[None]:
                        key = idMap[None][id_]
                    else:
                        key = len(idMap) - 1
                        idMap[None][id_] = key
                        idMap[key] = _sub

                    _result.append( {1:{key:1}} )
                else:
                    if compute_values:
                        _result.append( {0:_sub.value} )
                    else:
                        _result.append( {0:_sub} )

            elif not _sub._potentially_variable():
                #
                # Store a non-variable expression
                #
                if compute_values:
                    val = EXPR.evaluate_expression(_sub, only_fixed_vars=True, exception=False)
                    if val is None:
                        _result = [{-999: "Error evaluating expression: %s" % str(_sub)}] 
                    else:
                        _result.append( {0:val} )
                else:
                    _result.append( {0:_sub} )

            elif _sub.__class__ is EXPR._StaticLinearExpression:
                #
                # Extract data from the linear expression
                #
                val = {}
                constant = _sub.constant
                if len(_sub.linear_vars) > 0:
                    ans = {}
                    for c,v in zip(_sub.linear_coefs, _sub.linear_vars):
                        if v.fixed:
                            if compute_values:
                                constant += EXPR.evaluate_expression(c)*v.value
                            else:
                                constant += c*v
                        else:
                            #
                            # Store a variable 
                            #
                            id_ = id(v)
                            if id_ in idMap[None]:
                                key = idMap[None][id_]
                            else:
                                key = len(idMap) - 1
                                idMap[None][id_] = key
                                idMap[key] = v
                            if compute_values:
                                ans[key] = EXPR.evaluate_expression(c)
                            else:
                                ans[key] = c
                    val[1] = ans
                if not isclose(constant, 0):
                    val[0] = constant
                _result.append( val )
            else:
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)

                _stack.append( (_obj, _argList, _idx, _len, _compute_value, _result) )

                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = _sub.nargs()
                _result  = []

        #
        # POST-DIVE
        #
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj.to_string())
            print(_argList)
            print(_idx)
            print(_len)
            print(_compute_value)
            print(_result)
            print("STACK LEN %d" % len(_stack))

        if -999 in _result[-1]:
            #
            # "return" the recursion by putting the return value on the end of the results stack
            #
            if _stack:
                _stack[-1][-1].append( {-999:_result[-1][-999]} )
                continue
            else:
                ans = {}
                break

        if _obj.__class__ is EXPR._SumExpression or _obj.__class__ is EXPR._ViewSumExpression:
            ans = {}
            # Add nonlinear terms
            # Do some extra work to combine the arguments of 'Sum' expressions
            nonl = []
            if not linear:
                for res in _result:
                    if None in res:
                        if res[None].__class__ is EXPR._SumExpression or res[None].__class__ is EXPR._ViewSumExpression:
                            for arg in itertools.islice(res[None]._args, res[None].nargs()):
                                nonl.append(arg)
                        else:
                            nonl.append(res[None])
                if len(nonl) > 0:
                    nonl = Sum(x for x in nonl)
                    if not (nonl.__class__ in native_numeric_types and isclose(nonl,0)):
                        ans[None] = nonl
                        linear = False
            # Add constant terms
            cons = 0
            cons = 0 + sum(res[0] for res in _result if 0 in res)
            if not cons is 0:
                ans[0] = cons

            for res in _result:
                # Add linear terms
                if 1 in res:
                    if not 1 in ans:
                        ans[1] = {}
                    for key in res[1]:
                        if key in ans[1]:
                            coef = ans[1][key] + res[1][key]
                            if not (coef.__class__ in native_numeric_types and isclose(coef, 0.0)):     # coef != 0.0
                                ans[1][key] = coef
                            else:
                                del ans[1][key]
                        else:
                            ans[1][key] = res[1][key]           # We shouldn't need to check if this is zero
                # Add quadratic terms
                if quadratic and 2 in res:
                    if not 2 in ans:
                        ans[2] = {}
                    for key in res[2]:
                        if key in ans[2]:
                            coef = ans[2][key] + res[2][key]
                            if not (coef.__class__ in native_numeric_types and isclose(coef, 0.0)):     # coef != 0.0
                                ans[2][key] = coef
                            else:
                                del ans[2][key]
                        else:
                            ans[2][key] = res[2][key]           # We shouldn't need to check if this is zero

        elif _obj.__class__ is EXPR._ProductExpression or (_obj.__class__ is EXPR._PowExpression and len(_result) == 2):
            #
            # The POW expression is a special case.  This the length==2 indicates that this is a quadratic.
            #
            if _obj.__class__ is EXPR._PowExpression:
                _tmp, _l = _result
                _r = _l
            else:
                _l, _r = _result
            #print("_l")
            #print(_l)
            #print("_r")
            #print(_r)
            ans = {}
            #
            # Compute the product
            #
            # l\r   None    0       1       2
            # None  None    None    None    None
            # 0     None    0       1       2
            # 1     None    1       2       None
            # 2     None    2       None    None
            #

            #
            # GENERATING A NONLINEAR TERM
            #
            # Products that include a nonlinear term
            nonl = []
            if None in _l:
                rhs = 0
                if None in _r:
                    rhs += _r[None]
                if 0 in _r and \
                   not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):    # _r[0] != 0.0
                    rhs += _r[0]
                if 1 in _r:
                    rhs += Sum(_r[1][key]*idMap[key] for key in _r[1])
                if 2 in _r:
                    rhs += Sum(_r[2][key]*idMap[key[0]]*idMap[key[1]] for key in _r[2])
                nonl.append(_l[None]*rhs)
            if None in _r:
                lhs = 0
                if 0 in _l and \
                   not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):        # _l[0] != 0.0
                    lhs += _l[0]
                if 1 in _l:
                    lhs += Sum(_l[1][key]*idMap[key] for key in _l[1])
                if 2 in _l:
                    lhs += Sum(_l[2][key]*idMap[key[0]]*idMap[key[1]] for key in _l[2])
                nonl.append(lhs*_r[None])
            if quadratic:
                # Products that generate term with degree > 2
                if 2 in _l:
                    if 1 in _r:
                        for lkey in _l[2]:
                            v1_, v2_ = lkey
                            for rkey in _r[1]:
                                nonl.append(_l[2][lkey]*_r[1][rkey]*idMap[v1_]*idMap[v2_]*idMap[rkey])
                    if 2 in _r:
                        for lkey in _l[2]:
                            lv1_, lv2_ = lkey
                            for rkey in _r[2]:
                                rv1_, rv2_ = rkey
                                nonl.append(_l[2][lkey]*_r[2][rkey]*idMap[lv1_]*idMap[lv2_]*idMap[rv1_]*idMap[rv2_])
                if 1 in _l and 2 in _r:
                        for lkey in _l[1]:
                            for rkey in _r[2]:
                                v1_, v2_ = rkey
                                nonl.append(_l[1][lkey]*_r[2][rkey]*idMap[lkey]*idMap[v1_]*idMap[v2_])
            else:
                # Products that generate term with degree = 2
                if 1 in _l and 1 in _r:
                    # TODO: Consider creating Multsum objects here with the Sum() function
                    nonl.append( Sum(_l[1][i]*idMap[i] for i in _l[1]) * Sum(_r[1][i]*idMap[i] for i in _r[1]) )
            if len(nonl) > 0:
                nonl = Sum(x for x in nonl)
                if not (nonl.__class__ in native_numeric_types and isclose(nonl,0)):
                    ans[None] = nonl
                    linear = False

            #
            # GENERATING A CONSTANT TERM
            #
            if 0 in _l and 0 in _r:
                ans[0] = _l[0]*_r[0]

            #
            # GENERATING LINEAR TERMS
            #
            if (0 in _l and 1 in _r) or (1 in _l and 0 in _r):
                ans[1] = {}
                if 0 in _l and 1 in _r and \
                   not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):    # _l[0] != 0.0
                    for key in _r[1]:
                        ans[1][key] = _l[0]*_r[1][key]
                if 1 in _l and 0 in _r and \
                   not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):    # _r[0] != 0.0
                    for key in _l[1]:
                        if key in ans[1]:
                            ans[1][key] += _l[1][key]*_r[0]
                        else:
                            ans[1][key] = _l[1][key]*_r[0]

            #
            # GENERATING QUADRATIC TERMS
            #
            if quadratic:
                if (0 in _l and 2 in _r) or (2 in _l and 0 in _r) or (1 in _l and 1 in _r):
                    ans[2] = {}
                    if 0 in _l and 2 in _r and \
                       not (_l[0].__class__ in native_numeric_types and isclose(_l[0], 0.0)):
                        for key in _r[2]:
                            ans[2][key] = _l[0]*_r[2][key]
                    if 2 in _l and 0 in _r and \
                       not (_r[0].__class__ in native_numeric_types and isclose(_r[0], 0.0)):
                        for key in _l[2]:
                            if key in ans[2]:
                                ans[2][key] += _l[2][key]*_r[0]
                            else:
                                ans[2][key] = _l[2][key]*_r[0]
                    if 1 in _l and 1 in _r:
                        for lkey in _l[1]:
                            for rkey in _r[1]:
                                if id(idMap[lkey]) <= id(idMap[rkey]):
                                    key_ = (lkey,rkey)
                                else:
                                    key_ = (rkey,lkey)
                                if key_ in ans[2]:
                                    ans[2][key_] += _l[1][lkey]*_r[1][rkey]
                                else:
                                    ans[2][key_] = _l[1][lkey]*_r[1][rkey]

        elif _obj.__class__ is EXPR._NegationExpression:
            ans = _result[0]
            if None in ans:
                ans[None] *= -1
            if 0 in ans:
                ans[0] *= -1
            if 1 in ans:
                for i in ans[1]:
                    ans[1][i] *= -1
            if 2 in ans:
                for i in ans[2]:
                    ans[2][i] *= -1

        elif _obj.__class__ is EXPR._ReciprocalExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
                linear = False
            else:
                ans = {0:1/_result[0][0]}

        elif _obj.__class__ is EXPR._AbsExpression or _obj.__class__ is EXPR._UnaryFunctionExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
                linear = False
            else:
                ans = {0:_obj(_result[0][0])}

        elif _obj.__class__ is EXPR.Expr_if:
            ans = _result[0]

        else:
            try:
                assert(len(_result) == 1)
            except Exception as e:
                print("ERROR: "+str(type(_obj)))
                raise
            ans = _result[0]

        #print("ans")
        #print(ans)
        if verbose: #pragma:nocover
            print("*"*10 + " RETURN  " + "*"*10)
            print("."*30)
            print(type(_obj))
            print(_obj.to_string())
            print(_argList)
            print(_idx)
            print(_len)
            print(_compute_value)
            print(_result)
            print("STACK LEN %d" % len(_stack))

        if _stack:
            #
            # "return" the recursion by putting the return value on the end of the results stack
            #
            _stack[-1][-1].append( ans )
        else:
            break

    #
    # Create the final object here from 'ans'
    #
    repn.constant = _multiplier*ans.get(0,0)
    if 1 in ans:
        keys = list(ans[1].keys())
        repn.linear_vars  = tuple(idMap[i] for i in keys)
        repn.linear_coefs = tuple(_multiplier*ans[1][i] for i in keys)
    if 2 in ans:
        keys = list(ans[2].keys())
        repn.quadratic_vars  = tuple((idMap[v1_],idMap[v2_]) for v1_, v2_ in keys)
        repn.quadratic_coefs = tuple(_multiplier*ans[2][i] for i in keys)
    repn.nonlinear_expr = ans.get(None,None)
    if not repn.nonlinear_expr is None:
        repn.nonlinear_expr *= _multiplier
    repn.nonlinear_vars = {}
    if not repn.nonlinear_expr is None:
        repn.nonlinear_vars = []
        for v_ in EXPR.identify_variables(repn.nonlinear_expr, include_fixed=False, include_potentially_variable=False):
            repn.nonlinear_vars.append(v_)
            #
            # Update idMap in case we skipped nonlinear sub-expressions
            #
            # Q: Should we skip nonlinear sub-expressions?
            #
            id_ = id(v_)
            if not id_ in idMap[None]:
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


@pyomo.util.pyomo_api(namespace='pyomo.repn')
def compute_standard_repn(data, model=None):
    """
    This plugin computes the standard representation for all objectives
    and constraints. All results are stored in a ComponentMap named
    "_repn" at the block level.

    We break out preprocessing of the objectives and constraints
    in order to avoid redundant and unnecessary work, specifically
    in contexts where a model is iteratively solved and modified.
    we don't have finer-grained resolution, but we could easily
    pass in a Constraint and an Objective if warranted.

    Required:
        model:      A concrete model instance.
    """
    idMap = {}
    for block in model.block_data_objects(active=True):
        preprocess_block_constraints(block, idMap=idMap)
        preprocess_block_objectives(block, idMap=idMap)
