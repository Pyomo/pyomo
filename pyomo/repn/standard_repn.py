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


import logging
import math

from pyutilib.misc import Bunch

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
from pyomo.repn.canonical_repn import (collect_linear_canonical_repn,
                                       generate_canonical_repn)
from pyomo.core.base import expr_common
from pyomo.core.kernel.component_expression import IIdentityExpression
from pyomo.core.kernel.component_variable import IVariable

import six
from six import itervalues, iteritems, StringIO
from six.moves import xrange, zip
try:
    basestring
except:
    basestring = str

logger = logging.getLogger('pyomo.core')

using_py3 = six.PY3



class StandardRepn(object):
    """
    This class defines a standard/common representation for Pyomo expressions
    that provides an efficient interface for writing all models.

    TODO: define what "efficient" means to us.
    """

    __slots__ = ('_constant',               # The constant term
                 '_linear_terms_coef',      # Linear coefficients
                 '_linear_vars',            # Linear variables
                 '_quadratic_terms_coef',   # Quadratic coefficients
                 '_quadratic_vars',         # Quadratic variables
                 '_nonlinear_expr',         # TODO
                 '_nonlinear_vars')         # TODO

    def __init__(self):
        self._constant = 0
        self._linear_vars = {}
        self._linear_terms_coef = {}
        self._quadratic_vars = {}
        self._quadratic_terms_coef = {}
        self._nonlinear_expr = None
        self._nonlinear_vars = {}

    def __getstate__(self):
        """
        This method is required because this class uses slots.
        """
        return  (self._constant,
                 self._linear_terms_coef,
                 self._linear_vars,
                 self._quadratic_terms_coef,
                 self._quadratic_vars,
                 self._nonlinear_expr,
                 self._nonlinear_vars)

    def __setstate__(self, state):
        """
        This method is required because this class uses slots.
        """
        self._constant, \
        self._linear_terms_coef, \
        self._linear_vars, \
        self._quadratic_terms_coef, \
        self._quadratic_vars, \
        self._nonlinear_expr, \
        self._nonlinear_vars = state

    #
    # Although it is convenient to have the dictionaries
    # hashed by some variable id when generating the standard
    # representation, we can compress these to lists after
    # generation is complete
    #
    def _compress(self):
        if  self._linear_vars.__class__ is dict:
            linear_keys = self._linear_vars.keys()
            self._linear_vars = tuple(self._linear_vars[key] for key in linear_keys)
            self._linear_terms_coef = tuple(self._linear_terms_coef[key] for key in linear_keys)
        else:
            self._linear_terms_coef = tuple()
        if  self._quadratic_vars.__class__ is dict:
            quadratic_keys = self._quadratic_vars.keys()
            self._quadratic_vars = tuple(self._quadratic_vars[key] for key in quadratic_keys)
            self._quadratic_terms_coef = tuple(self._quadratic_terms_coef[key] for key in quadratic_keys)
        if self._nonlinear_vars.__class__ is dict:
            self._nonlinear_vars = tuple(itervalues(self._nonlinear_vars))

    #
    # Generate a string representation of the expression
    #
    def __str__(self):
        output = StringIO()
        output.write("\n")
        output.write("constant:       "+str(self._constant)+"\n")
        if  self._linear_vars.__class__ is dict:
            output.write("linear vars:    "+str([v_.name for _,v_ in sorted(self._linear_vars.items(), key=lambda x: x[0])])+"\n")
            output.write("linear var ids: "+str([id(v_) for _,v_ in sorted(self._linear_vars.items(), key=lambda x: x[0])])+"\n")
            output.write("linear coef:    "+str([c_ for _,c_ in sorted(self._linear_terms_coef.items(), key=lambda x: x[0])])+"\n")
        else:
            output.write("linear vars:    "+str([v_.name for v_ in self._linear_vars])+"\n")
            output.write("linear var ids: "+str([id(v_) for v_ in self._linear_vars])+"\n")
            output.write("linear coef:    "+str(list(self._linear_terms_coef))+"\n")
        if  self._quadratic_vars.__class__ is dict:
            output.write("quadratic vars:    "+str([(v_[0].name,v_[1].name) for _,v_ in sorted(self._quadratic_vars.items(), key=lambda x: x[0])])+"\n")
            output.write("quadratic var ids: "+str([(id(v_[0]),id(v_[1])) for _,v_ in sorted(self._quadratic_vars.items(), key=lambda x: x[0])])+"\n")
            output.write("quadratic coef:    "+str([c_ for _,c_ in sorted(self._quadratic_terms_coef.items(), key=lambda x: x[0])])+"\n")
        else:
            output.write("quadratic vars:    "+str([v_.name for v_ in self._quadratic_vars])+"\n")
            output.write("quadratic var ids: "+str([id(v_) for v_ in self._quadratic_vars])+"\n")
            output.write("quadratic coef:    "+str(list(self._quadratic_terms_coef))+"\n")
        if self._nonlinear_expr is None:
            output.write("nonlinear expr: None\n")
        else:
            output.write("nonlinear expr:\n")
            try:
                self._nonlinear_expr.to_string(ostream=output)
                output.write("\n")
            except AttributeError:
                output.write(str([(i,str(e)) for i,e in self._nonlinear_expr])+"\n")
        if  self._nonlinear_vars.__class__ is dict:
            output.write("nonlinear vars: "+str([v_.name for _,v_ in sorted(self._nonlinear_vars.items(), key=lambda x: x[0])])+"\n")
        else:
            output.write("nonlinear vars: "+str([v_.name for v_ in self._nonlinear_vars])+"\n")
        output.write("\n")
        ret_str = output.getvalue()
        output.close()
        return ret_str

    #
    # Disabled
    #
    def X__eq__(self, other):
        # Can only be equal to other StandardRepn instances
        if not isinstance(other, StandardRepn):
            return False

        # Immediately check constants
        if self._constant != other._constant:
            return False

        # Check linear term lengths before iterating
        if len(self._linear_vars) != len(other._linear_vars):
            return False

        if self._linear_vars.__class__ is dict:
            self_linear_vars = list(itervalues(self._linear_vars))
            self_linear_terms = self._linear_terms_coef
        else:
            self_linear_vars = self._linear_vars
            self_linear_terms = dict((id(var),coef)
                                     for var,coef in zip(self._linear_vars,
                                                         self._linear_terms_coef))

        if other._linear_vars.__class__ is dict:
            other_linear_vars = list(itervalues(other._linear_vars))
            other_linear_terms = other._linear_terms_coef
        else:
            other_linear_vars = other._linear_vars
            other_linear_terms = dict((id(var),coef)
                                      for var,coef in zip(other._linear_vars,
                                                          other._linear_terms_coef))

        # Establish a mapping between self's linear terms and other's
        found_match = []
        for var in self_linear_vars:
            match = False
            for other_var in other_linear_vars:
                if var is other_var:
                    match = True
                    break
            found_match.append(match)

        # We have to have found all our own vars
        if not all(found_match):
            return False

        # Check that linear term coefficients are equal
        for var_ID in self_linear_terms:
            if self_linear_terms[var_ID] != other_linear_terms[var_ID]:
                return False

        if self._nonlinear_expr is not None:
            if other._nonlinear_expr is None:
                return False

            if type(self._nonlinear_expr) is list:
                if type(other._nonlinear_expr) is not list:
                    return False

                # Check that nonlinear expressions are the same length
                if len(self._nonlinear_expr) != len(other._nonlinear_expr):
                    return False

                # Check that nonlinear expression have the same terms
                for i in xrange(len(self._nonlinear_expr)):
                    if self._nonlinear_expr[i][0] != other._nonlinear_expr[i][0]:
                        return False
                    if not self._nonlinear_expr[i][1] is other._nonlinear_expr[i][1]:
                        return False
            else:
                if not self._nonlinear_expr is other._nonlinear_expr:
                    return False
        else:
            if other._nonlinear_expr is not None:
                return False

        return True

    def X__ne__(self, other):
        return not self.__eq__(other)

    def is_fixed(self):
        if len(self._linear_vars) == 0 and len(self._nonlinear_vars) == 0 and len(self._quadratic_vars) == 0:
            return True
        return False

    def is_linear(self):
        if self._nonlinear_expr is None and len(self._quadratic_terms_coef) == 0:
            return True
        return False

    def is_quadratic(self):
        return len(self._quadratic_terms_coef) > 0

    def is_nonlinear(self):
        if self._nonlinear_expr is None and len(self._quadratic_terms_coef) == 0:
            return False
        return True


"""

Note:  This function separates linear terms from nonlinear terms.
Along the way, fixed variable and mutable paramter values *may* be
replaced with constants.  However, that is not guaranteed.  Thus,
the nonlinear expression may contain subexpressions whose value is
constant.  This was done to avoid additional work when a subexpression
is clearly nonlinear.  However, this requires that standard
representations be temporary.  They should be used to interface
to a solver and then be deleted.

"""
def generate_standard_repn(expr, idMap=None, compute_values=True, verbose=False, compress=False, quadratic=True):
    if idMap is None:
        idMap = {}
    idMap.setdefault(None, {})
    repn = StandardRepn()
    #
    # Eliminate top-level negations
    #
    _multiplier = 1.0
    while expr.__class__ == EXPR._NegationExpression:
        #
        # Replace a negation sub-expression
        #
        _multiplier *= -1
        expr = expr._args[0]
    #
    # The expression is a number or a non-variable constant
    # expression.
    #
    if type(expr) in native_numeric_types or not expr._potentially_variable():
        if compute_values:
            repn._constant = _multiplier*value(expr)
        else:
            repn._constant = _multiplier*expr
        if compress:
            repn._compress()
        return repn
    #
    # The expression is a variable
    #
    if isinstance(expr, (_VarData, IVariable)):
        if expr.fixed:
            if compute_values:
                repn._constant = _multiplier*value(expr)
            else:
                repn._constant = _multiplier*expr
            if compress:
                repn._compress()
            return repn
        var_ID = id(expr)
        repn._linear_terms_coef[var_ID] = _multiplier
        repn._linear_vars[var_ID] = expr
        if compress:
            repn._compress()
        return repn
    #
    # Unknown expression object
    #
    if not expr.is_expression():
        raise ValueError("Unexpected expression type: "+str(expr))

    ##
    ## Recurse through the expression tree, collecting variables and linear terms, etc
    ##
    #
    # The stack starts with the current expression
    #
    _stack = [ (expr, expr._args, 0, len(expr._args), _multiplier, False, [])]
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
        _obj, _argList, _idx, _len, _multiplier, _compute_value, _result = _stack.pop()
        if verbose: #pragma:nocover
            print("*"*10 + " POP  " + "*"*10)

        #
        # Iterate through the arguments
        #
        while _idx < _len:
            if verbose: #pragma:nocover
                print("-"*30)
                print(type(_obj))
                print(_obj)
                print(_argList)
                print(_idx)
                print(_len)
                print(_multiplier)
                print(_compute_value)
                print(_result)

            ##
            ## Process context based on _obj type
            ##

            # No special processing for *Sum* objects

            # No special processing for _ProductExpression

            if _obj.__class__ == EXPR._NegationExpression:
                _multiplier *= -1

            elif _obj.__class__ == EXPR._PowExpression:
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
                            break

            elif _obj.__class__ == EXPR.Expr_if:
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
            
            if -999 in _result:
                break

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
                try:
                    # TODO: disable ERROR logging message
                    _result.append( {0:EXPR.evaluate_expression(_sub, only_fixed_vars=True)} )
                except Exception as e:
                    _result = [{-999: "Error evaluating expression: %s" % str(e)}] 

            elif not _sub._potentially_variable():
                #
                # Store a non-variable expression
                #
                if compute_values:
                    _result.append( {0:value(_sub)} )
                else:
                    _result.append( {0:_sub} )

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

                    _result.append( {1:{key:1.0}} )
                else:
                    if compute_values:
                        _result.append( {0:value(_sub)} )
                    else:
                        _result.append( {0:_sub} )

            else:
                assert(_sub.is_expression())
                #
                # Push an expression onto the stack
                #
                if verbose: #pragma:nocover
                    print("*"*10 + " PUSH " + "*"*10)

                _stack.append( (_obj, _argList, _idx, _len, _multiplier, _compute_value, _result) )

                _obj     = _sub
                _argList = _sub._args
                _idx     = 0
                _len     = len(_argList)
                _result  = []



        #
        # POST-DIVE
        #
        if verbose: #pragma:nocover
            print("="*30)
            print(type(_obj))
            print(_obj)
            print(_argList)
            print(_idx)
            print(_len)
            print(_multiplier)
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
                break

        if _obj.__class__ in (EXPR._SumExpression, EXPR._MultiSumExpression, EXPR._CompressedSumExpression, EXPR._StaticMultiSumExpression):
            ans = {}
            for res in _result:
                # Add nonlinear terms
                if None in res:
                    if None in ans:
                        ans[None] += res[None]
                    else:
                        ans[None] = res[None]
                # Add constant terms
                if 0 in res:
                    if 0 in ans:
                        ans[0] += res[0]
                    else:
                        ans[0] = res[0]
                # Add linear terms
                if 1 in res:
                    if not 1 in ans:
                        ans[1] = {}
                    for key in res[1]:
                        if key in ans[1]:
                            coef = ans[1][key] + res[1][key]
                            if not math.isclose(coef, 0.0):     # coef != 0.0
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
                            if not math.isclose(coef, 0.0):     # coef != 0.0
                                ans[2][key] = coef
                            else:
                                del ans[2][key]
                        else:
                            ans[2][key] = res[2][key]           # We shouldn't need to check if this is zero

        elif _obj.__class__ == EXPR._ProductExpression or (_obj.__class__ == EXPR._PowExpression and len(_result) == 2):
            #
            # The POW expression is a special case.  This the length==2 indicates that this is a quadratic.
            #
            if _obj.__class__ == EXPR._PowExpression:
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
            nonl = 0
            if None in _l:
                if None in _r:
                    nonl += _l[None]*_r[None]
                if 0 in _r and not math.isclose(_r[0], 0.0):    # _r[0] != 0.0
                    nonl += _l[None]*_r[0]
                if 1 in _r:
                    for key in _r[1]:
                        nonl += _l[None]*_r[1][key]*idMap[key]
                if 2 in _r:
                    for key in _r[2]:
                        v1_, v2_ = key
                        nonl += _l[None]*_r[2][key]*idMap[v1_]*idMap[v2_]
            if None in _r:
                if 0 in _l and not math.isclose(_l[0], 0.0):        # _l[0] != 0.0
                    nonl += _l[0]*_r[None]
                if 1 in _l:
                    for key in _l[1]:
                        nonl += _l[1][key]*_r[None]*idMap[key]
                if 2 in _l:
                    for key in _l[2]:
                        v1_, v2_ = key
                        nonl += _r[None]*_l[2][key]*idMap[v1_]*idMap[v2_]
            if quadratic:
                # Products that generate term with degree > 2
                if 2 in _l:
                    if 1 in _r:
                        for lkey in _l[2]:
                            v1_, v2_ = lkey
                            for rkey in _r[1]:
                                nonl += _l[2][lkey]*_r[1][rkey]*idMap[v1_]*idMap[v2_]*idMap[rkey]
                    if 2 in _r:
                        for lkey in _l[2]:
                            lv1_, lv2_ = lkey
                            for rkey in _r[2]:
                                rv1_, rv2_ = rkey
                                nonl += _l[2][lkey]*_r[2][rkey]*idMap[lv1_]*idMap[lv2_]*idMap[rv1_]*idMap[rv2_]
                if 1 in _l and 2 in _r:
                        for lkey in _l[1]:
                            for rkey in _r[2]:
                                v1_, v2_ = rkey
                                nonl += _l[1][lkey]*_r[2][rkey]*idMap[lkey]*idMap[v1_]*idMap[v2_]
            else:
                # Products that generate term with degree = 2
                if 1 in _l and 1 in _r:
                    # TODO: Consider creating Multsum objects here with the Sum() function
                    nonl += sum(_l[1][i]*idMap[i] for i in _l[1]) * sum(_r[1][i]*idMap[i] for i in _r[1])
            if not nonl is 0:
                ans[None] = nonl

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
            if 0 in _l and 1 in _r and not math.isclose(_l[0], 0.0):    # _l[0] != 0.0
                for key in _r[1]:
                    ans[1][key] = _l[0]*_r[1][key]
            if 1 in _l and 0 in _r and not math.isclose(_r[0], 0.0):    # _r[0] != 0.0
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
                if 0 in _l and 2 in _r and not math.isclose(_l[0], 0.0):
                    for key in _r[2]:
                        ans[2][key] = _l[0]*_r[2][key]
                if 2 in _l and 0 in _r and not math.isclose(_r[0], 0.0):
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

        elif _obj.__class__ == EXPR._NegationExpression:
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

        elif _obj.__class__ == EXPR._ReciprocalExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
            else:
                ans = {0:1.0/_result[0][0]}

        elif _obj.__class__ == EXPR._AbsExpression or _obj.__class__ == EXPR._UnaryFunctionExpression:
            if None in _result[0] or 1 in _result[0] or 2 in _result[0]:
                ans = {None:_obj}
            else:
                ans = {0:_obj(_result[0][0])}

        elif _obj.__class__ == EXPR.Expr_if:
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
            print(_obj)
            print(_argList)
            print(_idx)
            print(_len)
            print(_multiplier)
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
    repn._constant = ans.get(0,0)
    if 1 in ans:
        for i in ans[1]:
            repn._linear_vars[i] = idMap[i]
            repn._linear_terms_coef[i] = ans[1][i]
    if 2 in ans:
        for i in ans[2]:
            v1_, v2_ = i
            repn._quadratic_vars[i] = (idMap[v1_],idMap[v2_])
            repn._quadratic_terms_coef[i] = ans[2][i]
    repn._nonlinear_expr = ans.get(None,None)
    repn._nonlinear_vars = {}
    if not repn._nonlinear_expr is None:
        for v_ in EXPR.identify_variables(repn._nonlinear_expr, include_fixed=False, include_potentially_variable=False):
            repn._nonlinear_vars[id(v_)] = v_
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

    if compress:
        repn._compress()
    return repn


