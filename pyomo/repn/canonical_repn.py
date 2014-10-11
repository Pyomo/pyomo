#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the FAST README.txt file.
#  _________________________________________________________________________

from __future__ import division

__all__ = [ 'generate_canonical_repn', 'as_expr', 'canonical_is_constant', 
            'canonical_is_linear', 'canonical_is_quadratic', 'canonical_is_nonlinear', 
            'canonical_degree', 'LinearCanonicalRepn', 'GeneralCanonicalRepn' ]

import logging
import copy
import six

from six import iterkeys, itervalues, iteritems, StringIO
from six.moves import xrange, reduce

using_py3 = six.PY3

from pyomo.core.base import IPyomoPresolver, IPyomoPresolveAction, Model, \
                             Constraint, Objective, value
from pyomo.core.base import param
from pyomo.core.base import expr
from pyomo.core.base.expression import _ExpressionData, SimpleExpression, Expression
from pyomo.core.base.connector import _ConnectorValue, SimpleConnector
from pyomo.core.base.var import _VarDataWithDomain, SimpleVar, Var, _VarData

logger = logging.getLogger('pyomo.core')

##############################################################################
##############################################################################
#### CLASSES AND ROUTINES FOR GENERAL CANONICAL REPRESENTATIONS START NOW ####
##############################################################################
##############################################################################

#
# A frozen dictionary that can be hashed.  This dictionary isn't _really_
# frozen, but it acts hashable.
#
class GeneralCanonicalRepn(dict):

    __slots__ = ['_hash']

    def __hash__(self):
        rval = getattr(self, '_hash', None)
        if rval is None:
            rval = self._hash = hash(frozenset(iteritems(self)))
        return rval

    def __lt__(self,other):
        return self.__hash__() < other.__hash__()

    # getstate and setstate are currently not defined in a generic manner, i.e.,
    # we assume the underlying dictionary is the only thing

    def __getstate__(self):
        return self.items()

    def __setstate__(self, dictionary):
        self.clear()
        self.update(dictionary)
        self._hash = None

    def __str__(self):
        return "GeneralCanonical{ %s%s }" % (
            as_expr(self, ignore_other=True),
            ", general: %s" % self[None] if None in self else "" )

#
# Generate a canonical representation of an expression.
#
# The canonical representation is a dictionary.  Each element is a mapping
# from a term degree to terms in the expression.  If the term degree is
# None, then the map value is simply an expression of terms without a
# specific degree.  Otherwise, the map value is a dictionary of terms to
# their coefficients.  A term is represented as a frozen dictionary that
# maps variable id to variable power.  A constant term is represented
# with None.
#
# Examples:
#  Let x[1] ... x[4] be the first 4 variables, and
#      y[1] ... y[4] be the next 4 variables
#
# 1.3                           {0:{ None :1.3}}
# 3.2*x[1]                      {1:{ {0:1} :3.2}}
# 2*x[1]*y[2] + 3*x[2] + 4      {0:{None:4.0}, 1:{{1:1}:3.0}, 2:{{0:1, 5:1}:2.0}}
# 2*x[1]**4*y[2]    + 4         {0:{None:4.0}, 1:{{1:1}:3.0}, 5:{{0:4, 5:1 }:2.0}}
# log(y[1]) + x[1]*x[1]         {2:{{0:2}:1.0}, None:log(y[1])}
#



def as_expr(rep, vars=None, model=None, ignore_other=False):
    """ Convert a canonical representation into an expression. """
    if isinstance(model, Model):
        vars = model._var
        id_offset=0
    elif not vars is None:
        id_offset=1
    exp = 0.0
    for d in sorted(rep):
        if d in (None, -1):
            continue
        for v in sorted(rep[d]):
            if v is None:
                exp += rep[d][v]
                continue
            e = rep[d][v]
            if type(v) is int:
                if vars is None:
                    e *= rep[-1][v]
                else:
                    e *= vars[id[1]+id_offset]
            else:
                for id in v:
                    if vars is None:
                        e *= rep[-1][id]**v[id]
                    else:
                        e *= vars[id[1]+id_offset]**v[id]
            exp += e
    # Stick general nonlineat at the end
    if None in rep and not ignore_other:
        exp += rep[None]
    return exp

def repn_add(lhs, rhs, coef=1.0):
    """
    lhs and rhs are the expressions being added together.
    'lhs' and 'rhs' are left-hand and right-hand side operands
    See generate_canonical_repn for explanation of pyomo expressions
    """
    for order in rhs:
        # For each order term, first-order might be 3*x,
        # second order 4*x*y or 5*x**2
        if order is None:
            # i.e., (currently) order is a constant or logarithm
            if order in lhs:
                lhs[order] += rhs[order]
            else:
                lhs[order] = rhs[order]
            continue
        if order < 0:
            # ignore now, handled below
            continue
        if not order in lhs:
            lhs[order] = {}
        for var in rhs[order]:
            # Add coefficients of variables in this order (e.g., third power)
            lhs[order][var] = coef*rhs[order][var] + lhs[order].get(var,0.0)
    #
    # Merge the _VarData maps
    #
    if -1 in rhs:
        if -1 in lhs:
            lhs[-1].update(rhs[-1])
        else:
            lhs[-1] = rhs[-1]
    return lhs

def repn_mult(r1, r2, coef=1.0):
    rep = {}
    for d1 in r1:
        for d2 in r2:
            if d1 == None or d2 == None or d1 < 0 or d2 < 0:
                pass
            else:
                d=d1+d2
                if not d in rep:
                    rep[d] = {}
                if d == 0:
                    rep[d][None] = coef * r1[0][None] * r2[0][None]
                elif d1 == 0:
                    for v2 in r2[d2]:
                        rep[d][v2]  = coef * r1[0][None] * r2[d2][v2] + rep[d].get(v2,0.0)
                elif d2 == 0:
                    for v1 in r1[d1]:
                        rep[d][v1]  = coef * r1[d1][v1] * r2[0][None] + rep[d].get(v1,0.0)
                else:
                    for v1 in r1[d1]:
                        for v2 in r2[d2]:
                            v = GeneralCanonicalRepn(v1)
                            for id in v2:
                                if id in v:
                                    v[id] += v2[id]
                                else:
                                    v[id]  = v2[id]
                            rep[d][v] = coef * r1[d1][v1] * r2[d2][v2] + rep[d].get(v,0.0)
    #
    # Handle other nonlinear terms
    #
    if None in r1:
        rep[None] = as_expr(r2, ignore_other=True) * copy.deepcopy(r1[None])
        if None in r2:
            rep[None] += copy.deepcopy(r1[None])*copy.deepcopy(r2[None])
    if None in r2:
        if None in rep:
            rep[None] += as_expr(r1, ignore_other=True) * copy.deepcopy(r2[None])
        else:
            rep[None]  = as_expr(r1, ignore_other=True) * copy.deepcopy(r2[None])
    #
    # Merge the _VarData maps
    #
    if -1 in r1:
        rep[-1] = r1[-1]
    if -1 in r2:
        if -1 in rep:
            rep[-1].update(r2[-1])
        else:
            rep[-1] = r2[-1]
    #
    # Return the canonical repn
    #
    return rep

def collect_variables(exp, idMap):
    if exp.is_expression():
        ans = {}
        if exp.__class__ is expr._ProductExpression:
            for subexp in exp._numerator:
                ans.update(collect_variables(subexp, idMap))
            for subexp in exp._denominator:
                ans.update(collect_variables(subexp, idMap))
        else:
            # This is fragile: we assume that all other expression
            # objects "play nice" and just use the _args member.
            for subexp in exp._args:
                ans.update(collect_variables(subexp, idMap))
        return ans
    elif exp.is_fixed():
        # NB: is_fixed() returns True for constants and variables with
        # fixed values
        return {}
    elif exp.__class__ is _VarData or exp.__class__ is _VarDataWithDomain or exp.type() is Var:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        return { key : exp }
    else:
        raise ValueError("Unexpected expression type: "+str(exp))

#
# Temporary canonical expressions
#
# temp_const = { 0: {None:0.0} }
# temp_var = { 1: {GeneralCanonicalRepn({None:1}):1.0} }
# temp_nonl = { None: None }

#
# Internal function for collecting canonical representation, which is
# called recursively.
#
def collect_general_canonical_repn(exp, idMap, compute_values):
#     global temp_const
#     global temp_var
#     global temp_nonl
    temp_const = { 0: {None:0.0} }
    temp_var = { 1: {GeneralCanonicalRepn({None:1}):1.0} }
    temp_nonl = { None: None }
    exp_type = type(exp)

    #
    # Constant
    #
    if exp.is_fixed():
        if compute_values:
            temp_const[0][None] = value(exp)
        else:
            temp_const[0][None] = exp
        return temp_const
    #
    # Expression
    #
    elif exp.is_expression():

        #
        # Sum
        #
        if exp_type is expr._SumExpression:
            if exp._const != 0.0:
                repn = { 0: {None:exp._const} }
            else:
                repn = {}
            for i in xrange(len(exp._args)):
                repn = repn_add(repn, collect_general_canonical_repn(exp._args[i], idMap, compute_values), coef=exp._coef[i] )
            return repn
        #
        # Product
        #
        elif exp_type is expr._ProductExpression:
            #
            # Iterate through the denominator.  If they aren't all constants, then
            # simply return this expresion.
            #
            denom=1.0
            for e in exp._denominator:
                if e.is_fixed():
                    denom *= e()
                else:
                    temp_nonl[None] = exp
                    return temp_nonl
                if denom == 0.0:
                    print("Divide-by-zero error - offending sub-expression:")
                    e.pprint()
                    raise ZeroDivisionError
            #
            # OK, the denominator is a constant.
            #
            repn = { 0: {None:exp._coef / denom} }
            for e in exp._numerator:
                repn = repn_mult(repn, collect_general_canonical_repn(e, idMap, compute_values))
            return repn
        #
        # Power Expression
        #
        elif exp_type is expr._PowExpression:
            if exp.polynomial_degree() is None:
                raise TypeError("Unsupported general power expression: "+str(exp._args))
                
            # If this is of the form EXPR**1, we can just get the
            # representation of EXPR
            if exp._args[1] == 1:
                return collect_general_canonical_repn(exp._args[0], idMap, compute_values)
            # The only other way to get a polynomial expression is if
            # exp=EXPR**p where p is fixed a nonnegative integer.  We
            # can expand this expression and generate a canonical
            # representation from there.  If p=0, this expression is
            # constant (and is processed by the is_fixed code above
            # NOTE: There is no check for 0**0
            return collect_general_canonical_repn(
                        reduce( lambda x,y: x*y, [exp._args[0]]*int(exp._args[1]), 1.0 ),
                        idMap,
                        compute_values)
        elif exp_type is expr.Expr_if:
            if exp._if.is_fixed():
                if exp._if():
                    return collect_general_canonical_repn(exp._then, idMap, compute_values)
                else:
                    return collect_general_canonical_repn(exp._else, idMap, compute_values)
            else:
                temp_nonl[None] = exp
                return temp_nonl
        #
        # Expression (the component)
        # 
        elif exp_type is _ExpressionData or exp.type() is Expression:
            return collect_general_canonical_repn(exp.value, idMap, compute_values)
        #
        # ERROR
        #
        else:
            raise ValueError("Unsupported expression type: "+str(exp))
    #
    # Variable
    #
    elif exp_type is _VarData or exp.__class__ is _VarDataWithDomain or exp.type() is Var:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        temp_var = { -1: {key:exp}, 1: {GeneralCanonicalRepn({key:1}):1.0} }
        return temp_var
    #
    # Connector
    #
    elif exp_type is _ConnectorValue or exp.type() is Connector:
        # Silently omit constraint...  The ConnectorExpander should
        # expand this constraint into indvidual constraints that
        # reference "real" variables.
        return {}
    #
    # ERROR
    #
    else:
        raise ValueError("Unexpected expression (type %s): " %
                         ( type(exp).__name__, str(exp) ))

##############################################################################
##############################################################################
#### CLASSES AND ROUTINES FOR LINEAR CANONICAL REPRESENTATIONS START NOW #####
##############################################################################
##############################################################################

# not a named tuple, because we want the fields mutable.
class LinearCanonicalRepn(object):

    __slots__ = ['variables', 'constant', 'linear']

    def __init__(self, **kwds):

        # a single scalar (int or float)
        self.constant = None 

        # a tuple of coefficents - one-for-one with self.variables
        self.linear = None    

        # a tuple of _VarValues
        self.variables = None

    def __getstate__(self):
        """
        This method is required because this class uses slots.
        """
        return dict((i, getattr(self, i, None)) for i in LinearCanonicalRepn.__slots__)

    def __setstate__(self, state):
        """
        This method is required because this class uses slots.
        """
        for (slot_name, value) in iteritems(state):
            setattr(self, slot_name, value)

    def __str__(self):
        ordered_vars = None
        if self.variables is not None:
            ordered_vars = sorted( 
                (v.cname(True), i) for i,v in enumerate(self.variables) )
        tmp_str = str(self.constant) if (self.constant) else ("")
        tmp_str += (" + ") if (self.constant and self.variables) else ("")
        tmp_str += (" + ".join("%s*%s"%(self.linear[i], v) for v,i in ordered_vars)) if (self.variables) else ("")
        return "LinearCanonical{ %s }" % (tmp_str)

def _collect_linear_sum(exp, idMap, multiplier, coef, varmap, compute_values):

    coef[None] += multiplier * exp._const  # None is the constant term in the coefficient map.
    
    arg_coef_iterator = exp._coef.__iter__()
    for arg in exp._args:
        # an arg can be anything - a product, a variable, whatever.

        # Special case... <sigh>
        if (arg.__class__ is _VarData or arg.__class__ is _VarDataWithDomain) and arg.fixed is False:
            # save an expensive recursion - this is by far the most common case.
            id_ = id(arg)
            if id_ in idMap[None]:
                key = idMap[None][id_]
            else:
                key = len(idMap) - 1
                idMap[None][id_] = key
                idMap[key] = arg
            #
            varmap[key]=arg
            if key in coef:
                coef[key] += multiplier * six.next(arg_coef_iterator)
            else:
                coef[key] = multiplier * six.next(arg_coef_iterator)
        else:
            _linear_collectors[arg.__class__](arg, idMap, multiplier * six.next(arg_coef_iterator), coef, varmap, compute_values)

def _collect_linear_prod(exp, idMap, multiplier, coef, varmap, compute_values):

    multiplier *= exp._coef
    _coef = { None : 0 }
    _varmap = {}
    
    for subexp in exp._denominator:
        if compute_values:
            x = value(subexp) # only have constants/fixed terms in the denominator.
            if x == 0:
                buf = StringIO()
                subexp.pprint(buf)
                logger.error("Divide-by-zero: offending sub-expression:\n   " + buf)
                raise ZeroDivisionError
            multiplier /= x
        else:
            multiplier /= subexp

    for subexp in exp._numerator:
        if _varmap:
            if compute_values:
                multiplier *= value(subexp)
            else:
                multiplier *= subexp
        else:
            _linear_collectors[subexp.__class__](subexp, idMap, 1, _coef, _varmap, compute_values)
            if not _varmap:
                multiplier *= _coef[None]
                _coef[None] = 0

    if _varmap:
        for key, val in iteritems(_coef):
            if key in coef:
                coef[key] += multiplier * val
            else:
                coef[key] = multiplier * val
        varmap.update(_varmap)
    else:
        # constant expression; i.e. 1/x
        coef[None] += multiplier
        

def _collect_linear_pow(exp, idMap, multiplier, coef, varmap, compute_values):

    if exp.is_fixed():
        if compute_values:
            coef[None] += multiplier * value(exp)
        else:
            coef[None] += multiplier * exp
    elif value(exp._args[1]) == 1:
        arg = exp._args[0]
        _linear_collectors[arg.__class__](arg, idMap, multiplier, coef, varmap, compute_values)
    else:
        raise TypeError( "Unsupported power expression: "+str(exp._args) )

def _collect_branching_expr(exp, idMap, multiplier, coef, varmap, compute_values):

    if exp._if.is_fixed():
        if exp._if():
            arg = exp._then
            _linear_collectors[arg.__class__](arg, idMap, multiplier, coef, varmap, compute_values)
        else:
            arg = exp._else
            _linear_collectors[arg.__class__](arg, idMap, multiplier, coef, varmap, compute_values)
    else:
        raise TypeError( "Unsupported dynamic If-Then-Else expression: "+str(exp._args) )

def _collect_linear_var(exp, idMap, multiplier, coef, varmap, compute_values):

    if exp.is_fixed():
        if compute_values:
            coef[None] += multiplier * value(exp)
        else:
            coef[None] += multiplier * exp
    else:
        id_ = id(exp)
        if id_ in idMap[None]:
            key = idMap[None][id_]
        else:
            key = len(idMap) - 1
            idMap[None][id_] = key
            idMap[key] = exp
        #
        if key in coef:
            coef[key] += multiplier
        else:
            coef[key] = multiplier
        varmap[key] = exp

def _collect_linear_const(exp, idMap, multiplier, coef, varmap, compute_values):
    if compute_values:
        coef[None] += multiplier * value(exp)
    else:
        coef[None] += multiplier * exp

def _collect_linear_connector(exp, idMap, multiplier, coef, varmap, compute_values):

    pass

def _collect_linear_intrinsic(exp, idMap, multiplier, coef, varmap, compute_values):

    if exp.is_fixed():
        if compute_values:
            coef[None] += multiplier * value(exp)
        else:
            coef[None] += multiplier * exp
    else:
        raise TypeError( "Unsupported intrinsic expression: %s: %s" % (exp, str(exp._args)) )

def _collect_identity(exp, idMap, multiplier, coef, varmap, compute_values):
    exp = exp.value
    if exp.is_fixed():
        if compute_values:
            coef[None] += multiplier * value(exp)
        else:
            coef[None] += multiplier * exp
    else:
        _linear_collectors[exp.__class__](exp, idMap, multiplier, coef, varmap, compute_values)


_linear_collectors = {
    expr._SumExpression               : _collect_linear_sum,
    expr._ProductExpression           : _collect_linear_prod,
    expr._PowExpression               : _collect_linear_pow,
    expr._IntrinsicFunctionExpression : _collect_linear_intrinsic,
    _ConnectorValue         : _collect_linear_connector,
    SimpleConnector         : _collect_linear_connector,
    expr.Expr_if            : _collect_branching_expr,
    param._ParamData        : _collect_linear_const,
    param.SimpleParam       : _collect_linear_const,
    param.Param             : _collect_linear_const,
    _VarData                : _collect_linear_var,
    _VarDataWithDomain      : _collect_linear_var,
    SimpleVar               : _collect_linear_var,
    Var                     : _collect_linear_var,
    _ExpressionData         : _collect_identity,
    SimpleExpression        : _collect_identity,
    Expression              : _collect_identity
    }

def collect_linear_canonical_repn(exp, idMap, compute_values=True):

    idMap.setdefault(None, {})
    coef = { None : 0 }
    varmap = {}
    try:
        _linear_collectors[exp.__class__](exp, idMap, 1, coef, varmap, compute_values)
    except KeyError:
        raise
        raise ValueError( "Unexpected expression (type %s): %s" %
                          (type(exp).__name__, str(exp)) )
    return coef, varmap

#########################################################################
#########################################################################
#### ROUTINES OPERATING ON BOTH LINEAR AND GENERAL CANONICAL REPNS  #####
#########################################################################
#########################################################################

def generate_canonical_repn(exp, idMap=None, compute_values=True):

    degree = exp.polynomial_degree()

    if idMap is None:
        idMap = {}
    idMap.setdefault(None, {})

    if degree == 0:
        ans = LinearCanonicalRepn()
        ans.constant = value(exp)
        return ans

    elif degree == 1:
        # varmap is a map from the variable id() to a _VarData.
        # coef is a map from the variable id() to its coefficient.
        coef, varmap = collect_linear_canonical_repn(exp, idMap, compute_values)
        ans = LinearCanonicalRepn()
        if None in coef:
            val = coef.pop(None)
            if val != 0.0:
                ans.constant = val

        # the six module is inefficient in terms of wrapping iterkeys
        # and itervalues, in the context of Python 2.7. use the native
        # dictionary methods where possible.
        if using_py3:
            ans.linear = tuple( itervalues(coef) )
            ans.variables = tuple(varmap[var_hash] for var_hash in iterkeys(coef) )
        else:
            ans.linear = tuple( coef.itervalues() )
            ans.variables = tuple(varmap[var_hash] for var_hash in coef.iterkeys() )
        return ans

    # **Py3k: degree > 1 comparision will error if degree is None
    elif degree and degree > 1:
        ans = collect_general_canonical_repn(exp, idMap, compute_values)
        if 1 in ans:
            linear_terms = {}
            for key, coef in iteritems(ans[1]):
                linear_terms[list(key.keys())[0]] = coef
            ans[1] = linear_terms
        return GeneralCanonicalRepn(ans)
    else:
        return GeneralCanonicalRepn(
            { None: exp, -1 : collect_variables(exp, idMap) } )

def canonical_is_constant(repn):
    """Return True if the canonical representation is a constant expression"""
    if isinstance(repn, dict):
        return max(i for i in iterkeys(repn) if not i is None) == 0 and None not in repn
    else:
        return (repn.constant is not None) and (repn.linear is None)

def canonical_is_nonlinear(repn):
    """Return True if the canonical representation is a nonlinear expression"""
    if isinstance(repn, dict):
        return max(i for i in iterkeys(repn) if not i is None) > 1 or None in repn
    else:
        return False

def canonical_is_linear(repn):
    """Return True if the canonical representation is a linear expression."""
    if isinstance(repn, dict):
        return max(i for i in iterkeys(repn) if not i is None) == 1 and None not in repn
    else:
        return (repn.linear is not None)

def canonical_is_quadratic(repn):
    """Return True if the canonical representation is a quadratic expression."""
    if isinstance(repn, dict):
        return max(i for i in iterkeys(repn) if not i is None) == 2 and None not in repn
    else:
        return False

def canonical_degree(repn):
    if isinstance(repn, dict):
        if None in repn:
            return None
        else:
            return max(i for i in iterkeys(repn) if not i is None)
    else:
        if repn.linear is not None:
            return 1
        else:
            return 0
