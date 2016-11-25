#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from __future__ import division

__all__ = ['generate_canonical_repn', 'as_expr', 'canonical_is_constant',
           'canonical_is_linear', 'canonical_is_quadratic', 'canonical_is_nonlinear',
           'canonical_degree', 'LinearCanonicalRepn', 'GeneralCanonicalRepn']

import logging
import copy

from pyomo.core.base import Model, value
from pyomo.core.base import param
from pyomo.core.base import expr
from pyomo.core.base.numvalue import native_numeric_types
from pyomo.core.base.expression import (_ExpressionData,
                                        _GeneralExpressionData,
                                        SimpleExpression,
                                        Expression)
from pyomo.core.base.objective import (_GeneralObjectiveData,
                                       SimpleObjective)
from pyomo.core.base.connector import _ConnectorData, SimpleConnector, Connector
from pyomo.core.base.var import SimpleVar, Var, _GeneralVarData, _VarData

from pyomo.core.base.expr_pyomo4 import TreeWalkerHelper

import six
from six import iterkeys, itervalues, iteritems, StringIO
from six.moves import xrange, reduce

using_py3 = six.PY3

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
    # Stick general nonlinear at the end
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
    elif (exp.__class__ is _GeneralVarData) or isinstance(exp, _VarData):
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
                repn = repn_add(
                    repn,
                    collect_general_canonical_repn(exp._args[i],
                                                   idMap,
                                                   compute_values),
                    coef=exp._coef[i])
            return repn
        #
        # Product
        #
        elif exp_type is expr._ProductExpression:
            #
            # Iterate through the denominator.  If they aren't all
            # constants, then simply return this expression.
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
                repn = repn_mult(
                    repn,
                    collect_general_canonical_repn(e,
                                                   idMap,
                                                   compute_values))
            return repn
        #
        # Power Expression
        #
        elif exp_type is expr._PowExpression:
            if exp.polynomial_degree() is None:
                raise TypeError("Unsupported general power expression: "
                                +str(exp._args))

            # If this is of the form EXPR**1, we can just get the
            # representation of EXPR
            if exp._args[1] == 1:
                return collect_general_canonical_repn(exp._args[0],
                                                      idMap,
                                                      compute_values)
            # The only other way to get a polynomial expression is if
            # exp=EXPR**p where p is fixed a nonnegative integer.  We
            # can expand this expression and generate a canonical
            # representation from there.  If p=0, this expression is
            # constant (and is processed by the is_fixed code above
            # NOTE: There is no check for 0**0
            return collect_general_canonical_repn(
                        reduce( lambda x,y: x*y, [exp._args[0]]*int(value(exp._args[1])), 1.0 ),
                        idMap,
                        compute_values)
        elif exp_type is expr.Expr_if:
            if exp._if.is_fixed():
                if exp._if():
                    return collect_general_canonical_repn(exp._then,
                                                          idMap,
                                                          compute_values)
                else:
                    return collect_general_canonical_repn(exp._else,
                                                          idMap,
                                                          compute_values)
            else:
                temp_nonl[None] = exp
                return temp_nonl
        #
        # Expression (the component)
        # (faster check)
        elif isinstance(exp, _ExpressionData):
            return collect_general_canonical_repn(exp.expr,
                                                  idMap,
                                                  compute_values)
        #
        # ERROR
        #
        else:
            raise ValueError("Unsupported expression type: "+str(exp))
    #
    # Variable
    #
    elif (exp.__class__ is _GeneralVarData) or isinstance(exp, _VarData):
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
    elif exp_type is _ConnectorData or exp.type() is Connector:
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

#
# A pure abstract class that defines an interface
# for linear canonical representations
#
class LinearCanonicalRepn(object):

    #
    # Abstract Interface
    #

    @property
    def variables(self):
        """A tuple of variables comprising the constraint body."""
        raise NotImplementedError

    @property
    def coefficients(self):
        """A tuple of coefficients associated with the variables."""
        raise NotImplementedError

    # for backwards compatibility
    linear=coefficients

    @property
    def constant(self):
        """The constant value associated with the constraint body."""
        raise NotImplementedError

# not a named tuple, because we want the fields mutable.
class coopr3_CompiledLinearCanonicalRepn(LinearCanonicalRepn):

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
        return dict((i, getattr(self, i, None)) for i in CompiledLinearCanonicalRepn.__slots__)

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
                (v.name, i) for i,v in enumerate(self.variables) )
        tmp_str = str(self.constant) if (self.constant) else ("")
        tmp_str += (" + ") if (self.constant and self.variables) else ("")
        tmp_str += (" + ".join("%s*%s"%(self.linear[i], v) for v,i in ordered_vars)) if (self.variables) else ("")
        return "LinearCanonical{ %s }" % (tmp_str)

CompiledLinearCanonicalRepn_Pool = []

class pyomo4_CompiledLinearCanonicalRepn(LinearCanonicalRepn):
    __slots__ = ['variables', 'constant', 'linear']

    def __init__(self):
        self.variables = []
        self.linear = {}
        self.constant = 0.

    def __iadd__(self, other):
        _type = other.__class__
        if _type in native_numeric_types:
            self.constant += other
        elif _type is CompiledLinearCanonicalRepn:
            self.constant += other.constant
            for v in other.variables:
                _id = id(v)
                if _id in self.linear:
                    self.linear[_id] += other.linear[_id]
                else:
                    self.variables.append(v)
                    self.linear[_id] = other.linear[_id]
            CompiledLinearCanonicalRepn_Pool.append(other)
        elif other.is_fixed():
            self.constant += value(other)
        else:
            assert isinstance(other, _VarData)
            _id = id(other)
            if _id in self.linear:
                self.linear[_id] += 1.
            else:
                self.variables.append(other)
                self.linear[_id] = 1.
        return self

    def __imul__(self, other):
        _type = other.__class__
        if _type in native_numeric_types:
            pass
        elif _type is CompiledLinearCanonicalRepn:
            if other.variables:
                self, other = other, self
            assert(not other.variables)
            CompiledLinearCanonicalRepn_Pool.append(other)
            other = other.constant
        elif other.is_fixed():
            other = value(other)
        else:
            assert isinstance(other, _VarData)
            assert not self.variables
            self.variables.append(other)
            self.linear[id(other)] = self.constant
            self.constant = 0.
            return self

        if other:
            for _id in self.linear:
                self.linear[_id] *= other
        else:
            self.linear = {}
            self.variables = []
        self.constant *= other
        return self


    def X__add__(self, other):
        assert(type(other) == CompiledLinearCanonicalRepn)
        #if len(self.variables) < len(other.variables):
        #    self, other = other, self
        self.constant += other.constant
        for v in other.variables:
            _id = id(v)
            if _id in self.linear:
                self.linear[_id] += other.linear[_id]
            else:
                self.variables.append(v)
                self.linear[_id] = other.linear[_id]

        CompiledLinearCanonicalRepn_Pool.append(other)
        return self

    def X__radd__(self, other):
        assert(type(other) in native_numeric_types)
        self.constant += other
        return self

    def X__mul__(self, other):
        assert(type(other) == CompiledLinearCanonicalRepn)
        if len(self.variables) < len(other.variables):
            self, other = other, self
        assert(len(other.variables) == 0)

        mul = other.constant
        coef = self.linear
        self.constant *= mul
        for _v in coef:
            coef[_v] *= mul

        CompiledLinearCanonicalRepn_Pool.append(other)
        return self

    def X__div__(self, other):
        assert(type(other) == CompiledLinearCanonicalRepn)
        assert(len(other.variables) == 0)

        mul = other.constant
        coef = self.linear
        self.constant /= mul
        for _v in coef:
            coef[_v] /= mul

        CompiledLinearCanonicalRepn_Pool.append(other)
        return self

    def X__pow__(self, other):
        assert(type(other) == CompiledLinearCanonicalRepn)
        assert(len(other.variables) == 0)

        if other.constant == 0:
            self.__init__(1.,None)
        else:
            assert(other.constant == 1)

        CompiledLinearCanonicalRepn_Pool.append(other)
        return self

    def X__neg__(self):
        self.constant *= -1.
        coef = self.linear
        for _v in coef:
            coef[_v] *= -1.
        return self


def _collect_linear_sum(exp, idMap, multiplier, coef, varmap, compute_values):

    coef[None] += multiplier * exp._const  # None is the constant term in the coefficient map.

    arg_coef_iterator = exp._coef.__iter__()
    for arg in exp._args:
        # an arg can be anything - a product, a variable, whatever.

        # Special case... <sigh>
        if ((arg.__class__ is _GeneralVarData) or isinstance(arg, _VarData)) and (not arg.fixed):
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
    exp = exp.expr
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
    _ConnectorData          : _collect_linear_connector,
    SimpleConnector         : _collect_linear_connector,
    expr.Expr_if            : _collect_branching_expr,
    param._ParamData        : _collect_linear_const,
    param.SimpleParam       : _collect_linear_const,
    param.Param             : _collect_linear_const,
    _GeneralVarData         : _collect_linear_var,
    SimpleVar               : _collect_linear_var,
    Var                     : _collect_linear_var,
    _GeneralExpressionData  : _collect_identity,
    SimpleExpression        : _collect_identity,
    _GeneralObjectiveData  : _collect_identity,
    SimpleObjective        : _collect_identity
    }

def collect_linear_canonical_repn(exp, idMap, compute_values=True):

    idMap.setdefault(None, {})
    coef = { None : 0 }
    varmap = {}
    try:
        _linear_collectors[exp.__class__](exp, idMap, 1, coef, varmap, compute_values)
    except KeyError:
        if isinstance(exp, _VarData):
            _collect_linear_var(exp, idMap, 1, coef, varmap, compute_values)
        elif isinstance(exp, _ExpressionData):
            _collect_identity(exp, idMap, 1, coef, varmap, compute_values)
        else:
            raise ValueError( "Unexpected expression (type %s): %s" %
                              (type(exp).__name__, str(exp)) )
    return coef, varmap

#########################################################################
#########################################################################
#### ROUTINES OPERATING ON BOTH LINEAR AND GENERAL CANONICAL REPNS  #####
#########################################################################
#########################################################################

def coopr3_generate_canonical_repn(exp, idMap=None, compute_values=True):
    degree = exp.polynomial_degree()

    if idMap is None:
        idMap = {}
    idMap.setdefault(None, {})

    if degree == 0:
        ans = CompiledLinearCanonicalRepn()
        ans.constant = value(exp)
        return ans

    elif degree == 1:
        # varmap is a map from the variable id() to a _VarData.
        # coef is a map from the variable id() to its coefficient.
        coef, varmap = collect_linear_canonical_repn(exp, idMap, compute_values)
        ans = CompiledLinearCanonicalRepn()
        if None in coef:
            val = coef.pop(None)
            if type(val) not in [int,float] or val != 0.0:
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


_stack = [[0,0,0,0,0,pyomo4_CompiledLinearCanonicalRepn()]]

def pyomo4_generate_canonical_repn(exp, idMap=None, compute_values=True):
    # A **very** special case
    if TreeWalkerHelper.typeList.get(exp.__class__,0) == 4: # _LinearExpression:
        ans = CompiledLinearCanonicalRepn()

        # old format
        ans.constant = exp._const
        ans.variables = list( exp._args )
        _l = exp._coef
        ans.linear = [_l[id(v)] for v in exp._args]

        if idMap:
            if None not in idMap:
                idMap[None] = {}
            _test = idMap[None]
            _key = len(idMap) - 1
            for v in exp._args:
                if id(v) not in _test:
                    _test[id(v)] = _key
                    idMap[_key] = v
                    _key += 1
        return ans
    else:
        degree = exp.polynomial_degree()

    if degree == 1:
        _typeList = TreeWalkerHelper.typeList
        _stackMax = len(_stack)
        _stackIdx = 0
        _stackPtr = _stack[0]

        _stackPtr[0] = exp
        try:
            _stackPtr[1] = exp._args
        except AttributeError:
            ans = CompiledLinearCanonicalRepn()
            ans.variables.append(exp)
            # until we can redefine CompiledLinearCanonicalRepn, restore
            # old format
            #ans.linear[id(exp)] = 1.
            ans.linear = [1.]
            return ans
        try:
            _stackPtr[2] = _type = _typeList[exp.__class__]
            if _stackPtr[2] == 2:
                _stackPtr[5].constant = 1.
        except KeyError:
            _stackPtr[2] = _type = 0
        _stackPtr[3] = len(_stackPtr[1])
        _stackPtr[4] = 0
        #_stackPtr[5] = CompiledLinearCanonicalRepn()

        if _type == 4: # _LinearExpression
            _stackPtr[4] = _stackPtr[3]
            _stackPtr[5].constant = exp._const
            _stackPtr[5].linear = dict(exp._coef)
            _stackPtr[5].variables = list(exp._args)

        while 1: # Note: 1 is faster than True for Python 2.x
            if _stackPtr[4] < _stackPtr[3]:
                _sub = _stackPtr[1][_stackPtr[4]]
                _stackPtr[4] += 1
                _test = _sub.__class__ in native_numeric_types
                if _test or not _sub.is_expression():
                    if not _test and _sub.is_fixed():
                        _sub = value(_sub)
                        _test = 1 # True
                    if _test:
                        if _type == 2:
                            _stackPtr[5].constant *= _sub
                            _l = _stackPtr[5].linear
                            if _l:
                                for _id in _l:
                                    _l[_id] *= _sub
                        elif _type == 1:
                            _stackPtr[5].constant += _sub
                        elif _type == 3:
                            _stackPtr[5].constant = -1. * _sub
                        else:
                            raise RuntimeError("HELP")
                    else:
                        _id = id(_sub)
                        if _type == 2:
                            _lcr = _stackPtr[5]
                            _lcr.variables.append(_sub)
                            _lcr.linear[_id] = _lcr.constant
                            _lcr.constant = 0
                        elif _type == 1:
                            if _id in _stackPtr[5].linear:
                                _stackPtr[5].linear[_id] += 1.
                            else:
                                _stackPtr[5].variables.append(_sub)
                                _stackPtr[5].linear[_id] = 1.
                        elif _type == 3:
                            _lcr = _stackPtr[5]
                            _lcr.variables.append(_sub)
                            _lcr.linear[_id] = -1.
                        else:
                            raise RuntimeError("HELP")
                else:
                    _stackIdx += 1
                    if _stackMax == _stackIdx:
                        _stackMax += 1
                        _stack.append([0,0,0,0,0, CompiledLinearCanonicalRepn()])
                    _stackPtr = _stack[_stackIdx]

                    _stackPtr[0] = _sub
                    _stackPtr[1] = _sub._args
                    #_stackPtr[2] = _type = _typeList.get(_sub.__class__, 0)
                    #if _type == 2:
                    #    _stackPtr[5].constant = 1.
                    try:
                        _stackPtr[2] = _type = _typeList[_sub.__class__]
                        if _type == 2:
                            _stackPtr[5].constant = 1.
                    except KeyError:
                        _stackPtr[2] = _type = 0
                    _stackPtr[3] = len(_stackPtr[1])
                    _stackPtr[4] = 0
                    #_stackPtr[5] = CompiledLinearCanonicalRepn()

                    if _type == 4: # _LinearExpression
                        _stackPtr[4] = _stackPtr[3]
                        _stackPtr[5].constant = _sub._const
                        _stackPtr[5].linear = dict(_sub._coef)
                        _stackPtr[5].variables = list(_sub._args)
            else:
                old = _stackPtr[5]
                if not _type:
                    old.constant = _stackPtr[0]._apply_operation(old.variables)
                    old.variables = []

                if _stackIdx == 0:
                    ans = CompiledLinearCanonicalRepn()
                    ans.variables, old.variables = old.variables, ans.variables
                    ans.linear, old.linear = old.linear, ans.linear
                    ans.constant, old.constant = old.constant, ans.constant
                    # until we can redefine CompiledLinearCanonicalRepn, restore
                    # old format
                    ans.linear = [ans.linear[id(v)] for v in ans.variables]

                    if idMap:
                        if None not in idMap:
                            idMap[None] = {}
                        _test = idMap[None]
                        _key = len(idMap) - 1
                        for v in ans.variables:
                            if id(v) not in _test:
                                _test[id(v)] = _key
                                idMap[_key] = v
                                _key += 1
                    return ans

                _stackIdx -= 1
                _stackPtr = _stack[_stackIdx]
                new = _stackPtr[5]
                _type = _stackPtr[2]
                if _type == 1:
                    new.constant += old.constant
                    _nl = new.linear
                    # Note: append the variables in the order that they
                    # were originally added to the CompiledLinearCanonicalRepn.
                    # This keeps things deterministic.
                    for v in old.variables:
                        _id = id(v)
                        if _id in _nl:
                            _nl[_id] += old.linear[_id]
                        else:
                            new.variables.append(v)
                            _nl[_id] = old.linear[_id]
                    old.constant = 0.
                    old.variables = []
                    old.linear = {}
                elif _type == 2:
                    if old.variables:
                        old.variables, new.variables = new.variables, old.variables
                        old.linear, new.linear = new.linear, old.linear
                        old.constant, new.constant = new.constant, old.constant
                    _c = old.constant
                    new.constant *= _c
                    _nl = new.linear
                    for _id in _nl:
                        _nl[_id] *= _c
                    old.constant = 0.
                elif _type == 3:
                    old.variables, new.variables = new.variables, old.variables
                    old.linear, new.linear = new.linear, old.linear
                    new.constant = -1 * old.constant
                    old.constant = 0.
                    _nl = new.linear
                    for _id in _nl:
                        _nl[_id] *= -1
                else:
                    raise RuntimeError("HELP")

    elif degree == 0:
        if CompiledLinearCanonicalRepn_Pool:
            ans = CompiledLinearCanonicalRepn_Pool.pop()
            ans.__init__()
        else:
            ans = CompiledLinearCanonicalRepn()
        ans.constant = value(exp)
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

import pyomo.core.base.expr_common as common
def generate_canonical_repn(exp, idMap=None, compute_values=True):
    if common.mode is common.Mode.coopr3_trees:
        globals()['CompiledLinearCanonicalRepn'] = coopr3_CompiledLinearCanonicalRepn
        return coopr3_generate_canonical_repn(exp, idMap, compute_values)
    elif common.mode is common.Mode.pyomo4_trees:
        globals()['CompiledLinearCanonicalRepn'] = pyomo4_CompiledLinearCanonicalRepn
        return pyomo4_generate_canonical_repn(exp, idMap, compute_values)
    else:
        raise RuntimeError("Unrecognized expression tree mode")

if common.mode is common.Mode.coopr3_trees:
    CompiledLinearCanonicalRepn = coopr3_CompiledLinearCanonicalRepn
elif common.mode is common.Mode.pyomo4_trees:
    CompiledLinearCanonicalRepn = pyomo4_CompiledLinearCanonicalRepn
else:
    raise RuntimeError("Unrecognized expression tree mode")
