#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from __future__ import division

#__all__ = ( 'log', 'log10', 'sin', 'cos', 'tan', 'cosh', 'sinh', 'tanh',
#            'asin', 'acos', 'atan', 'exp', 'sqrt', 'asinh', 'acosh', 
#            'atanh', 'ceil', 'floor' )

import copy
import logging
import math
import sys
from six import advance_iterator
from weakref import ref

logger = logging.getLogger('pyomo.core')

try:
    from sys import getrefcount
    _getrefcount_available = True
except ImportError:
    logger.warning(
        "This python interpreter does not support sys.getrefcount()\n"
        "Pyomo cannot automatically guarantee that expressions do not become\n"
        "entangled (multiple expressions that share common subexpressions).\n")
    _getrefcount_available = False

from six import StringIO, next
from six.moves import xrange
try:
    basestring
except:
    basestring = str

#from pyomo.core.plugin import *

from pyomo.core.base.component import Component
#from pyomo.core.base.plugin import *
from pyomo.core.base.numvalue import *
from pyomo.core.base.numvalue import native_numeric_types
from pyomo.core.base.var import _VarData
from pyomo.core.base.param import _ParamData
from pyomo.core.base import expr_common as common
import pyomo.core.base.expr_common
from pyomo.core.base.expr_common import \
    ensure_independent_trees as safe_mode, bypass_backreference, \
    _add, _sub, _mul, _div, _pow, _neg, _abs, _inplace, _unary, \
    _radd, _rsub, _rmul, _rdiv, _rpow, _iadd, _isub, _imul, _idiv, _ipow, \
    _lt, _le, _eq

_stack = []


def _const_to_string(*args):
    args[1].write("%s" % args[0])


def clone_expression(exp):
    if exp.__class__ in native_numeric_types:
        return exp
    if exp.is_expression():
        # It is important that this function calls the clone method not
        # __copy__.  __copy__ and clone are the same for all classes
        # that advertise "is_expression = True" except the
        # _ExpressionData class (in which case clone does nothing so
        # that the object remains persistent when generating
        # expressions)
        return exp.clone()
    else:
        return exp

def chainedInequalityErrorMessage(msg=None):
    if msg is None:
        msg = "Nonconstant relational expression used in an "\
              "unexpected Boolean context."
    buf = StringIO()
    generate_relational_expression.chainedInequality.to_string(buf)
    # We are about to raise an exception, so it's OK to reset chainedInequality
    generate_relational_expression.chainedInequality = None
    msg += """
The inequality expression:
    %s
contains non-constant terms (variables) appearing in a Boolean context, e.g.:
    if expression <= 5:
This is generally invalid.  If you want to obtain the Boolean value of
the expression based on the current variable values, explicitly evaluate
the expression, e.g.:
    if value(expression) <= 5:
or
    if value(expression <= 5):
""" % ( buf.getvalue().strip(), )
    return msg


def identify_variables(expr, include_fixed=True):
    if expr.is_expression():
        if type(expr) is _ProductExpression:
            for arg in expr._numerator:
                for var in identify_variables(arg, include_fixed):
                    yield var
            for arg in expr._denominator:
                for var in identify_variables(arg, include_fixed):
                    yield var
        elif type(expr) is _ExternalFunctionExpression:
            for arg in expr._args:
                if isinstance(arg, basestring):
                    continue
                for var in identify_variables(arg, include_fixed):
                    yield var
        else:
            for arg in expr._args:
                for var in identify_variables(arg, include_fixed):
                    yield var
    elif include_fixed or not expr.is_fixed():
        if isinstance(expr, _VarData):
            yield expr


class _ExpressionBase(NumericValue):
    """An object that defines a mathematical expression that can be evaluated"""

    __slots__ = ( '__weakref__', '_args' ) + \
                ( ('_parent_expr',) if safe_mode else () )
    PRECEDENCE = 10

    def __init__(self, args):
        self._args = args
        if safe_mode:
            self._parent_expr = None

    def __copy__(self):
        """Clone this object using the specified arguments"""
        return self.__class__( self._args.__class__(
            (clone_expression(a) for a in self._args) ))

    def __getstate__(self):
        state = super(_ExpressionBase, self).__getstate__()
        for i in _ExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def __str__(self):
        buf = StringIO()
        self.to_string(buf)
        return buf.getvalue()

    def __call__(self, exception=None):
        argList = self._args
        _stack = [ (self, argList, 0, len(argList)) ]
        _result = []
        while _stack:
            _obj, _argList, _idx, _len = _stack.pop()
            if _idx < _len:
                _sub = _argList[_idx]
                _stack.append((_obj, _argList, _idx+1, _len))
                if type(_sub) in native_numeric_types:
                    _result.append(_sub)
                    continue

                if _sub.is_expression():
                    argList = _sub._args
                    _stack.append(( _sub, argList, 0, len(argList) ))
                else:
                    _result.append(value(_sub))
            else:
                _result.append( _obj._apply_operation(_result) )
        assert(len(_result)==1)
        return _result[0]


    def clone(self):
        ans = copy.copy(self)
        ans._parent_expr = None
        return ans

    def cname(self):
        """The text name of this Expression function"""
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement cname()" % ( str(self.__class__), ))

    #
    # this method contrast with the is_fixed() method.  This method
    # returns True if the expression is an atomic constant, that is it
    # is composed exclusively of constants and immutable parameters.
    # NumericValue objects returning is_constant() == True may be
    # simplified to their numeric value at any point without warning.
    # In contrast, the is_fixed() method returns iff there are no free
    # variables within this expression (i.e., all arguments are
    # constants, params, and fixed variables).  The parameter values can
    # of course change over time, but at any point in time, they are
    # "fixed". hence, the name.
    #
    # FIXME: These need to be made non-recursive
    def is_constant(self):
        for a in self._args:
            if a.__class__ not in native_numeric_types and not a.is_constant():
                return False
        return True

    # FIXME: These need to be made non-recursive
    def is_fixed(self):
        for a in self._args:
            if a.__class__ not in native_numeric_types and not a.is_fixed():
                return False
        return True

    def is_expression(self):
        return True

    def polynomial_degree(self):
        _typeList = TreeWalkerHelper.typeList

        _stackMax = len(_stack) 
        _stackIdx = 0

        _obj = self
        _argList = _obj._args
        _idx = 0
        _len = len(_argList)
        try:
            _type = _typeList[_obj.__class__]
            _ans = 0
        except KeyError:
            _type = 0
            _ans = []
        #_ans = 0 if _type else []
        while 1: # Note: 1 is faster than True for Python 2.x
            if _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                if _sub.__class__ in native_numeric_types:
                    ans = 0
                elif not _sub.is_expression():
                    ans = 0 if _sub.is_fixed() else 1
                else:
                    if _stackIdx == _stackMax:
                        _stackMax += 1
                        _stack.append((_obj, _argList, _len, _type, _idx, _ans))
                    else:
                        _stack[_stackIdx] = \
                                _obj, _argList, _len, _type, _idx, _ans

                    _obj = _sub
                    _argList = _obj._args
                    _idx = 0
                    _len = len(_argList)
                    try:
                        _type = _typeList[_obj.__class__]
                    except KeyError:
                        _type = 0
                    _ans = 0 if _type else []

                    _stackIdx += 1
                    continue
            else:
                if not _type:
                    _ans = _obj._polynomial_degree(_ans)
                if _stackIdx == 0:
                    return _ans
                ans = _ans
                _stackIdx -= 1
                _obj, _argList, _len, _type, _idx, _ans = _stack[_stackIdx]

            #_objType = type(_obj)
            if _type is 1:#_objType is _SumExpression:
                if _ans is not None:
                    if ans is None or ans > _ans:
                        _ans = ans
            elif _type is 2:#_objType is _ProductExpression:
                if _ans is not None:
                    if ans is None:
                        _ans = None
                    else:
                        _ans += ans
            elif _type is 3:#_objType is _NegationExpression:
                _ans = ans
            elif _type is 4:#_objType is _LinearExpression_Pool
                if ans == 1:
                    _ans = ans
                    _idx = _len
            else:
                _ans.append(ans)


    def to_string(self, ostream=None, verbose=None):
        _name_buffer = {}
        if ostream is None:
            ostream = sys.stdout
        verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose

        _infix = False
        _bypass_prefix = False
        argList = self._args
        _stack = [ [self, argList, 0, len(argList), _ExpressionBase.PRECEDENCE-1] ]
        while _stack:
            _parent, _args, _idx, _len, _prec = _stack[-1]
            if _idx < _len:
                _my_precedence = _parent._precedence()
                _sub = _args[_idx]
                _stack[-1][2] += 1
                if _infix:
                    _bypass_prefix = _parent._to_string_infix(ostream, verbose)
                else:
                    if not _bypass_prefix:
                        _parent._to_string_prefix(ostream, verbose)
                    else:
                        _bypass_prefix = False
                    if _my_precedence > _prec or verbose:
                        ostream.write("( ")
                    _infix = True
                if hasattr(_sub, '_args'): # _args is a proxy for Expression
                    argList = _sub._arguments()
                    _stack.append([ _sub, argList, 0, len(argList), _my_precedence ])
                    _infix = False
                elif _sub.__class__ in native_numeric_types:
                    ostream.write(str(_sub))
                else:
                    ostream.write(_sub.cname(True, _name_buffer))
            else:
                _stack.pop()
                if _my_precedence > _prec or verbose:
                    ostream.write(" )")
       

    def simplify(self, model): #pragma:nocover
        print("""
WARNING: _ExpressionBase.simplify() has been deprecated and removed from
     Pyomo Expressions.  Please remove references to simplify() from your
     code.
""")
        return self

    def _arguments(self):
        return self._args

    def _precedence(self):
        return _ExpressionBase.PRECEDENCE

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.cname())

    def _to_string_infix(self, ostream, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            ostream.write(self._inline_operator())


class _NegationExpression(_ExpressionBase):
    __slots__ = ()

    PRECEDENCE = 3

    def cname(self):
        return 'neg'

    def _polynomial_degree(self, result):
        return result.pop()

    def _precedence(self):
        return _NegationExpression.PRECEDENCE

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.cname())
        else:
            ostream.write("-")        

    def _apply_operation(self, result):
        return -result.pop()

    

class _UnaryFunctionExpression(_ExpressionBase):
    """An object that defines a mathematical expression that can be evaluated"""

    # TODO: Unary functions should define their own subclasses so as to
    # eliminate the need for the fcn and name slots
    __slots__ = ('_fcn', '_name')

    def __init__(self, arg, name, fcn):
        """Construct an expression with an operation and a set of arguments"""
        if safe_mode:
            self._parent_expr = None
        self._args = arg
        self._fcn = fcn
        self._name = name

    def __getstate__(self):
        result = super(_UnaryFunctionExpression, self).__getstate__()
        for i in _UnaryFunctionExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def __copy__(self):
        """Clone this object using the specified arguments"""
        return self.__class__( clone_expression(self._args[0]), 
                               self._name,
                               self._fcn )

    def cname(self):
        return self._name

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.cname())

    def _polynomial_degree(self, result):
        if result.pop() == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result.pop())



class _XXX_BinaryExpression(_ExpressionBase):

    # Almost all binary expressions are linear, so to not repeat
    # ourselves, we will define the linear version here and override for
    # the (2) nonlinear cases.
    def _polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        a = result.pop()
        b = result.pop()
        if a is None or b is None:
            return None
        else:
            return a if a > b else b

    def _inline_operator(self):
        return ', '

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.cname())

    def _to_string_infix(self, ostream, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            ostream.write(self._inline_operator())



class _ExternalFunctionExpression(_ExpressionBase):
    __slots__ = ()

    def cname(self):
        return self._fcn.cname()

    def _polynomial_degree(self, result):
        if result.pop() == 0:
            return 0
        else:
            return None

    def is_constant(self):
        for arg in self._args:
            try:
                if not arg.is_constant():
                    return False
            except: 
                pass
        return True

    def is_fixed(self):
        for arg in self._args:
            try:
                if not arg.is_fixed():
                    return False
            except: 
                pass
        return True

    def x__call__(self, exception=True):
        """Evaluate the expression"""
        try:
            return self._apply_operation(
                self._evaluate_arglist(self._args, exception=exception))
        except (ValueError, TypeError):
            if exception:
                e = sys.exc_info()[1]
                logger.error("evaluating expression: %s\n    (expression: %s)",
                             str(e), str(self))
                raise
            return None

    def _evaluate_arglist(self, arglist, exception=True):
        for arg in arglist:
            try:
                yield value(arg, exception=exception)
            except Exception:
                if exception:
                    e = sys.exc_info()[1]
                    logger.error("evaluating expression: %s\n"
                                 "    (expression: %s)",
                                 str(e), str(self))
                    raise
                yield None

    def _inline_operator(self):
        return ', '


# Should this actually be a special class, or just an instance of
# _UnaryFunctionExpression (like sin, cos, etc)?
class _AbsExpression(_UnaryFunctionExpression):

    __slots__ = ()

    def __init__(self, arg):
        _UnaryFunctionExpression.__init__(self, arg, 'abs', abs)

    def _polynomial_degree(self, result):
        if result.pop() == 0:
            return 0
        else:
            return None



class _PowExpression(_ExpressionBase):

    __slots__ = ()
    PRECEDENCE = 2

    def _polynomial_degree(self, result):
        # _PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        r = result.pop()
        l = result.pop()
        if r == 0:
            if l == 0:
                return 0
            try:
                exp = int(self._args[0])
                if exp == value(self._args[1]):
                    if l is not None and exp > 0:
                        return l * exp
                    elif exp == 0:
                        return 0
            except:
                pass
        return None

    def is_fixed(self):
        if self._args[1].is_fixed():
            return self._args[0].is_fixed() or value(self._args[1]) == 0
        return False

    def _precedence(self):
        return _PowExpression.PRECEDENCE

    def cname(self):
        return 'pow'

    def _inline_operator(self):
        return '**'



class _InequalityExpression(_ExpressionBase):
    """An object that defines a series of less-than or
    less-than-or-equal expressions"""

    __slots__ = ('_strict', '_cloned_from')
    PRECEDENCE = 9

    def __init__(self, lhs, rhs, strict, cloned_from):
        """Constructor"""
        super(_InequalityExpression,self).__init__((lhs, rhs))
        self._strict = strict
        self._cloned_from = cloned_from

    def __getstate__(self):
        result = super(_InequalityExpression, self).__getstate__()
        for i in _InequalityExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def __nonzero__(self):
        if generate_relational_expression.chainedInequality is not None:
            raise TypeError(chainedInequalityErrorMessage())
        if not self.is_constant():
            generate_relational_expression.chainedInequality = self
            return True

        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _InequalityExpression.PRECEDENCE

    def __copy__(self):
        return self.__class__( clone_expression(self._args[0]),
                               clone_expression(self._args[1]),
                               copy.copy(self._strict),
                               copy.copy(self._cloned_from) )

    def _apply_operation(self, result):
        _r = result.pop()
        _l = result.pop()
        if self._strict:
            return _l < _r
        else:
            return _l <= _r

    def _inline_operator(self):
        return '  <  ' if self._strict else '  <=  '


class _EqualityExpression(_ExpressionBase):
    """An object that defines a equal-to expression"""

    __slots__ = ()
    PRECEDENCE = 9

    def __nonzero__(self):
        if generate_relational_expression.chainedInequality is not None:
            raise TypeError(chainedInequalityErrorMessage())
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _EqualityExpression.PRECEDENCE

    def _apply_operation(self, result):
        return result.pop() == result.pop()

    def _inline_operator(self):
        return '  ==  ' 


class _ProductExpression(_ExpressionBase):
    """An object that defines a product expression"""

    __slots__ = ()
    PRECEDENCE = 4

    def _precedence(self):
        return _ProductExpression.PRECEDENCE

    def _polynomial_degree(self, result):
        # NB: We can't use sum() here because None (non-polynomial)
        # overrides a numeric value (and sum() just ignores it - or
        # errors in py3k)
        a = result.pop()
        b = result.pop()
        if a is None or b is None:
            return None
        else:
            return a + b


    def cname(self):
        return 'prod'

    def _inline_operator(self):
        return ' * '

    def _apply_operation(self, result):
        return result.pop() * result.pop()


class _DivisionExpression(_ExpressionBase):
    """An object that defines a division expression"""

    __slots__ = ()
    PRECEDENCE = 3

    def _precedence(self):
        return _DivisionExpression.PRECEDENCE

    def _polynomial_degree(self, result):
        # NB: We can't use sum() here because None (non-polynomial)
        # overrides a numeric value (and sum() just ignores it - or
        # errors in py3k)
        d = result.pop()
        n = result.pop()
        if d == 0:
            return n
        else:
            return None


    def cname(self):
        return 'div'

    def _inline_operator(self):
        return ' / '

    def _apply_operation(self, result):
        _r = result.pop()
        _l = result.pop()
        return _l / _r


# a * b / ( a * b )  :or:  (a + b) / (c + d)
# a / b / c  :or: a / (b / c)
# (a / b) * c
# Note: Precedence of numerator == multiple, denominator < multiply




class _SumExpression(_ExpressionBase):
    """An object that defines a weighted summation of expressions"""

    __slots__ = ()
    PRECEDENCE = 6

    def _precedence(self):
        return _SumExpression.PRECEDENCE

    def _arguments(self):
        return self._args

    def _inline_operator(self):
        return ' + '

    def _to_string_infix(self, ostream, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            if type(self._args[1]) is _NegationExpression:
                ostream.write(' - ')
                return True
            ostream.write(self._inline_operator())

    def _polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in self._args:
            _pd = result.pop()
            if ans is not None:
                if _pd is not None:
                    if _pd > ans:
                        ans = _pd
                else:
                    ans = None
        return ans

    def _apply_operation(self, result):
        return sum(result.pop() for x in self._args)

    def cname(self):
        return 'sum'

    def __iadd__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            self._args.append(other)
            return self

        try:
            _other_expr = other.is_expression()
        except AttributeError:
            # This exception gets raised rarely in production models: to
            # get here, other must be a numeric, but non-NumericValue,
            # type that we haven't seen before.  Therefore, we can be a
            # bit inefficient.
            other = as_numeric(other)
            _other_expr = other.is_expression()
        if _other_expr:
            if safe_mode and other._parent_expr:# is not None:
                raise EntangledExpressionError(self, other)
            if _type is _SumExpression:
                self._args.extend(other._args)
                other._args = [] # for safety
                return self
            if safe_mode:
                other._parent_expr = bypass_backreference or ref(self)

        elif other.is_indexed():
            raise TypeError(
                "Argument for expression '%s' is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?" 
                % (etype, other.cname(), ) )
        elif other.is_constant():
            other = other()
        
        self._args.append(other)
        return self

    # As we do all addition "in-place", all additions are the same as
    # in-place additions.
    __add__ = __iadd__

    # Note: treating __radd__ the same as iadd is fine, as it will only be
    # called when other is not a NumericValue object ... that is, a
    # constant.  SO, we don't have to worry about preserving the
    # variable order.
    __radd__ = __iadd__

    def __isub__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            self._args.append(-other)
            return self

        try:
            _other_expr = other.is_expression()
        except AttributeError:
            # This exception gets raised rarely in production models: to
            # get here, other must be a numeric, but non-NumericValue,
            # type that we haven't seen before.  Therefore, we can be a
            # bit inefficient.
            other = as_numeric(other)
            _other_expr = other.is_expression()
        if _other_expr:
            if safe_mode and other._parent_expr:# is not None:
                raise EntangledExpressionError(self, other)
            if _type is _SumExpression:
                self._args.extend(-i for i in other._args)
                other._args = [] # for safety
                return self
            tmp = other
            other = _NegationExpression((tmp,))
            if safe_mode:
                tmp._parent_expr = bypass_backreference or ref(other)
                other._parent_expr = bypass_backreference or ref(self)
        elif other.is_indexed():
            raise TypeError(
                "Argument for expression '%s' is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?" 
                % (etype, other.cname(), ) )
        elif other.is_constant():
            other = - ( other() )
        
        self._args.append(other)
        return self


    # As we do all subtraction "in-place", all subtractions are the same as
    # in-place subtractions.
    __sub__ = __isub__

    # Note: __rsub__ of _SumExpression is very rare (basically, it needs
    # to be "non-NumericValue - _SumExpression"), and the easiest
    # thing to do is to just use the underlying NumericValue logic to
    # construct sum(other, -self)
    #def __rsub__(self, other):
    #    return other + ( - self )


class Expr_if(_ExpressionBase):
    """An object that defines a dynamic if-then-else expression"""

    __slots__ = ('_if','_then','_else')

    # **NOTE**: This class evaluates the branching "_if" expression
    #           on a number of occasions. It is important that
    #           one uses __call__ for value() and NOT bool().

    def __init__(self, IF=None, THEN=None, ELSE=None):
        """Constructor"""
        _ExpressionBase.__init__(self, None)
        
        self._if = as_numeric(IF)
        self._then = as_numeric(THEN)
        self._else = as_numeric(ELSE)
        self._args = (self._if, self._then, self._else)

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in Expr_if.__slots__:
            result[i] = getattr(self, i)
        return result

    def _arguments(self):
        return ( self._if, self._then, self._else )

    def cname(self):
        return "Expr_if"

    def is_constant(self):
        if self._if.is_constant():
            if self._if():
                return self._then.is_constant()
            else:
                return self._else.is_constant()
        else:
            return False

    def is_fixed(self):
        if self._if.is_fixed():
            if self._if():
                return self._then.is_fixed() 
            else:
                return self._else.is_fixed()
        else:
            return False

    def _polynomial_degree(self):
        _else = result.pop()
        _then = result.pop()
        _if = result.pop()
        if _if == 0:
            if self._if():
                return _then
            else:
                return _else
        else:
            return None

    def xto_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _my_precedence = self._precedence()
        ostream.write("Expr_if( if=( ")

        try:
            to_string = self._if.to_string
        except AttributeError:
            to_string = lambda o,v,p: _const_to_string(self._if,o,v,p)
        to_string( ostream, verbose, self._precedence() )

        ostream.write(" ), then=( ")

        try:
            to_string = self._then.to_string
        except AttributeError:
            to_string = lambda o,v,p: _const_to_string(self._then,o,v,p)
        to_string( ostream, verbose, self._precedence() )

        ostream.write(" ), else=( ")

        try:
            to_string = self._else.to_string
        except AttributeError:
            to_string = lambda o,v,p: _const_to_string(self._else,o,v,p)
        to_string( ostream, verbose, self._precedence() )

        ostream.write(" ) )")

    def __copy__(self):
        """Clone this object using the specified arguments"""
        return self.__class__(IF=clone_expression(self._if),
                              THEN=clone_expression(self._then),
                              ELSE=clone_expression(self._else))

    def _apply_operation(self, result):
        _e = result.pop()
        _t = result.pop()
        _i = result.pop()
        return _t if _i else _e

_LinearExpression_Pool = []

class _LinearExpression(_ExpressionBase):
    __slots__ = ('constant', 'linear')

    def __init__(self):
        if safe_mode:
            self._parent_expr = None
        self._args = []
        self.linear = {}
        self.constant = 0.

    def __getstate__(self):
        state = super(_LinearExpression, self).__getstate__()
        for i in _LinearExpression.__slots__:
           state[i] = getattr(self,i)
        # ID's do not persist from instance to instance (they are pointers!)
        # ...so we will convert them to a (temporary, but portable) index
        state['linear'] = dict( (i, self.linear[id(v)]) 
                                for i, v in enumerate(self._args) )
        return state

    def __setstate__(self, state):
        super(_LinearExpression, self).__setstate__(state)
        # ID's do not persist from instance to instance (they are pointers!)
        self.linear = dict( (id(v), self.linear[i])
                            for i, v in enumerate(self._args) )

    def __iadd__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__

        if _type is _LinearExpression:
            self.constant += other.constant
            for v in other._args:
                _id = id(v)
                if _id in self.linear:
                    self.linear[_id] += other.linear[_id]
                else:
                    self._args.append(v)
                    self.linear[_id] = other.linear[_id]
            other.constant = 0
            other._args = []
            other.linear = {}
            _LinearExpression_Pool.append(other)
            return self
        elif isinstance(other, _VarData):
            _id = id(other)
            if _id in self.linear:
                self.linear[_id] += 1.
            else:
                self._args.append(other)
                self.linear[_id] = 1.
            return self  
        elif _type in native_numeric_types:
            self.constant += other
            return self

        try:
            _is_expr = other.is_expression()
        except AttributeError:
            other = as_numeric(other)
            _is_expr = other.is_expression()
        
        if _is_expr:
            if safe_mode and other._parent_expr:
                raise EntangledExpressionError(self, other)
            return super(_LinearExpression, self).__iadd__(other)
        elif other.is_constant():
            self.constant += value(other)
        elif isinstance(other, _ParamData):
            self.constant += other
        else:
            return super(_LinearExpression, self).__iadd__(other)

        return self

    # As we do all addition "in-place", all additions are the same as
    # in-place additions.
    __add__ = __iadd__

    # Note: treating __radd__ the same as iadd is fine, as it will only be
    # called when other is not a NumericValue object ... that is, a
    # constant.  SO, we don't have to worry about preserving the
    # variable order.
    __radd__ = __iadd__

    def __isub__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            self.constant -= other
            return self

        try:
            _is_expr = other.is_expression()
        except AttributeError:
            other = as_numeric(other)
            _is_expr = other.is_expression()
        
        if _is_expr:
            if safe_mode and other._parent_expr:
                raise EntangledExpressionError(self, other)
            if _type is _LinearExpression:
                self.constant -= other.constant
                for v in other._args:
                    _id = id(v)
                    if _id in self.linear:
                        self.linear[_id] -= other.linear[_id]
                    else:
                        self._args.append(v)
                        self.linear[_id] = -1. * other.linear[_id]
                other.constant = 0
                other._args = []
                other.linear = {}
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__iadd__(other)
        elif isinstance(other, _VarData):
            _id = id(other)
            if _id in self.linear:
                self.linear[_id] -= 1.
            else:
                self._args.append(other)
                self.linear[_id] = -1.
        elif other.is_constant():
            self.constant -= value(other)
        elif isinstance(other, _ParamData):
            self.constant -= other
        else:
            return super(_LinearExpression, self).__iadd__(other)

        return self

    # As we do all subtraction "in-place", all subtractions are the same as
    # in-place subtractions.
    __sub__ = __isub__

    # Note: __rsub__ of _LinearExpressions is rare, and the easiest thing
    # to do is just negate ourselves and add.
    def __rsub__(self, other):
        self *= -1.
        return self.__iadd__(other)

    def __imul__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            self.constant *= other
            for i in self.linear:
                self.linear[i] *= other
            return self

        try:
            _is_expr = other.is_expression()
        except AttributeError:
            other = as_numeric(other)
            _is_expr = other.is_expression()

        if _is_expr:
            if safe_mode and other._parent_expr:
                raise EntangledExpressionError(self, other)

            if _type is _LinearExpression:
                if self._args:
                    if other._args:
                        return super(_LinearExpression, self).__imul__(other)
                else:
                    self, other = other, self
                self.constant *= other.constant
                for i in self.linear:
                    self.linear[i] *= other.constant
                other.constant = 0.
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__imul__(other)
        elif isinstance(other, _VarData):
            if self._args:
                return super(_LinearExpression, self).__imul__(other)
            _id = id(other)
            self._args.append(other)
            self.linear[_id] = self.constant
            self.constant = 0
        elif other.is_constant():
            other = value(other)
            self.constant *= other
            for i in self.linear:
                self.linear[i] *= other
        elif isinstance(other, _ParamData):
            self.constant *= other
            for i in self.linear:
                self.linear[i] *= other
        else:
            return super(_LinearExpression, self).__imul__(other)

        return self

    # As we do all multiplication "in-place", all multiplications are
    # the same as in-place multiplications.
    __mul__ = __imul__

    # Note: treating __rmul__ the same as imul is fine, as it will only be
    # called when other is not a NumericValue object ... that is, a
    # constant.  SO, we don't have to worry about preserving the
    # variable order.
    __rmul__ = __imul__

    def __idiv__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            self.constant /= other
            for i in self.linear:
                self.linear[i] /= other
            return self

        try:
            _is_expr = other.is_expression()
        except AttributeError:
            other = as_numeric(other)
            _is_expr = other.is_expression()

        if _is_expr:
            if _type is _LinearExpression:
                if other._args:
                    return super(_LinearExpression, self).__imul__(other)
                if safe_mode and other._parent_expr:
                    raise EntangledExpressionError(self, other)
                self.constant /= other.constant
                for i in self.linear:
                    self.linear[i] /= other.constant
                other.constant = 0.
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__imul__(other)
        elif other.is_constant():
            other = value(other)
            self.constant /= other
            for i in self.linear:
                self.linear[i] /= other
        elif isinstance(other, _ParamData):
            self.constant /= other
            for i in self.linear:
                self.linear[i] /= other
        else:
            return super(_LinearExpression, self).__imul__(other)

        return self

    # As we do all division "in-place", all divisions are
    # the same as in-place divisions.
    __mul__ = __imul__

    # Note: __rdiv__ must fall back on the underlying division operator,
    # unless this is a constant "linear expression"
    def __rdiv__(self, other):
        if self._args:
            return super(_LinearExpression, self).__rdiv__(other)
        # We should only get here if this is a constant and other is a
        # non-NumericValue object.
        self.constant = other / self.constant
        return self
            
    def __neg__(self):
        self *= -1.
        return self

    def _inline_operator(self):
        return ' + '

    def _apply_operation(self, result):
        if not self._args:
            return value(self.constant)

        result[-len(self._args):] = []
        ans = value(self.constant)
        for i,v in self._args:
            result.pop()
            ans += v() * value(self.linear[id(v)])
        return ans


def generate_expression(etype, _self, _other):
    if etype > _inplace: #and etype < 2*_inplace:#etype[0] == 'i':
        etype -= _inplace

    # Note: because generate_expression is called by the __op__ methods
    # on NumericValue, we are guaranteed that _self is a NumericValue.
    _self_expr = _self.is_expression()
    if _self_expr:
        if safe_mode and _self._parent_expr:# is not None:
            raise EntangledExpressionError(_self, _other)
            #_self = _self.clone()
            #generate_expression.clone_counter += 1
    elif _self.is_constant():
        _self = _self()

    if etype >= _unary:
        if etype == _neg:
            ans = _NegationExpression((_self,))
        elif etype == _abs:
            ans = _AbsExpression((_self,))
        if safe_mode and _self_expr:
            _self._parent_expr = bypass_backreference or ref(ans)
        return ans
    
    if _other.__class__ in native_numeric_types:
        _other_expr = False
    else:
        try:
            _other_expr = _other.is_expression()
        except AttributeError:
            # This exception gets raised rarely in production models: to
            # get here, other must be a numeric, but non-NumericValue,
            # type that we haven't seen before.  Therefore, we can be a
            # bit inefficient.
            _other = as_numeric(_other)
            _other_expr = _other.is_expression()
        if _other_expr:
            if safe_mode and _other._parent_expr:# is not None:
                raise EntangledExpressionError(_self, _other)
        elif _other.is_indexed():
            raise TypeError(
                "Argument for expression '%s' is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?" 
                % (etype, _other.cname(), ) )
        elif _other.is_constant():
            _other = _other()

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here, and the _clone_if_needed logic
        # above will make sure that we don't accidentally clobber
        # someone else's expression (fragment).
        #
        etype *= -1
        _self, _other = _other, _self
        _self_expr, _other_expr = _other_expr, _self_expr

    if etype == _mul:
        if _self_expr or _other_expr:
            ans = _ProductExpression((_self, _other))
        else:
            if isinstance(_self, _VarData):
                if isinstance(_other, _VarData):
                    ans = _ProductExpression((_self, _other))
                else:
                    if _LinearExpression_Pool:
                        ans = _LinearExpression_Pool.pop()
                    else:
                        ans = _LinearExpression()
                    ans._args.append(_self)
                    ans.linear[id(_self)] = _other
                    return ans
            elif isinstance(_other, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans.linear[id(_other)] = _self
                return ans
            else:
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans.constant = _ProductExpression((_self, _other))
                return ans
                
    elif etype == _add:
        #if _self_expr and type(_self) is _SumExpression:
        #    if _other_expr and type(_other) is _SumExpression:
        #        _self._args.extend( _other._args )
        #    else:
        #        _self._args.append(_other)
        #    ans = _self
        #elif _other_expr and type(_other) is _SumExpression:
        #    _self, _other = _other, _self
        #    _self_expr, _other_expr = _other_expr, _self_expr
        #    _self._args.append(_other)
        #    ans = _self
        #else:
        if _self_expr or _other_expr:
            ans = _SumExpression([_self, _other])
        else:
            if isinstance(_self, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_self)
                ans.linear[id(_self)] = 1.
                if isinstance(_other, _VarData):
                    ans._args.append(_other)
                    ans.linear[id(_other)] = 1.
                else:
                    ans.constant = _other
            elif isinstance(_other, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans.linear[id(_other)] = 1.
                ans.constant = _self
            else:
                ans = _SumExpression([_self, _other])
            return ans

    elif etype == _sub:
        if _self_expr or _other_expr:
            ans = _SumExpression([_self, -_other])
        else:
            if isinstance(_self, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_self)
                ans.linear[id(_self)] = 1.
                if isinstance(_other, _VarData):
                    ans._args.append(_other)
                    ans.linear[id(_other)] = -1.
                else:
                    ans.constant = -_other
            elif isinstance(_other, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans.linear[id(_other)] = -1.
                ans.constant = _self
            else:
                return _SumExpression([_self, -_other])
            return ans

        tmp = _other
        _other = _NegationExpression((tmp,))
        #if _self_expr and type(_self) is _SumExpression:
        #    _self._args.append(_other)
        #    ans = _self
        #else:
        ans = _SumExpression([_self, _other])
        if safe_mode and _other_expr:
            tmp._parent_expr = bypass_backreference or ref(_other)
        _other_expr = True
    elif etype == _div:
        ans = _DivisionExpression((_self, _other))
    elif etype == _pow:
        ans = _PowExpression((_self, _other))
    else:
        raise RuntimeError("Unknown expression type '%s'" % etype)

    if safe_mode:
        if _self_expr and ans is not _self:
            _self._parent_expr = bypass_backreference or ref(ans)
        if _other_expr and ans is not _other:
            _other._parent_expr = bypass_backreference or ref(ans)
    return ans

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_expression.clone_counter = 0


def generate_relational_expression(etype, lhs, rhs):
    cloned_from = (id(lhs), id(rhs))
    rhs_is_relational = False
    lhs_is_relational = False

    #
    # TODO: It would be nice to reduce all Constants to literals (and
    # not carry around the overhead of the NumericConstants). For
    # consistency, we will not do that yet, as many things downstream
    # would break; in particular within Constraint.add.  This way, all
    # arguments in the relational Expression's _args will be guaranteed
    # to be NumericValues (just as they are for all other Expressions).
    #
    lhs = as_numeric(lhs)
    if lhs.is_indexed():
        raise TypeError(
            "Argument for expression '%s' is an indexed numeric value "
            "specified without an index: %s\n    Is variable or parameter "
            "'%s' defined over an index that you did not specify?" 
            % ({_eq:'==',_lt:'<',_le:'<='}.get(etype, etype), 
               lhs.cname(), lhs.cname()))
    elif lhs.is_expression():
        if lhs.is_relational():
            lhs_is_relational = True

    rhs = as_numeric(rhs)
    if rhs.is_indexed():
        raise TypeError(
            "Argument for expression '%s' is an indexed numeric value "
            "specified without an index: %s\n    Is variable or parameter "
            "'%s' defined over an index that you did not specify?" 
            % ({_eq:'==',_lt:'<',_le:'<='}.get(etype, etype), 
               rhs.cname(), rhs.cname()))
    elif rhs.is_expression():
        if rhs.is_relational():
            rhs_is_relational = True

    if generate_relational_expression.chainedInequality is not None:
        raise RuntimeError("Not Implemented")

        prevExpr = generate_relational_expression.chainedInequality
        match = []
        # This is tricky because the expression could have been posed
        # with >= operators, so we must figure out which arguments
        # match.  One edge case is when the upper and lower bounds are
        # the same (implicit equality) - in which case *both* arguments
        # match, and this should be converted into an equality
        # expression.
        for i,arg in enumerate(prevExpr._cloned_from):
            if arg == cloned_from[0]:
                match.append(1)
            elif arg == cloned_from[1]:
                match.append(0)
        if len(match) == 1:
            if match[0]:
                cloned_from = prevExpr._cloned_from + (cloned_from[1],)
                lhs = prevExpr
                lhs_is_relational = True
            else:
                cloned_from = (cloned_from[0],) + prevExpr._cloned_from
                rhs = prevExpr
                rhs_is_relational = True
        elif len(match) == 2:
            # Special case: implicit equality constraint posed as a <= b <= a
            if prevExpr._strict[0] or etype == _lt:
                generate_relational_expression.chainedInequality = None
                buf = StringIO()
                prevExpr.to_string(buf)
                raise TypeError("Cannot create a compound inequality with "
                      "identical upper and lower\n\tbounds with strict "
                      "inequalities: constraint infeasible:\n\t%s and "
                      "%s < %s" % ( buf.getvalue().strip(), lhs, rhs ))
            etype = _eq
        else:
            raise TypeError(chainedInequalityErrorMessage())
        generate_relational_expression.chainedInequality = None

    if etype == _eq:
        if lhs_is_relational or rhs_is_relational:
            buf = StringIO()
            if lhs_is_relational:
                lhs.to_string(buf)
            else:
                rhs.to_string(buf)
            raise TypeError("Cannot create an EqualityExpression where "\
                  "one of the sub-expressions is a relational expression:\n"\
                  "    " + buf.getvalue().strip())
        return _EqualityExpression((lhs,rhs))
    else:
        if etype == _le:
            strict = False
        elif etype == _lt:
            strict = True
        else:
            raise ValueError("Unknown relational expression type '%s'" % etype)
        if lhs_is_relational or rhs_is_relational:
            raise RuntimeError('Not Implemented')
        elif lhs_is_relational:
            if lhs.__class__ is _InequalityExpression:
                if rhs_is_relational:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where both sub-expressions are also relational "\
                          "expressions (we support no more than 3 terms "\
                          "in an inequality expression).")
                if len(lhs._args) > 2:
                    raise ValueError("Cannot create an InequalityExpression "\
                          "with more than 3 terms.")
                lhs._args.append(rhs)
                lhs._strict.append(strict)
                lhs._cloned_from = cloned_from
                return lhs
            else:
                buf = StringIO()
                lhs.to_string(buf)
                raise TypeError("Cannot create an InequalityExpression "\
                      "where one of the sub-expressions is an equality "\
                      "expression:\n    " + buf.getvalue().strip())
        elif rhs_is_relational:
            if rhs.__class__ is _InequalityExpression:
                if len(rhs._args) > 2:
                    raise ValueError("Cannot create an InequalityExpression "\
                          "with more than 3 terms.")
                rhs._args.insert(0, lhs)
                rhs._strict.insert(0, strict)
                rhs._cloned_from = cloned_from
                return rhs
            else:
                buf = StringIO()
                rhs.to_string(buf)
                raise TypeError("Cannot create an InequalityExpression "\
                      "where one of the sub-expressions is an equality "\
                      "expression:\n    " + buf.getvalue().strip())
        else:
            return _InequalityExpression(lhs, rhs, strict, cloned_from)

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_relational_expression.clone_counter = 0

# [functionality] chainedInequality allows us to generate symbolic
# expressions of the type "a < b < c".  This provides a buffer to hold
# the first inequality so the second inequality can access it later.
generate_relational_expression.chainedInequality = None

def generate_intrinsic_function_expression(arg, name, fcn):
    # Special handling: if there are no Pyomo Modeling Objects in the
    # argument list, then evaluate the expression and return the result.
    pyomo_expression = False
    if isinstance(arg, NumericValue) or isinstance(arg, Component):
        # TODO: efficiency: we already know this is a NumericValue -
        # so we should be able to avoid the call to as_numeric()
        # below (expecially since all intrinsic functions are unary
        # operators.
        pyomo_expression = True
    if not pyomo_expression:
        return fcn(arg)
    elif arg.is_constant():
        return fcn(arg())
    
    if arg.is_indexed():
        raise ValueError("Argument for intrinsic function '%s' is an "\
            "n-ary numeric value: %s\n    Have you given variable or "\
            "parameter '%s' an index?" % (name, arg.cname(), arg.cname()))
    return _UnaryFunctionExpression((arg,), name, fcn)

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_intrinsic_function_expression.clone_counter = 0


generate_expression_bypassCloneCheck = generate_expression


def _rmul_override(_self, _other):
    _self_expr = _self.is_expression()
    if _self_expr:
        if safe_mode and _self._parent_expr:# is not None:
            raise EntangledExpressionError(_self, _other)
            #_self = _self.clone()
            #generate_expression.clone_counter += 1
    elif _self.is_constant():
        _self = _self()
    
    if _other.__class__ not in native_numeric_types:
        _other = as_numeric(_other)
        if _other.is_constant():
            _other = _other()

    if _self_expr:
        ans = _ProductExpression((_self, _other))
    else:
        if isinstance(_self, _VarData):
            if _LinearExpression_Pool:
                ans = _LinearExpression_Pool.pop()
            else:
                ans = _LinearExpression()
            ans._args.append(_self)
            ans.linear[id(_self)] = _other
            return ans
        else:
            if _LinearExpression_Pool:
                ans = _LinearExpression_Pool.pop()
            else:
                ans = _LinearExpression()
            ans.constant = _ProductExpression((_self, _other))
            return ans

#NumericValue.__rmul__ = _rmul_override


class TreeWalkerHelper(object):
    stack = []
    max = 0
    inuse = False
    typeList = { _SumExpression: 1, 
                 _ProductExpression: 2, 
                 _NegationExpression: 3,
                 _LinearExpression: 4,
    }
