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

import logging
import math
import sys
import traceback
from six import advance_iterator
from weakref import ref

logger = logging.getLogger('pyomo.core')

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
from pyomo.core.base.numvalue import native_types, native_numeric_types
from pyomo.core.base.var import _VarData, Var
from pyomo.core.base.param import _ParamData
from pyomo.core.base import expr_common as common
import pyomo.core.base.expr_common
from pyomo.core.base.expr_common import \
    ensure_independent_trees as safe_mode, bypass_backreference, \
    _add, _sub, _mul, _div, _pow, _neg, _abs, _inplace, _unary, \
    _radd, _rsub, _rmul, _rdiv, _rpow, _iadd, _isub, _imul, _idiv, _ipow, \
    _lt, _le, _eq, clone_expression, chainedInequalityErrorMessage as cIEM

# Wrap the common chainedInequalityErrorMessage to pass the local context
chainedInequalityErrorMessage \
    = lambda *x: cIEM(generate_relational_expression, *x)

_stack = []


def _const_to_string(*args):
    args[1].write("%s" % args[0])


class EntangledExpressionError(Exception):
    pass


def identify_variables( expr,
                        include_fixed=True,
                        allow_duplicates=False,
                        include_potentially_variable=False ):
    if not allow_duplicates:
        _seen = set()
    _stack = [ ([expr], 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if type(_sub) in native_types:
                pass
            elif _sub.is_expression():
                _stack.append(( _argList, _idx, _len ))
                _argList = _sub._args
                _idx = 0
                _len = len(_argList)
            elif isinstance(_sub, _VarData):
                if ( include_fixed
                     or not _sub.is_fixed()
                     or include_potentially_variable ):
                    if not allow_duplicates:
                        if id(_sub) in _seen:
                            continue
                        _seen.add(id(_sub))
                    yield _sub
            elif include_potentially_variable and _sub._potentially_variable():
                if not allow_duplicates:
                    if id(_sub) in _seen:
                        continue
                    _seen.add(id(_sub))
                yield _sub


class _ExpressionBase(NumericValue):
    """An object that defines a mathematical expression that can be evaluated"""

    __pickle_slots__ = ('_args',)
    __slots__ =  __pickle_slots__ + (
        ('__weakref__', '_parent_expr') if safe_mode else () )
    PRECEDENCE = 0

    def __init__(self, args):
        if safe_mode:
            self._parent_expr = None
        self._args = args

    def __getstate__(self):
        state = super(_ExpressionBase, self).__getstate__()
        for i in _ExpressionBase.__pickle_slots__:
           state[i] = getattr(self,i)
        if safe_mode:
            if not bypass_backreference and self._parent_expr is not None:
                state['_parent_expr'] = self._parent_expr()
            else:
                state['_parent_expr'] = self._parent_expr
        return state

    def __setstate__(self, state):
        super(_ExpressionBase, self).__setstate__(state)
        if safe_mode:
            if self._parent_expr is not None and not bypass_backreference:
                self._parent_expr = ref(self._parent_expr)

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def __str__(self):
        buf = StringIO()
        self.to_string(buf)
        return buf.getvalue()

    def __call__(self, exception=None):
        _stack = [ (self, self._args, 0, len(self._args), []) ]
        while 1:  # Note: 1 is faster than True for Python 2.x
            _obj, _argList, _idx, _len, _result = _stack.pop()
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                if type(_sub) in native_numeric_types:
                    _result.append( _sub )
                elif _sub.is_expression():
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj     = _sub
                    _argList = _sub._args
                    _idx     = 0
                    _len     = len(_argList)
                    _result  = []
                else:
                    _result.append( value(_sub) )
            ans = _obj._apply_operation(_result)
            if _stack:
                _stack[-1][-1].append( ans )
            else:
                return ans


    def clone(self, substitute=None):
        ans = clone_expression(self, substitute)
        if safe_mode:
            ans._parent_expr = None
        return ans

    def getname(self, *args, **kwds):
        """The text name of this Expression function"""
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement getname()" % ( str(self.__class__), ))

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

    # FIXME: These need to be made non-recursive
    def _potentially_variable(self):
        for a in self._args:
            if a.__class__ not in native_numeric_types and a._potentially_variable():
                return True
        return False

    def is_expression(self):
        return True


    def polynomial_degree(self):
        _stack = [ (self, self._args, 0, len(self._args), []) ]
        while 1:  # Note: 1 is faster than True for Python 2.x
            _obj, _argList, _idx, _len, _result = _stack.pop()
            while _idx < _len:
                _sub = _argList[_idx]
                _idx += 1
                if type(_sub) in native_numeric_types:
                    _result.append( 0 )
                elif _sub.is_expression():
                    _stack.append( (_obj, _argList, _idx, _len, _result) )
                    _obj     = _sub
                    _argList = _sub._args
                    _idx     = 0
                    _len     = len(_argList)
                    _result  = []
                else:
                    _result.append( 0 if _sub.is_fixed() else 1 )
            ans = _obj._polynomial_degree(_result)
            if _stack:
                _stack[-1][-1].append( ans )
            else:
                return ans


    def _polynomial_degree(self, ans):
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _polynomial_degree()" % ( str(self.__class__), ))


    def to_string(self, ostream=None, verbose=None, precedence=None):
        _name_buffer = {}
        if ostream is None:
            ostream = sys.stdout
        verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose

        _infix = False
        _bypass_prefix = False
        argList = self._arguments()
        _stack = [ [ self, argList, 0, len(argList),
                     precedence if precedence is not None else self._precedence() ] ]
        while _stack:
            _parent, _args, _idx, _len, _prec = _stack[-1]
            _my_precedence = _parent._precedence()
            if _idx < _len:
                _sub = _args[_idx]
                _stack[-1][2] += 1
                if _infix:
                    _bypass_prefix = _parent._to_string_infix(ostream, _idx, verbose)
                else:
                    if not _bypass_prefix:
                        _parent._to_string_prefix(ostream, verbose)
                    else:
                        _bypass_prefix = False
                    if _my_precedence > _prec or not _my_precedence or verbose:
                        ostream.write("( ")
                    _infix = True
                if hasattr(_sub, '_args'): # _args is a proxy for Expression
                    argList = _sub._arguments()
                    _stack.append([ _sub, argList, 0, len(argList), _my_precedence ])
                    _infix = False
                else:
                    _parent._to_string_term(ostream, _idx, _sub, _name_buffer, verbose)
            else:
                _stack.pop()
                #print _stack
                if (_my_precedence > _prec) or not _my_precedence or verbose:
                    ostream.write(" )")

    def _arguments(self):
        return self._args

    def _precedence(self):
        return _ExpressionBase.PRECEDENCE

    def _to_string_term(self, ostream, _idx, _sub, _name_buffer, verbose):
        if _sub.__class__ in native_numeric_types:
            ostream.write(str(_sub))
        elif _sub.__class__ is NumericConstant:
            ostream.write(str(_sub()))
        else:
            ostream.write(_sub.getname(True, _name_buffer))

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.getname())

    def _to_string_infix(self, ostream, idx, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            ostream.write(self._inline_operator())


class _NegationExpression(_ExpressionBase):
    __slots__ = ()

    PRECEDENCE = 4

    def getname(self, *args, **kwds):
        return 'neg'

    def _polynomial_degree(self, result):
        return result[0]

    def _precedence(self):
        return _NegationExpression.PRECEDENCE

    def _to_string_prefix(self, ostream, verbose):
        if verbose:
            ostream.write(self.getname())
        elif not self._args[0].is_expression and _NegationExpression.PRECEDENCE <= self._args[0]._precedence():
            ostream.write("-")
        else:
            ostream.write("- ")

    def _apply_operation(self, result):
        return -result[0]

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

    def getname(self, *args, **kwds):
        return self._name

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.getname())

    def _polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result[0])

# Backwards compatibility: Coopr 3.x expected a slightly less informative name
_IntrinsicFunctionExpression =  _UnaryFunctionExpression


class _ExternalFunctionExpression(_ExpressionBase):
    __slots__ = ('_fcn',)

    def __init__(self, fcn, args):
        """Construct a call to an external function"""
        if safe_mode:
            self._parent_expr = None
            for x in args:
                if isinstance(x, _ExpressionBase):
                    if x._parent_expr:
                        raise EntangledExpressionError(x)
                    x._parent_expr = bypass_backreference or ref(self)
        self._args = tuple(
            x if isinstance(x, basestring) else as_numeric(x)
            for x in args )
        self._fcn = fcn

    def getname(self, *args, **kwds):
        return self._fcn.getname(*args, **kwds)

    def _polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        """Evaluate the expression"""
        return self._fcn.evaluate( result )

    def _inline_operator(self):
        return ', '


# Should this actually be a special class, or just an instance of
# _UnaryFunctionExpression (like sin, cos, etc)?
class _AbsExpression(_UnaryFunctionExpression):

    __slots__ = ()

    def __init__(self, arg):
        super(_AbsExpression, self).__init__(arg, 'abs', abs)


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
        l,r = result
        if r == 0:
            if l == 0:
                return 0
            try:
                # NOTE: use value before int() so that we don't
                #       run into the disabled __int__ method on
                #       NumericValue
                exp = value(self._args[1])
                if exp == int(exp):
                    if l is not None and exp > 0:
                        return l * exp
                    elif exp == 0:
                        return 0
            except:
                pass
        return None

    def is_fixed(self):
        if self._args[1].__class__ not in native_numeric_types and not self._args[1].is_fixed():
            return False
        return value(self._args[1]) == 0 or \
            self._args[0].__class__ in native_numeric_types or \
            self._args[0].is_fixed()

    def is_constant(self):
        if self._args[1].__class__ not in native_numeric_types and not self._args[1].is_constant():
            return False
        return value(self._args[1]) == 0 or \
            self._args[0].__class__ in native_numeric_types or \
            self._args[0].is_constant()

    # the base class implementation is fine
    #def _potentially_variable(self)

    def _precedence(self):
        return _PowExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _inline_operator(self):
        return '**'


class _LinearOperatorExpression(_ExpressionBase):
    """An 'abstract' class that defines the polynominal degree for a simple
    linear operator
    """

    __slots__ = ()

    def _polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans


class _InequalityExpression(_LinearOperatorExpression):
    """An object that defines a series of less-than or
    less-than-or-equal expressions"""

    __slots__ = ('_strict', '_cloned_from')
    PRECEDENCE = 9

    def __init__(self, args, strict, cloned_from):
        """Constructor"""
        super(_InequalityExpression,self).__init__(args)
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
        if not self.is_constant() and len(self._args) == 2:
            generate_relational_expression.call_info \
                = traceback.extract_stack(limit=2)[-2]
            generate_relational_expression.chainedInequality = self
            return True

        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _InequalityExpression.PRECEDENCE

    def _apply_operation(self, result):
        for i, a in enumerate(result):
            if not i:
                pass
            elif self._strict[i-1]:
                if not _l < a:
                    return False
            else:
                if not _l <= a:
                    return False
            _l = a
        return True

    def _to_string_prefix(self, ostream, verbose):
        pass

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write( '  <  ' if self._strict[idx-1] else '  <=  ' )


class _EqualityExpression(_LinearOperatorExpression):
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
        _l, _r = result
        return _l == _r

    def _to_string_prefix(self, ostream, verbose):
        pass

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write('  ==  ' )


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
        a, b = result
        if a is None or b is None:
            return None
        else:
            return a + b


    def getname(self, *args, **kwds):
        return 'prod'

    def _inline_operator(self):
        return ' * '

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r


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
        n,d = result
        if d == 0:
            return n
        else:
            return None


    def getname(self, *args, **kwds):
        return 'div'

    def _inline_operator(self):
        return ' / '

    def _apply_operation(self, result):
        _l, _r = result
        return _l / _r


# a * b / ( a * b )  :or:  (a + b) / (c + d)
# a / b / c  :or: a / (b / c)
# (a / b) * c
# Note: Precedence of numerator == multiple, denominator < multiply




class _SumExpression(_LinearOperatorExpression):
    """An object that defines a simple summation of expressions"""

    __slots__ = ()
    PRECEDENCE = 6

    def _precedence(self):
        return _SumExpression.PRECEDENCE

    def _to_string_infix(self, ostream, idx, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            if type(self._args[idx]) is _NegationExpression:
                ostream.write(' - ')
                return True
            else:
                ostream.write(' + ')

    def _apply_operation(self, result):
        return sum(result)

    def getname(self, *args, **kwds):
        return 'sum'

    def __iadd__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            if other:
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
                if safe_mode:
                    for x in other._args:
                        if x.is_expression():
                            x._parent_expr = bypass_backreference or ref(self)
                other._args = [] # for safety
                return self
            if safe_mode:
                other._parent_expr = bypass_backreference or ref(self)

        elif other.is_indexed():
            raise TypeError(
                "Argument for expression '%s' is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?"
                % (etype, other.name, ) )
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
            if other:
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
                if safe_mode:
                    for x in other._args:
                        if x.is_expression():
                            x._parent_expr = None
                _other_args = tuple(-i for i in other._args)
                self._args.extend(_other_args)
                if safe_mode:
                    for x in _other_args:
                        if x.is_expression():
                            x._parent_expr = bypass_backreference or ref(self)
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
                % (etype, other.name, ) )
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
        if safe_mode:
            self._parent_expr = None

        self._if = as_numeric(IF)
        self._then = as_numeric(THEN)
        self._else = as_numeric(ELSE)
        self._args = (self._if, self._then, self._else)

    def __getstate__(self):
        state = super(Expr_if, self).__getstate__()
        for i in Expr_if.__slots__:
            state[i] = getattr(self, i)
        return state

    def _arguments(self):
        return ( self._if, self._then, self._else )

    def getname(self, *args, **kwds):
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

    # the base class implementation is fine
    #def _potentially_variable(self)

    def _polynomial_degree(self, result):
        _if, _then, _else = result
        if _if == 0:
            try:
                return _then if self._if() else _else
            except:
                pass
        return None

    def _to_string_term(self, ostream, _idx, _sub, _name_buffer, verbose):
        ostream.write("%s=( " % ('if','then','else')[_idx], )
        self._args[_idx].to_string(ostream=ostream, verbose=verbose)
        ostream.write(" )")

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.getname())

    def _to_string_infix(self, ostream, idx, verbose):
        ostream.write(", ")

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

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class _GetItemExpression(_ExpressionBase):
    """Expression to call "__getitem__" on the base"""

    __slots__ = ('_base',)
    PRECEDENCE = 1

    def _precedence(self):
        return _GetItemExpression.PRECEDENCE

    def __init__(self, base, args):
        """Construct an expression with an operation and a set of arguments"""
        if safe_mode:
            self._parent_expr = None
        self._args = args
        self._base = base

    def __getstate__(self):
        result = super(_GetItemExpression, self).__getstate__()
        for i in _GetItemExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def is_constant(self):
        return False

    def is_fixed(self):
        return not isinstance(self._base, Var)

    def _polynomial_degree(self, result):
        return 0 if self.is_fixed() else 1

    def _apply_operation(self, result):
        return value(self._base.__getitem__( tuple(result) ))

    def _to_string_prefix(self, ostream, verbose):
        ostream.write(self.name)

    def resolve_template(self):
        return self._base.__getitem__(tuple(value(i) for i in self._args))


_LinearExpression_Pool = []

class _LinearExpression(_ExpressionBase):
    __slots__ = ('_const', '_coef')

    def __init__(self, const=None, args=None, coef=None):
        if safe_mode:
            self._parent_expr = None
        if const is not None:
            self._const = const
            self._args = args
            self._coef = dict((id(self._args[i]),c) for i,c in enumerate(coef))
        else:
            self._const = 0
            self._args = []
            self._coef = {}

    def __getstate__(self):
        state = super(_LinearExpression, self).__getstate__()
        for i in _LinearExpression.__slots__:
           state[i] = getattr(self,i)
        # ID's do not persist from instance to instance (they are pointers!)
        # ...so we will convert them to a (temporary, but portable) index
        state['_coef'] = tuple(self._coef[id(v)] for v in self._args)
        return state

    def __setstate__(self, state):
        super(_LinearExpression, self).__setstate__(state)
        # ID's do not persist from instance to instance (they are pointers!)
        self._coef = dict((id(v), self._coef[i])
                          for i, v in enumerate(self._args))

    def _precedence(self):
        if len(self._args) > 1:
            return _SumExpression.PRECEDENCE
        elif len(self._args) and not (
                self._const.__class__ in native_numeric_types
                and self._const == 0 ):
            return _SumExpression.PRECEDENCE
        else:
            return _ProductExpression.PRECEDENCE

    def _arguments(self):
        if self._const.__class__ in native_numeric_types and self._const == 0:
            return self._args
        else:
            ans = [ self._const ]
            ans.extend(self._args)
            return ans

    def getname(self, *args, **kwds):
        return 'linear'

    def is_constant(self):
        if self._const.__class__ not in native_numeric_types \
           and not self._const.is_constant():
            return False
        return super(_LinearExpression, self).is_constant()

    def _inline_operator(self):
        return ' + '

    def _to_string_term(self, ostream, _idx, _sub, _name_buffer, verbose):
        if _idx == 0 and self._const != 0:
            ostream.write("%s" % (self._const, ))
        else:
            coef = self._coef[id(_sub)]
            _coeftype = coef.__class__
            if _idx and _coeftype is _NegationExpression:
                coef = coef._args[0]
                _coeftype = coef.__class__
            if _coeftype in native_numeric_types:
                if _idx:
                    coef = abs(coef)
                if coef == 1:
                    ostream.write(_sub.getname(True, _name_buffer))
                    return
                ostream.write(str(coef))
            elif coef.is_expression():
                coef.to_string( ostream=ostream, verbose=verbose,
                                precedence=_ProductExpression.PRECEDENCE )
            else:
                ostream.write(str(coef))
            ostream.write("*%s" % (_sub.getname(True, _name_buffer)))

    def _to_string_infix(self, ostream, idx, verbose):
        if verbose:
            ostream.write(" , ")
        else:
            hasConst = not ( self._const.__class__ in native_numeric_types
                             and self._const == 0 )
            if hasConst:
                idx -= 1
            _l = self._coef[id(self._args[idx])]
            _lt = _l.__class__
            if _lt is _NegationExpression or (
                    _lt in native_numeric_types and _l < 0 ):
                ostream.write(' - ')
            else:
                ostream.write(' + ')

    def _polynomial_degree(self, result):
        if result:
            return max(result)
        else:
            return 0

    def _apply_operation(self, result):
        assert( len(result) == len(self._args) )
        ans = value(self._const)
        for i,v in enumerate(self._args):
            ans += value(self._coef[id(v)]) * result[i]
        return ans

    def __iadd__(self, other, _reversed=0):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__

        if _type is _LinearExpression:
            self._const += other._const
            for v in other._args:
                _id = id(v)
                if _id in self._coef:
                    self._coef[_id] += other._coef[_id]
                else:
                    self._args.append(v)
                    self._coef[_id] = other._coef[_id]
            other._const = 0
            other._args = []
            other._coef = {}
            if safe_mode:
                other._parent_expr = None
            _LinearExpression_Pool.append(other)
            return self
        elif isinstance(other, _VarData):
            _id = id(other)
            if _id in self._coef:
                self._coef[_id] += 1
            else:
                if _reversed:
                    self._args.insert(0,other)
                else:
                    self._args.append(other)
                self._coef[_id] = 1
            return self
        elif _type in native_numeric_types:
            if other:
                self._const += other
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
            self._const += value(other)
        elif isinstance(other, _ParamData):
            self._const += other
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
            if other:
                self._const -= other
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
                self._const -= other._const
                for v in other._args:
                    _id = id(v)
                    if _id in self._coef:
                        self._coef[_id] -= other._coef[_id]
                    else:
                        self._args.append(v)
                        self._coef[_id] = -other._coef[_id]
                other._const = 0
                other._args = []
                other._coef = {}
                if safe_mode:
                    other._parent_expr = None
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__isub__(other)
        elif isinstance(other, _VarData):
            _id = id(other)
            if _id in self._coef:
                self._coef[_id] -= 1
            else:
                self._args.append(other)
                self._coef[_id] = -1
        elif other.is_constant():
            self._const -= value(other)
        elif isinstance(other, _ParamData):
            self._const -= other
        else:
            return super(_LinearExpression, self).__iadd__(other)

        return self

    # As we do all subtraction "in-place", all subtractions are the same as
    # in-place subtractions.
    __sub__ = __isub__

    # Note: __rsub__ of _LinearExpressions is rare, and the easiest thing
    # to do is just negate ourselves and add.
    def __rsub__(self, other):
        self *= -1
        return self.__radd__(other, 1)

    def __imul__(self, other):
        if safe_mode and self._parent_expr:
            raise EntangledExpressionError(self, other)
        _type = other.__class__
        if _type in native_numeric_types:
            if other != 1:
                self._const *= other
                for i in self._coef:
                    self._coef[i] *= other
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
                self._const *= other._const
                for i in self._coef:
                    self._coef[i] *= other._const
                other._const = 0
                if safe_mode:
                    other._parent_expr = None
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__imul__(other)
        elif isinstance(other, _VarData):
            if self._args:
                return super(_LinearExpression, self).__imul__(other)
            _id = id(other)
            self._args.append(other)
            self._coef[_id] = self._const
            self._const = 0
        elif other.is_constant():
            other = value(other)
            self._const *= other
            for i in self._coef:
                self._coef[i] *= other
        elif isinstance(other, _ParamData):
            self._const *= other
            for i in self._coef:
                self._coef[i] *= other
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
            if other != 1:
                self._const /= other
                for i in self._coef:
                    self._coef[i] /= other
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
                self._const /= other._const
                for i in self._coef:
                    self._coef[i] /= other._const
                other._const = 0
                if safe_mode:
                    other._parent_expr = None
                _LinearExpression_Pool.append(other)
            else:
                return super(_LinearExpression, self).__imul__(other)
        elif other.is_constant():
            other = value(other)
            self._const /= other
            for i in self._coef:
                self._coef[i] /= other
        elif isinstance(other, _ParamData):
            self._const /= other
            for i in self._coef:
                self._coef[i] /= other
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
        self._const = other / self._const
        return self

    def __neg__(self):
        self *= -1
        return self



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
            if _self.__class__ in native_numeric_types:
                ans = -_self
            else:
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
                % (etype, _other.name, ) )
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
                if _other.__class__ in native_numeric_types:
                    if not _other:
                        return 0
                    elif _other == 1:
                        return _self
                if isinstance(_other, _VarData):
                    ans = _ProductExpression((_self, _other))
                else:
                    if _LinearExpression_Pool:
                        ans = _LinearExpression_Pool.pop()
                    else:
                        ans = _LinearExpression()
                    ans._args.append(_self)
                    ans._coef[id(_self)] = _other
                    return ans
            elif isinstance(_other, _VarData):
                if _self.__class__ in native_numeric_types:
                    if not _self:
                        return 0
                    elif _self == 1:
                        return _other
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans._coef[id(_other)] = _self
                return ans
            elif _self.__class__ in native_numeric_types \
                 and _other.__class__ in native_numeric_types:
                ans = _self * _other
            else:
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._const = _ProductExpression((_self, _other))
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
            if _other.__class__ is _LinearExpression:
                return _other.__iadd__(_self, 1)
            ans = _SumExpression([_self, _other])
        else:
            if isinstance(_self, _VarData):
                if _other.__class__ in native_numeric_types and not _other:
                    return _self
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_self)
                ans._coef[id(_self)] = 1
                if isinstance(_other, _VarData):
                    if id(_other) in ans._coef:
                        ans._coef[id(_other)] += 1
                    else:
                        ans._args.append(_other)
                        ans._coef[id(_other)] = 1
                else:
                    ans._const = _other
            elif isinstance(_other, _VarData):
                if _self.__class__ in native_numeric_types and not _self:
                    return _other
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans._coef[id(_other)] = 1
                ans._const = _self
            elif _self.__class__ in native_numeric_types \
                 and _other.__class__ in native_numeric_types:
                ans = _self + _other
            else:
                ans = _SumExpression([_self, _other])
            return ans

    elif etype == _sub:
        if _self_expr or _other_expr:
            if _other.__class__ is _LinearExpression:
                return _other.__rsub__(_self)
            ans = _SumExpression([_self, -_other])
        else:
            if isinstance(_self, _VarData):
                if _other.__class__ in native_numeric_types and not _other:
                    return _self
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_self)
                ans._coef[id(_self)] = 1
                if isinstance(_other, _VarData):
                    if id(_other) in ans._coef:
                        ans._coef[id(_other)] -= 1
                    else:
                        ans._args.append(_other)
                        ans._coef[id(_other)] = -1
                else:
                    ans._const = -_other
            elif isinstance(_other, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_other)
                ans._coef[id(_other)] = -1
                ans._const = _self
            elif _self.__class__ in native_numeric_types \
                 and _other.__class__ in native_numeric_types:
                ans = _self - _other
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
        if _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                ans = _self / _other
            elif _other == 1:
                ans = _self
            elif isinstance(_self, _VarData):
                if _LinearExpression_Pool:
                    ans = _LinearExpression_Pool.pop()
                else:
                    ans = _LinearExpression()
                ans._args.append(_self)
                ans._coef[id(_self)] = 1/_other
            else:
                ans = _DivisionExpression((_self, _other))
        else:
            if _self.__class__ in native_numeric_types and not _self:
                ans = _self
            else:
                ans = _DivisionExpression((_self, _other))
    elif etype == _pow:
        if _other.__class__ in native_numeric_types and \
                _self.__class__ in native_numeric_types:
            ans = _self ** _other
        else:
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
               lhs.name, lhs.name))
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
               rhs.name, rhs.name))
    elif rhs.is_expression():
        if rhs.is_relational():
            rhs_is_relational = True

    if generate_relational_expression.chainedInequality is not None:
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
                match.append((i,0))
            elif arg == cloned_from[1]:
                match.append((i,1))
        if etype == _eq:
            raise TypeError(chainedInequalityErrorMessage())
        if len(match) == 1:
            if match[0][0] == match[0][1]:
                raise TypeError(chainedInequalityErrorMessage(
                    "Attempting to form a compound inequality with two "
                    "%s bounds" % ('lower' if match[0][0] else 'upper',)))
            if not match[0][1]:
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
                      "identical upper and lower\n\tbounds using strict "
                      "inequalities: constraint infeasible:\n\t%s and "
                      "%s < %s" % ( buf.getvalue().strip(), lhs, rhs ))
            if match[0] == (0,0):
                # This is a particularly weird case where someone
                # evaluates the *same* inequality twice in a row.  This
                # should always be an error (you can, for example, get
                # it with "0 <= a >= 0").
                raise TypeError(chainedInequalityErrorMessage())
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
            strict = (False,)
        elif etype == _lt:
            strict = (True,)
        else:
            raise ValueError("Unknown relational expression type '%s'" % etype)
        if lhs_is_relational:
            if lhs.__class__ is _InequalityExpression:
                if rhs_is_relational:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where both sub-expressions are also relational "\
                          "expressions (we support no more than 3 terms "\
                          "in an inequality expression).")
                if len(lhs._args) > 2:
                    raise ValueError("Cannot create an InequalityExpression "\
                          "with more than 3 terms.")
                lhs._args = lhs._args + (rhs,)
                lhs._strict = lhs._strict + strict
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
                rhs._args = (lhs,) + rhs._args
                rhs._strict = strict + rhs._strict
                rhs._cloned_from = cloned_from
                return rhs
            else:
                buf = StringIO()
                rhs.to_string(buf)
                raise TypeError("Cannot create an InequalityExpression "\
                      "where one of the sub-expressions is an equality "\
                      "expression:\n    " + buf.getvalue().strip())
        else:
            return _InequalityExpression((lhs, rhs), strict, cloned_from)

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
            "parameter '%s' an index?" % (name, arg.name, arg.name))
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
            ans._coef[id(_self)] = _other
            return ans
        else:
            if _LinearExpression_Pool:
                ans = _LinearExpression_Pool.pop()
            else:
                ans = _LinearExpression()
            ans._const = _ProductExpression((_self, _other))
            return ans

#NumericValue.__rmul__ = _rmul_override


class TreeWalkerHelper(object):
    stack = []
    max = 0
    inuse = False
    typeList = { _SumExpression: 1,
                 _InequalityExpression: 1,
                 _EqualityExpression: 1,
                 _ProductExpression: 2,
                 _NegationExpression: 3,
                 _LinearExpression: 4,
    }

def _clear_expression_pool():
    global _LinearExpression_Pool
    _LinearExpression_Pool = []
