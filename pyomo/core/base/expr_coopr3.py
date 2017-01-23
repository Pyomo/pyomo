#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from __future__ import division

#__all__ = ( 'log', 'log10', 'sin', 'cos', 'tan', 'cosh', 'sinh', 'tanh',
#            'asin', 'acos', 'atan', 'exp', 'sqrt', 'asinh', 'acosh',
#            'atanh', 'ceil', 'floor' )
import logging
import math
import sys
import traceback

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

#from pyomo.util.plugin import *

from pyomo.core.base.component import Component
#from pyomo.core.base.plugin import *
from pyomo.core.base.numvalue import *
from pyomo.core.base.numvalue import native_numeric_types, native_types
from pyomo.core.base.var import _VarData, Var

import pyomo.core.base.expr_common
from pyomo.core.base.expr_common import \
    _add, _sub, _mul, _div, _pow, _neg, _abs, _inplace, _unary, \
    _radd, _rsub, _rmul, _rdiv, _rpow, _iadd, _isub, _imul, _idiv, _ipow, \
    _lt, _le, _eq, clone_expression, chainedInequalityErrorMessage as cIEM

# Wrap the common chainedInequalityErrorMessage to pass the local context
chainedInequalityErrorMessage \
    = lambda *x: cIEM(generate_relational_expression, *x)


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
                if type(_sub) is _ProductExpression:
                    if _sub._denominator:
                        _stack.append(
                            (_sub._denominator, 0, len(_sub._denominator)) )
                    _argList = _sub._numerator
                else:
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

    __slots__ = ('_args', )
    PRECEDENCE = 10

    def __init__(self, args):
        """Construct an expression with an operation and a set of arguments"""
        self._args=args

    def __getstate__(self):
        result = NumericValue.__getstate__(self)
        for i in _ExpressionBase.__slots__:
            result[i] = getattr(self, i)
        return result

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose
        ostream.write(self.getname() + "( ")
        first = True
        for arg in self._args:
            if first:
                first = False
            elif _verbose:
                ostream.write(" , ")
            else:
                ostream.write(", ")
            try:
                arg.to_string( ostream=ostream, precedence=self._precedence(),
                               verbose=verbose )
            except AttributeError:
                ostream.write("(%s)" % (arg,))
        ostream.write(" )")

    def clone(self, substitute=None):
        return clone_expression(self, substitute)

    def simplify(self, model): #pragma:nocover
        print("""
WARNING: _ExpressionBase.simplify() has been deprecated and removed from
     Pyomo Expressions.  Please remove references to simplify() from your
     code.
""")
        return self

    #
    # this method contrast with the is_fixed() method.
    # the is_fixed() method returns true iff the value is
    # an atomic constant.
    # this method returns true iff all composite arguments
    # in this sum expression are constant, i.e., numeric
    # constants or parametrs. the parameter values can of
    # course change over time, but at any point in time,
    # they are constant. hence, the name.
    #
    def is_constant(self):
        for arg in self._args:
            if not arg.is_constant():
                return False
        return True

    def is_fixed(self):
        for arg in self._args:
            if not arg.is_fixed():
                return False
        return True

    def _potentially_variable(self):
        for arg in self._args:
            if arg._potentially_variable():
                return True
        return False

    def is_expression(self):
        return True

    def polynomial_degree(self, ):
        return None

    def _precedence(self):
        return _ExpressionBase.PRECEDENCE

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """Evaluate the expression"""
        try:
            return self._apply_operation(
                self._evaluate_arglist(self._args,
                                       exception=exception))
        except (ValueError, TypeError):
            if exception:
                raise
            return None

    def _evaluate_arglist(self, arglist, exception=True):
        for arg in arglist:
            try:
                yield value(arg, exception=exception)
            except Exception:
                if exception:
                    e = sys.exc_info()[1]
                    logger.error(
                        "evaluating expression: %s\n    (expression: %s)",
                        str(e), str(self) )
                    raise
                yield None

    def _apply_operation(self, values):
        """Method that can be overwritten to define the operation in
        this expression"""
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _apply_operation()" % ( str(self.__class__), ))

    def __str__(self):
        buf = StringIO()
        self.to_string(buf)
        return buf.getvalue()

class _ExternalFunctionExpression(_ExpressionBase):
    __slots__ = ('_fcn',)

    def __init__(self, fcn, args):
        """Construct a call to an external function"""
        _args = tuple(
            _generate_expression__clone_if_needed(
                x,-2) if isinstance(x, _ExpressionBase)
            else x if isinstance(x, basestring)
            else as_numeric(x)
            for x in args )
        _ExpressionBase.__init__(self, _args)
        self._fcn = fcn

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in _ExternalFunctionExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._fcn.getname(*args, **kwds)

    def polynomial_degree(self):
        return None

    def is_constant(self):
        for arg in self._args:
            if isinstance(arg, basestring):
                continue
            if not arg.is_constant():
                return False
        return True

    def is_fixed(self):
        for arg in self._args:
            if isinstance(arg, basestring):
                continue
            if not arg.is_fixed():
                return False
        return True

    def _potentially_variable(self):
        for arg in self._args:
            if isinstance(arg, basestring):
                continue
            if arg._potentially_variable():
                return True
        return False

    def _apply_operation(self, values):
        return self._fcn.evaluate(values)


class _IntrinsicFunctionExpression(_ExpressionBase):
    __slots__ = ('_operator', '_name')

    def __init__(self, name, nargs, args, operator):
        """Construct an expression with an operation and a set of arguments"""
        if nargs and nargs != len(args):
            raise ValueError("%s() takes exactly %d arguments (%d given)" % \
                ( self.name, nargs, len(args) ))
        _ExpressionBase.__init__(self, args)
        self._operator = operator
        self._name = name

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in _IntrinsicFunctionExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def _apply_operation(self, values):
        return self._operator(*tuple(values))

    def getname(self, *args, **kwds):
        return self._name

    def polynomial_degree(self):
        if self.is_fixed():
            return 0
        return None


# Should this actually be a special class, or just an instance of
# _IntrinsicFunctionExpression (like sin, cos, etc)?
class _AbsExpression(_IntrinsicFunctionExpression):

    __slots__ = ()

    def __init__(self, args):
        _IntrinsicFunctionExpression.__init__(self, 'abs', 1, args, abs)

    #def __getstate__(self):
    #    return _IntrinsicFunctionExpression.__getstate__(self)

# Should this actually be a special class, or just an instance of
# _IntrinsicFunctionExpression (like sin, cos, etc)?
class _PowExpression(_IntrinsicFunctionExpression):

    __slots__ = ()
    PRECEDENCE = 2

    def __init__(self, args):
        _IntrinsicFunctionExpression.__init__(self, 'pow', 2, args, pow)

    #def __getstate__(self):
    #    return _IntrinsicFunctionExpression.__getstate__(self)

    def polynomial_degree(self):
        # _PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        if self._args[1].is_fixed():
            if self._args[0].is_fixed():
                return 0
            try:
                # NOTE: use value before int() so that we don't
                #       run into the disabled __int__ method on
                #       NumericValue
                exp = value(self._args[1])
                if exp == int(exp):
                    base = self._args[0].polynomial_degree()
                    if base is not None and exp > 0:
                        return base * exp
                    elif exp == 0:
                        return 0
            except TypeError:
                pass
        return None

    def is_fixed(self):
        if self._args[1].is_fixed():
            return self._args[0].is_fixed() or bool(self._args[1] == 0)
        return False

    # the base class implementation is fine
    #def _potentially_variable(self)

    def _precedence(self):
        return _PowExpression.PRECEDENCE

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        # For verbose mode, rely on the underlying base expression
        # (prefix) expression printer
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose
        if _verbose:
            return super(_PowExpression, self).to_string(
                ostream, verbose, precedence)

        if ostream is None:
            ostream = sys.stdout
        _my_precedence = self._precedence()
        if precedence and _my_precedence > precedence:
            ostream.write("( ")
        first = True
        for arg in self._args:
            if first:
                first = False
            else:
                ostream.write("**")
            try:
                arg.to_string( ostream=ostream, verbose=verbose,
                               precedence=self._precedence() )
            except AttributeError:
                ostream.write("(%s)" % (arg,))
        if precedence and _my_precedence > precedence:
            ostream.write(" )")

class _LinearExpression(_ExpressionBase):

    __slots__ = ()

     # the constructor for a _LinearExpression does nothing, as there are
     # no class-specific slots. thus, one is not provided - providing one
     # simply incurs overhead. the original constructor is commented out
     # below, to provide documentation as to how arguments are munged.

    def polynomial_degree(self):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        degree = 0
        for x in self._args:
            x_degree = x.polynomial_degree()
            if x_degree is None:
                return None
            if x_degree > degree:
               degree = x_degree
        return degree


class _InequalityExpression(_LinearExpression):
    """An object that defines a series of less-than or
    less-than-or-equal expressions"""

    __slots__ = ('_strict', '_cloned_from')
    PRECEDENCE = 5

    def __init__(self, args, strict, cloned_from):
        """Constructor"""
        _LinearExpression.__init__(self, args)
        self._strict = strict
        self._cloned_from = cloned_from
        if len(self._args)-1 != len(self._strict):
            raise ValueError("_InequalityExpression(): len(args)-1 != " \
                  "len(strict)(%d and %d given)" % (len(args), len(strict)))

    def __getstate__(self):
        result = _LinearExpression.__getstate__(self)
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

    def _apply_operation(self, values):
        """Method that defines the less-than-or-equal operation"""
        arg1 = next(values)
        for strict in self._strict:
            arg2 = next(values)
            if strict:
                if not (arg1 < arg2):
                    return False
            else:
                if not (arg1 <= arg2):
                    return False
            arg1 = arg2
        return True

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _my_precedence = self._precedence()
        if precedence and _my_precedence > precedence:
            ostream.write("( ")
        for i, strict in enumerate(self._strict):
            arg = self._args[i]
            try:
                arg.to_string( ostream=ostream, verbose=verbose,
                               precedence=_my_precedence )
            except AttributeError:
                ostream.write("(%s)" % (arg,))
            if strict:
                ostream.write("  <  ")
            else:
                ostream.write("  <=  ")
        arg = self._args[-1]
        try:
            arg.to_string( ostream=ostream, verbose=verbose,
                           precedence=_my_precedence )
        except AttributeError:
            ostream.write("(%s)" % (arg,))
        if precedence and _my_precedence > precedence:
            ostream.write(" )")


class _EqualityExpression(_LinearExpression):
    """An object that defines a equal-to expression"""

    __slots__ = ()
    PRECEDENCE = 5

    def __init__(self, args):
        """Constructor"""
        if 2 != len(args):
            raise ValueError("%s() takes exactly 2 arguments (%d given)" % \
                ( self.name, len(args) ))
        _LinearExpression.__init__(self, args)

    def __nonzero__(self):
        if generate_relational_expression.chainedInequality is not None:
            raise TypeError(chainedInequalityErrorMessage())
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return _EqualityExpression.PRECEDENCE

    def _apply_operation(self, values):
        """Method that defines the equal-to operation"""
        return next(values) == next(values)

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _my_precedence = self._precedence()
        if precedence and _my_precedence > precedence:
            ostream.write("( ")
        first = True
        for arg in self._args:
            if first:
                first = False
            else:
                ostream.write("  ==  ")
            try:
                arg.to_string( ostream=ostream, verbose=verbose,
                               precedence=_my_precedence)
            except AttributeError:
                ostream.write("(%s)" % (arg,))
        if precedence and _my_precedence > precedence:
            ostream.write(" )")


# It is common to generate sums of const*var (which immediately get
# thrown away by the simplifications in generate_expression): this gives
# us a pool for re-using the temporary _ProductExpression objects.
_ProdExpression_Pool = []

class _ProductExpression(_ExpressionBase):
    """An object that defines a product expression"""

    __slots__ = ('_denominator','_numerator','_coef')
    PRECEDENCE = 3

    def __init__(self):
        """Constructor"""
        _ExpressionBase.__init__(self, None)

        # generate_expression will create these for us.  Creating them
        # here is significantly wasteful as they would just end up being
        # thrown away.
        #self._denominator = []
        #self._numerator = []
        self._coef = 1

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in _ProductExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def is_constant(self):
        for arg in self._numerator:
            if not arg.is_constant():
                return False
        for arg in self._denominator:
            if not arg.is_constant():
                return False
        return True

    def is_fixed(self):
        for arg in self._numerator:
            if not arg.is_fixed():
                return False
        for arg in self._denominator:
            if not arg.is_fixed():
                return False
        return True

    def _potentially_variable(self):
        for arg in self._numerator:
            if arg._potentially_variable():
                return True
        for arg in self._denominator:
            if arg._potentially_variable():
                return True
        return False

    def _precedence(self):
        return _ProductExpression.PRECEDENCE

    def polynomial_degree(self):
        for x in self._denominator:
            if x.polynomial_degree() != 0:
                return None
        try:
            return sum(x.polynomial_degree() for x in self._numerator)
        except TypeError:
            return None
        except ValueError:
            return None

    def invert(self):
        tmp = self._denominator
        self._denominator = self._numerator
        self._numerator = tmp
        self._coef = 1.0/self._coef

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose
        _my_precedence = self._precedence()
        if _verbose:
            ostream.write("prod( num=( ")
        elif precedence and _my_precedence > precedence:
            ostream.write("( ")
        first = True
        if self._coef != 1:
            ostream.write(str(self._coef))
            first = False
        for arg in self._numerator:
            if first:
                first = False
            elif _verbose:
                ostream.write(" , ")
            else:
                ostream.write(" * ")
            try:
                arg.to_string( ostream=ostream, verbose=verbose,
                               precedence=_my_precedence )
            except AttributeError:
                ostream.write("(%s)" % (arg,))
        if first:
            ostream.write('1')
        if len(self._denominator) > 0:
            if _verbose:
                ostream.write(" ) , denom=( ")
            elif len(self._denominator) == 1:
                ostream.write(" / ")
            else:
                ostream.write(" / ( ")
            first = True
            for arg in self._denominator:
                if first:
                    first = False
                elif _verbose:
                    ostream.write(" , ")
                else:
                    ostream.write(" * ")
                try:
                    arg.to_string( ostream=ostream, verbose=verbose,
                                   precedence=_my_precedence )
                except AttributeError:
                    ostream.write("(%s)" % (arg,))
            if len(self._denominator) > 1 and not _verbose:
                ostream.write(" )")
        if _verbose:
            ostream.write(" ) )")
        elif precedence and _my_precedence > precedence:
            ostream.write(" )")

    def __call__(self, exception=True):
        """Evaluate the expression"""
        try:
            ans = self._coef
            for n in self._evaluate_arglist(self._numerator,
                                            exception=exception):
                ans *= n
            for n in self._evaluate_arglist(self._denominator,
                                            exception=exception):
                ans /= n
            return ans
        except (TypeError, ValueError):
            if exception:
                raise
            return None

# It is common to generate sums of sums (which immediately get
# thrown away by the simplifications in generate_expression): this gives
# us a pool for re-using the temporary _SumExpression objects.
_SumExpression_Pool = []

class _SumExpression(_LinearExpression):
    """An object that defines a weighted summation of expressions"""

    __slots__ = ('_coef','_const')
    PRECEDENCE = 4

    def __init__(self):
        """Constructor"""
        _LinearExpression.__init__(self, [])

        self._const = 0
        # generate_expression will create this for us.  Creating it
        # here is significantly wasteful as they would just end up being
        # thrown away.
        #self._coef = []
        # TODO: determine if we can do something similar with _args for
        # all _LinearExpressions

    def __getstate__(self):
        result = _LinearExpression.__getstate__(self)
        for i in _SumExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def _precedence(self):
        return _SumExpression.PRECEDENCE

    def scale(self, val):
        for i in xrange(len(self._coef)):
            self._coef[i] *= val
        self._const *= val

    def negate(self):
        self.scale(-1)

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE \
                   if verbose is None else verbose
        _my_precedence = self._precedence()
        if _verbose:
            ostream.write("sum( ")
        elif precedence and _my_precedence > precedence:
                ostream.write("( ")
        first = True
        if self._const != 0:
            ostream.write(str(self._const))
            first = False
        for i, arg in enumerate(self._args):
            if first:
                first = False
                if self._coef[i] < 0:
                    ostream.write(" - ")
            elif _verbose:
                ostream.write(" , ")
            elif self._coef[i] < 0:
                ostream.write(" - ")
            else:
                ostream.write(" + ")
            if self._coef[i] == 1:
                _sub_precedence = _my_precedence
            elif _verbose:
                ostream.write(str(self._coef[i])+" *  ")
                _sub_precedence = _my_precedence
            elif self._coef[i] == -1:
                _sub_precedence = _ProductExpression.PRECEDENCE
            else:
                ostream.write(str(abs(self._coef[i]))+"*")
                _sub_precedence = _ProductExpression.PRECEDENCE
            try:
                arg.to_string( ostream=ostream, verbose=verbose,
                               precedence=_sub_precedence )
            except AttributeError:
                ostream.write("(%s)" % arg)
            first=False
        if _verbose or ( precedence and _my_precedence > precedence ):
            ostream.write(" )")

    def _apply_operation(self, values):
        """Evaluate the expression"""
        return sum(c*next(values) for c in self._coef) + self._const


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

    def polynomial_degree(self):
        if self._if.is_fixed():
            if self._if():
                return self._then.polynomial_degree()
            else:
                return self._else.polynomial_degree()
        else:
            return None

    def to_string(self, ostream=None, verbose=None, precedence=0):
        """Print this expression"""
        if ostream is None:
            ostream = sys.stdout
        _my_precedence = self._precedence()
        ostream.write("Expr_if( if=( ")
        self._if.to_string( ostream=ostream, verbose=verbose,
                            precedence=self._precedence() )
        ostream.write(" ), then=( ")
        self._then.to_string( ostream=ostream, verbose=verbose,
                              precedence=self._precedence() )
        ostream.write(" ), else=( ")
        self._else.to_string( ostream=ostream, verbose=verbose,
                              precedence=self._precedence())
        ostream.write(" ) )")

    def __call__(self, exception=True):
        """Evaluate the expression"""
        if self._if(exception=exception):
            return self._then(exception=exception)
        else:
            return self._else(exception=exception)


class _GetItemExpression(_ExpressionBase):
    __slots__ = ('_base',)

    def __init__(self, base, args):
        """Construct a call to an external function"""
        _ExpressionBase.__init__(self, args)
        self._base = base

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in _GetItemExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def polynomial_degree(self):
        return 0 if self.is_fixed() else 1

    def is_constant(self):
        return False

    def is_fixed(self):
        return not isinstance(self._base, Var)

    def _apply_operation(self, values):
        return value(self._base[values])

    def resolve_template(self):
        return self._base.__getitem__(tuple(value(i) for i in self._args))


def _generate_expression__clone_if_needed(obj, target):
    #print(getrefcount(obj) - UNREFERENCED_EXPR_COUNT, target)
    if getrefcount(obj) - UNREFERENCED_EXPR_COUNT == target:
        return obj
    elif getrefcount(obj) - UNREFERENCED_EXPR_COUNT > target:
        generate_expression.clone_counter += 1
        return obj.clone()
    else: #pragma:nocover
        raise RuntimeError("Expression entered generate_expression() " \
              "with too few references (%s<0); this is indicative of a " \
              "SERIOUS ERROR in the expression reuse detection scheme." \
              % ( getrefcount(obj) - UNREFERENCED_EXPR_COUNT, ))

def _generate_expression__noCloneCheck(obj, target):
    return obj

global _bypassing_clonecheck
_bypassing_clonecheck = False
def generate_expression_bypassCloneCheck(etype, _self, other):
    global _bypassing_clonecheck
    global _generate_expression__noCloneCheck
    global _generate_expression__clone_if_needed

    if _bypassing_clonecheck:
        return generate_expression(etype, _self, other)

    try:
        _bypassing_clonecheck = True
        # Swap the cloneCheck and no cloneCheck functions
        _generate_expression__noCloneCheck, \
            _generate_expression__clone_if_needed \
            = _generate_expression__clone_if_needed, \
              _generate_expression__noCloneCheck

        ans = generate_expression(etype, _self, other)

    finally:
        # Swap the cloneCheck and no cloneCheck functions back
        _generate_expression__noCloneCheck, \
            _generate_expression__clone_if_needed \
            = _generate_expression__clone_if_needed, \
              _generate_expression__noCloneCheck
        _bypassing_clonecheck = False

    return ans




_old_relational_strings = {
    '<'  : _lt,
    '<=' : _le,
    '==' : _eq,
    }

def generate_expression(etype, _self, other):
    if _self.is_indexed():
        raise TypeError("Argument for expression '%s' is an indexed "\
              "numeric value specified without an index: %s\n    Is "\
              "variable or parameter '%s' defined over an index that "\
              "you did not specify?" % (etype, _self.name, _self.name))

    self_type = _self.__class__
    # In-place operators should only clone `self` if someone else (other
    # than the self) holds a reference to it.  This also enforces that
    # there MUST be an external `self` reference for in-place operators.
    if etype > _inplace: #and etype < 2*_inplace:#etype[0] == 'i':
        #etype = etype[1:]
        etype -= _inplace
        if _self.is_expression():
            _self = _generate_expression__clone_if_needed(_self, 1)
        elif _self.is_constant():
            _self = _self()
            self_type = None
    else:
        if _self.is_expression():
            _self = _generate_expression__clone_if_needed(_self, 0)
        elif _self.is_constant():
            _self = _self()
            self_type = None

    #
    # First, handle the special cases of unary operators and
    # "degenerate" binary operators (subtraction)
    #
    #if etype in _generate_expression__specialCases:
    #    if etype[-1] == 'b': # This covers sub and rsub
    #        etype = etype[:-3] + 'add'
    #        multiplier = -1
    if etype >= _unary:
        if etype == _neg:
            if self_type is _SumExpression:
                _self.negate()
                return _self
            elif self_type is _ProductExpression:
                _self._coef *= -1
                return _self
            else:
                etype = _rmul#'rmul'
                other = -1
        elif etype == _abs:
            if self_type is None:
                return abs(_self)
            else:
                return _AbsExpression([_self])

    if other.__class__ in native_numeric_types:
        other_type = None
    else:
        try:
            other = other.as_numeric()
        except AttributeError:
            other = as_numeric(other)
        other_type = other.__class__
        if other.is_indexed():
            raise TypeError(
                "Argument for expression '%s' is an indexed numeric "
                "value\nspecified without an index:\n\t%s\nIs this "
                "value defined over an index that you did not specify?"
                % (etype, other.name, ) )
        if other.is_expression():
            other = _generate_expression__clone_if_needed(other, 0)
        elif other.is_constant():
            other = other()
            other_type = None

    #
    # Binary operators can either be "normal" or "reversed"; reverse all
    # "reversed" opertors
    #
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
        _self, other = other, _self
        self_type, other_type = other_type, self_type

    if etype == _sub:
        multiplier = -1
        etype = _add
    else:
        multiplier = 1

    #
    # Now, handle all binary operators
    #
    if etype == _add:
        #
        # self + other
        #
        if self_type is _SumExpression:
            if other_type is _ProductExpression \
                     and len(other._numerator) == 1 \
                     and not other._denominator:
                _self._args += other._numerator
                _self._coef.append(multiplier*other._coef)
                other._coef = 1
                other._numerator = None
                _ProdExpression_Pool.append(other)
            elif other_type is _SumExpression:
                if multiplier < 0:
                    other.negate()
                _self._args += other._args
                _self._coef += other._coef
                _self._const += other._const
                other._args = []
                other._coef = []
                other._const = 0
                _SumExpression_Pool.append(other)
            elif other_type is None: #NumericConstant:
                _self._const += multiplier * other
                if _self._const==0 and len(_self._coef)==1 and \
                        _self._coef[0]==1:
                    _self._coef = []
                    _SumExpression_Pool.append(_self)
                    return _self._args.pop()
            else:
                _self._args.append(other)
                _self._coef.append(multiplier)
            return _self
        elif other_type is _SumExpression:
            if multiplier < 0:
                other.negate()
            if self_type is _ProductExpression and \
                   len(_self._numerator) == 1 and \
                   not _self._denominator:
                _self._numerator += other._args
                other._args = _self._numerator
                other._coef.insert(0,_self._coef)
                _self._coef = 1
                _self._numerator = None
                _ProdExpression_Pool.append(_self)
            elif self_type is None: #NumericConstant:
                other._const += _self
                if other._const==0 and len(other._coef)==1 and \
                        other._coef[0]==1:
                    other._coef = []
                    _SumExpression_Pool.append(other)
                    return other._args.pop()
            else:
                other._args.insert(0,_self)
                other._coef.insert(0,1)
            return other
        else:
            if other_type is None:
                if self_type is None:
                    return _self + multiplier * other
                elif other == 0: #NB: multiplier doesn't matter (this is x-0)!
                    return _self
                elif _SumExpression_Pool:
                    ans = _SumExpression_Pool.pop()
                else:
                    ans = _SumExpression()
                ans._const = multiplier * other
                if self_type is _ProductExpression and \
                       len(_self._numerator) == 1 and \
                       not _self._denominator:
                    ans._coef = [ _self._coef ]
                    ans._args = _self._numerator
                    _self._coef = 1
                    _self._numerator = None
                    _ProdExpression_Pool.append(_self)
                else:
                    ans._coef = [ 1 ]
                    ans._args = [ _self ]
            elif self_type is None:
                if _self == 0 and multiplier == 1:
                    return other
                elif _SumExpression_Pool:
                    ans = _SumExpression_Pool.pop()
                else:
                    ans = _SumExpression()
                ans._const = _self
                if other_type is _ProductExpression and \
                       len(other._numerator) == 1 and \
                       not other._denominator:
                    ans._coef = [ multiplier * other._coef ]
                    ans._args = other._numerator
                    other._coef = 1
                    other._numerator = None
                    _ProdExpression_Pool.append(other)
                else:
                    ans._coef = [ multiplier ]
                    ans._args = [ other ]
            else:
                if _SumExpression_Pool:
                    ans = _SumExpression_Pool.pop()
                else:
                    ans = _SumExpression()
                ans._const = 0
                if self_type is _ProductExpression and \
                       len(_self._numerator) == 1 and \
                       not _self._denominator:
                    ans._coef = [ _self._coef ]
                    ans._args = _self._numerator
                    _self._coef = 1
                    _self._numerator = None
                    _ProdExpression_Pool.append(_self)
                else:
                    ans._coef = [ 1 ]
                    ans._args = [ _self ]
                if other_type is _ProductExpression and \
                       len(other._numerator) == 1 and \
                       not other._denominator:
                    ans._coef.append( multiplier * other._coef )
                    ans._args += other._numerator
                    other._coef = 1
                    other._numerator = None
                    _ProdExpression_Pool.append(other)
                else:
                    ans._coef.append( multiplier )
                    ans._args.append( other )
            return ans

    elif etype == _mul:#'mul':
        #
        # self * other
        #
        if self_type is _ProductExpression:
            if other_type is None: #NumericConstant:
                _self._coef *= other
            elif other_type is _ProductExpression:
                _self._numerator += other._numerator
                _self._denominator += other._denominator
                _self._coef *= other._coef
                other._coef = 1
                other._numerator = None
                other._denominator = None
                _ProdExpression_Pool.append(other)
            else:
                _self._numerator.append(other)
            ans = _self
        elif other_type is _ProductExpression:
            if self_type is None: #NumericConstant:
                other._coef *= _self
            else:
                other._numerator.insert(0,_self)
            ans = other
        else:
            if other_type is None:
                if self_type is None:
                    return _self * other
                elif other == 1:
                    return _self
                elif _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = other
                ans._numerator = [ _self ]
                ans._denominator = []
            elif self_type is None:
                if _self == 1:
                    return other
                elif _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = _self
                ans._numerator = [ other ]
                ans._denominator = []
            else:
                if _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = 1
                ans._numerator = [ _self, other ]
                ans._denominator = []

        # Special cases for simplifying expressions
        if ans._coef == 0:
            ans._coef = 1
            ans._numerator = None
            ans._denominator = None
            _ProdExpression_Pool.append(ans)
            return 0 #ZeroConstant
        return ans

    elif etype == _div:#'div':
        #
        # self / other
        #
        if self_type is _ProductExpression:
            if other_type is None: #NumericConstant:
                _self._coef /= other
            elif other_type is _ProductExpression:
                _self._numerator += other._denominator
                _self._denominator += other._numerator
                _self._coef /= other._coef
                other._coef = 1
                other._numerator = None
                other._denominator = None
                _ProdExpression_Pool.append(other)
            else:
                _self._denominator.append(other)
            return _self
        elif other_type is _ProductExpression:
            other.invert()
            if self_type is None: #NumericConstant:
                if _self == 0:
                    return 0
                else:
                    other._coef *= _self
                if len(other._denominator) == 0 and len(other._numerator) == 1\
                       and other._coef == 1:
                    _ProdExpression_Pool.append(other)
                    return other._numerator.pop()
            else:
                other._numerator.insert(0,_self)
            return other
        else:
            if other_type is None: #NumericConstant
                if self_type is None:
                    return _self / other
                elif other == 1:
                    return _self
                elif _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = 1/other
                ans._numerator = [ _self ]
                ans._denominator = []
            elif self_type is None:
                if _self == 0:
                    return 0
                elif _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = _self
                ans._numerator = []
                ans._denominator = [ other ]
            else:
                if _ProdExpression_Pool:
                    ans = _ProdExpression_Pool.pop()
                else:
                    ans = _ProductExpression()
                ans._coef = 1
                ans._numerator = [ _self ]
                ans._denominator = [ other ]
            return ans

    elif etype == _pow:#'pow':
        #
        # self ** other
        #
        if other_type is None: #NumericConstant:
            if self_type is None: #NumericConstant:
                # Special case: constant expressions should be
                # immediately reduced
                return _self ** other
            if other == 1:
                return _self
            if other == 0:
                return 1
            other = as_numeric(other)
        elif self_type is None: #NumericConstant:
            if _self == 0:
                return 0
            _self = as_numeric(_self)
        return _PowExpression((_self, other))

    else:
        if etype in _old_etype_strings:
            return generate_expression( _old_etype_strings[etype],
                                        as_numeric(_self), other )
        else:
            raise RuntimeError("Unknown expression type '%s'" % etype)

##
## "static" variables within the generate_expression function
##

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_expression.clone_counter = 0

# [configuration] UNREFERENCED_EXPR_COUNT is a "magic number" that
# indicates the stack depth between "normal" modeling and
# _clone_if_needed().  If an expression enters _clone_if_needed() with
# UNREFERENCED_EXPR_COUNT references, then there are no other variables
# that hold a reference to the expression and cloning is not necessary.
# If there are more references than UNREFERENCED_EXPR_COUNT, then we
# must clone the expression before operating on it.  It should be an
# error to hit _clone_if_needed() with fewer than
# UNREFERENCED_EXPR_COUNT references.
UNREFERENCED_EXPR_COUNT = 9


def _generate_relational_expression__clone_if_needed(obj):
    count = getrefcount(obj) - UNREFERENCED_RELATIONAL_EXPR_COUNT
    if count == 0:
        return obj
    elif count > 0:
        generate_relational_expression.clone_counter += 1
        return obj.clone()
    else:
        raise RuntimeError("Expression entered " \
              "generate_relational_expression() " \
              "with too few references (%s<0); this is indicative of a " \
              "SERIOUS ERROR in the expression reuse detection scheme." \
              % ( count, ))

def _generate_relational_expression__noCloneCheck(obj):
    return obj


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
        lhs = _generate_relational_expression__clone_if_needed(lhs)
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
        rhs = _generate_relational_expression__clone_if_needed(rhs)
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
            strict = False
        elif etype == _lt:
            strict = True
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
            return _InequalityExpression([lhs,rhs], [strict], cloned_from)


##
## "static" variables within the generate_relational_expression function
##

# [functionality] chainedInequality allows us to generate symbolic
# expressions of the type "a < b < c".  This provides a buffer to hold
# the first inequality so the second inequality can access it later.
generate_relational_expression.chainedInequality = None
generate_relational_expression.call_info = None

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_relational_expression.clone_counter = 0

# [configuration] UNREFERENCED_EXPR_COUNT is a "magic number" that
# indicates the stack depth between "normal" modeling and
# _clone_if_needed().  If an expression enters _clone_if_needed() with
# UNREFERENCED_EXPR_COUNT references, then there are no other variables
# that hold a reference to the expression and cloning is not necessary.
# If there are more references than UNREFERENCED_EXPR_COUNT, then we
# must clone the expression before operating on it.  It should be an
# error to hit _clone_if_needed() with fewer than
# UNREFERENCED_EXPR_COUNT references.
#
# Note: when generating compound inequality expressions of the type "a <
# b1+b2 < c", Python creates the inner expression and assigns it to a
# temporary before forming the first inequality.  After evaluating
# __nonzero__() on the inequality, the temporary is passed in when
# forming the compound inequality.  Unfortunately, that means that the
# inner expression is *always* cloned when forming the first half of the
# compound inequality.
UNREFERENCED_RELATIONAL_EXPR_COUNT = 9



def _generate_intrinsic_function_expression__clone_if_needed(obj):
    if getrefcount(obj) - UNREFERENCED_INTRINSIC_EXPR_COUNT == 0:
        return obj
    elif getrefcount(obj) - UNREFERENCED_INTRINSIC_EXPR_COUNT > 0:
        generate_intrinsic_function_expression.clone_counter += 1
        return obj.clone()
    else:
        raise RuntimeError("Expression entered " \
              "generate_intrinsic_function_expression() " \
              "with too few references (%s<0); this is indicative of a " \
              "SERIOUS ERROR in the expression reuse detection scheme." \
              % ( count, ))

def _generate_intrinsic_function_expression__noCloneCheck(obj):
    return obj

def generate_intrinsic_function_expression(arg, name, fcn):
    # Special handling: if there are no Pyomo Modeling Objects in the
    # argument list, then evaluate the expression and return the result.
    pyomo_expression = False
    # FIXME: does anyone know why we also test for 'Component' here? [JDS]
    if isinstance(arg, NumericValue) or isinstance(arg, Component):
        # TODO: efficiency: we already know this is a NumericValue -
        # so we should be able to avoid the call to as_numeric()
        # below (expecially since most intrinsic functions are unary
        # operators.
        pyomo_expression = True
    if not pyomo_expression:
        return fcn(arg)

    new_arg = as_numeric(arg)
    if new_arg.is_expression():
        new_arg = _generate_intrinsic_function_expression__clone_if_needed(new_arg)
    elif new_arg.is_indexed():
        raise ValueError("Argument for intrinsic function '%s' is an "\
            "n-ary numeric value: %s\n    Have you given variable or "\
            "parameter '%s' an index?" % (name, new_arg.name, new_arg.name))
    return _IntrinsicFunctionExpression(name, 1, (new_arg,), fcn)

# [debugging] clone_counter is a count of the number of calls to
# expr.clone() made during expression generation.
generate_intrinsic_function_expression.clone_counter = 0

# [configuration] UNREFERENCED_EXPR_COUNT is a "magic number" that
# indicates the stack depth between "normal" modeling and
# _clone_if_needed().  If an expression enters _clone_if_needed() with
# UNREFERENCED_EXPR_COUNT references, then there are no other variables
# that hold a reference to the expression and cloning is not necessary.
# If there are more references than UNREFERENCED_EXPR_COUNT, then we
# must clone the expression before operating on it.  It should be an
# error to hit _clone_if_needed() with fewer than
# UNREFERENCED_EXPR_COUNT references.
UNREFERENCED_INTRINSIC_EXPR_COUNT = 8

#
# If you want to completely disable clone checking (e.g., for
# line-profiling this file), set the following to "if True".
#
if not _getrefcount_available:
    _generate_relational_expression__clone_if_needed = \
        _generate_relational_expression__noCloneCheck
    _generate_expression__clone_if_needed = \
        _generate_expression__noCloneCheck
    _generate_intrinsic_function_expression__clone_if_needed = \
        _generate_intrinsic_function_expression__noCloneCheck

def _clear_expression_pool():
    global _SumExpression_Pool
    global _ProdExpression_Pool
    _SumExpression_Pool = []
    _ProdExpression_Pool = []
