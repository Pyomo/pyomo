#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

import math
import logging
from operator import attrgetter
from itertools import islice

logger = logging.getLogger('pyomo.core')

from math import isclose
from pyomo.common.deprecation import deprecated, deprecation_warning

from .expr_common import (
    OperatorAssociativity,
    ExpressionType,
    clone_counter,
    _add, _sub, _mul, _div,
    _pow, _neg, _abs, _inplace,
    _unary
)
from .base import ExpressionBase
from .numvalue import (
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    native_numeric_types,
    as_numeric,
    value,
    is_potentially_variable,
    is_constant,
)

from .visitor import (
    evaluate_expression, expression_to_string, polynomial_degree,
    clone_expression, sizeof_expression, _expression_is_fixed
)


class nonlinear_expression(object):
    """ Context manager for mutable sums.

    This context manager is used to compute a sum while
    treating the summation as a mutable object.
    """

    def __enter__(self):
        self.e = _MutableSumExpression([])
        return self.e

    def __exit__(self, *args):
        if self.e.__class__ == _MutableSumExpression:
            self.e.__class__ = SumExpression


class linear_expression(object):
    """ Context manager for mutable linear sums.

    This context manager is used to compute a linear sum while
    treating the summation as a mutable object.
    """

    def __enter__(self):
        """
        The :class:`_MutableLinearExpression <pyomo.core.expr.current._MutableLinearExpression>`
        class is the context that is used to to
        hold the mutable linear sum.
        """
        self.e = _MutableLinearExpression()
        return self.e

    def __exit__(self, *args):
        """
        The context is changed to the
        :class:`LinearExpression <pyomo.core.expr.current.LinearExpression>`
        class to transform the context into a nonmutable
        form.
        """
        if self.e.__class__ == _MutableLinearExpression:
            self.e.__class__ = LinearExpression


#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------


class NumericExpression(ExpressionBase, NumericValue):
    """
    The base class for Pyomo expressions.

    This class is used to define nodes in a numeric expression
    tree.

    Args:
        args (list or tuple): Children of this node.
    """

    # Previously, we used _args to define expression class arguments.
    # Here, we use _args_ to force errors for code that was referencing this
    # data.  There are now accessor methods, so in most cases users
    # and developers should not directly access the _args_ data values.
    __slots__ =  ('_args_',)
    EXPRESSION_SYSTEM = ExpressionType.NUMERIC
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        # by default, Pyomo numeric operators are binary operators
        return 2

    @property
    def args(self):
        """
        Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.
        """
        return self._args_

    @deprecated('The implicit recasting of a "not potentially variable" '
                'expression node to a potentially variable one is no '
                'longer supported (this violates that immutability '
                'promise for Pyomo5 expression trees).', version='6.4.3')
    def create_potentially_variable_object(self):
        """
        Create a potentially variable version of this object.

        This method returns an object that is a potentially variable
        version of the current object.  In the simplest
        case, this simply sets the value of `__class__`:

            self.__class__ = self.__class__.__mro__[1]

        Note that this method is allowed to modify the current object
        and return it.  But in some cases it may create a new
        potentially variable object.

        Returns:
            An object that is potentially variable.
        """
        if not self.is_potentially_variable():
            logger.error(
                'recasting a non-potentially variable expression to a '
                'potentially variable one violates the immutability '
                'promise for Pyomo5 expression trees.')
            cls = list(self.__class__.__bases__)
            cls.remove(NPV_Mixin)
            assert len(cls) == 1
            self.__class__ = cls[0]
        return self

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return polynomial_degree(self)

    def _compute_polynomial_degree(self, values):
        """
        Compute the polynomial degree of this expression given
        the degree values of its children.

        This method is called by the :class:`_PolynomialDegreeVisitor
        <pyomo.core.expr.current._PolynomialDegreeVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of values that indicate the degree
                of the children expression.

        Returns:
            A nonnegative integer that is the polynomial degree of the
            expression, or :const:`None`.  Default is :const:`None`.
        """
        return None


class NPV_Mixin(object):
    __slots__ = ()

    def is_potentially_variable(self):
        return False

    def create_node_with_local_data(self, args, classtype=None):
        assert classtype is None
        try:
            npv_args = all(
                type(arg) in native_types or not arg.is_potentially_variable()
                for arg in args
            )
        except AttributeError:
            # We can hit this during expression replacement when the new
            # type is not a PyomoObject type, but is not in the
            # native_types set.  We will play it safe and clear the NPV flag
            npv_args = False
        if npv_args:
            return super().create_node_with_local_data(args, None)
        else:
            return super().create_node_with_local_data(
                args, self.potentially_variable_base_class())

    def potentially_variable_base_class(self):
        cls = list(self.__class__.__bases__)
        cls.remove(NPV_Mixin)
        assert len(cls) == 1
        return cls[0]


class NegationExpression(NumericExpression):
    """
    Negation expressions::

        - x
    """

    __slots__ = ()

    PRECEDENCE = 4

    def nargs(self):
        return 1

    def getname(self, *args, **kwds):
        return 'neg'

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({values[0]})"
        tmp = values[0]
        if tmp[0] == '-':
            i = 1
            while tmp[i] == ' ':
                i += 1
            return tmp[i:]
        return "- " + tmp

    def _apply_operation(self, result):
        return -result[0]


class NPV_NegationExpression(NPV_Mixin, NegationExpression):
    __slots__ = ()


class ExternalFunctionExpression(NumericExpression):
    """
    External function expressions

    Example::

        model = ConcreteModel()
        model.a = Var()
        model.f = ExternalFunction(library='foo.so', function='bar')
        expr = model.f(model.a)

    Args:
        args (tuple): children of this node
        fcn: a class that defines this external function
    """
    __slots__ = ('_fcn',)

    def __init__(self, args, fcn=None):
        self._args_ = args
        self._fcn = fcn

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._fcn)

    def getname(self, *args, **kwds):           #pragma: no cover
        return self._fcn.getname(*args, **kwds)

    def _compute_polynomial_degree(self, result):
        return 0 if all(arg == 0 for arg in result) else None

    def _apply_operation(self, result):
        return self._fcn.evaluate( result )

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"

    def get_arg_units(self):
        """ Return the units for this external functions arguments """
        return self._fcn.get_arg_units()

    def get_units(self):
        """ Get the units of the return value for this external function """
        return self._fcn.get_units()

class NPV_ExternalFunctionExpression(NPV_Mixin, ExternalFunctionExpression):
    __slots__ = ()


class PowExpression(NumericExpression):
    """
    Power expressions::

        x**y
    """

    __slots__ = ()
    PRECEDENCE = 2

    # "**" is right-to-left associative in Python (so this should
    # return -1), however, as this rule is not widely known and can
    # confuse novice users, we will make our "**" operator
    # non-associative (forcing parens)
    ASSOCIATIVITY = OperatorAssociativity.NON_ASSOCIATIVE

    def _compute_polynomial_degree(self, result):
        # PowExpression is a tricky thing.  In general, a**b is
        # nonpolynomial, however, if b == 0, it is a constant
        # expression, and if a is polynomial and b is a positive
        # integer, it is also polynomial.  While we would like to just
        # call this a non-polynomial expression, these exceptions occur
        # too frequently (and in particular, a**2)
        l,r = result
        if r == 0:
            if l == 0:
                return 0
            # NOTE: use value before int() so that we don't
            #       run into the disabled __int__ method on
            #       NumericValue
            exp = value(self._args_[1], exception=False)
            if exp is None:
                return None
            if exp == int(exp):
                if l is not None and exp > 0:
                    return l * exp
                elif exp == 0:
                    return 0
        return None

    def _is_fixed(self, args):
        assert(len(args) == 2)
        if not args[1]:
            return False
        return args[0] or value(self._args_[1]) == 0

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        return f"{values[0]}**{values[1]}"


class NPV_PowExpression(NPV_Mixin, PowExpression):
    __slots__ = ()


class MaxExpression(NumericExpression):
    """
    Maximum expressions::

        max(x, y, ...)
    """

    __slots__ = ()

    # This operator does not have an infix representation
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def _apply_operation(self, result):
        return max(result)

    def getname(self, *args, **kwds):
        return 'max'

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"


class NPV_MaxExpression(NPV_Mixin, MaxExpression):
    __slots__ = ()


class MinExpression(NumericExpression):
    """
    Minimum expressions::

        min(x, y, ...)
    """

    __slots__ = ()

    # This operator does not have an infix representation
    PRECEDENCE = None

    def nargs(self):
        return len(self._args_)

    def _apply_operation(self, result):
        return min(result)

    def getname(self, *args, **kwds):
        return 'min'

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"


class NPV_MinExpression(NPV_Mixin, MinExpression):
    __slots__ = ()


class ProductExpression(NumericExpression):
    """
    Product expressions::

        x*y
    """

    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
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

    def _is_fixed(self, args):
        # Anything times 0 equals 0, so one of the children is
        # fixed and has a value of 0, then this expression is fixed
        assert(len(args) == 2)
        if all(args):
            return True
        for i in (0, 1):
            if args[i] and value(self._args_[i]) == 0:
                return True
        return False

    def _apply_operation(self, result):
        _l, _r = result
        return _l * _r

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        if values[0] in self._to_string.one:
            return values[1]
        if values[0] in self._to_string.minus_one:
            return f"- {values[1]}"
        return f"{values[0]}*{values[1]}"

    # Store these reference sets on the function for quick lookup
    _to_string.one = {"1", "1.0", "(1)", "(1.0)"}
    _to_string.minus_one = {"-1", "-1.0", "(-1)", "(-1.0)"}


class NPV_ProductExpression(NPV_Mixin, ProductExpression):
    __slots__ = ()


class MonomialTermExpression(ProductExpression):
    __slots__ = ()

    def getname(self, *args, **kwds):
        return 'mon'

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            # If this doesn't look like a MonomialTermExpression, then
            # fall back on the expression generation system to sort out
            # what the appropriate return type is.
            try:
                if not (args[0].__class__ in native_types
                        or not args[0].is_potentially_variable()):
                    return args[0] * args[1]
                elif (args[1].__class__ in native_types
                      or not args[1].is_variable_type()):
                    return args[0] * args[1]
            except AttributeError:
                # Fall back on general expression generation
                return args[0] * args[1]
        return self.__class__(args)


class DivisionExpression(NumericExpression):
    """
    Division expressions::

        x/y
    """
    __slots__ = ()
    PRECEDENCE = 4

    def _compute_polynomial_degree(self, result):
        if result[1] == 0:
            return result[0]
        return None

    def getname(self, *args, **kwds):
        return 'div'

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        return f"{values[0]}/{values[1]}"

    def _apply_operation(self, result):
        return result[0] / result[1]


class NPV_DivisionExpression(NPV_Mixin, DivisionExpression):
    __slots__ = ()


class SumExpressionBase(NumericExpression):
    """
    A base class for simple summation of expressions

    The class hierarchy for summation is different than for other
    expression types.  For example, ProductExpression defines
    the class for representing binary products, and sub-classes are
    specializations of that class.

    By contrast, the SumExpressionBase is not directly used to
    represent expressions.  Rather, this base class provides
    commonly used methods and data.  The reason is that some
    subclasses of SumExpressionBase are binary while others
    are n-ary.

    Thus, developers will need to treat checks for summation
    classes differently, depending on whether the binary/n-ary
    operations are different.
    """

    __slots__ = ()
    PRECEDENCE = 6

    def getname(self, *args, **kwds):
        return 'sum'

    def _apply_operation(self, result):
        return sum(result)

    def _compute_polynomial_degree(self, result):
        # NB: We can't use max() here because None (non-polynomial)
        # overrides a numeric value (and max() just ignores it)
        ans = 0
        for x in result:
            if x is None:
                return None
            elif ans < x:
                ans = x
        return ans

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"

        for i in range(1, len(values)):
            val = values[i]
            if val[0] == '-':
                values[i] = ' - ' + val[1:].strip()
            elif len(val) > 3 and val[:2] == '(-' and val[-1] == ')' \
                 and _balanced_parens(val[1:-1]):
                values[i] = ' - ' + val[2:-1].strip()
            else:
                values[i] = ' + ' + val
        return ''.join(values)


class SumExpression(SumExpressionBase):
    """
    Sum expression::

        x + y

    Args:
        args (list): Children nodes
    """
    __slots__ = ('_nargs','_shared_args')
    PRECEDENCE = 6

    def __init__(self, args):
        self._args_ = args
        self._shared_args = False
        self._nargs = len(self._args_)

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and new_arg == 0:
            return self
        # Clone 'self', because SumExpression are immutable
        self._shared_args = True
        self = self.__class__(self._args_)
        #
        if new_arg.__class__ is SumExpression or new_arg.__class__ is _MutableSumExpression:
            self._args_.extend( islice(new_arg._args_, new_arg._nargs) )
        elif not new_arg is None:
            self._args_.append(new_arg)
        self._nargs = len(self._args_)
        return self

    def nargs(self):
        return self._nargs

    @property
    def args(self):
        return self._args_[:self._nargs]

    def create_node_with_local_data(self, args, classtype=None):
        return super().create_node_with_local_data(list(args), classtype)


class NPV_SumExpression(NPV_Mixin, SumExpression):
    __slots__ = ()

    def create_potentially_variable_object(self):
        return SumExpression( self._args_ )

    def _apply_operation(self, result):
        l_, r_ = result
        return l_ + r_

    def _to_string(self, values, verbose, smap):
        if verbose:
            return f"{self.getname()}({', '.join(values)})"
        if values[1][0] == '-':
            return f"{values[0]} {values[1]}"
        return f"{values[0]} + {values[1]}"

    def create_node_with_local_data(self, args, classtype=None):
        assert classtype is None
        try:
            npv_args = all(
                type(arg) in native_types or not arg.is_potentially_variable()
                for arg in args
            )
        except AttributeError:
            # We can hit this during expression replacement when the new
            # type is not a PyomoObject type, but is not in the
            # native_types set.  We will play it safe and clear the NPV flag
            npv_args = False
        if npv_args:
            return NPV_SumExpression(args)
        else:
            return SumExpression(args)


class _MutableSumExpression(SumExpression):
    """
    A mutable SumExpression

    The :func:`add` method is slightly different in that it
    does not create a new sum expression, but modifies the
    :attr:`_args_` data in place.
    """

    __slots__ = ()

    def add(self, new_arg):
        if new_arg.__class__ in native_numeric_types and new_arg == 0:
            return self
        # Do not clone 'self', because _MutableSumExpression are mutable
        #self._shared_args = True
        #self = self.__class__(list(self.args))
        #
        if new_arg.__class__ is SumExpression or new_arg.__class__ is _MutableSumExpression:
            self._args_.extend( islice(new_arg._args_, new_arg._nargs) )
        elif not new_arg is None:
            self._args_.append(new_arg)
        self._nargs = len(self._args_)
        return self


class Expr_ifExpression(NumericExpression):
    """
    A logical if-then-else expression::

        Expr_if(IF_=x, THEN_=y, ELSE_=z)

    Args:
        IF_ (expression): A relational expression
        THEN_ (expression): An expression that is used if :attr:`IF_` is true.
        ELSE_ (expression): An expression that is used if :attr:`IF_` is false.
    """
    __slots__ = ('_if','_then','_else')

    # This operator does not have an infix representation
    PRECEDENCE = None

    # **NOTE**: This class evaluates the branching "_if" expression
    #           on a number of occasions. It is important that
    #           one uses __call__ for value() and NOT bool().

    def __init__(self, IF_=None, THEN_=None, ELSE_=None):
        if type(IF_) is tuple and THEN_==None and ELSE_==None:
            IF_, THEN_, ELSE_ = IF_
        self._args_ = (IF_, THEN_, ELSE_)
        self._if = IF_
        self._then = THEN_
        self._else = ELSE_
        if self._if.__class__ in native_numeric_types:
            self._if = as_numeric(self._if)

    def nargs(self):
        return 3

    def getname(self, *args, **kwds):
        return "Expr_if"

    def _is_fixed(self, args):
        assert(len(args) == 3)
        if args[1] and args[2]:
            return True
        if args[0]: # self._if.is_fixed():
            if value(self._if):
                return args[1] # self._then.is_fixed()
            else:
                return args[2] # self._else.is_fixed()
        else:
            return False

    def is_potentially_variable(self):
        return any(map(is_potentially_variable, self._args_))

    def _compute_polynomial_degree(self, result):
        _if, _then, _else = result
        if _if == 0:
            if _then == _else:
                return _then
            try:
                return _then if value(self._if) else _else
            except ValueError:
                pass
        return None

    def _to_string(self, values, verbose, smap):
        return f'{self.getname()}( ( {values[0]} ), then=( {values[1]} ), ' \
            f'else=( {values[2]} ) )'

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class UnaryFunctionExpression(NumericExpression):
    """
    An expression object used to define intrinsic functions (e.g. sin, cos, tan).

    Args:
        args (tuple): Children nodes
        name (string): The function name
        fcn: The function that is used to evaluate this expression
    """
    __slots__ = ('_fcn', '_name')

    # This operator does not have an infix representation
    PRECEDENCE = None

    def __init__(self, args, name=None, fcn=None):
        if type(args) is not tuple:
            args = (args,)
        self._args_ = args
        self._name = name
        self._fcn = fcn

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._name, self._fcn)

    def getname(self, *args, **kwds):
        return self._name

    def _to_string(self, values, verbose, smap):
        return f"{self.getname()}({', '.join(values)})"

    def _compute_polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result[0])


class NPV_UnaryFunctionExpression(NPV_Mixin, UnaryFunctionExpression):
    __slots__ = ()


# NOTE: This should be a special class, since the expression generation relies
# on the Python __abs__ method.
class AbsExpression(UnaryFunctionExpression):
    """
    An expression object for the :func:`abs` function.

    Args:
        args (tuple): Children nodes
    """
    __slots__ = ()

    def __init__(self, arg):
        super(AbsExpression, self).__init__(arg, 'abs', abs)

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args)


class NPV_AbsExpression(NPV_Mixin, AbsExpression):
    __slots__ = ()


class LinearExpression(NumericExpression):
    """
    An expression object linear polynomials.

    Args:
        args (tuple): Children nodes
    """
    __slots__ = (
        'constant',          # The constant term
        'linear_coefs',      # Linear coefficients
        'linear_vars',       # Linear variables
        '_args_cache_',
    )

    PRECEDENCE = 6

    def __init__(self, args=None, constant=None, linear_coefs=None, linear_vars=None):
        """A linear expression of the form `const + sum_i(c_i*x_i).

        You can specify args OR (constant, linear_coefs, and
        linear_vars).  If args is provided, it should be a list that
        contains the constant, followed by a series of
        :py:class:`MonomialTermExpression` objects. Alternatively, you
        can specify the constant, the list of linear_coeffs and the list
        of linear_vars separately. Note that these lists are NOT copied.

        """
        # I am not sure why LinearExpression allows omitting args, but
        # it does.  If they are provided, they should be the (non-zero)
        # constant followed by MonomialTermExpressions.
        if args:
            if any(arg is not None for arg in
                   (constant, linear_coefs, linear_vars)):
                raise ValueError("Cannot specify both args and any of "
                                 "{constant, linear_coeffs, or linear_vars}")
            if len(args) > 1 and (args[1].__class__ in native_types
                                  or not args[1].is_potentially_variable()):
                deprecation_warning(
                    "LinearExpression has been updated to expect args= to "
                    "be a constant followed by MonomialTermExpressions.  "
                    "The older format (`[const, coefficient_1, ..., "
                    "variable_1, ...]`) is deprecated.", version='6.2')
                args = args[:1] + list(map(
                    MonomialTermExpression,
                    zip(args[1:1+len(args)//2], args[1+len(args)//2:])))
            self._args_ = args
        else:
            self.constant = constant if constant is not None else 0
            self.linear_coefs = linear_coefs if linear_coefs else []
            self.linear_vars = linear_vars if linear_vars else []
            self._args_cache_ = []

    def nargs(self):
        return len(self.linear_vars) + (
            0 if (self.constant is None
                  or (self.constant.__class__ in native_numeric_types
                      and not self.constant)) else 1
        )

    @property
    def _args_(self):
        nargs = self.nargs()
        if len(self._args_cache_) != nargs:
            if len(self.linear_vars) == nargs:
                self._args_cache_ = []
            else:
                self._args_cache_ = [self.constant]
            self._args_cache_.extend(
                map(MonomialTermExpression,
                    zip(self.linear_coefs, self.linear_vars)))
        elif len(self.linear_vars) != nargs:
            self._args_cache_[0] = self.constant
        return self._args_cache_

    @_args_.setter
    def _args_(self, value):
        self._args_cache_ = list(value)
        if not self._args_cache_:
            self.constant = 0
            self.linear_coefs = []
            self.linear_vars = []
            return
        if self._args_cache_[0].__class__ is not MonomialTermExpression:
            self.constant = value[0]
            first_var = 1
        else:
            self.constant = 0
            first_var = 0
        self.linear_coefs, self.linear_vars = zip(
            *map(attrgetter('args'), value[first_var:]))
        self.linear_coefs = list(self.linear_coefs)
        self.linear_vars = list(self.linear_vars)

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is not None:
            return classtype(args)
        else:
            const = 0
            new_args = []
            for arg in args:
                if arg.__class__ is MonomialTermExpression:
                    new_args.append(arg)
                elif arg.__class__ in native_types or arg.is_constant():
                    const += arg
                else:
                    return SumExpression(args)
            if not new_args:
                return const
            if const:
                new_args.insert(0, const)
            return self.__class__(new_args)

    def getname(self, *args, **kwds):
        return 'sum'

    def _compute_polynomial_degree(self, result):
        return 1 if not self.is_fixed() else 0

    def _is_fixed(self, values=None):
        return all(v.fixed for v in self.linear_vars)

    def is_fixed(self):
        return self._is_fixed()

    def _to_string(self, values, verbose, smap):
        if not values:
            values = ['0']
        if verbose:
            return f"{self.getname()}({', '.join(values)})"

        for i in range(1, len(values)):
            term = values[i]
            if term[0] not in '+-':
                values[i] = '+ ' + term
            elif term[1] != ' ':
                values[i] = term[0] + ' ' + term[1:]
        return ' '.join(values)

    def is_potentially_variable(self):
        return len(self.linear_vars) > 0

    def _apply_operation(self, result):
        return sum(result)

    #@profile
    def _combine_expr(self, etype, _other):
        if etype == _add or etype == _sub or etype == -_add or etype == -_sub:
            #
            # if etype == _sub,  then _MutableLinearExpression - VAL
            # if etype == -_sub, then VAL - _MutableLinearExpression
            #
            if etype == _sub:
                omult = -1
            else:
                omult = 1
            if etype == -_sub:
                self.constant *= -1
                for i,c in enumerate(self.linear_coefs):
                    self.linear_coefs[i] = -c

            if _other.__class__ in native_numeric_types or not _other.is_potentially_variable():
                self.constant = self.constant + omult * _other
            #
            # WEH - These seem like uncommon cases, so I think we should defer processing them
            #       until _decompose_linear_terms
            #
            #elif _other.__class__ is _MutableLinearExpression:
            #    self.constant = self.constant + omult * _other.constant
            #    for c,v in zip(_other.linear_coefs, _other.linear_vars):
            #        self.linear_coefs.append(omult*c)
            #        self.linear_vars.append(v)
            #elif _other.__class__ is SumExpression or _other.__class__ is _MutableSumExpression:
            #    for e in _other._args_:
            #        for c,v in _decompose_linear_terms(e, multiplier=omult):
            #            if v is None:
            #                self.constant += c
            #            else:
            #                self.linear_coefs.append(c)
            #                self.linear_vars.append(v)
            else:
                for c,v in _decompose_linear_terms(_other, multiplier=omult):
                    if v is None:
                        self.constant += c
                    else:
                        self.linear_coefs.append(c)
                        self.linear_vars.append(v)

        elif etype == _mul or etype == -_mul:
            if _other.__class__ in native_numeric_types:
                multiplier = _other
            elif _other.is_potentially_variable():
                if len(self.linear_vars) > 0:
                    raise ValueError("Cannot multiply a linear expression with a variable expression")
                #
                # The linear expression is a constant, so re-initialize it with
                # a single term that multiplies the expression by the constant value.
                #
                c_ = self.constant
                self.constant = 0
                for c,v in _decompose_linear_terms(_other):
                    if v is None:
                        self.constant = c*c_
                    else:
                        self.linear_vars.append(v)
                        self.linear_coefs.append(c*c_)
                return self
            else:
                multiplier = _other

            if multiplier.__class__ in native_numeric_types and multiplier == 0:
                self.constant = 0
                self.linear_vars = []
                self.linear_coefs = []
            else:
                self.constant *= multiplier
                for i,c in enumerate(self.linear_coefs):
                    self.linear_coefs[i] = c*multiplier

        elif etype == _div:
            if _other.__class__ in native_numeric_types:
                divisor = _other
            elif _other.is_potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            else:
                divisor = _other
            self.constant /= divisor
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = c/divisor

        elif etype == -_div:
            if self.is_potentially_variable():
                raise ValueError("Unallowed operation on linear expression: division with a variable RHS")
            return _other / self.constant

        elif etype == _neg:
            self.constant *= -1
            for i,c in enumerate(self.linear_coefs):
                self.linear_coefs[i] = - c

        else:
            raise ValueError("Unallowed operation on mutable linear expression: %d" % etype)    #pragma: no cover

        return self


class _MutableLinearExpression(LinearExpression):
    __slots__ = ()


#-------------------------------------------------------
#
# Functions used to generate expressions
#
#-------------------------------------------------------

def decompose_term(expr):
    """
    A function that returns a tuple consisting of (1) a flag indicated
    whether the expression is linear, and (2) a list of tuples that
    represents the terms in the linear expression.

    Args:
        expr (expression): The root node of an expression tree

    Returns:
        A tuple with the form ``(flag, list)``.  If :attr:`flag` is :const:`False`, then
        a nonlinear term has been found, and :const:`list` is :const:`None`.
        Otherwise, :const:`list` is a list of tuples: ``(coef, value)``.
        If :attr:`value` is :const:`None`, then this
        represents a constant term with value :attr:`coef`.  Otherwise,
        :attr:`value` is a variable object, and :attr:`coef` is the
        numeric coefficient.
    """
    if expr.__class__ in nonpyomo_leaf_types or not expr.is_potentially_variable():
        return True, [(expr,None)]
    elif expr.is_variable_type():
        return True, [(1,expr)]
    else:
        try:
            terms = [t_ for t_ in _decompose_linear_terms(expr)]
            return True, terms
        except LinearDecompositionError:
            return False, None

class LinearDecompositionError(Exception):

    def __init__(self, message):
        super(LinearDecompositionError, self).__init__(message)


def _decompose_linear_terms(expr, multiplier=1):
    """
    A generator function that yields tuples for the linear terms
    in an expression.  If nonlinear terms are encountered, this function
    raises the :class:`LinearDecompositionError` exception.

    Args:
        expr (expression): The root node of an expression tree

    Yields:
        Tuples: ``(coef, value)``.  If :attr:`value` is :const:`None`,
        then this represents a constant term with value :attr:`coef`.
        Otherwise, :attr:`value` is a variable object, and :attr:`coef`
        is the numeric coefficient.

    Raises:
        :class:`LinearDecompositionError` if a nonlinear term is encountered.
    """
    if expr.__class__ in native_numeric_types or not expr.is_potentially_variable():
        yield (multiplier*expr,None)
    elif expr.is_variable_type():
        yield (multiplier,expr)
    elif expr.__class__ is MonomialTermExpression:
        yield (multiplier*expr._args_[0], expr._args_[1])
    elif expr.__class__ is ProductExpression:
        if expr._args_[0].__class__ in native_numeric_types or not expr._args_[0].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[1], multiplier*expr._args_[0])
        elif expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[0], multiplier*expr._args_[1])
        else:
            raise LinearDecompositionError("Quadratic terms exist in a product expression.")
    elif expr.__class__ is DivisionExpression:
        if expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            yield from _decompose_linear_terms(expr._args_[0], multiplier/expr._args_[1])
        else:
            raise LinearDecompositionError("Unexpected nonlinear term (division)")
    elif expr.__class__ is SumExpression or expr.__class__ is _MutableSumExpression:
        for arg in expr.args:
            yield from _decompose_linear_terms(arg, multiplier)
    elif expr.__class__ is NegationExpression:
        yield from _decompose_linear_terms(expr._args_[0], -multiplier)
    elif expr.__class__ is LinearExpression or expr.__class__ is _MutableLinearExpression:
        if not (expr.constant.__class__ in native_numeric_types and expr.constant == 0):
            yield (multiplier*expr.constant,None)
        if len(expr.linear_coefs) > 0:
            for c,v in zip(expr.linear_coefs, expr.linear_vars):
                yield (multiplier*c,v)
    else:
        raise LinearDecompositionError("Unexpected nonlinear term")   #pragma: no cover


def _process_arg(obj):
    # Note: caller is responsible for filtering out native types and
    # expressions.
    if not obj.is_numeric_type():
        if hasattr(obj, 'as_binary'):
            # We assume non-numeric types that have an as_binary method
            # are instances of AutoLinkedBooleanVar.  Calling as_binary
            # will return a valid Binary Var (and issue the appropriate
            # deprecation warning)
            obj = obj.as_binary()
        else:
            # User assistance: provide a helpful exception when using an
            # indexed object in an expression
            if obj.is_component_type() and obj.is_indexed():
                raise TypeError(
                    "Argument for expression is an indexed numeric "
                    "value\nspecified without an index:\n\t%s\nIs this "
                    "value defined over an index that you did not specify?"
                    % (obj.name, ) )

            raise TypeError(
                "Attempting to use a non-numeric type (%s) in a "
                "numeric context." % (obj.__class__.__name__,))
    elif obj.is_constant():
        # Resolve constants (e.g., immutable scalar Params & NumericConstants)
        return value(obj)
    return obj


#@profile
def _generate_sum_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _MutableLinearExpression:
        try:
            if etype >= _unary:
                return _self._combine_expr(etype, None)
            if _other.__class__ is not _MutableLinearExpression:
                if not (_other.__class__ in native_types or _other.is_expression_type()):
                    _other = _process_arg(_other)
            return _self._combine_expr(etype, _other)
        except LinearDecompositionError:
            pass
    elif _other.__class__ is _MutableLinearExpression:
        try:
            if not (_self.__class__ in native_types or _self.is_expression_type()):
                _self = _process_arg(_self)
            return _other._combine_expr(-etype, _self)
        except LinearDecompositionError:
            pass

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    if etype == _neg:
        if _self.__class__ in native_numeric_types:
            return - _self
        elif _self.__class__ is MonomialTermExpression:
            tmp = _self._args_[0]
            if tmp.__class__ in native_numeric_types:
                return MonomialTermExpression((-tmp, _self._args_[1]))
            else:
                return MonomialTermExpression((NPV_NegationExpression((tmp,)), _self._args_[1]))
        elif _self.is_variable_type():
            return MonomialTermExpression((-1, _self))
        elif _self.is_potentially_variable():
            return NegationExpression((_self,))
        else:
            if _self.__class__ is NPV_NegationExpression:
                return _self._args_[0]
            return NPV_NegationExpression((_self,))

    if not (_other.__class__ in native_types or _other.is_expression_type()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _add:
        #
        # x + y
        #
        if (_self.__class__ is SumExpression and not _self._shared_args) or \
           _self.__class__ is _MutableSumExpression:
            return _self.add(_other)
        elif (_other.__class__ is SumExpression and not _other._shared_args) or \
            _other.__class__ is _MutableSumExpression:
            return _other.add(_self)
        elif _other.__class__ in native_numeric_types:
            if _other == 0:
                return _self
            if _self.__class__ in native_numeric_types:
                return _self + _other
            if _self.is_potentially_variable():
                return SumExpression([_self, _other])
            return NPV_SumExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return _other
            if _other.is_potentially_variable():
                #return _LinearSumExpression((_self, _other))
                return SumExpression([_self, _other])
            return NPV_SumExpression((_self, _other))
        elif _other.is_potentially_variable():
            #return _LinearSumExpression((_self, _other))
            return SumExpression([_self, _other])
        elif _self.is_potentially_variable():
            #return _LinearSumExpression((_other, _self))
            #return SumExpression([_other, _self])
            return SumExpression([_self, _other])
        else:
            return NPV_SumExpression((_self, _other))

    elif etype == _sub:
        #
        # x - y
        #
        if (_self.__class__ is SumExpression and not _self._shared_args) or \
           _self.__class__ is _MutableSumExpression:
            return _self.add(-_other)
        elif _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self - _other
            elif _other == 0:
                return _self
            if _self.is_potentially_variable():
                return SumExpression([_self, -_other])
            return NPV_SumExpression((_self, -_other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                if _other.__class__ is MonomialTermExpression:
                    tmp = _other._args_[0]
                    if tmp.__class__ in native_numeric_types:
                        return MonomialTermExpression((-tmp, _other._args_[1]))
                    return MonomialTermExpression((NPV_NegationExpression((_other._args_[0],)), _other._args_[1]))
                elif _other.is_variable_type():
                    return MonomialTermExpression((-1, _other))
                elif _other.is_potentially_variable():
                    return NegationExpression((_other,))
                return NPV_NegationExpression((_other,))
            elif _other.__class__ is MonomialTermExpression:
                return SumExpression([_self, MonomialTermExpression((-_other._args_[0], _other._args_[1]))])
            elif _other.is_variable_type():
                return SumExpression([_self, MonomialTermExpression((-1,_other))])
            elif _other.is_potentially_variable():
                return SumExpression([_self, NegationExpression((_other,))])
            return NPV_SumExpression((_self, NPV_NegationExpression((_other,))))
        elif _other.__class__ is MonomialTermExpression:
            return SumExpression([_self, MonomialTermExpression((-_other._args_[0], _other._args_[1]))])
        elif _other.is_variable_type():
            return SumExpression([_self, MonomialTermExpression((-1,_other))])
        elif _other.is_potentially_variable():
            return SumExpression([_self, NegationExpression((_other,))])
        elif _self.is_potentially_variable():
            return SumExpression([_self, NPV_NegationExpression((_other,))])
        else:
            return NPV_SumExpression((_self, NPV_NegationExpression((_other,))))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover

#@profile
def _generate_mul_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    if _self.__class__ is _MutableLinearExpression:
        try:
            if _other.__class__ is not _MutableLinearExpression:
                if not (_other.__class__ in native_types or _other.is_expression_type()):
                    _other = _process_arg(_other)
            return _self._combine_expr(etype, _other)
        except LinearDecompositionError:
            pass
    elif _other.__class__ is _MutableLinearExpression:
        try:
            if not (_self.__class__ in native_types or _self.is_expression_type()):
                _self = _process_arg(_self)
            return _other._combine_expr(-etype, _self)
        except LinearDecompositionError:
            pass

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    if not (_other.__class__ in native_types or _other.is_expression_type()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _mul:
        #
        # x * y
        #
        if _other.__class__ in native_numeric_types:
            if _self.__class__ in native_numeric_types:
                return _self * _other
            elif _other == 0:
                return 0
            elif _other == 1:
                return _self
            if _self.is_variable_type():
                return MonomialTermExpression((_other, _self))
            elif _self.__class__ is MonomialTermExpression:
                tmp = _self._args_[0]
                if tmp.__class__ in native_numeric_types:
                    return MonomialTermExpression((_other*tmp, _self._args_[1]))
                else:
                    return MonomialTermExpression((NPV_ProductExpression((_other,tmp)), _self._args_[1]))
            elif _self.is_potentially_variable():
                return ProductExpression((_self, _other))
            return NPV_ProductExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return 0
            elif _self == 1:
                return _other
            if _other.is_variable_type():
                return MonomialTermExpression((_self, _other))
            elif _other.__class__ is MonomialTermExpression:
                tmp = _other._args_[0]
                if tmp.__class__ in native_numeric_types:
                    return MonomialTermExpression((_self*tmp, _other._args_[1]))
                else:
                    return MonomialTermExpression((NPV_ProductExpression((_self,tmp)), _other._args_[1]))
            elif _other.is_potentially_variable():
                return ProductExpression((_self, _other))
            return NPV_ProductExpression((_self, _other))
        elif _other.is_variable_type():
            if _self.is_potentially_variable():
                return ProductExpression((_self, _other))
            return MonomialTermExpression((_self, _other))
        elif _other.is_potentially_variable():
            return ProductExpression((_self, _other))
        elif _self.is_variable_type():
            return MonomialTermExpression((_other, _self))
        elif _self.is_potentially_variable():
            return ProductExpression((_self, _other))
        else:
            return NPV_ProductExpression((_self, _other))

    elif etype == _div:
        #
        # x / y
        #
        if _other.__class__ in native_numeric_types:
            if _other == 1:
                return _self
            elif not _other:
                raise ZeroDivisionError()
            elif _self.__class__ in native_numeric_types:
                return _self / _other
            if _self.is_variable_type():
                return MonomialTermExpression((1/_other, _self))
            elif _self.__class__ is MonomialTermExpression:
                return MonomialTermExpression((_self._args_[0]/_other, _self._args_[1]))
            elif _self.is_potentially_variable():
                return DivisionExpression((_self, _other))
            return NPV_DivisionExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _self == 0:
                return 0
            elif _other.is_potentially_variable():
                return DivisionExpression((_self, _other))
            return NPV_DivisionExpression((_self, _other))
        elif _other.is_potentially_variable():
            return DivisionExpression((_self, _other))
        elif _self.is_potentially_variable():
            if _self.is_variable_type():
                return MonomialTermExpression((NPV_DivisionExpression((1, _other)), _self))
            return DivisionExpression((_self, _other))
        else:
            return NPV_DivisionExpression((_self, _other))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover


#@profile
def _generate_other_expression(etype, _self, _other):

    if etype > _inplace:
        etype -= _inplace

    #
    # A mutable sum is used as a context manager, so we don't
    # need to process it to see if it's entangled.
    #
    if not (_self.__class__ in native_types or _self.is_expression_type()):
        _self = _process_arg(_self)

    #
    # abs(x)
    #
    if etype == _abs:
        if _self.__class__ in native_numeric_types:
            return abs(_self)
        elif _self.is_potentially_variable():
            return AbsExpression(_self)
        else:
            return NPV_AbsExpression(_self)

    if not (_other.__class__ in native_types or _other.is_expression_type()):
        _other = _process_arg(_other)

    if etype < 0:
        #
        # This may seem obvious, but if we are performing an
        # "R"-operation (i.e. reverse operation), then simply reverse
        # self and other.  This is legitimate as we are generating a
        # completely new expression here.
        #
        etype *= -1
        _self, _other = _other, _self

    if etype == _pow:
        if _other.__class__ in native_numeric_types:
            if _other == 1:
                return _self
            elif not _other:
                return 1
            elif _self.__class__ in native_numeric_types:
                return _self ** _other
            elif _self.is_potentially_variable():
                return PowExpression((_self, _other))
            return NPV_PowExpression((_self, _other))
        elif _self.__class__ in native_numeric_types:
            if _other.is_potentially_variable():
                return PowExpression((_self, _other))
            return NPV_PowExpression((_self, _other))
        elif _self.is_potentially_variable() or _other.is_potentially_variable():
            return PowExpression((_self, _other))
        else:
            return NPV_PowExpression((_self, _other))

    raise RuntimeError("Unknown expression type '%s'" % etype)      #pragma: no cover

def _generate_intrinsic_function_expression(arg, name, fcn):
    if not (arg.__class__ in native_types or arg.is_expression_type()):
        arg = _process_arg(arg)

    if arg.__class__ in native_types:
        return fcn(arg)
    elif arg.is_potentially_variable():
        return UnaryFunctionExpression(arg, name, fcn)
    else:
        return NPV_UnaryFunctionExpression(arg, name, fcn)

def _balanced_parens(arg):
    """Verify the string argument contains balanced parentheses.

    This checks that every open paren is balanced by a closed paren.
    That is, the infix string expression is likely to be valid.  This is
    primarily used to determine if a string that starts and ends with
    parens can have those parens removed.

    Examples:
        >>> a = "(( x + y ) * ( z - w ))"
        >>> _balanced_parens(a[1:-1])
        True
        >>> a = "( x + y ) * ( z - w )"
        >>> _balanced_parens(a[1:-1])
        False
    """
    _parenCount = 0
    for c in arg:
        if c == '(':
            _parenCount += 1
        elif c == ')':
            _parenCount -= 1
            if _parenCount < 0:
                return False
    return _parenCount == 0


NPV_expression_types = set(
   [NPV_NegationExpression,
    NPV_ExternalFunctionExpression,
    NPV_PowExpression,
    NPV_ProductExpression,
    NPV_DivisionExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    NPV_AbsExpression])

