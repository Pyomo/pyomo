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

import math
import logging
from itertools import islice

logger = logging.getLogger('pyomo.core')

from pyutilib.math.util import isclose
from pyomo.common.deprecation import deprecated

from .expr_common import (
    _add, _sub, _mul, _div,
    _pow, _neg, _abs, _inplace,
    _unary
)
from .numvalue import (
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    native_numeric_types,
    as_numeric,
    value,
)

from .visitor import (
    evaluate_expression, expression_to_string, polynomial_degree,
    clone_expression, sizeof_expression, _expression_is_fixed
)


class clone_counter(object):
    """ Context manager for counting cloning events.

    This context manager counts the number of times that the
    :func:`clone_expression <pyomo.core.expr.current.clone_expression>`
    function is executed.
    """

    _count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def count(self):
        """A property that returns the clone count value.
        """
        return clone_counter._count


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


class ExpressionBase(NumericValue):
    """
    The base class for Pyomo expressions.

    This class is used to define nodes in an expression
    tree.

    Args:
        args (list or tuple): Children of this node.
    """

    # Previously, we used _args to define expression class arguments.
    # Here, we use _args_ to force errors for code that was referencing this
    # data.  There are now accessor methods, so in most cases users
    # and developers should not directly access the _args_ data values.
    __slots__ =  ('_args_',)
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        """
        Returns the number of child nodes.

        By default, Pyomo expressions represent binary operations
        with two arguments.

        Note:
            This function does not simply compute the length of
            :attr:`_args_` because some expression classes use
            a subset of the :attr:`_args_` array.  Thus, it
            is imperative that developers use this method!

        Returns:
            A nonnegative integer that is the number of child nodes.
        """
        return 2

    def arg(self, i):
        """
        Return the i-th child node.

        Args:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argument: %d" % i)
        if i < 0:
            return self._args_[self.nargs()+i]
        return self._args_[i]

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[:self.nargs()]


    def __getstate__(self):
        """
        Pickle the expression object

        Returns:
            The pickled state.
        """
        state = super(ExpressionBase, self).__getstate__()
        for i in ExpressionBase.__slots__:
           state[i] = getattr(self,i)
        return state

    def __nonzero__(self):      #pragma: no cover
        """
        Compute the value of the expression and convert it to
        a boolean.

        Returns:
            A boolean value.
        """
        return bool(self())

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """
        Evaluate the value of the expression tree.

        Args:
            exception (bool): If :const:`False`, then
                an exception raised while evaluating
                is captured, and the value returned is
                :const:`None`.  Default is :const:`True`.

        Returns:
            The value of the expression or :const:`None`.
        """
        return evaluate_expression(self, exception)

    def __str__(self):
        """
        Returns a string description of the expression.

        Note:
            The value of ``pyomo.core.expr.expr_common.TO_STRING_VERBOSE``
            is used to configure the execution of this method.
            If this value is :const:`True`, then the string
            representation is a nested function description of the expression.
            The default is :const:`False`, which is an algebraic
            description of the expression.

        Returns:
            A string.
        """
        return expression_to_string(self)

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of the expression tree.

        Args:
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
                Defaults to :const:`False`.
            labeler: An object that generates string labels for
                variables in the expression tree.  Defaults to :const:`None`.
            smap:  If specified, this :class:`SymbolMap <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.  Default is :const:`False`.

        Returns:
            A string representation for the expression tree.
        """
        return expression_to_string(self, verbose=verbose, labeler=labeler, smap=smap, compute_values=compute_values)

    def _precedence(self):
        return ExpressionBase.PRECEDENCE

    def _associativity(self):
        """Return the associativity of this operator.

        Returns 1 if this operator is left-to-right associative or -1 if
        it is right-to-left associative.  Any other return value will be
        interpreted as "not associative" (implying any arguments that
        are at this operator's _precedence() will be enclosed in parens).
        """
        # Most operators in Python are left-to-right associative
        return 1

    def _to_string(self, values, verbose, smap, compute_values):            #pragma: no cover
        """
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        Args:
            values (list): The string representations of the children of this
                node.
            verbose (bool): If :const:`True`, then the the string
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
            smap:  If specified, this :class:`SymbolMap
                <pyomo.core.expr.symbol_map.SymbolMap>` is
                used to cache labels for variables.
            compute_values (bool): If :const:`True`, then
                parameters and fixed variables are evaluated before the
                expression string is generated.

        Returns:
            A string representation for this node.
        """
        pass

    def getname(self, *args, **kwds):                       #pragma: no cover
        """
        Return the text name of a function associated with this expression object.

        In general, no arguments are passed to this function.

        Args:
            *arg: a variable length list of arguments
            **kwds: keyword arguments

        Returns:
            A string name for the function.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement getname()" % ( str(self.__class__), ))

    def clone(self, substitute=None):
        """
        Return a clone of the expression tree.

        Note:
            This method does not clone the leaves of the
            tree, which are numeric constants and variables.
            It only clones the interior nodes, and
            expression leaf nodes like
            :class:`_MutableLinearExpression<pyomo.core.expr.current._MutableLinearExpression>`.
            However, named expressions are treated like
            leaves, and they are not cloned.

        Args:
            substitute (dict): a dictionary that maps object ids to clone
                objects generated earlier during the cloning process.

        Returns:
            A new expression tree.
        """
        return clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, args):
        """
        Construct a node using given arguments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this simply returns::

            self.__class__(args)

        But in general this creates an expression object using local
        data as well as arguments that represent the child nodes.

        Args:
            args (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argument is needed to clone objects that are
                owned by a model, and it can be safely ignored for
                most expression classes.

        Returns:
            A new expression object with the same type as the current
            class.
        """
        return self.__class__(args)

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
        self.__class__ = self.__class__.__mro__[1]
        return self

    def is_constant(self):
        """Return True if this expression is an atomic constant

        This method contrasts with the is_fixed() method.  This method
        returns True if the expression is an atomic constant, that is it
        is composed exclusively of constants and immutable parameters.
        NumericValue objects returning is_constant() == True may be
        simplified to their numeric value at any point without warning.

        Note:  This defaults to False, but gets redefined in sub-classes.
        """
        return False

    def is_fixed(self):
        """
        Return :const:`True` if this expression contains no free variables.

        Returns:
            A boolean.
        """
        return _expression_is_fixed(self)

    def _is_fixed(self, values):
        """
        Compute whether this expression is fixed given
        the fixed values of its children.

        This method is called by the :class:`_IsFixedVisitor
        <pyomo.core.expr.current._IsFixedVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of boolean values that indicate whether
                the children of this expression are fixed

        Returns:
            A boolean that is :const:`True` if the fixed values of the
            children are all :const:`True`.
        """
        return all(values)

    def is_potentially_variable(self):
        """
        Return :const:`True` if this expression might represent
        a variable expression.

        This method returns :const:`True` when (a) the expression
        tree contains one or more variables, or (b) the expression
        tree contains a named expression. In both cases, the
        expression cannot be treated as constant since (a) the variables
        may not be fixed, or (b) the named expressions may be changed
        at a later time to include non-fixed variables.

        Returns:
            A boolean.  Defaults to :const:`True` for expressions.
        """
        return True

    def is_named_expression_type(self):
        """
        Return :const:`True` if this object is a named expression.

        This method returns :const:`False` for this class, and it
        is included in other classes within Pyomo that are not named
        expressions, which allows for a check for named expressions
        without evaluating the class type.

        Returns:
            A boolean.
        """
        return False

    def is_expression_type(self):
        """
        Return :const:`True` if this object is an expression.

        This method obviously returns :const:`True` for this class, but it
        is included in other classes within Pyomo that are not expressions,
        which allows for a check for expressions without
        evaluating the class type.

        Returns:
            A boolean.
        """
        return True

    def size(self):
        """
        Return the number of nodes in the expression tree.

        Returns:
            A nonnegative integer that is the number of interior and leaf
            nodes in the expression tree.
        """
        return sizeof_expression(self)

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return polynomial_degree(self)

    def _compute_polynomial_degree(self, values):                          #pragma: no cover
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

    def _apply_operation(self, result):     #pragma: no cover
        """
        Compute the values of this node given the values of its children.

        This method is called by the :class:`_EvaluationVisitor
        <pyomo.core.expr.current._EvaluationVisitor>` class.  It must
        be over-written by expression classes to customize this logic.

        Note:
            This method applies the logical operation of the
            operator to the arguments.  It does *not* evaluate
            the arguments in the process, but assumes that they
            have been previously evaluated.  But noted that if
            this class contains auxilliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        Args:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _apply_operation()" % ( str(self.__class__), ))


class NegationExpression(ExpressionBase):
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

    def _precedence(self):
        return NegationExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        tmp = values[0]
        if tmp[0] == '-':
            i = 1
            while tmp[i] == ' ':
                i += 1
            return tmp[i:]
        return "- "+tmp

    def _apply_operation(self, result):
        return -result[0]


class NPV_NegationExpression(NegationExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class ExternalFunctionExpression(ExpressionBase):
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

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._fcn)

    def __getstate__(self):
        state = super(ExternalFunctionExpression, self).__getstate__()
        for i in ExternalFunctionExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):           #pragma: no cover
        return self._fcn.getname(*args, **kwds)

    def _compute_polynomial_degree(self, result):
        return 0 if all(arg == 0 for arg in result) else None

    def _apply_operation(self, result):
        return self._fcn.evaluate( result )

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}({1})".format(self.getname(), ", ".join(values))

    def get_arg_units(self):
        """ Return the units for this external functions arguments """
        return self._fcn.get_arg_units()

    def get_units(self):
        """ Get the units of the return value for this external function """
        return self._fcn.get_units()

class NPV_ExternalFunctionExpression(ExternalFunctionExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class PowExpression(ExpressionBase):
    """
    Power expressions::

        x**y
    """

    __slots__ = ()
    PRECEDENCE = 2

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

    def _precedence(self):
        return PowExpression.PRECEDENCE

    def _associativity(self):
        # "**" is right-to-left associative in Python (so this should
        # return -1), however, as this rule is not widely known and can
        # confuse novice users, we will make our "**" operator
        # non-associative (forcing parens)
        return 0

    def _apply_operation(self, result):
        _l, _r = result
        return _l ** _r

    def getname(self, *args, **kwds):
        return 'pow'

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        return "{0}**{1}".format(values[0], values[1])


class NPV_PowExpression(PowExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class ProductExpression(ExpressionBase):
    """
    Product expressions::

        x*y
    """

    __slots__ = ()
    PRECEDENCE = 4

    def _precedence(self):
        return ProductExpression.PRECEDENCE

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

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        if values[0] in self._to_string.one:
            return values[1]
        if values[0] in self._to_string.minus_one:
            return "- {0}".format(values[1])
        return "{0}*{1}".format(values[0],values[1])
    # Store these reference sets on the function for quick lookup
    _to_string.one = {"1", "1.0", "(1)", "(1.0)"}
    _to_string.minus_one = {"-1", "-1.0", "(-1)", "(-1.0)"}


class NPV_ProductExpression(ProductExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class MonomialTermExpression(ProductExpression):
    __slots__ = ()

    def getname(self, *args, **kwds):
        return 'mon'

class DivisionExpression(ExpressionBase):
    """
    Division expressions::

        x/y
    """
    __slots__ = ()
    PRECEDENCE = 4

    def nargs(self):
        return 2

    def _precedence(self):
        return DivisionExpression.PRECEDENCE

    def _compute_polynomial_degree(self, result):
        if result[1] == 0:
            return result[0]
        return None

    def getname(self, *args, **kwds):
        return 'div'

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        return "{0}/{1}".format(values[0], values[1])

    def _apply_operation(self, result):
        return result[0] / result[1]


class NPV_DivisionExpression(DivisionExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class ReciprocalExpression(ExpressionBase):
    """
    Reciprocal expressions::

        1/x
    """
    __slots__ = ()
    PRECEDENCE = 4

    @deprecated("ReciprocalExpression is deprecated. Use DivisionExpression",
                version='5.6.7')
    def __init__(self, args):
        super(ReciprocalExpression, self).__init__(args)

    def nargs(self):
        return 1

    def _precedence(self):
        return ReciprocalExpression.PRECEDENCE

    def _associativity(self):
        return 0

    def _compute_polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        return None

    def getname(self, *args, **kwds):
        return 'recip'

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        return "1/{0}".format(values[0])

    def _apply_operation(self, result):
        return 1 / result[0]


class NPV_ReciprocalExpression(ReciprocalExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class _LinearOperatorExpression(ExpressionBase):
    """
    An 'abstract' class that defines the polynomial degree for a simple
    linear operator
    """

    __slots__ = ()

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


class SumExpressionBase(_LinearOperatorExpression):
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

    def _precedence(self):
        return SumExpressionBase.PRECEDENCE

    def getname(self, *args, **kwds):
        return 'sum'


class NPV_SumExpression(SumExpressionBase):
    __slots__ = ()

    def create_potentially_variable_object(self):
        return SumExpression( self._args_ )

    def _apply_operation(self, result):
        l_, r_ = result
        return l_ + r_

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1}, {2})".format(self.getname(), values[0], values[1])
        if values[1][0] == '-':
            return "{0} {1}".format(values[0],values[1])
        return "{0} + {1}".format(values[0],values[1])

    def is_potentially_variable(self):
        return False


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

    def _precedence(self):
        return SumExpression.PRECEDENCE

    def _apply_operation(self, result):
        return sum(result)

    def create_node_with_local_data(self, args):
        return self.__class__(list(args))

    def __getstate__(self):
        state = super(SumExpression, self).__getstate__()
        for i in SumExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def is_constant(self):
        #
        # In most normal contexts, a SumExpression is
        # non-constant.  When Forming expressions, constant
        # parameters are turned into numbers, which are
        # simply added.  Mutable parameters, variables and
        # expressions are not constant.
        #
        return False

    def is_potentially_variable(self):
        for v in islice(self._args_, self._nargs):
            if v.__class__ in nonpyomo_leaf_types:
                continue
            if v.is_variable_type() or v.is_potentially_variable():
                return True
        return False

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            tmp = [values[0]]
            for i in range(1,len(values)):
                tmp.append(", ")
                tmp.append(values[i])
            return "{0}({1})".format(self.getname(), "".join(tmp))

        tmp = [values[0]]
        for i in range(1,len(values)):
            if values[i][0] == '-':
                tmp.append(' - ')
                tmp.append(values[i][1:].strip())
            elif len(values[i]) > 3 and values[i][:2] == '(-' \
                 and values[i][-1] == ')' and _balanced_parens(values[i][1:-1]):
                tmp.append(' - ')
                tmp.append(values[i][2:-1].strip())
            else:
                tmp.append(' + ')
                tmp.append(values[i])
        return ''.join(tmp)


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


class Expr_ifExpression(ExpressionBase):
    """
    A logical if-then-else expression::

        Expr_if(IF_=x, THEN_=y, ELSE_=z)

    Args:
        IF_ (expression): A relational expression
        THEN_ (expression): An expression that is used if :attr:`IF_` is true.
        ELSE_ (expression): An expression that is used if :attr:`IF_` is false.
    """
    __slots__ = ('_if','_then','_else')

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

    def __getstate__(self):
        state = super(Expr_ifExpression, self).__getstate__()
        for i in Expr_ifExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return "Expr_if"

    def _is_fixed(self, args):
        assert(len(args) == 3)
        if args[0]: # self._if.is_fixed():
            if args[1] and args[2]:
                return True
            if value(self._if):
                return args[1] # self._then.is_fixed()
            else:
                return args[2] # self._else.is_fixed()
        else:
            return False

    def is_constant(self):
        if self._if.__class__ in native_numeric_types or self._if.is_constant():
            if value(self._if):
                return (self._then.__class__ in native_numeric_types or self._then.is_constant())
            else:
                return (self._else.__class__ in native_numeric_types or self._else.is_constant())
        else:
            return (self._then.__class__ in native_numeric_types or self._then.is_constant()) and \
                (self._else.__class__ in native_numeric_types or self._else.is_constant())

    def is_potentially_variable(self):
        return ((not self._if.__class__ in native_numeric_types) and self._if.is_potentially_variable()) or \
            ((not self._then.__class__ in native_numeric_types) and self._then.is_potentially_variable()) or \
            ((not self._else.__class__ in native_numeric_types) and self._else.is_potentially_variable())

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

    def _to_string(self, values, verbose, smap, compute_values):
        return '{0}( ( {1} ), then=( {2} ), else=( {3} ) )'.\
            format(self.getname(), self._if, self._then, self._else)

    def _apply_operation(self, result):
        _if, _then, _else = result
        return _then if _if else _else


class UnaryFunctionExpression(ExpressionBase):
    """
    An expression object used to define intrinsic functions (e.g. sin, cos, tan).

    Args:
        args (tuple): Children nodes
        name (string): The function name
        fcn: The function that is used to evaluate this expression
    """
    __slots__ = ('_fcn', '_name')

    def __init__(self, args, name=None, fcn=None):
        if not type(args) is tuple:
            args = (args,)
        self._args_ = args
        self._name = name
        self._fcn = fcn

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._name, self._fcn)

    def __getstate__(self):
        state = super(UnaryFunctionExpression, self).__getstate__()
        for i in UnaryFunctionExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return self._name

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "{0}({1})".format(self.getname(), values[0])
        if values[0] and values[0][0] == '(' and values[0][-1] == ')' \
           and _balanced_parens(values[0][1:-1]):
            return '{0}{1}'.format(self._name, values[0])
        else:
            return '{0}({1})'.format(self._name, values[0])

    def _compute_polynomial_degree(self, result):
        if result[0] == 0:
            return 0
        else:
            return None

    def _apply_operation(self, result):
        return self._fcn(result[0])


class NPV_UnaryFunctionExpression(UnaryFunctionExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


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

    def create_node_with_local_data(self, args):
        return self.__class__(args)


class NPV_AbsExpression(AbsExpression):
    __slots__ = ()

    def is_potentially_variable(self):
        return False


class LinearExpression(ExpressionBase):
    """
    An expression object linear polynomials.

    Args:
        args (tuple): Children nodes
    """
    __slots__ = ('constant',          # The constant term
                 'linear_coefs',      # Linear coefficients
                 'linear_vars')       # Linear variables

    PRECEDENCE = 6

    def __init__(self, args=None, constant=None, linear_coefs=None, linear_vars=None):
        """ 
        Build a linear expression object that stores the constant, as well as 
        coefficients and variables to represent const + sum_i(c_i*x_i)
        
        You can specify args OR (constant, linear_coefs, and linear_vars)
        If args is provided, it should be a list that contains the constant,
        followed by the coefficients, followed by the variables.
        
        Alternatively, you can specify the constant, the list of linear_coeffs
        and the list of linear_vars separately. Note that these lists are NOT
        copied.
        """
        # I am not sure why LinearExpression allows omitting args, but
        # it does.  If they are provided, they should be the constant
        # followed by the coefficients followed by the variables.
        if args:
            self.constant = args[0]
            n = (len(args)-1) // 2
            self.linear_coefs = args[1:n+1]
            self.linear_vars = args[n+1:]
        else:
            self.constant = constant if constant is not None else 0
            self.linear_coefs = linear_coefs if linear_coefs else []
            self.linear_vars = linear_vars if linear_vars else []
            
        self._args_ = tuple()

    def nargs(self):
        return 0

    def _precedence(self):
        return LinearExpression.PRECEDENCE

    def __getstate__(self):
        state = super(LinearExpression, self).__getstate__()
        for i in LinearExpression.__slots__:
           state[i] = getattr(self,i)
        return state

    def create_node_with_local_data(self, args):
        return self.__class__(args)

    def getname(self, *args, **kwds):
        return 'sum'

    def _compute_polynomial_degree(self, result):
        return 1 if not self.is_fixed() else 0

    def is_constant(self):
        return len(self.linear_vars) == 0

    def _is_fixed(self, values=None):
        return all(v.fixed for v in self.linear_vars)

    def is_fixed(self):
        return self._is_fixed()

    def _to_string(self, values, verbose, smap, compute_values):
        tmp = []
        if compute_values:
            const_ = value(self.constant)
            if not isclose(const_,0):
                tmp = [str(const_)]
        elif self.constant.__class__ in native_numeric_types:
            if not isclose(self.constant, 0):
                tmp = [str(self.constant)]
        else:
            tmp = [self.constant.to_string(compute_values=False)]
        if verbose:
            for c,v in zip(self.linear_coefs, self.linear_vars):
                if smap:                        # TODO: coverage
                    v_ = smap.getSymbol(v)
                else:
                    v_ = str(v)
                if c.__class__ in native_numeric_types or compute_values:
                    c_ = value(c)
                    if isclose(c_,1):
                        tmp.append(str(v_))
                    elif isclose(c_,0):
                        continue
                    else:
                        tmp.append("prod(%s, %s)" % (str(c_),str(v_)))
                else:
                    tmp.append("prod(%s, %s)" % (str(c), v_))
            return "{0}({1})".format(self.getname(), ', '.join(tmp))
        for c,v in zip(self.linear_coefs, self.linear_vars):
            if smap:
                v_ = smap.getSymbol(v)
            else:
                v_ = str(v)
            if c.__class__ in native_numeric_types or compute_values:
                c_ = value(c)
                if isclose(c_,1):
                   tmp.append(" + %s" % v_)
                elif isclose(c_,0):
                    continue
                elif isclose(c_,-1):
                   tmp.append(" - %s" % v_)
                elif c_ < 0:
                   tmp.append(" - %s*%s" % (str(math.fabs(c_)), v_))
                else:
                   tmp.append(" + %s*%s" % (str(c_), v_))
            else:
                c_str = str(c)
                if any(_ in c_str for _ in '+-*/'):
                    c_str = '('+c_str+')'
                tmp.append(" + %s*%s" % (c_str, v_))
        s = "".join(tmp)
        if len(s) == 0:                 #pragma: no cover
            return s
        if s[0] == " ":
            if s[1] == "+":
                return s[3:]
            return s[1:]
        return s

    def is_potentially_variable(self):
        return len(self.linear_vars) > 0

    def _apply_operation(self, result):
        return value(self.constant) + sum(value(c)*v.value for c,v in zip(self.linear_coefs, self.linear_vars))

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
            for term in _decompose_linear_terms(expr._args_[1], multiplier*expr._args_[0]):
                yield term
        elif expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            for term in _decompose_linear_terms(expr._args_[0], multiplier*expr._args_[1]):
                yield term
        else:
            raise LinearDecompositionError("Quadratic terms exist in a product expression.")
    elif expr.__class__ is DivisionExpression:
        if expr._args_[1].__class__ in native_numeric_types or not expr._args_[1].is_potentially_variable():
            for term in _decompose_linear_terms(expr._args_[0], multiplier/expr._args_[1]):
                yield term
        else:
            raise LinearDecompositionError("Unexpected nonlinear term (division)")
    elif expr.__class__ is ReciprocalExpression:
        # The argument is potentially variable, so this represents a nonlinear term
        #
        # NOTE: We're ignoring possible simplifications
        raise LinearDecompositionError("Unexpected nonlinear term")
    elif expr.__class__ is SumExpression or expr.__class__ is _MutableSumExpression:
        for arg in expr.args:
            for term in _decompose_linear_terms(arg, multiplier):
                yield term
    elif expr.__class__ is NegationExpression:
        for term in  _decompose_linear_terms(expr._args_[0], -multiplier):
            yield term
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
    if obj.is_numeric_type() and obj.is_constant():
        # Resolve constants (e.g., immutable scalar Params & NumericConstants)
        return value(obj)
    # User assistance: provide a helpful exception when using an indexed
    # object in an expression
    if obj.is_component_type() and obj.is_indexed():
        raise TypeError(
            "Argument for expression is an indexed numeric "
            "value\nspecified without an index:\n\t%s\nIs this "
            "value defined over an index that you did not specify?"
            % (obj.name, ) )
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
            if _self.__class__ in native_numeric_types:
                return _self + _other
            elif _other == 0:
                return _self
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
    NPV_ReciprocalExpression,
    NPV_SumExpression,
    NPV_UnaryFunctionExpression,
    NPV_AbsExpression])

