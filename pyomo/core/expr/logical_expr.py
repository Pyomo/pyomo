#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


"""
Notes. Delete later
"""

from __future__ import division

_using_chained_inequality = True

import logging
import traceback

logger = logging.getLogger('pyomo.core')
from pyomo.common.deprecation import deprecation_warning
from .numvalue import (
     native_types,
     native_numeric_types,
     as_numeric,
)

from .logicalvalue import (
    LogicalValue,
    LogicalConstant,
    as_logical,
    value,
)

from .expr_common import (
    _add, _sub, _mul, _div,
    _pow, _neg, _abs, _inplace,
    _unary, _radd, _rsub, _rmul,
    _rdiv, _rpow, _iadd, _isub,
    _imul, _idiv, _ipow, _lt, _le,
    _eq,
)

from .visitor import (
    evaluate_expression, expression_to_string, polynomial_degree,
    clone_expression, sizeof_expression, _expression_is_fixed
)

from .numeric_expr import _LinearOperatorExpression, _process_arg
import operator


if _using_chained_inequality:               #pragma: no cover
    class _chainedInequality(object):

        prev = None
        call_info = None
        cloned_from = []

        @staticmethod
        def error_message(msg=None):
            if msg is None:
                msg = "Relational expression used in an unexpected Boolean context."
            val = _chainedInequality.prev.to_string()
            # We are about to raise an exception, so it's OK to reset chainedInequality
            info = _chainedInequality.call_info
            _chainedInequality.call_info = None
            _chainedInequality.prev = None

            args = ( str(msg).strip(), val.strip(), info[0], info[1],
                     ':\n    %s' % info[3] if info[3] is not None else '.' )
            return """%s

        The inequality expression:
            %s
        contains non-constant terms (variables) that were evaluated in an
        unexpected Boolean context at
          File '%s', line %s%s

        Evaluating Pyomo variables in a Boolean context, e.g.
            if expression <= 5:
        is generally invalid.  If you want to obtain the Boolean value of the
        expression based on the current variable values, explicitly evaluate the
        expression using the value() function:
            if value(expression) <= 5:
        or
            if value(expression <= 5):
        """ % args

else:                               #pragma: no cover
    _chainedInequality = None


#-------------------------------------------------------
#
# Expression classes
#
#-------------------------------------------------------


class RangedExpression(_LinearOperatorExpression):
    """
    Ranged expressions, which define relations with a lower and upper bound::

        x < y < z
        x <= y <= z

    args:
        args (tuple): child nodes
        strict (tuple): flags that indicates whether the inequalities are strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super(RangedExpression,self).__init__(args)
        self._strict = strict

    def nargs(self):
        return 3

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def __getstate__(self):
        state = super(RangedExpression, self).__getstate__()
        for i in RangedExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return RangedExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _b, _r = result
        if not self._strict[0]:
            if not self._strict[1]:
                return _l <= _b and _b <= _r
            else:
                return _l <= _b and _b < _r
        elif not self._strict[1]:
            return _l < _b and _b <= _r
        else:
            return _l < _b and _b < _r

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}  {1}  {2}  {3}  {4}".format(values[0], '<' if self._strict[0] else '<=', values[1], '<' if self._strict[1] else '<=', values[2])

    def is_constant(self):
        return (self._args_[0].__class__ in native_numeric_types or self._args_[0].is_constant()) and \
               (self._args_[1].__class__ in native_numeric_types or self._args_[1].is_constant()) and \
               (self._args_[2].__class__ in native_numeric_types or self._args_[2].is_constant())

    def is_potentially_variable(self):
        return (self._args_[1].__class__ not in native_numeric_types and \
                self._args_[1].is_potentially_variable()) or \
               (self._args_[0].__class__ not in native_numeric_types and \
                self._args_[0].is_potentially_variable()) or \
               (self._args_[2].__class__ not in native_numeric_types and \
                self._args_[2].is_potentially_variable())


class InequalityExpression(_LinearOperatorExpression):
    """
    Inequality expressions, which define less-than or
    less-than-or-equal relations::

        x < y
        x <= y

    args:
        args (tuple): child nodes
        strict (bool): a flag that indicates whether the inequality is strict
    """

    __slots__ = ('_strict',)
    PRECEDENCE = 9

    def __init__(self, args, strict):
        super(InequalityExpression,self).__init__(args)
        self._strict = strict

    def nargs(self):
        return 2

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._strict)

    def __getstate__(self):
        state = super(InequalityExpression, self).__getstate__()
        for i in InequalityExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def __nonzero__(self):
        if _using_chained_inequality and not self.is_constant():    #pragma: no cover
            deprecation_warning("Chained inequalities are deprecated. "
                                "Use the inequality() function to "
                                "express ranged inequality expressions.")     # Remove in Pyomo 6.0
            _chainedInequality.call_info = traceback.extract_stack(limit=2)[-2]
            _chainedInequality.prev = self
            return True
            #return bool(self())                # This is needed to apply simple evaluation of inequalities

        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return InequalityExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        if self._strict:
            return _l < _r
        return _l <= _r

    def _to_string(self, values, verbose, smap, compute_values):
        if len(values) == 2:
            return "{0}  {1}  {2}".format(values[0], '<' if self._strict else '<=', values[1])

    def is_constant(self):
        return (self._args_[0].__class__ in native_numeric_types or self._args_[0].is_constant()) and \
               (self._args_[1].__class__ in native_numeric_types or self._args_[1].is_constant())

    def is_potentially_variable(self):
        return (self._args_[0].__class__ not in native_numeric_types and \
                self._args_[0].is_potentially_variable()) or \
               (self._args_[1].__class__ not in native_numeric_types and \
                self._args_[1].is_potentially_variable())


def inequality(lower=None, body=None, upper=None, strict=False):
    """
    A utility function that can be used to declare inequality and
    ranged inequality expressions.  The expression::

        inequality(2, model.x)

    is equivalent to the expression::

        2 <= model.x

    The expression::

        inequality(2, model.x, 3)

    is equivalent to the expression::

        2 <= model.x <= 3

    .. note:: This ranged inequality syntax is deprecated in Pyomo.
        This function provides a mechanism for expressing
        ranged inequalities without chained inequalities.

    args:
        lower: an expression defines a lower bound
        body: an expression defines the body of a ranged constraint
        upper: an expression defines an upper bound
        strict (bool): A boolean value that indicates whether the inequality
            is strict.  Default is :const:`False`.

    Returns:
        A relational expression.  The expression is an inequality
        if any of the values :attr:`lower`, :attr:`body` or
        :attr:`upper` is :const:`None`.  Otherwise, the expression
        is a ranged inequality.
    """
    if lower is None:
        if body is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return InequalityExpression((body, upper), strict)
    if body is None:
        if lower is None or upper is None:
            raise ValueError("Invalid inequality expression.")
        return InequalityExpression((lower, upper), strict)
    if upper is None:
        return InequalityExpression((lower, body), strict)
    return RangedExpression((lower, body, upper), (strict, strict))

class EqualityExpression(_LinearOperatorExpression):
    """
    Equality expression::

        x == y
    """

    __slots__ = ()
    PRECEDENCE = 9

    def nargs(self):
        return 2

    def __nonzero__(self):
        return bool(self())

    __bool__ = __nonzero__

    def is_relational(self):
        return True

    def _precedence(self):
        return EqualityExpression.PRECEDENCE

    def _apply_operation(self, result):
        _l, _r = result
        return _l == _r

    def _to_string(self, values, verbose, smap, compute_values):
        return "{0}  ==  {1}".format(values[0], values[1])

    def is_constant(self):
        return self._args_[0].is_constant() and self._args_[1].is_constant()

    def is_potentially_variable(self):
        return self._args_[0].is_potentially_variable() or self._args_[1].is_potentially_variable()



if _using_chained_inequality:
    def _generate_relational_expression(etype, lhs, rhs):               #pragma: no cover
        # We cannot trust Python not to recycle ID's for temporary POD data
        # (e.g., floats).  So, if it is a "native" type, we will record the
        # value, otherwise we will record the ID.  The tuple for native
        # types is to guarantee that a native value will *never*
        # accidentally match an ID
        cloned_from = (\
            id(lhs) if lhs.__class__ not in native_numeric_types else (0,lhs),
            id(rhs) if rhs.__class__ not in native_numeric_types else (0,rhs)
            )
        rhs_is_relational = False
        lhs_is_relational = False

        if not (lhs.__class__ in native_types or lhs.is_expression_type()):
            lhs = _process_arg(lhs)
        if not (rhs.__class__ in native_types or rhs.is_expression_type()):
            rhs = _process_arg(rhs)

        if lhs.__class__ in native_numeric_types:
            lhs = as_numeric(lhs)
        elif lhs.is_relational():
            lhs_is_relational = True

        if rhs.__class__ in native_numeric_types:
            rhs = as_numeric(rhs)
        elif rhs.is_relational():
            rhs_is_relational = True

        if _chainedInequality.prev is not None:
            prevExpr = _chainedInequality.prev
            match = []
            # This is tricky because the expression could have been posed
            # with >= operators, so we must figure out which argsuments
            # match.  One edge case is when the upper and lower bounds are
            # the same (implicit equality) - in which case *both* argsuments
            # match, and this should be converted into an equality
            # expression.
            for i,args in enumerate(_chainedInequality.cloned_from):
                if args == cloned_from[0]:
                    match.append((i,0))
                elif args == cloned_from[1]:
                    match.append((i,1))
            if etype == _eq:
                raise TypeError(_chainedInequality.error_message())
            if len(match) == 1:
                if match[0][0] == match[0][1]:
                    raise TypeError(_chainedInequality.error_message(
                        "Attempting to form a compound inequality with two "
                        "%s bounds" % ('lower' if match[0][0] else 'upper',)))
                if not match[0][1]:
                    cloned_from = _chainedInequality.cloned_from + (cloned_from[1],)
                    lhs = prevExpr
                    lhs_is_relational = True
                else:
                    cloned_from = (cloned_from[0],) + _chainedInequality.cloned_from
                    rhs = prevExpr
                    rhs_is_relational = True
            elif len(match) == 2:
                # Special case: implicit equality constraint posed as a <= b <= a
                if prevExpr._strict or etype == _lt:
                    _chainedInequality.prev = None
                    raise TypeError("Cannot create a compound inequality with "
                          "identical upper and lower\n\tbounds using strict "
                          "inequalities: constraint infeasible:\n\t%s and "
                          "%s < %s" % ( prevExpr.to_string(), lhs, rhs ))
                if match[0] == (0,0):
                    # This is a particularly weird case where someone
                    # evaluates the *same* inequality twice in a row.  This
                    # should always be an error (you can, for example, get
                    # it with "0 <= a >= 0").
                    raise TypeError(_chainedInequality.error_message())
                etype = _eq
            else:
                raise TypeError(_chainedInequality.error_message())
            _chainedInequality.prev = None

        if etype == _eq:
            if lhs_is_relational or rhs_is_relational:
                if lhs_is_relational:
                    val = lhs.to_string()
                else:
                    val = rhs.to_string()
                raise TypeError("Cannot create an EqualityExpression where "\
                      "one of the sub-expressions is a relational expression:\n"\
                      "    " + val)
            _chainedInequality.prev = None
            return EqualityExpression((lhs,rhs))
        else:
            if etype == _le:
                strict = False
            elif etype == _lt:
                strict = True
            else:
                raise ValueError("Unknown relational expression type '%s'" % etype) #pragma: no cover
            if lhs_is_relational:
                if lhs.__class__ is InequalityExpression:
                    if rhs_is_relational:
                        raise TypeError("Cannot create an InequalityExpression "\
                              "where both sub-expressions are relational "\
                              "expressions.")
                    _chainedInequality.prev = None
                    return RangedExpression(lhs._args_ + (rhs,), (lhs._strict,strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + lhs.to_string())
            elif rhs_is_relational:
                if rhs.__class__ is InequalityExpression:
                    _chainedInequality.prev = None
                    return RangedExpression((lhs,) + rhs._args_, (strict, rhs._strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + rhs.to_string())
            else:
                obj = InequalityExpression((lhs, rhs), strict)
                #_chainedInequality.prev = obj
                _chainedInequality.cloned_from = cloned_from
                return obj

else:

    def _generate_relational_expression(etype, lhs, rhs):               #pragma: no cover
        rhs_is_relational = False
        lhs_is_relational = False

        if not (lhs.__class__ in native_types or lhs.is_expression_type()):
            lhs = _process_arg(lhs)
        if not (rhs.__class__ in native_types or rhs.is_expression_type()):
            rhs = _process_arg(rhs)

        if lhs.__class__ in native_numeric_types:
            # TODO: Why do we need this?
            lhs = as_numeric(lhs)
        elif lhs.is_relational():
            lhs_is_relational = True

        if rhs.__class__ in native_numeric_types:
            # TODO: Why do we need this?
            rhs = as_numeric(rhs)
        elif rhs.is_relational():
            rhs_is_relational = True

        if etype == _eq:
            if lhs_is_relational or rhs_is_relational:
                if lhs_is_relational:
                    val = lhs.to_string()
                else:
                    val = rhs.to_string()
                raise TypeError("Cannot create an EqualityExpression where "\
                      "one of the sub-expressions is a relational expression:\n"\
                      "    " + val)
            return EqualityExpression((lhs,rhs))
        else:
            if etype == _le:
                strict = False
            elif etype == _lt:
                strict = True
            else:
                raise ValueError("Unknown relational expression type '%s'" % etype) #pragma: no cover
            if lhs_is_relational:
                if lhs.__class__ is InequalityExpression:
                    if rhs_is_relational:
                        raise TypeError("Cannot create an InequalityExpression "\
                              "where both sub-expressions are relational "\
                              "expressions.")
                    return RangedExpression(lhs._args_ + (rhs,), (lhs._strict,strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + lhs.to_string())
            elif rhs_is_relational:
                if rhs.__class__ is InequalityExpression:
                    return RangedExpression((lhs,) + rhs._args_, (strict, rhs._strict))
                else:
                    raise TypeError("Cannot create an InequalityExpression "\
                          "where one of the sub-expressions is an equality "\
                          "or ranged expression:\n    " + rhs.to_string())
            else:
                return InequalityExpression((lhs, rhs), strict)



class LogicalExpressionBase(LogicalValue):
    """
    Logical expressions base expression.

    This class is used to define nodes in an expression
    tree.
    
    Abstract

    args:
        args (list or tuple): Children of this node.
    """

    # 0-0 do we need this and used this new base for the expressions above?
    __slots__ =  ('_args_',)
    PRECEDENCE = 10

    def __init__(self, args):
        self._args_ = args


    def nargs(self):
        """
        Returns the number of child nodes.
        By default, logical expression represents binary expression.
        #0-0 should we make this 1 or like in numexpr 2?
        """
        return 2

    def args(self, i):
        #0-0
        """
        Return the i-th child node.

        args:
            i (int): Nonnegative index of the child that is returned.

        Returns:
            The i-th child node.
        """
        if i >= self.nargs():
            raise KeyError("Invalid index for expression argsument: %d" % i)
        if i < 0:
            return self._args_[self.nargs()+i]
            #0-0 send a warning?
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

    def __call__(self, exception=True):
        """
        Evaluate the value of the expression tree.
        #0-0 leave it for now
        args:
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
        #leave it for now
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
        #leave it for now
        args:
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
        interpreted as "not associative" (implying any argsuments that
        are at this operator's _precedence() will be enclosed in parens).
        """
        #0-0 do we need this?
        return 1

    def _to_string(self, values, verbose, smap, compute_values):            #pragma: no cover
        """
        #0-0 pass for now
        Construct a string representation for this node, using the string
        representations of its children.

        This method is called by the :class:`_ToStringVisitor
        <pyomo.core.expr.current._ToStringVisitor>` class.  It must
        must be defined in subclasses.

        args:
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

        In general, no argsuments are passed to this function.

        args:
            *arg: a variable length list of argsuments
            **kwds: keyword argsuments

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

        args:
            substitute (dict): a dictionary that maps object ids to clone
                objects generated earlier during the cloning process.

        Returns:
            A new expression tree.
        """
        return clone_expression(self, substitute=substitute)

    def create_node_with_local_data(self, args):
        """
        Construct a node using given argsuments.

        This method provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.  In the simplest
        case, this simply returns::

            self.__class__(args)

        But in general this creates an expression object using local
        data as well as argsuments that represent the child nodes.

        args:
            args (list): A list of child nodes for the new expression
                object
            memo (dict): A dictionary that maps object ids to clone
                objects generated earlier during a cloning process.
                This argsument is needed to clone objects that are
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

        # 0-0 leave it for now?

        Compute whether this expression is fixed given
        the fixed values of its children.

        This method is called by the :class:`_IsFixedVisitor
        <pyomo.core.expr.current._IsFixedVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        args:
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

    #def polynomial_degree(self):
    #not needed for logical expresion
    #def _compute_polynomial_degree(self, values):                          #pragma: no cover


    def _apply_operation(self, result):     #pragma: no cover
        """
        Compute the values of this node given the values of its children.

        This method is called by the :class:`_EvaluationVisitor
        <pyomo.core.expr.current._EvaluationVisitor>` class.  It must
        be over-written by expression classes to customize this logic.

        Note:
            This method applies the logical operation of the
            operator to the argsuments.  It does *not* evaluate
            the argsuments in the process, but assumes that they
            have been previously evaluated.  But noted that if
            this class contains auxilliary data (e.g. like the
            numeric coefficients in the :class:`LinearExpression
            <pyomo.core.expr.current.LinearExpression>` class, then
            those values *must* be evaluated as part of this
            function call.  An uninitialized parameter value
            encountered during the execution of this method is
            considered an error.

        args:
            values (list): A list of values that indicate the value
                of the children expressions.

        Returns:
            A floating point value for this expression.
        """
        raise NotImplementedError("Derived expression (%s) failed to "\
            "implement _apply_operation()" % ( str(self.__class__), ))
    
    """
    ---------------------**********************--------------------
    The following are nodes creators that should be used to create
    new nodes properly.
    """
        
    #NotExpression Creator
    def __invert__(self):
        return NotExpression(self)


    #EquivalanceExpression Creator
    def __eq__(self, other):
        return EquivalanceExpression(self, other)

    def equals(self, other):
        return EquivalanceExpression(self, other)

    #XorExpression Creator
    def __xor__(self, other):
        return XorExpression(self, other)

    def Xor(self, other):
        return XorExpression(self, other)

    def implies(self, other):
        return Implication(self, other)

    #AndExpressionCreator
    #Create a new node iff neither node is an AndNode
    #If we have an "AndNode" already, safe_add new node to the exisiting one.
    def __and__(self, other):
        if (self.getname() != "AndExpression"):  
            if (other.getname() != "AndExpression"):
                #return AndExpression(set([self, other])) #set version
                return AndExpression(list([self, other]))
            else :
                other._add(self)
                self = other
                return self
        else :
            self._add(other)
        return self

    #class method for AndExpression,basically the same logic as above
    '''
    #This section is documented just in case the class method is needed
    #in the future
    def LogicalAnd(self, other):
        if (self.getname() != "AndExpression"):  
            if (other.getname() != "AndExpression"):
                #return AndExpression(set([self, other])) #set version
                return AndExpression(list([self, other]))
            else :
                other._add(self)
                # 0-0 This step is a safety consideration, we can also 
                # use the python add and access the list to make things 
                # faster(not by much I guess)
                self = other
                return self
        else:
            self._add(other)
        return self
    '''
    
    #OrExpressionCreator
    #Create a new node iff neither node is an OrNode
    def __or__(self, other):
        if (self.getname() != "OrExpression"):  
            if (other.getname() != "OrExpression"):
                #return OrExpression(set([self, other])) #set version
                return OrExpression(list([self, other]))
            else :
                other._add(self)
                self = other
                return self
        else :
            self._add(other)
        return self

    '''
        This section is documented just in case the class method is needed
        in the future
    #class method for OrExpression,basically the same logic as above
    def LogicalOr(self, other):
        if (self.getname() != "OrExpression"):  
            if (other.getname() != "OrExpression"):
                #return OrExpression(set([self, other])) #set version
                return OrExpression(list([self, other]))
            else :
                other._add(self)
                # 0-0 This step is a safety consideration, we can also 
                # use the python add and access the list to make things 
                # faster(not by much I guess)
                self = other
                return self
        else:
            self._add(other)
        return self
    '''

"""
---------------------------******************--------------------
The following methods are static methods for nodes creator. Those should
do the exact same thing as the class methods as well as overloaded operators.
"""

# static method for NotExpression creator 
def Not(self):
    return NotExpression(self)

# static method for EquivalenceExpression creator 
def Equivalence(arg1, arg2):
    return EquivalenceExpression(arg1, arg2) 

# static method for XorExpression creator
def LogicalXor(arg1, arg2):
    return XorExpression(arg1, arg2)

# 0-0 add a static method for impies>
def Implies(arg1, arg2):
    return Implication(arg1, arg2)

# static method for AndExpression creator
# create a new node iff neither node is an AndNode

#combine 2 function and name it And()
def LogicalAnd(*argv):
    # 0-0 Do we need to take care of the safety of this method?
    # argsList is a set of LogicalValues, RaiseError if not?
    # checking requires a full loop from my understanding
    # Do we need to take care of empty set of set with length 1?
    argsList = list(argv)
    parent = argsList[0]
    for tmp in argsList:
        if isinstance(tmp, AndExpression):
            parent = tmp
            argList.remove(tmp)
            for target in argsList:
                parent._add(target)
            return parent

    res = AndExpression(list([parent]))
    argsList.remove(parent)
    while (len(argsList) != 0):
        res._add(argsList.pop())
    return res

# static method for OrExpression creator
# create a new node iff neither node is an OrNode, same logic

def LogicalOr(*argv):
    argsList = list(argv)
    parent = argsList[0]
    for tmp in argsList:
        if isinstance(tmp, OrExpression):
            parent = tmp
            argList.remove(tmp)
            for target in argsList:
                parent._add(target)
            return parent

    res = OrExpression(list([parent]))
    argsList.remove(parent)   
    while (len(argsList) != 0):
        res._add(argsList.pop())
    return res



# static Method for ExactlyExpression, AtMostExpression and AtLeastExpression
# make it support tuples?
def Exactly(req, argsList):
    result = ExactlyExpression(list(argsList))
    result._args_.insert(0, req)
    return result

def AtMost(req, argsList):
    result = AtMostExpression(list(argsList))
    result._args_.insert(0, req)
    return result

def AtLeast(req, argsList):
    result = AtLeastExpression(list(argsList))
    result._args_.insert(0, req)
    return result


#-------------------------*************------------------------------



class UnaryExpression(LogicalExpressionBase):
    """ 
    An abstract class for NotExpression
    There should only be one child under this kind of nodes
    This class should never be created directly. 
    #0-0
    This is the tempting naming. "args" should be one subclass of UnaryExpression,
    BinaryExperssion or MultiNodesExpression.
    """

    __slots__ = ("_args_",)

    """
    #0-0
    The precedence of an abstract class should not be a concern here, so it will be set
    to zero for now.
    """
    def __init__(self, args):
        self._args_ = args
        #print("The variable is initialized using UnaryExpression")
        #for tracing purpose only, delete later.
        #0-0 

    PRECEDENCE = 10

    def nargs(self):
        return 1

    def getname(self, *arg, **kwd):
        return 'UnaryExpression'

    def _precedence(self):
        return UnaryExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))

    def _apply_operation(self):
        raise TypeError("Please use Notexpression instead.")
        #0-0 ok to do this?
        


class NotExpression(UnaryExpression):
        """
        This is the node for a NotExpression, this node should have exactly one child
        """

        __slots__ = ()

        PRECEDENCE = 1
        #This operation should have the highest precedence among all, for now 
        #use 10 and adjust that later 0-0

        def getname(self, *arg, **kwd):
            return 'NotExpression'

        def _precendence(self):
            return Notexpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        def _apply_operetion(self, result):
            """
            result should be a tuple in general
            """
            return not result

class BinaryExpression(LogicalExpressionBase):
    """
    The abstract class for binary expression. This class should never be initialized.
    with __init__ .  largs and rargs are tempting names for its child nodes.
    """
    #0-0 changed larg and rarg to args
    __slots__ = ("_args_",)

    def __init__(self, larg, rarg):
        self._args_ = list([larg, rarg])
        #
        #print("The variable is initialized using BinaryExpression")
        #for tracing purpose only, delete later.
        #0-0 

    PRECEDENCE = 10
    #As this class should never be used in practice.

    def nargs(self):
        return 2

    def getname(self, *arg, **kwd):
        return 'BinaryExpression'

    def _precedence(self):
        return BinaryExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))

    def _apply_operation(self):
        raise TypeError("Please use the approriate binary expression instead.")
        #0-0 fine like this?




class EquivalenceExpression(BinaryExpression):
        """
        This is the node for EquivalanceExpression, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 7
        #0-0 not really sure... Is there a reference I can use?

        def getname(self, *arg, **kwd):
            return 'EquivalanceExpression'

        def _precendence(self):
            return EquivalanceExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            pass

        #change it to (self, result):
        def _apply_operation(self, resList):
            """
            #0-0 safety check?
            """
            return (resList[0] == resList[1])


class XorExpression(BinaryExpression):
        """
        This is the node for XorExpression, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 6
        #0-0 same as above

        def getname(self, *arg, **kwd):
            return 'XorExpression'

        def _precendence(self):
            return XorExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            return "XorExpression_toString_fornow"

        def _apply_operation(self,resList):
            """
            #0-0 
            """
            #return (res1 + res2 == 1)
            return operator.xor(resList[0], resList[1])


class Implication(BinaryExpression):
        """
        This is the node for Implication, this node should have exactly two children
        """

        __slots__ = ()

        PRECEDENCE = 4
        #0-0 same as above

        def getname(self, *arg, **kwd):
            return 'Implication'

        def _precendence(self):
            return XorExpression.PRECEDENCE

        def _to_string(self, values, verbose, smap, compute_values):
            #pass this one for now 0-0
            return "Implication_toString_fornow"

        def _apply_operation(self,resList):
            return ((not resList[0]) or (resList[1]))


class MultiArgsExpression(LogicalExpressionBase):
    """
    The abstract class for MultiargsExpression. This class should never be initialized.
    with __init__ .  args is a tempting name.
    """

    #args should be a set from Romeo's prototype

    __slots__ = ("_args_")

    def __init__(self, ChildList):
        #self._args_ =  list([v for v in ChildList]) #if we want the set version
        self._args_ =  list([v for v in ChildList])

    PRECEDENCE = 10
    #As this class should never be used in practice.

    def nargs(self):
        return len(self._args_)

    def getname(self, *arg, **kwd):
        return 'MultiArgsExpression'

    def _add(self, other):
        #a private method that adds another logicalexpression into this node
        #add elements into the list,while not creating a node if they share the same type
        #Always use this safe_add to add elements into a multinode
        '''
        #set version
        if (other.getname() != self.getname()):
            self._args_.add(other)
        else:
            self._args_.update(other._args_)
            #should we remove other in some way here?
        '''
        #list version
        if (type(other) != type(self)):
            self._args_.append(other)
        else:
            self._args_.extend(other._args_) 

    def _precedence(self):
        return MultiargsExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #question: how should this one work in general, though this function should not
        #be called ever imo. 
        #fine to raise this error?
        raise NotImplementedError("Derived expression (%s) failed to "\
                "implement _apply_operation()" % ( str(self.__class__), ))

    def _apply_operation(self):
        raise TypeError("Please use the approriate MultiargsExpression instead.")


class AndExpression(MultiArgsExpression):
    """
    This is the node for AndExpression.
    For coding only, given that AndExpression should have & as the
    overloaded operator, it is necessary to perform a check. 
    """

    __slots__ = ()

    PRECEDENCE = 2

    def getname(self, *arg, **kwd):
        return 'AndExpression'

    def _precendence(self):
        return AndExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        return "AndExpression_toString_fornow"

    def _apply_operation(self, result):
        if (len(self._args_) != len(result)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        return all(result)


class OrExpression(MultiArgsExpression):
    __slots__ = ()

    PRECEDENCE = 3

    def getname(self, *arg, **kwd):
        return 'OrExpression'

    def _precendence(self):
        return OrExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        return "OrExpression_toString_fornow"

    def _apply_operation(self, result):
        """
        """
        if (len(self._args_) != len(result)):
            raise KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        return any(result)



'''for Exactly, ...
'''
class ExactlyExpression(MultiArgsExpression):
    __slots__ = ()

    PRECEDENCE = 8

    def getname(self, *arg, **kwd):
        return 'ExactlyExpression'

    def _precendence(self):
        return ExactlyExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, result):
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        #0-0 Had some logic error here, is this fine now
        return sum(result) == 2*self._args_[0]


class AtMostExpression(MultiArgsExpression):
    __slots__ = ()

    PRECEDENCE = 8

    def getname(self, *arg, **kwd):
        return 'AtMostExpression'

    def _precendence(self):
        return AtMostExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, res_list):
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        counter = 0
        for tmp in res_list[1:]:
            if(tmp == True):
                counter += 1
        return (counter >= self._args_[0])

        

class AtLeastExpression(MultiArgsExpression):
    __slots__ = ()

    PRECEDENCE = 8

    def getname(self, *arg, **kwd):
        return 'AtLeastExpression'

    def _precendence(self):
        return AtLeastExpression.PRECEDENCE

    def _to_string(self, values, verbose, smap, compute_values):
        #pass this one for now 0-0
        pass

    def _apply_operation(self, res_list):
        if (len(self._args_)-1 != len(res_list)):
            KeyError("Make sure number of truth values matches number"\
             "of children for this node")
        counter = 0
        for tmp in res_list[1:]:
            if(tmp == True):
                counter += 1
        return (counter <= self._args_[0])



Expressions = (LogicalExpressionBase, NotExpression, AndExpression, OrExpression,
    Implication, EquivalenceExpression, XorExpression, ExactlyExpression,
    AtMostExpression, AtLeastExpression, Not, Equivalence, LogicalOr, Implies,
    LogicalAnd, Exactly, AtMost, AtLeast, LogicalXor
    )





