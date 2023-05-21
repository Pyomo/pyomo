# -*- coding: utf-8 -*-
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

import types
from itertools import islice

import logging
import traceback

logger = logging.getLogger('pyomo.core')

from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
    deprecation_warning,
    RenamedClass,
    relocated_module_attribute,
)
from .numvalue import (
    native_types,
    native_numeric_types,
    as_numeric,
    native_logical_types,
    value,
    is_potentially_variable,
)
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType

import operator

relocated_module_attribute(
    'EqualityExpression',
    'pyomo.core.expr.relational_expr.EqualityExpression',
    version='6.4.3',
    f_globals=globals(),
)
relocated_module_attribute(
    'InequalityExpression',
    'pyomo.core.expr.relational_expr.InequalityExpression',
    version='6.4.3',
    f_globals=globals(),
)
relocated_module_attribute(
    'RangedExpression',
    'pyomo.core.expr.relational_expr.RangedExpression',
    version='6.4.3',
    f_globals=globals(),
)
relocated_module_attribute(
    'inequality',
    'pyomo.core.expr.relational_expr.inequality',
    version='6.4.3',
    f_globals=globals(),
)


def _generate_logical_proposition(etype, lhs, rhs):
    if (
        lhs.__class__ in native_types and lhs.__class__ not in native_logical_types
    ) and not isinstance(lhs, BooleanValue):
        return NotImplemented
    if (
        (rhs.__class__ in native_types and rhs.__class__ not in native_logical_types)
        and not isinstance(rhs, BooleanValue)
        and not (rhs is None and etype == _inv)
    ):
        return NotImplemented

    if etype == _equiv:
        return EquivalenceExpression((lhs, rhs))
    elif etype == _inv:
        assert rhs is None
        return NotExpression((lhs,))
    elif etype == _xor:
        return XorExpression((lhs, rhs))
    elif etype == _impl:
        return ImplicationExpression((lhs, rhs))
    elif etype == _and:
        return land(lhs, rhs)
    elif etype == _or:
        return lor(lhs, rhs)
    else:
        raise ValueError(
            "Unknown logical proposition type '%s'" % etype
        )  # pragma: no cover


class BooleanExpression(ExpressionBase, BooleanValue):
    """
    Logical expression base class.

    This class is used to define nodes in an expression
    tree.

    Abstract

    args:
        args (list or tuple): Children of this node.
    """

    __slots__ = ('_args_',)
    EXPRESSION_SYSTEM = ExpressionType.LOGICAL
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    @property
    def args(self):
        """
        Return the child nodes

        Returns: Either a list or tuple (depending on the node storage
            model) containing only the child nodes of this node
        """
        return self._args_[: self.nargs()]


class BooleanExpressionBase(metaclass=RenamedClass):
    __renamed__new_class__ = BooleanExpression
    __renamed__version__ = '6.4.3'


"""
---------------------------******************--------------------
The following methods are static methods for nodes creator. Those should
do the exact same thing as the class methods as well as overloaded operators.
"""


def lnot(Y):
    """
    Construct a NotExpression for the passed BooleanValue.
    """
    return NotExpression((Y,))


def equivalent(Y1, Y2):
    """
    Construct an EquivalenceExpression Y1 == Y2
    """
    return EquivalenceExpression((Y1, Y2))


def xor(Y1, Y2):
    """
    Construct an XorExpression Y1 xor Y2
    """
    return XorExpression((Y1, Y2))


def implies(Y1, Y2):
    """
    Construct an Implication using function, where Y1 implies Y2
    """
    return ImplicationExpression((Y1, Y2))


def _flattened(args):
    """Flatten any potentially indexed arguments."""
    for arg in args:
        if arg.__class__ in native_types:
            yield arg
        else:
            if isinstance(arg, (types.GeneratorType, list)):
                for _argdata in arg:
                    yield _argdata
            elif arg.is_indexed():
                for _argdata in arg.values():
                    yield _argdata
            else:
                yield arg


def land(*args):
    """
    Construct an AndExpression between passed arguments.
    """
    result = AndExpression([])
    for argdata in _flattened(args):
        result = result.add(argdata)
    return result


def lor(*args):
    """
    Construct an OrExpression between passed arguments.
    """
    result = OrExpression([])
    for argdata in _flattened(args):
        result = result.add(argdata)
    return result


def exactly(n, *args):
    """Creates a new ExactlyExpression

    Require exactly n arguments to be True, to make the expression True

    Usage: exactly(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = ExactlyExpression([n] + list(_flattened(args)))
    return result


def atmost(n, *args):
    """Creates a new AtMostExpression

    Require at most n arguments to be True, to make the expression True

    Usage: atmost(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = AtMostExpression([n] + list(_flattened(args)))
    return result


def atleast(n, *args):
    """Creates a new AtLeastExpression

    Require at least n arguments to be True, to make the expression True

    Usage: atleast(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = AtLeastExpression([n] + list(_flattened(args)))
    return result


class UnaryBooleanExpression(BooleanExpression):
    """
    Abstract class for single-argument logical expressions.
    """

    def nargs(self):
        """
        Returns number of arguments in expression
        """
        return 1


class NotExpression(UnaryBooleanExpression):
    """
    This is the node for a NotExpression, this node should have exactly one child
    """

    PRECEDENCE = 2

    def getname(self, *arg, **kwd):
        return 'Logical Negation'

    def _to_string(self, values, verbose, smap):
        return "~%s" % values[0]

    def _apply_operation(self, result):
        return not result[0]


class BinaryBooleanExpression(BooleanExpression):
    """
    Abstract class for binary logical expressions.
    """

    def nargs(self):
        """
        Return the number of argument the expression has
        """
        return 2


class EquivalenceExpression(BinaryBooleanExpression):
    """
    Logical equivalence statement: Y_1 iff Y_2.

    """

    __slots__ = ()

    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'iff'

    def _to_string(self, values, verbose, smap):
        return " iff ".join(values)

    def _apply_operation(self, result):
        return result[0] == result[1]


class XorExpression(BinaryBooleanExpression):
    """
    Logical Exclusive OR statement: Y_1 ⊻ Y_2
    """

    __slots__ = ()

    PRECEDENCE = 4

    def getname(self, *arg, **kwd):
        return 'xor'

    def _to_string(self, values, verbose, smap):
        return " ⊻ ".join(values)

    def _apply_operation(self, result):
        return operator.xor(result[0], result[1])


class ImplicationExpression(BinaryBooleanExpression):
    """
    Logical Implication statement: Y_1 --> Y_2.
    """

    __slots__ = ()

    PRECEDENCE = 6

    def getname(self, *arg, **kwd):
        return 'implies'

    def _to_string(self, values, verbose, smap):
        return " --> ".join(values)

    def _apply_operation(self, result):
        return (not result[0]) or result[1]


class NaryBooleanExpression(BooleanExpression):
    """
    The abstract class for NaryBooleanExpression.

    This class should never be initialized.
    """

    __slots__ = ('_nargs',)

    def __init__(self, args):
        self._args_ = args
        self._nargs = len(self._args_)

    def nargs(self):
        """
        Return the number of expression arguments
        """
        return self._nargs

    def getname(self, *arg, **kwd):
        return 'NaryBooleanExpression'


def _add_to_and_or_expression(orig_expr, new_arg):
    """
    Since AND and OR are Nary expressions, we extend the existing expression
    instead of creating a nested expression object if the types are compatible.
    """
    # Clone 'self', because AndExpression/OrExpression are immutable
    if new_arg.__class__ is orig_expr.__class__:
        # adding new AndExpression/OrExpression on the right
        new_expr = orig_expr.__class__(orig_expr._args_)
        new_expr._args_.extend(islice(new_arg._args_, new_arg._nargs))
    else:
        # adding new singleton on the right
        new_expr = orig_expr.__class__(orig_expr._args_)
        new_expr._args_.append(new_arg)

    # TODO set up id()-based scheme for avoiding duplicate entries

    new_expr._nargs = len(new_expr._args_)
    return new_expr


class AndExpression(NaryBooleanExpression):
    """
    This is the node for AndExpression.
    """

    __slots__ = ()

    PRECEDENCE = 3

    def getname(self, *arg, **kwd):
        return 'and'

    def _to_string(self, values, verbose, smap):
        return " ∧ ".join(values)

    def _apply_operation(self, result):
        return all(result)

    def add(self, new_arg):
        if new_arg.__class__ in native_logical_types:
            if new_arg is False:
                return BooleanConstant(False)
            elif new_arg is True:
                return self
        return _add_to_and_or_expression(self, new_arg)


class OrExpression(NaryBooleanExpression):
    """
    This is the node for OrExpression.
    """

    __slots__ = ()

    PRECEDENCE = 5

    def getname(self, *arg, **kwd):
        return 'or'

    def _to_string(self, values, verbose, smap):
        return " ∨ ".join(values)

    def _apply_operation(self, result):
        return any(result)

    def add(self, new_arg):
        if new_arg.__class__ in native_logical_types:
            if new_arg is False:
                return self
            elif new_arg is True:
                return BooleanConstant(True)
        return _add_to_and_or_expression(self, new_arg)


class ExactlyExpression(NaryBooleanExpression):
    """
    Logical constraint that exactly N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: exactly(1, True, False, False) --> True

    """

    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'exactly'

    def _to_string(self, values, verbose, smap):
        return "exactly(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) == result[0]


class AtMostExpression(NaryBooleanExpression):
    """
    Logical constraint that at most N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: atmost(1, True, False, False) --> True

    """

    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'atmost'

    def _to_string(self, values, verbose, smap):
        return "atmost(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) <= result[0]


class AtLeastExpression(NaryBooleanExpression):
    """
    Logical constraint that at least N child statements are True.

    The first argument N is expected to be a numeric non-negative integer.
    Subsequent arguments are expected to be Boolean.

    Usage: atleast(1, True, False, False) --> True

    """

    __slots__ = ()

    PRECEDENCE = 9

    def getname(self, *arg, **kwd):
        return 'atleast'

    def _to_string(self, values, verbose, smap):
        return "atleast(%s: [%s])" % (values[0], ", ".join(values[1:]))

    def _apply_operation(self, result):
        return sum(result[1:]) >= result[0]


special_boolean_atom_types = {ExactlyExpression, AtMostExpression, AtLeastExpression}
