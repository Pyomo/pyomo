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

import enum

from pyomo.common.backports import nullcontext
from pyomo.common.deprecation import deprecated

TO_STRING_VERBOSE = False

_add = 1
_sub = 2
_mul = 3
_div = 4
_pow = 5
_neg = 6
_abs = 7
_inplace = 10
_unary = _neg

_radd = -_add
_iadd = _inplace + _add
_rsub = -_sub
_isub = _inplace + _sub
_rmul = -_mul
_imul = _inplace + _mul
_rdiv = -_div
_idiv = _inplace + _div
_rpow = -_pow
_ipow = _inplace + _pow

_eq = 0
_le = 1
_lt = 2

# logical propositions
_and = 0
_or = 1
_inv = 2
_equiv = 3
_xor = 4
_impl = 5


class OperatorAssociativity(enum.IntEnum):
    """Enum for indicating the associativity of an operator.

    LEFT_TO_RIGHT(1) if this operator is left-to-right associative or
    RIGHT_TO_LEFT(-1) if it is right-to-left associative.  Any other
    values will be interpreted as "not associative" (implying any
    arguments that are at this operator's PRECEDENCE will be enclosed
    in parens).

    """

    RIGHT_TO_LEFT = -1
    NON_ASSOCIATIVE = 0
    LEFT_TO_RIGHT = 1


class ExpressionType(enum.Enum):
    NUMERIC = 0
    RELATIONAL = 1
    LOGICAL = 2


@deprecated(
    """The clone counter has been removed and will always return 0.

Beginning with Pyomo5 expressions, expression cloning (detangling) no
longer occurs automatically within expression generation.  As a result,
the 'clone counter' has lost its utility and is no longer supported.
This context manager will always report 0.""",
    version='6.4.3',
)
class clone_counter(nullcontext):
    """Context manager for counting cloning events.

    This context manager counts the number of times that the
    :func:`clone_expression <pyomo.core.expr.current.clone_expression>`
    function is executed.
    """

    _count = 0

    def __init__(self):
        super().__init__(enter_result=self)

    @property
    def count(self):
        """A property that returns the clone count value."""
        return clone_counter._count
