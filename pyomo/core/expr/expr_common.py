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
from contextlib import nullcontext

from pyomo.common.deprecation import deprecated

TO_STRING_VERBOSE = False

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


#
# Provide a global value that indicates which expression system is being used
#
class Mode(enum.IntEnum):
    # coopr: Original Coopr/Pyomo expression system
    coopr_trees = 1
    # coopr3: leverage reference counts to reduce the amount of required
    # expression cloning to ensure independent expression trees.
    coopr3_trees = 3
    # pyomo4: rework the expression system to remove reliance on
    # reference counting.  This enables pypy support (which doesn't have
    # reference counting).  This version never became the default.
    pyomo4_trees = 4
    # pyomo5: refinement of pyomo4.  Expressions are now immutable by
    # contract, which tolerates "entangled" expression trees.  Added
    # specialized classes for NPV expressions and LinearExpressions.
    pyomo5_trees = 5
    # pyomo6: refinement of pyomo5 expression generation to leverage
    # multiple dispatch.  Standardized expression storage and argument
    # handling (significant rework of the LinearExpression structure).
    pyomo6_trees = 6
    #
    CURRENT = pyomo6_trees


_mode = Mode.CURRENT
# We no longer support concurrent expression systems.  _mode is left
# primarily so we can support expression system-specific baselines
assert _mode == Mode.pyomo6_trees


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
