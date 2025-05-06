#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from contextlib import nullcontext

from pyomo.common import enums
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET

TO_STRING_VERBOSE = False

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
class Mode(enums.IntEnum):
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


class OperatorAssociativity(enums.IntEnum):
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


class ExpressionType(enums.Enum):
    NUMERIC = 0
    RELATIONAL = 1
    LOGICAL = 2


class NUMERIC_ARG_TYPE(enums.IntEnum):
    MUTABLE = -2
    ASNUMERIC = -1
    INVALID = 0
    NATIVE = 1
    NPV = 2
    PARAM = 3
    VAR = 4
    MONOMIAL = 5
    LINEAR = 6
    SUM = 7
    OTHER = 8


class RELATIONAL_ARG_TYPE(enums.IntEnum, metaclass=enums.ExtendedEnumType):
    __base_enum__ = NUMERIC_ARG_TYPE

    INEQUALITY = 100
    INVALID_RELATIONAL = 101


def _invalid(*args):
    return NotImplemented


def _recast_mutable(expr):
    expr.make_immutable()
    if expr._nargs > 1:
        return expr
    elif not expr._nargs:
        return 0
    else:
        return expr._args_[0]


def _unary_op_dispatcher_type_mapping(dispatcher, updates, TYPES=NUMERIC_ARG_TYPE):
    #
    # Special case (wrapping) operators
    #
    def _asnumeric(a):
        a = a.as_numeric()
        return dispatcher[a.__class__](a)

    def _mutable(a):
        a = _recast_mutable(a)
        return dispatcher[a.__class__](a)

    mapping = {
        TYPES.ASNUMERIC: _asnumeric,
        TYPES.MUTABLE: _mutable,
        TYPES.INVALID: _invalid,
    }

    mapping.update(updates)
    return mapping


def _binary_op_dispatcher_type_mapping(dispatcher, updates, TYPES=NUMERIC_ARG_TYPE):
    #
    # Special case (wrapping) operators
    #
    def _any_asnumeric(a, b):
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_any(a, b):
        a = a.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_asnumeric(a, b):
        a = a.as_numeric()
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _any_mutable(a, b):
        b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_any(a, b):
        a = _recast_mutable(a)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_mutable(a, b):
        if a is b:
            # Note: _recast_mutable is an in-place operation: make sure
            # that we don't call it twice on the same object.
            a = b = _recast_mutable(a)
        else:
            a = _recast_mutable(a)
            b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)

    mapping = {}

    # Because ASNUMERIC and MUTABLE re-call the dispatcher, we want to
    # resolve ASNUMERIC first, MUTABLE second, and INVALID last.  That
    # means we will add them to the dispatcher dict in opposite order so
    # "higher priority" callbacks override lower priority ones.

    mapping.update({(i, TYPES.INVALID): _invalid for i in TYPES})
    mapping.update({(TYPES.INVALID, i): _invalid for i in TYPES})

    mapping.update({(i, TYPES.MUTABLE): _any_mutable for i in TYPES})
    mapping.update({(TYPES.MUTABLE, i): _mutable_any for i in TYPES})
    mapping[TYPES.MUTABLE, TYPES.MUTABLE] = _mutable_mutable

    mapping.update({(i, TYPES.ASNUMERIC): _any_asnumeric for i in TYPES})
    mapping.update({(TYPES.ASNUMERIC, i): _asnumeric_any for i in TYPES})
    mapping[TYPES.ASNUMERIC, TYPES.ASNUMERIC] = _asnumeric_asnumeric

    mapping.update(updates)
    return mapping


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


def _type_check_exception_arg(cls, exception):
    if exception is NOTSET:
        return True
    elif type(exception) is not bool:
        raise ValueError(
            f"{cls.ctype.__name__} '{cls.name}' was called with a non-bool "
            f"argument for 'exception': {exception}"
        )
    else:
        return exception
