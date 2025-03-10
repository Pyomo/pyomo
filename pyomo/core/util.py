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

#
# Utility functions
#

from pyomo.common.deprecation import deprecation_warning
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.numeric_expr import mutable_expression, NPV_SumExpression
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.component import ComponentBase
import logging

logger = logging.getLogger(__name__)


def prod(terms):
    """
    A utility function to compute the product of a list of terms.

    Args:
        terms (list): A list of terms that are multiplied together.

    Returns:
        The value of the product, which may be a Pyomo expression object.
    """
    ans = 1
    for term in terms:
        ans *= term
    return ans


def quicksum(args, start=0, linear=None):
    """A utility function to compute a sum of Pyomo expressions.

    The behavior of :func:`quicksum` is similar to the builtin
    :func:`sum` function, but this function can avoid the generation and
    disposal of intermediate objects, and thus is slightly more
    performant.

    Parameters
    ----------
    args: Iterable
        A generator for terms in the sum.

    start: Any
        A value that initializes the sum.  If this value is not a
        numeric constant, then the += operator is used to add terms to
        this object.  Defaults to 0.

    linear: bool
        DEPRECATED: the linearity of the resulting expression is
        determined automatically.  This option is ignored.

    Returns
    -------
    The value of the sum, which may be a Pyomo expression object.

    """

    # Ensure that args is an iterator (this manages things like
    # IndexedComponent_slice objects)
    try:
        args = iter(args)
    except:
        logger.error('The argument `args` to quicksum() is not iterable!')
        raise

    if linear is not None:
        deprecation_warning(
            "The quicksum(linear=...) argument is deprecated and ignored.",
            version='6.6.0',
        )

    #
    # If we're starting with a numeric value, then
    # create a new nonlinear sum expression but
    # return a static version to the user.
    #
    if start.__class__ in native_numeric_types:
        with mutable_expression() as e:
            e += start
            for arg in args:
                e += arg
        # Special case: reduce NPV sums of native types to a single
        # constant
        if e.__class__ is NPV_SumExpression and all(
            arg.__class__ in native_numeric_types for arg in e.args
        ):
            return e()
        if e.nargs() > 1:
            return e
        elif not e.nargs():
            return 0
        else:
            return e.arg(0)
    #
    # Otherwise, use the context that is provided and return it.
    #
    e = start
    for arg in args:
        e += arg
    return e


def sum_product(*args, **kwds):
    """
    A utility function to compute a generalized dot product.

    This function accepts one or more components that provide terms
    that are multiplied together.  These products are added together
    to form a sum.

    Args:
        *args: Variable length argument list of generators that
            create terms in the summation.
        **kwds: Arbitrary keyword arguments.

    Keyword Args:
        index: A set that is used to index the components used to
            create the terms
        denom: A component or tuple of components that are used to
            create the denominator of the terms
        start: The initial value used in the sum

    Returns:
        The value of the sum.
    """
    denom = kwds.pop('denom', tuple())
    if type(denom) not in (list, tuple):
        denom = [denom]
    nargs = len(args)
    ndenom = len(denom)

    if nargs == 0 and ndenom == 0:
        raise ValueError(
            "The sum_product() command requires at least an "
            + "argument or a denominator term"
        )

    if 'index' in kwds:
        index = kwds['index']
    else:
        if nargs > 0:
            iarg = args[-1]
            if not isinstance(iarg, Var) and not isinstance(iarg, Expression):
                raise ValueError(
                    "Error executing sum_product(): The last argument value must be a variable or expression object if no 'index' option is specified"
                )
        else:
            iarg = denom[-1]
            if not isinstance(iarg, Var) and not isinstance(iarg, Expression):
                raise ValueError(
                    "Error executing sum_product(): The last denom argument value must be a variable or expression object if no 'index' option is specified"
                )
        index = iarg.index_set()

    start = kwds.get("start", 0)

    if ndenom == 0:
        #
        # Sum of polynomial terms
        #
        with mutable_expression() as expr:
            expr += start
            if nargs == 1:
                arg1 = args[0]
                for i in index:
                    expr += arg1[i]
            elif nargs == 2:
                arg1, arg2 = args
                for i in index:
                    expr += arg1[i] * arg2[i]
            else:
                for i in index:
                    expr += prod(arg[i] for arg in args)
        if expr.nargs() > 1:
            return expr
        elif not expr.nargs():
            return 0
        else:
            return expr.arg(0)
    elif nargs == 0:
        #
        # Sum of reciprocals
        #
        return quicksum((1 / prod(den[i] for den in denom) for i in index), start)
    else:
        #
        # Sum of fractions
        #
        return quicksum(
            (
                prod(arg[i] for arg in args) / prod(den[i] for den in denom)
                for i in index
            ),
            start,
        )


#: An alias for :func:`sum_product <pyomo.core.expr.util>`
dot_product = sum_product

#: An alias for :func:`sum_product <pyomo.core.expr.util>`
summation = sum_product


def sequence(*args):
    """
    sequence([start,] stop[, step]) -> generator for a list of integers

    Return a generator that containing an arithmetic
    progression of integers.

       - ``sequence(i, j)`` returns ``[i, i+1, i+2, ..., j]``;
       - start defaults to 1.
       - step specifies the increment (or decrement)

    For example, ``sequence(4)`` returns ``[1, 2, 3, 4]``.
    """
    if len(args) == 0:
        raise ValueError('sequence expected at least 1 arguments, got 0')
    if len(args) > 3:
        raise ValueError('sequence expected at most 3 arguments, got %d' % len(args))
    if len(args) == 1:
        return range(1, args[0] + 1)
    if len(args) == 2:
        return range(args[0], args[1] + 1)
    return range(args[0], args[1] + 1, args[2])


def target_list(x):
    if isinstance(x, ComponentBase):
        return [x]
    elif hasattr(x, '__iter__'):
        ans = []
        for i in x:
            if isinstance(i, ComponentBase):
                ans.append(i)
            else:
                raise ValueError(
                    "Expected Component or list of Components."
                    "\n\tReceived %s" % (type(i),)
                )
        return ans
    else:
        raise ValueError(
            "Expected Component or list of Components.\n\tReceived %s" % (type(x),)
        )
