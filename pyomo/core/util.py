#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Utility functions
#

__all__ = ['summation', 'dot_product', 'sequence', 'prod', 'Prod', 'Sum']

from six.moves import xrange
from functools import reduce
import operator
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.expr_pyomo5 import decompose_term
from pyomo.core import expr as EXPR
import pyomo.core.base.var


def prod(factors):
    """
    A utility function to compute the product of a list of factors.

    Args:
        factors (list): A list of terms that are multiplied together.

    Returns:
        The value of the product, which may be a Pyomo expression object.
    """
    return reduce(operator.mul, factors, 1)

Prod = prod


def Sum(args, start=0, linear=None):
    """
    A utility function to compute a sum of Pyomo expressions.

    The behavior of :func:`Sum` is similar to the builtin :func:`sum`
    function, but this function generates a more compact Pyomo
    expression.

    Args:
        args: A generator for terms in the sum.
        start: A value that is initializes the sum.  This value may be
            any Python object, which allow the user to define the type
            that contains the sum that is generated. Defaults to zero.
        linear: If :attr:`start` is zero, then this value indicates whether 
            the terms in the sum
            are linear.  Otherwise, this option is ignored.
    
            If the value is :const:`False`, then the terms are treated
            as nonlinears, and if :const:`True`, then the terms
            are treated as linear.  Default is :const:`None`, which 
            indicates that the first term in the :attr:`args` is used
            to determine this value.

    Returns:
        The value of the sum, which may be a Pyomo expression object.
    """
    #
    # If we're starting with a numeric value, then 
    # create a new nonlinear sum expression but 
    # return a static version to the user.
    #
    if start.__class__ in native_numeric_types:
        if linear is None:
            #
            # Get the first term, which we will test for linearity
            #
            first = next(args, None)
            if first is None:
                return start
            #
            # Check if the first term is linear, and if so return the terms
            #
            linear, terms = decompose_term(first)
            #
            # Right now Pyomo5 expressions can only handle single linear
            # terms.
            #
            if linear:
                nvar=0
                for term in terms:
                    c,v = term
                    if not v is None:
                        nvar += 1
                #
                # NOTE: We treat constants as nonlinear since we want to 
                # simply keep them in a sum
                #
                if nvar == 0 or nvar > 1:
                    linear = False
            start = start+first
        if linear:
            with EXPR.linear_expression as e:
                e += start
                for arg in args:
                    e += arg
            return e
        else:
            with EXPR.nonlinear_expression as e:
                e += start
                for arg in args:
                    e += arg
            if len(e._args) == 0:
                return 0
            elif len(e._args) == 1:
                return e._args[0]
            return e
    #
    # Otherwise, use the context that is provided and return it.
    #
    e = start
    for arg in args:
        e += arg
    return e


def summation(*args, **kwds):
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
    denom = kwds.pop('denom', tuple() )
    if type(denom) not in (list, tuple):
        denom = [denom]
    nargs = len(args)
    ndenom = len(denom)

    if nargs == 0 and ndenom == 0:
        raise ValueError("The summation() command requires at least an " + \
              "argument or a denominator term")

    if 'index' in kwds:
        index=kwds['index']
    else:
        if nargs > 0:
            iarg=args[-1]
            if not isinstance(iarg,pyomo.core.base.var.Var) and not isinstance(iarg, pyomo.core.base.expression.Expression):
                raise ValueError("Error executing summation(): The last argument value must be a variable or expression object if no 'index' option is specified")
        else:
            iarg=denom[-1]
            if not isinstance(iarg,pyomo.core.base.var.Var) and not isinstance(iarg, pyomo.core.base.expression.Expression):
                raise ValueError("Error executing summation(): The last denom argument value must be a variable or expression object if no 'index' option is specified")
        index = iarg.index_set()

    start = kwds.get("start", 0)
    nvars = sum(1 if isinstance(arg,pyomo.core.base.var.Var) else 0 for arg in args)

    num_index = range(0,nargs)
    if ndenom == 0:
        #
        # Sum of polynomial terms
        #
        if start.__class__ in native_numeric_types:
            if nvars == 1:
                with EXPR.linear_expression as expr:
                    expr += start
                    Sum((prod(args[j][i] for j in num_index) for i in index), expr)
                return expr
            #elif nvars == 2:
            #    with EXPR.quadratic_expression as expr:
            #        expr += start
            #        Sum((prod(args[j][i] for j in num_index) for i in index), expr)
            #    return expr
            return Sum((prod(args[j][i] for j in num_index) for i in index), start)
        return Sum((prod(args[j][i] for j in num_index) for i in index), start)
    elif nargs == 0:
        #
        # Sum of reciprocals
        #
        denom_index = range(0,ndenom)
        return Sum((1/prod(denom[j][i] for j in denom_index) for i in index), start)
    else:
        #
        # Sum of fractions
        #
        denom_index = range(0,ndenom)
        return Sum((prod(args[j][i] for j in num_index)/prod(denom[j][i] for j in denom_index) for i in index), start)


#: An alias for :func:`summation <pyomo.core.expr.util>`
dot_product = summation


def sequence(*args):
    """
    sequence([start,] stop[, step]) -> generator for a list of integers

    Return a generator that creates a list containing an arithmetic
    progression of integers.  
       sequence(i, j) returns [i, i+1, i+2, ..., j]; 
       start defaults to 1.  
       step specifies the increment (or decrement)
    For example, sequence(4) returns [1, 2, 3, 4].
    """
    if len(args) == 0:
        raise ValueError('sequence expected at least 1 arguments, got 0')
    if len(args) > 3:
        raise ValueError('sequence expected at most 3 arguments, got %d' % len(args))
    if len(args) == 1:
        return xrange(1,args[0]+1)
    if len(args) == 2:
        return xrange(args[0],args[1]+1)
    return xrange(args[0],args[1]+1,args[2])


def xsequence(*args):
    print("WARNING: The xsequence function is deprecated.  Use the sequence function, which returns a generator.") 
    return sequence(*args)

