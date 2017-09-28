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
from pyomo.core.base import expr_common
from pyomo.core.kernel.numvalue import native_numeric_types
from pyomo.core.kernel import expr as EXPR
import pyomo.core.base.var


def prod(factors):
    """
    A utility function to compute the product of a list of factors.
    """
    return reduce(operator.mul, factors, 1)

Prod = prod


def Sum(args, start=0):
    """
    A utility function to compute a sum of Pyomo expressions.  The behavior is similar to the
    builtin 'sum' function, but this generates a compact expression.
    """
    if expr_common.mode == expr_common.Mode.pyomo5_trees:
        #
        # If we're starting with a numeric value, then 
        # create a new nonlinear sum expression but 
        # return a static version to the user.
        #
        if start.__class__ in native_numeric_types:
            with EXPR.nonlinear_expression as e:
                e += start
                for arg in args:
                    e += arg
            if len(e._args) == 0:
                return 0
            elif len(e._args) == 1:
                return e._args[0]
            else:
                return e
        #
        # Otherwise, use the context that is provided and return it.
        #
        e = start
        for arg in args:
            e += arg
        return e
    else:
        return sum(*args)


def summation(*args, **kwds):
    """
    A utility function to compute a generalized dot product.  The following examples illustrate
    the use of this function:

    summation(x)
    Sum the elements of x

    summation(x,y)
    Sum the product of elements in x and y

    summation(x,y, index=z)
    Sum the product of elements in x and y, over the index set z

    summation(x, denom=a)
    Sum the product of x_i/a_i

    summation(denom=(a,b))
    Sum the product of 1/(a_i*b_i)
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
    nvars = sum(1 if isinstance(arg,pyomo.core.base.var.Var) else 0 for arg in args[:])

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


def dot_product(*args, **kwds):
    """
    A synonym for the summation() function.
    """
    return summation(*args, **kwds)


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

