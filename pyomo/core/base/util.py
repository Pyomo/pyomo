#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

#
# Utility functions
#

__all__ = ['summation', 'dot_product', 'sequence', 'prod']

import pyomo.core.base.var
import inspect
from six.moves import xrange
from functools import reduce
import operator


def prod(factors):
    """
    A utility function to compute the product of a list of factors.
    """
    return reduce(operator.mul, factors, 1)


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
    if len(args) == 0 and len(denom) == 0:
        raise ValueError("The summation() command requires at least an " + \
              "argument or a denominator term")
    if 'index' in kwds:
        index=kwds['index']
    else:
        if len(args) > 0:
            iarg=args[-1]
            if not isinstance(iarg,pyomo.core.base.var.Var) and not isinstance(iarg, pyomo.core.base.expression.Expression):
                raise ValueError("Error executing summation(): The last argument value must be a variable or expression object if no 'index' option is specified")
        else:
            iarg=denom[-1]
            if not isinstance(iarg,pyomo.core.base.var.Var) and not isinstance(iarg, pyomo.core.base.expression.Expression):
                raise ValueError("Error executing summation(): The last denom argument value must be a variable or expression object if no 'index' option is specified")
        index = iarg.index_set()

    ans = 0
    num_index = range(0,len(args))
    denom_index = range(0,len(denom))
    #
    # Iterate through all indices
    #
    for i in index:
        #
        # Iterate through all arguments
        #
        item = 1
        for j in num_index:
            item *= args[j][i]
        for j in denom_index:
            item /= denom[j][i]
        ans += item
    return ans


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


def is_functor(obj):
    """
    Returns true iff obj.__call__ is defined.
    """
    return inspect.isfunction(obj) or hasattr(obj,'__call__')

