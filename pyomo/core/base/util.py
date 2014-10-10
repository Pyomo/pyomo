#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

#
# Utility functions
#

__all__ = ['summation', 'dot_product', 'sequence', 'xsequence']

try:
    xrange = xrange
except:
    xrange = range
    __all__.append('xrange')

import pyomo.core.base.var
import inspect
#import pyomo.core.base.expr

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
            if not isinstance(iarg,pyomo.core.base.var.Var):
                raise ValueError("Error executing summation(): The last argument value must be a variable object if no 'index' option is specified")
        else:
            iarg=denom[-1]
            if not isinstance(iarg,pyomo.core.base.var.Var):
                raise ValueError("Error executing summation(): The last denom argument value must be a variable object if no 'index' option is specified")
        index = iarg.keys()

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

# This is deprecated.  The same functionality is provided by
# itertools.product
def multisum(*args):
    """
    Returns a generator that produces all possible permutations from an
    arbitrary set of containers.

    Example:

    >>> a = [1,2,3]
    >>> b = ['foo','bar']
    >>> c = [4,5,6]

    >>> for (i,j,k) in multisum(a,b,c):
            print i,j,k

    1 foo 4
    2 foo 4
    3 foo 4
    1 bar 4
    2 bar 4
    3 bar 4
    1 foo 5
    2 foo 5
    3 foo 5
    1 bar 5
    2 bar 5
    3 bar 5
    1 foo 6
    2 foo 6
    3 foo 6
    1 bar 6
    2 bar 6
    3 bar 6

    """
    nargs = len(args)

    # Generators
    gens = [arg.__iter__() for arg in args]

    # Current value list; note that this will fail if any container is empty,
    # which is the desired behavior
    current = [g.next() for g in gens]

    i = 0
    while i < nargs:
        yield current

        # Find the next legal iterator, or break if we run out
        while True:
            if i >= nargs:
                break
            try:
                current[i] = gens[i].next()
                i = 0
                break
            except StopIteration:
                gens[i] = args[i].__iter__()
                current[i] = gens[i].next()
                i += 1


def sequence(*args):
    """
    sequence([start,] stop[, step]) -> list of integers

    Return a list containing an arithmetic progression of integers.
    sequence(i, j) returns [i, i+1, i+2, ..., j]; start (!) defaults to 1.
    When step is given, it specifies the increment (or decrement).
    For example, sequence(4) returns [1, 2, 3, 4].
    """
    if len(args) == 0:
        raise TypeError('sequence expected at least 1 arguments, got 0')
    if len(args) > 3:
        raise TypeError('sequence expected at most 3 arguments, got %d' % len(args))
    if len(args) == 1:
        return range(1,args[0]+1)
    if len(args) == 2:
        return range(args[0],args[1]+1)
    return range(args[0],args[1]+1,args[2])

def xsequence(*args):
    """
    xsequence([start,] stop[, step]) -> xrange object

    Like sequence(), but instead of returning a list, returns an object that
    generates the numbers in the sequence on demand.  For looping, this is
    slightly faster than sequence() and more memory efficient.
    """
    if len(args) == 0:
        raise TypeError('xsequence expected at least 1 arguments, got 0')
    if len(args) > 3:
        raise TypeError('xsequence expected at most 3 arguments, got %d' % len(args))
    if len(args) == 1:
        return xrange(1,args[0]+1)
    if len(args) == 2:
        return xrange(args[0],args[1]+1)
    return xrange(args[0],args[1]+1,args[2])

def is_functor(obj):
    """
    Returns true iff obj.__call__ is defined.
    """
    return inspect.isfunction(obj) or hasattr(obj,'__call__')

