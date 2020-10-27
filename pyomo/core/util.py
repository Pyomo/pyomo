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

__all__ = ['sum_product', 'summation', 'dot_product', 'sequence', 'prod', 'quicksum']

from six.moves import xrange
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.numeric_expr import decompose_term
from pyomo.core.expr import current as EXPR
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression


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
    """
    A utility function to compute a sum of Pyomo expressions.

    The behavior of :func:`quicksum` is similar to the builtin :func:`sum`
    function, but this function generates a more compact Pyomo
    expression.

    Args:
        args: A generator for terms in the sum.

        start: A value that is initializes the sum.  If
            this value is not a numeric constant, then the += 
            operator is used to add terms to this object.
            Defaults to zero.

        linear: If :attr:`start` is not a numeric constant, then this 
            option is ignored.  Otherwise, this value indicates
            whether the terms in the sum are linear.  If the value
            is :const:`False`, then the terms are
            treated as nonlinear, and if :const:`True`, then
            the terms are treated as linear.  Default is
            :const:`None`, which indicates that the first term
            in the :attr:`args` is used to determine this value.

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
            try:
                first = next(args, None)
            except:
                try:
                    args = args.__iter__()
                    first = next(args, None)
                except:
                    raise RuntimeError("The argument to quicksum() is not iterable!")
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
            # Also, we treat linear expressions as nonlinear if the constant
            # term is not a native numeric type.  Otherwise, large summation
            # objects are created for the constant term.
            #
            if linear:
                nvar=0
                for term in terms:
                    c,v = term
                    if not v is None:
                        nvar += 1
                    elif not c.__class__ in native_numeric_types:
                        linear = False
                if nvar > 1:
                    linear = False
            start = start+first
        if linear:
            with EXPR.linear_expression() as e:
                e += start
                for arg in args:
                    e += arg
            # Return the constant term if the linear expression does not contains variables
            if e.is_constant():
                return e.constant
            return e
        else:
            with EXPR.nonlinear_expression() as e:
                e += start
                for arg in args:
                    e += arg
            if e.nargs() == 0:
                return 0
            elif e.nargs() == 1:
                return e.arg(0)
            return e
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
    denom = kwds.pop('denom', tuple() )
    if type(denom) not in (list, tuple):
        denom = [denom]
    nargs = len(args)
    ndenom = len(denom)

    if nargs == 0 and ndenom == 0:
        raise ValueError("The sum_product() command requires at least an " + \
              "argument or a denominator term")

    if 'index' in kwds:
        index=kwds['index']
    else:
        if nargs > 0:
            iarg=args[-1]
            if not isinstance(iarg,Var) and not isinstance(iarg, Expression):
                raise ValueError("Error executing sum_product(): The last argument value must be a variable or expression object if no 'index' option is specified")
        else:
            iarg=denom[-1]
            if not isinstance(iarg,Var) and not isinstance(iarg, Expression):
                raise ValueError("Error executing sum_product(): The last denom argument value must be a variable or expression object if no 'index' option is specified")
        index = iarg.index_set()

    start = kwds.get("start", 0)
    vars_ = []
    params_ = []
    for arg in args:
        if isinstance(arg, Var):
            vars_.append(arg)
        else:
            params_.append(arg)
    nvars = len(vars_)

    num_index = range(0,nargs)
    if ndenom == 0:
        #
        # Sum of polynomial terms
        #
        if start.__class__ in native_numeric_types:
            if nvars == 1:
                v = vars_[0]
                if len(params_) == 0:
                    with EXPR.linear_expression() as expr:
                        expr += start
                        for i in index:
                            expr += v[i]
                elif len(params_) == 1:    
                    p = params_[0]
                    with EXPR.linear_expression() as expr:
                        expr += start
                        for i in index:
                            expr += p[i]*v[i]
                else:
                    with EXPR.linear_expression() as expr:
                        expr += start
                        for i in index:
                            term = 1
                            for j in params_:
                                term *= params_[j][i]
                            expr += term * v[i]
                return expr
            #
            with EXPR.nonlinear_expression() as expr:
                expr += start
                for i in index:
                    term = 1
                    for j in num_index:
                        term *= args[j][i]
                    expr += term
            return expr
        #
        return quicksum((prod(args[j][i] for j in num_index) for i in index), start)
    elif nargs == 0:
        #
        # Sum of reciprocals
        #
        denom_index = range(0,ndenom)
        return quicksum((1/prod(denom[j][i] for j in denom_index) for i in index), start)
    else:
        #
        # Sum of fractions
        #
        denom_index = range(0,ndenom)
        return quicksum((prod(args[j][i] for j in num_index)/prod(denom[j][i] for j in denom_index) for i in index), start)


#: An alias for :func:`sum_product <pyomo.core.expr.util>`
dot_product = sum_product

#: An alias for :func:`sum_product <pyomo.core.expr.util>`
summation = sum_product


def sequence(*args):
    """
    sequence([start,] stop[, step]) -> generator for a list of integers

    Return a generator that containing an arithmetic
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
    from pyomo.common.deprecation import deprecation_warning
    deprecation_warning("The xsequence function is deprecated.  Use the sequence() function, which returns a generator.")  # Remove in Pyomo 6.0
    return sequence(*args)

