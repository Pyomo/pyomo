#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.sympy_tools import sympy_available, sympyify_expression, sympy2pyomo_expression

# A "public" attribute indicating that differentiate() can be called
# ... this provides a bit of future-proofing for alternative approaches
# to symbolic differentiation.
differentiate_available = sympy_available


def differentiate(expr, wrt=None, wrt_list=None):
    """Return derivative of expression.

    This function returns an expression or list of expression objects
    corresponding to the derivative of the passed expression 'expr' with
    respect to a variable 'wrt' or list of variables 'wrt_list'

    Args:
        expr (Expression): Pyomo expression
        wrt (Var): Pyomo variable
        wrt_list (list): list of Pyomo variables

    Returns:
        Expression or list of Expression objects

    """
    if not sympy_available:
        raise RuntimeError(
            "The sympy module is not available.\n\t"
            "Cannot perform automatic symbolic differentiation.")
    if not (( wrt is None ) ^ ( wrt_list is None )):
        raise ValueError(
            "differentiate(): Must specify exactly one of wrt and wrt_list")
    import sympy
    #
    # Convert the Pyomo expression to a sympy expression
    #
    objectMap, sympy_expr = sympyify_expression(expr)
    #
    # The partial_derivs dict holds intermediate sympy expressions that
    # we can re-use.  We will prepopulate it with None for all vars that
    # appear in the expression (so that we can detect wrt combinations
    # that are, by definition, 0)
    #
    partial_derivs = {x:None for x in objectMap.sympyVars()}
    #
    # Setup the WRT list
    #
    if wrt is not None:
        wrt_list = [ wrt ]
    else:
        # Copy the list because we will normalize things in place below
        wrt_list = list(wrt_list)
    #
    # Convert WRT vars into sympy vars
    #
    ans = [None]*len(wrt_list)
    for i, target in enumerate(wrt_list):
        if target.__class__ is not tuple:
            target = (target,)
        wrt_list[i] = tuple(objectMap.getSympySymbol(x) for x in target)
        for x in wrt_list[i]:
            if x not in partial_derivs:
                ans[i] = 0.
                break
    #
    # We assume that users will not request duplicate derivatives.  We
    # will only cache up to the next-to last partial, and if a user
    # requests the exact same derivative twice, then we will just
    # re-calculate it.
    #
    last_partial_idx = max(len(x) for x in wrt_list) - 1
    #
    # Calculate all the derivatives
    #
    for i, target in enumerate(wrt_list):
        if ans[i] is not None:
            continue
        part = sympy_expr
        for j, wrt_var in enumerate(target):
            if j == last_partial_idx:
                part = sympy.diff(part, wrt_var)
            else:
                partial_target = target[:j+1]
                if partial_target in partial_derivs:
                    part = partial_derivs[partial_target]
                else:
                    part = sympy.diff(part, wrt_var)
                    partial_derivs[partial_target] = part
        ans[i] = sympy2pyomo_expression(part, objectMap)
    #
    # Return the answer
    #
    return ans if wrt is None else ans[0]
