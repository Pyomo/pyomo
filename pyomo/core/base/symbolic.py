#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import deprecated
import pyomo.core.expr.calculus.derivatives as diff_core
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.common.errors import NondifferentiableError


@deprecated(msg=('The differentiate function in pyomo.core.base.symbolic has been deprecated. Please use the ' +
                 'differentiate function in pyomo.core.expr.'),
            version='5.6.7',
            remove_in='5.7')
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
    return diff_core.differentiate(expr=expr, wrt=wrt, wrt_list=wrt_list, mode=diff_core.Modes.sympy)
