#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import native_numeric_types, value
from pyomo.core.base.symbolic import differentiate

def calculate_variable_from_constraint(variable, constraint,
                                       eps=1e-8, iterlim=1000):
    """Calculate the variable value given a specified equality constraint

    This function calculates the value of the specified variable
    necessary to make the provided equality constraint feasible
    (assuming any other variables values are fixed).  The method first
    attempts to solve for the variable value assuming it appears
    linearly in the constraint.  If that doesn't converge the constraint
    residual, it falls back on Newton's method using exact (symbolic)
    derivatives.

    Parameters:
    -----------
    variable: `pyomo.core.base.var._VarData`
        The variable to solve for
    constraint: `pyomo.core.base.constraint._ConstraintData`
        The equality constraint to use to solve for the variable value
    eps: `float`
        The tolerance to use to determine equality [default=1e-8].
    iterlim: `int`
        The maximum number of iterations if this method has to fall back
        on using Newton's method.  Raises RuntimeError on iteration
        limit [default=1000]

    Returns:
    --------
    None

    Note: this is an unconstrained solver and is NOT guaranteed to
    respect the variable bounds.

    """
    upper = value(constraint.upper)
    if value(constraint.lower) != upper:
        raise ValueError("Constraint must be an equality constraint")

    if variable.value is None:
        if variable.lb is None:
            if variable.ub is None:
                variable.set_value(0)
            else:
                variable.set_value(min(0, variable.ub))
        elif variable.ub is None:
            variable.set_value(max(0, variable.lb))
        else:
            if variable.lb <= 0 and variable.ub >= 0:
                variable.set_value(0)
            else:
                variable.set_value((variable.lb+variable.ub)/2.0)
    x1 = value(variable)
    residual_1 = value(constraint.body)

    variable.set_value(x1 - (residual_1-upper))
    residual_2 = value(constraint.body)

    # if the variable appears linearly with a coefficient of 1, then we
    # are done
    if abs(residual_2-upper) < eps:
        return

    # Assume the variable appears linearly and calculate the coefficient
    x2 = value(variable)
    slope = float(residual_1 - residual_2) / (x1 - x2)
    intercept = (residual_1-upper) - slope*x1
    if slope:
        variable.set_value(-intercept/slope)
        if abs(value(constraint.body)-upper) < eps:
            return

    # Variable appears nonlinearly; solve using Newton's method
    expr = constraint.body - constraint.upper
    expr_deriv = differentiate(expr, wrt=variable)
    if type(expr_deriv) in native_numeric_types and expr_deriv == 0:
        raise ValueError("variable derivative == 0, cannot solve for variable")

    iter_left = iterlim
    while abs(value(expr)) > eps and iter_left:
        iter_left -= 1
        if not iter_left:
            raise RuntimeError(
                "Iteration limit (%s) reached; remaining residual=%s"
                % (iterlim, value(expr)) )
        variable.set_value(value(variable) - value(expr)/value(expr_deriv))
        
