#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import native_numeric_types, value, is_fixed
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.base.constraint import Constraint, _ConstraintData

import logging
logger = logging.getLogger(__name__)

def calculate_variable_from_constraint(variable, constraint,
                                       eps=1e-8, iterlim=1000,
                                       linesearch=True, alpha_min=1e-8):
    """Calculate the variable value given a specified equality constraint

    This function calculates the value of the specified variable
    necessary to make the provided equality constraint feasible
    (assuming any other variables values are fixed).  The method first
    attempts to solve for the variable value assuming it appears
    linearly in the constraint.  If that doesn't converge the constraint
    residual, it falls back on Newton's method using exact (symbolic)
    derivatives.

    Notes
    -----
    This is an unconstrained solver and is NOT guaranteed to respect the
    variable bounds or domain.  The solver may leave the variable value
    in an infeasible state (outside the declared bounds or domain bounds).

    Parameters:
    -----------
    variable: :py:class:`_VarData`
        The variable to solve for
    constraint: :py:class:`_ConstraintData` or relational expression or `tuple`
        The equality constraint to use to solve for the variable value.
        May be a `ConstraintData` object or any valid argument for
        ``Constraint(expr=<>)`` (i.e., a relational expression or 2- or
        3-tuple)
    eps: `float`
        The tolerance to use to determine equality [default=1e-8].
    iterlim: `int`
        The maximum number of iterations if this method has to fall back
        on using Newton's method.  Raises RuntimeError on iteration
        limit [default=1000]
    linesearch: `bool`
        Decides whether or not to use the linesearch (recommended).
        [default=True]
    alpha_min: `float`
        The minimum fractional step to use in the linesearch [default=1e-8].

    Returns:
    --------
    None

    """
    # Leverage all the Constraint logic to process the incoming tuple/expression
    if not isinstance(constraint, _ConstraintData):
        constraint = Constraint(expr=constraint, name=type(constraint).__name__)
        constraint.construct()

    body = constraint.body
    lower = constraint.lb
    upper = constraint.ub

    if lower != upper:
        raise ValueError("Constraint must be an equality constraint")

    if variable.value is None:
        # Note that we use "skip_validation=True" here as well, as the
        # variable domain may not admit the calculated initial guesses,
        # and we want to bypass that check.
        if variable.lb is None:
            if variable.ub is None:
                # no variable values, and no lower or upper bound - set
                # initial value to 0.0
                variable.set_value(0, skip_validation=True)
            else:
                # no variable value or lower bound - set to 0 or upper
                # bound whichever is lower
                variable.set_value(min(0, variable.ub), skip_validation=True)
        elif variable.ub is None:
            # no variable value or upper bound - set to 0 or lower
            # bound, whichever is higher
            variable.set_value(max(0, variable.lb), skip_validation=True)
        else:
            # we have upper and lower bounds
            if variable.lb <= 0 and variable.ub >= 0:
                # set the initial value to 0 if bounds bracket 0
                variable.set_value(0, skip_validation=True)
            else:
                # set the initial value to the midpoint of the bounds
                variable.set_value(
                    (variable.lb+variable.ub)/2.0, skip_validation=True)

    # store the initial value to use later if necessary
    orig_initial_value = variable.value

    # solve the common case where variable is linear with coefficient of 1.0
    x1 = value(variable)
    # Note: both the direct (linear) calculation and Newton's method
    # below rely on a numerically valid initial starting point.
    # While we have strategies for dealing with hitting numerically
    # invalid (e.g., sqrt(-1)) conditions below, if the initial point is
    # not valid, we will allow that exception to propagate up
    try:
        residual_1 = value(body)
    except:
        logger.error(
            "Encountered an error evaluating the expression at the "
            "initial guess.\n\tPlease provide a different initial guess.")
        raise

    variable.set_value(x1 - (residual_1 - upper), skip_validation=True)
    residual_2 = value(body, exception=False)

    # If we encounter an error while evaluating the expression at the
    # linear intercept calculated assuming the derivative was 1.  This
    # is most commonly due to nonlinear expressions (like sqrt())
    # becoming invalid/complex.  We will skip the rest of the
    # "shortcuts" that assume the expression is linear and move directly
    # to using Newton's method.

    if residual_2 is not None and type(residual_2) is not complex:
        # if the variable appears linearly with a coefficient of 1, then we
        # are done
        if abs(residual_2 - upper) < eps:
            # Re-set the variable value to trigger any warnings WRT the
            # final variable state
            variable.set_value(variable.value)
            return

        # Assume the variable appears linearly and calculate the coefficient
        x2 = value(variable)
        slope = float(residual_1 - residual_2) / (x1 - x2)
        intercept = (residual_1 - upper) - slope*x1
        if slope:
            variable.set_value(-intercept/slope, skip_validation=True)
            body_val = value(body, exception=False)
            if body_val is not None and abs(body_val - upper) < eps:
                # Re-set the variable value to trigger any warnings WRT
                # the final variable state
                variable.set_value(variable.value)
                return

    # Variable appears nonlinearly; solve using Newton's method
    #
    # restore initial value
    variable.set_value(orig_initial_value, skip_validation=True)
    expr = body - upper
    expr_deriv = differentiate(expr, wrt=variable,
                               mode=differentiate.Modes.sympy)

    if type(expr_deriv) in native_numeric_types and expr_deriv == 0:
        raise ValueError("Variable derivative == 0, cannot solve for variable")

    if abs(value(expr_deriv)) < 1e-12:
        raise RuntimeError(
            'Initial value for variable results in a derivative value that is '
            'very close to zero.\n\tPlease provide a different initial guess.')

    iter_left = iterlim
    fk = residual_1 - upper
    while abs(fk) > eps and iter_left:
        iter_left -= 1
        if not iter_left:
            raise RuntimeError(
                "Iteration limit (%s) reached; remaining residual = %s"
                % (iterlim, value(expr)) )

        # compute step
        xk = value(variable)
        try:
            fk = value(expr)
            if type(fk) is complex:
                raise ValueError(
                    "Complex numbers are not allowed in Newton's method.")
        except:
            # We hit numerical problems with the last step (possible if
            # the line search is turned off)
            logger.error(
                "Newton's method encountered an error evaluating the "
                "expression.\n\tPlease provide a different initial guess "
                "or enable the linesearch if you have not.")
            raise
        fpk = value(expr_deriv)
        if abs(fpk) < 1e-12:
            raise RuntimeError(
                "Newton's method encountered a derivative that was too "
                "close to zero.\n\tPlease provide a different initial guess "
                "or enable the linesearch if you have not.")
        pk = -fk/fpk
        alpha = 1.0
        xkp1 = xk + alpha * pk
        variable.set_value(xkp1, skip_validation=True)

        # perform line search
        if linesearch:
            c1 = 0.999 # ensure sufficient progress
            while alpha > alpha_min:
                # check if the value at xkp1 has sufficient reduction in
                # the residual
                fkp1 = value(expr, exception=False)
                # HACK for Python3 support, pending resolution of #879
                # Issue #879 also pertains to other checks for "complex"
                # in this method.
                if type(fkp1) is complex:
                    # We cannot perform computations on complex numbers
                    fkp1 = None
                if fkp1 is not None and fkp1**2 < c1*fk**2:
                    # found an alpha value with sufficient reduction
                    # continue to the next step
                    fk = fkp1
                    break
                alpha /= 2.0
                xkp1 = xk + alpha * pk
                variable.set_value(xkp1, skip_validation=True)

            if alpha <= alpha_min:
                residual = value(expr, exception=False)
                if residual is None or type(residual) is complex:
                    residual = "{function evaluation error}"
                raise RuntimeError(
                    "Linesearch iteration limit reached; remaining "
                    "residual = %s." % (residual,))
    #
    # Re-set the variable value to trigger any warnings WRT the final
    # variable state
    variable.set_value(variable.value)
