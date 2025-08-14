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

from pyomo.common.errors import IterationLimitError
from pyomo.common.numeric_types import native_numeric_types, native_complex_types, value
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.base.constraint import Constraint

import logging

logger = logging.getLogger(__name__)

_default_differentiation_mode = differentiate.Modes.sympy
_symbolic_modes = {
    None,
    differentiate.Modes.sympy,
    differentiate.Modes.reverse_symbolic,
}


def calculate_variable_from_constraint(
    variable,
    constraint,
    eps=1e-8,
    iterlim=1000,
    linesearch=True,
    alpha_min=1e-8,
    diff_mode=None,
):
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
    variable: :py:class:`VarData`
        The variable to solve for
    constraint: :py:class:`ConstraintData` or relational expression or `tuple`
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
    diff_mode: :py:enum:`pyomo.core.expr.calculus.derivatives.Modes`
        The mode to use to differentiate the expression.  If
        unspecified, defaults to `Modes.sympy`

    Returns:
    --------
    None

    """
    # Leverage all the Constraint logic to process the incoming tuple/expression
    if not getattr(constraint, 'ctype', None) is Constraint:
        constraint = Constraint(expr=constraint, name=type(constraint).__name__)
        constraint.construct()

    if constraint.is_indexed():
        raise ValueError(
            'calculate_variable_from_constraint(): constraint must be a '
            'scalar constraint or a single ConstraintData.  Received '
            f'{constraint.__class__.__name__} ("{constraint.name}")'
        )

    body = constraint.body
    lower = constraint.lb
    upper = constraint.ub

    if lower != upper:
        raise ValueError(f"Constraint '{constraint}' must be an equality constraint")

    _invalid_types = set(native_complex_types)
    _invalid_types.add(type(None))

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
                    (variable.lb + variable.ub) / 2.0, skip_validation=True
                )

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
            "initial guess.\n\tPlease provide a different initial guess."
        )
        raise

    try:
        variable.set_value(x1 - (residual_1 - upper), skip_validation=True)
        residual_2 = value(body, exception=False)
    except OverflowError:
        # If we encounter an error while evaluating the expression at the
        # linear intercept calculated assuming the derivative was 1.  This
        # is most commonly due to nonlinear expressions (like sqrt())
        # becoming invalid/complex.  We will skip the rest of the
        # "shortcuts" that assume the expression is linear and move directly
        # to using Newton's method.
        residual_2 = None

    if residual_2.__class__ not in _invalid_types:
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
        intercept = (residual_1 - upper) - slope * x1
        if slope:
            variable.set_value(-intercept / slope, skip_validation=True)
            try:
                body_val = value(body, exception=False)
            except OverflowError:
                body_val = None
            if body_val.__class__ not in _invalid_types and abs(body_val - upper) < eps:
                # Re-set the variable value to trigger any warnings WRT
                # the final variable state
                variable.set_value(variable.value)
                return

    # Variable appears nonlinearly; solve using Newton's method
    #
    # restore initial value
    variable.set_value(orig_initial_value, skip_validation=True)
    expr = body - upper

    expr_deriv = None
    if diff_mode in _symbolic_modes:
        try:
            expr_deriv = differentiate(
                expr, wrt=variable, mode=diff_mode or _default_differentiation_mode
            )
        except:
            if diff_mode is None:
                # If the user didn't care how we differentiate, try to
                # (mostly silently) revert to numeric differentiation.
                logger.debug(
                    'Calculating symbolic derivative of expression failed. '
                    'Reverting to numeric differentiation'
                )
                diff_mode = differentiate.Modes.reverse_numeric
            else:
                raise

        if type(expr_deriv) in native_numeric_types and expr_deriv == 0:
            raise ValueError(
                f"Variable '{variable}' derivative == 0 in constraint "
                f"'{constraint}', cannot solve for variable"
            )

    if expr_deriv is None:
        fp0 = differentiate(expr, wrt=variable, mode=diff_mode)
    else:
        fp0 = value(expr_deriv)

    if abs(value(fp0)) < 1e-12:
        raise ValueError(
            f"Initial value for variable '{variable}' results in a derivative "
            f"value for constraint '{constraint}' that is very close to zero.\n"
            "\tPlease provide a different initial guess."
        )

    iter_left = iterlim
    fk = residual_1 - upper
    while abs(fk) > eps and iter_left:
        iter_left -= 1
        if not iter_left:
            raise IterationLimitError(
                f"Iteration limit (%s) reached solving for variable '{variable}' "
                f"using constraint '{constraint}'; remaining residual = %s"
                % (iterlim, value(expr))
            )

        # compute step
        xk = value(variable)
        try:
            fk = value(expr)
            if fk.__class__ in _invalid_types and fk is not None:
                raise ValueError("Complex numbers are not allowed in Newton's method.")
        except:
            # We hit numerical problems with the last step (possible if
            # the line search is turned off)
            logger.error(
                "Newton's method encountered an error evaluating the "
                f"expression for constraint '{constraint}'.\n\tPlease provide a "
                "different initial guess or enable the linesearch if you have not."
            )
            raise

        if expr_deriv is None:
            fpk = differentiate(expr, wrt=variable, mode=diff_mode)
        else:
            fpk = value(expr_deriv)

        if abs(fpk) < 1e-12:
            # TODO: should this raise a ValueError or a new
            # DerivativeError (subclassing ArithmeticError)?
            raise RuntimeError(
                "Newton's method encountered a derivative of constraint "
                f"'{constraint}' with respect to variable '{variable}' that was too "
                "close to zero.\n\tPlease provide a different initial guess "
                "or enable the linesearch if you have not."
            )
        pk = -fk / fpk
        alpha = 1.0
        xkp1 = xk + alpha * pk
        variable.set_value(xkp1, skip_validation=True)

        # perform line search
        if linesearch:
            c1 = 0.999  # ensure sufficient progress
            while alpha > alpha_min:
                # check if the value at xkp1 has sufficient reduction in
                # the residual
                try:
                    fkp1 = value(expr, exception=False)
                    # HACK for Python3 support, pending resolution of #879
                    # Issue #879 also pertains to other checks for "complex"
                    # in this method.
                    if fkp1.__class__ in _invalid_types:
                        # We cannot perform computations on complex numbers
                        fkp1 = None
                    if fkp1 is not None and fkp1**2 < c1 * fk**2:
                        # found an alpha value with sufficient reduction
                        # continue to the next step
                        fk = fkp1
                        break
                except OverflowError:
                    # Encountered an overflow, either from evaluating
                    # this point in the line search (to get fkp1) or
                    # from squaring fkp1.  (The example from #3540
                    # actually triggers both).  Reject this alpha value.
                    pass
                alpha /= 2.0
                xkp1 = xk + alpha * pk
                variable.set_value(xkp1, skip_validation=True)

            if alpha <= alpha_min:
                residual = value(expr, exception=False)
                if residual.__class__ in _invalid_types:
                    residual = "{function evaluation error}"
                raise IterationLimitError(
                    f"Linesearch iteration limit reached solving for "
                    f"variable '{variable}' using constraint '{constraint}'; "
                    f"remaining residual = {residual}."
                )
    #
    # Re-set the variable value to trigger any warnings WRT the final
    # variable state
    variable.set_value(variable.value)
