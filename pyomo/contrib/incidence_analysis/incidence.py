#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""Functionality for identifying variables that participate in expressions
"""
import enum
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn import generate_standard_repn
from pyomo.common.backports import nullcontext
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.contrib.incidence_analysis.config import IncidenceMethod, IncidenceConfig


#
# Handlers for different methods of generating the incidence graph
#
def _get_incident_via_identify_variables(expr, include_fixed):
    # Note that identify_variables will not identify the same variable
    # more than once.
    return list(identify_variables(expr, include_fixed=include_fixed))


def _get_incident_via_standard_repn(expr, include_fixed, linear_only):
    if include_fixed:
        to_unfix = [
            var for var in identify_variables(expr, include_fixed=True) if var.fixed
        ]
        context = TemporarySubsystemManager(to_unfix=to_unfix)
    else:
        context = nullcontext()

    with context:
        repn = generate_standard_repn(expr, compute_values=False)

    linear_vars = list(repn.linear_vars)
    # TODO: Check coefficients
    # This is only necessary when handling mutable parameters and fixed
    # variables as coefficients, which we're punting on for now. Variables
    # with constant coefficients of zero don't appear here.
    if linear_only:
        return linear_vars
    else:
        # Combine linear, quadratic, and nonlinear variables and filter out
        # duplicates
        variables = linear_vars
        for var1, var2 in repn.quadratic_vars:
            variables.extend((var1, var2))
        variables.extend(repn.nonlinear_vars)
        unique_variables = []
        id_set = set()
        for var in variables:
            v_id = id(var)
            if v_id not in id_set:
                id_set.add(v_id)
                unique_variables.append(var)
        return unique_variables


def get_incident_variables(expr, **kwds):
    """Get variables that participate in an expression

    The exact variables returned depends on the method used to determine incidence.
    For example, ``method=IncidenceMethod.identify_variables`` will return all
    variables participating in the expression, while
    ``method=IncidenceMethod.standard_repn`` will return only the variables
    identified by ``generate_standard_repn`` which ignores variables that only
    appear multiplied by a constant factor of zero.

    Keyword arguments must be valid options for ``IncidenceConfig``.

    Parameters
    ----------
    expr: ``NumericExpression``
        Expression to search for variables

    Returns
    -------
    list of VarData
        List containing the variables that participate in the expression

    """
    config = IncidenceConfig(kwds)
    method = config.method
    include_fixed = config.include_fixed
    linear_only = config.linear_only
    if linear_only and method is IncidenceMethod.identify_variables:
        raise RuntimeError(
            "linear_only=True is not supported when using identify_variables"
        )
    if method is IncidenceMethod.identify_variables:
        return _get_incident_via_identify_variables(expr, include_fixed)
    elif method is IncidenceMethod.standard_repn:
        return _get_incident_via_standard_repn(expr, include_fixed, linear_only)
    else:
        raise ValueError(
            f"Unrecognized value {method} for the method used to identify incident"
            f" variables. Valid options are {IncidenceMethod.identify_variables}"
            f" and {IncidenceMethod.standard_repn}."
        )
