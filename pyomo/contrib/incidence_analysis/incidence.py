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
"""Functionality for identifying variables that participate in expressions"""

from contextlib import nullcontext

from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.repn import generate_standard_repn
from pyomo.util.subsystems import TemporarySubsystemManager
from pyomo.contrib.incidence_analysis.config import (
    IncidenceMethod,
    get_config_from_kwds,
)


#
# Handlers for different methods of generating the incidence graph
#
def _get_incident_via_identify_variables(expr, include_fixed):
    # Note that identify_variables will not identify the same variable
    # more than once.
    return list(identify_variables(expr, include_fixed=include_fixed))


def _get_incident_via_standard_repn(
    expr, include_fixed, linear_only, compute_values=False
):
    if include_fixed:
        to_unfix = [
            var for var in identify_variables(expr, include_fixed=True) if var.fixed
        ]
        context = TemporarySubsystemManager(to_unfix=to_unfix)
    else:
        context = nullcontext()

    with context:
        repn = generate_standard_repn(
            expr, compute_values=compute_values, quadratic=False
        )

    linear_vars = []
    # Check coefficients to make sure we don't include linear variables with
    # fixed coefficients of zero.
    # Note that linear variables with constant coefficients of zero are already
    # filtered in generate_standard_repn
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        try:
            value = pyo_value(coef)
        except ValueError as err:
            # Catch error evaluating expression with uninitialized variables
            # TODO: Suppress logged error?
            if "No value for uninitialized VarData" not in str(err):
                raise err
            value = None
        if value != 0:
            linear_vars.append(var)
    if linear_only:
        nl_var_id_set = set(id(var) for var in repn.nonlinear_vars)
        return [var for var in linear_vars if id(var) not in nl_var_id_set]
    else:
        # Combine linear and nonlinear variables and filter out duplicates. Note
        # that quadratic=False, so we don't need to include repn.quadratic_vars.
        variables = linear_vars + list(repn.nonlinear_vars)
        unique_variables = []
        id_set = set()
        for var in variables:
            v_id = id(var)
            if v_id not in id_set:
                id_set.add(v_id)
                unique_variables.append(var)
        return unique_variables


def _get_incident_via_ampl_repn(expr, linear_only, visitor):
    def _nonlinear_var_id_collector(idlist):
        for _id in idlist:
            if _id in visitor.subexpression_cache:
                info = visitor.subexpression_cache[_id][1]
                if info.nonlinear:
                    yield from _nonlinear_var_id_collector(info.nonlinear[1])
                if info.linear:
                    yield from _nonlinear_var_id_collector(info.linear)
            else:
                yield _id

    var_map = visitor.var_map
    repn = visitor.walk_expression((expr, None, 0, 1.0))

    nonlinear_var_id_set = set()
    unique_nonlinear_var_ids = []
    if repn.nonlinear:
        for v_id in _nonlinear_var_id_collector(repn.nonlinear[1]):
            if v_id not in nonlinear_var_id_set:
                nonlinear_var_id_set.add(v_id)
                unique_nonlinear_var_ids.append(v_id)

    nonlinear_vars = [var_map[v_id] for v_id in unique_nonlinear_var_ids]
    linear_only_vars = [
        var_map[v_id]
        for v_id, coef in repn.linear.items()
        if coef != 0.0 and v_id not in nonlinear_var_id_set
    ]
    if linear_only:
        return linear_only_vars
    else:
        variables = linear_only_vars + nonlinear_vars
        return variables


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

    Example
    -------

    .. doctest::

       >>> import pyomo.environ as pyo
       >>> from pyomo.contrib.incidence_analysis import get_incident_variables
       >>> m = pyo.ConcreteModel()
       >>> m.x = pyo.Var([1, 2, 3])
       >>> expr = m.x[1] + 2*m.x[2] + 3*m.x[3]**2
       >>> print([v.name for v in get_incident_variables(expr)])
       ['x[1]', 'x[2]', 'x[3]']
       >>> print([v.name for v in get_incident_variables(expr, linear_only=True)])
       ['x[1]', 'x[2]']

    """
    config = get_config_from_kwds(**kwds)
    method = config.method
    include_fixed = config.include_fixed
    linear_only = config.linear_only
    amplrepnvisitor = config._ampl_repn_visitor

    # Check compatibility of arguments
    if linear_only and method is IncidenceMethod.identify_variables:
        raise RuntimeError(
            "linear_only=True is not supported when using identify_variables"
        )
    if include_fixed and method is IncidenceMethod.ampl_repn:
        raise RuntimeError("include_fixed=True is not supported when using ampl_repn")
    if method is IncidenceMethod.ampl_repn and amplrepnvisitor is None:
        # Developer error, this should never happen!
        raise RuntimeError("_ampl_repn_visitor must be provided when using ampl_repn")

    # Dispatch to correct method
    if method is IncidenceMethod.identify_variables:
        return _get_incident_via_identify_variables(expr, include_fixed)
    elif method is IncidenceMethod.standard_repn:
        return _get_incident_via_standard_repn(
            expr, include_fixed, linear_only, compute_values=False
        )
    elif method is IncidenceMethod.standard_repn_compute_values:
        return _get_incident_via_standard_repn(
            expr, include_fixed, linear_only, compute_values=True
        )
    elif method is IncidenceMethod.ampl_repn:
        return _get_incident_via_ampl_repn(expr, linear_only, amplrepnvisitor)
    else:
        raise ValueError(
            f"Unrecognized value {method} for the method used to identify incident"
            f" variables. See the IncidenceMethod enum for valid methods."
        )
