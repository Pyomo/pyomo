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

#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.set import Set

from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData


def _get_quadratic_penalty_at_time(var, t, target, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - target) ** 2


def _get_penalty_expressions_at_time(
    variables, t, target_data, weight_data=None, time_set=None
):
    """A private helper function to process data and construct penalty
    expressions

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        # We pass time_set as an argument in case the user provides a
        # ComponentMap of VarData -> values. In this case knowing the
        # time set is necessary to recover the indexed CUID.
        weight_data = ScalarData(weight_data, time_set=time_set)
    if not isinstance(target_data, ScalarData):
        target_data = ScalarData(target_data, time_set=time_set)

    for var in variables:
        if not target_data.contains_key(var):
            raise KeyError(
                "Target data does not contain a key for variable %s" % var.name
            )
        if not weight_data.contains_key(var):
            raise KeyError(
                "Penalty weight data does not contain a key for variable %s" % var.name
            )

    penalties = [
        _get_quadratic_penalty_at_time(
            var,
            t,
            target_data.get_data_from_key(var),
            weight_data.get_data_from_key(var),
        )
        for var in variables
    ]
    return penalties


def get_penalty_at_time(
    variables, t, target_data, weight_data=None, time_set=None, variable_set=None
):
    """Returns an Expression penalizing the deviation of the specified
    variables at the specified point in time from the specified target

    Arguments
    ---------
    variables: List
        List of time-indexed variables that will be penalized
    t: Float
        Time point at which to apply the penalty
    target_data: ~scalar_data.ScalarData
        ScalarData object containing the target for (at least) the variables
        to be penalized
    weight_data: ~scalar_data.ScalarData (optional)
        ScalarData object containing the penalty weights for (at least) the
        variables to be penalized
    time_set: Set (optional)
        Time set that indexes the provided variables. This is only used if
        target or weight data are provided as a ComponentMap with VarData
        as keys. In this case the Set is necessary to recover the CUIDs
        used internally as keys
    variable_set: Set (optional)
        Set indexing the list of variables provided, if such a set already
        exists

    Returns
    -------
    Set, Expression
        Set indexing the list of variables provided and an Expression,
        indexed by this set, containing the weighted penalty expressions

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    penalty_expressions = _get_penalty_expressions_at_time(
        variables, t, target_data, weight_data=weight_data, time_set=time_set
    )

    def penalty_rule(m, i):
        return penalty_expressions[i]

    penalty = Expression(variable_set, rule=penalty_rule)
    return variable_set, penalty


def get_terminal_penalty(
    variables, time_set, target_data, weight_data=None, variable_set=None
):
    """Returns an Expression penalizing the deviation of the specified
    variables at the final point in time from the specified target

    Arguments
    ---------
    variables: List
        List of time-indexed variables that will be penalized
    time_set: Set
        Time set that indexes the provided variables. Penalties are applied
        at the last point in this set.
    target_data: ~scalar_data.ScalarData
        ScalarData object containing the target for (at least) the variables
        to be penalized
    weight_data: ~scalar_data.ScalarData (optional)
        ScalarData object containing the penalty weights for (at least) the
        variables to be penalized
    variable_set: Set (optional)
        Set indexing the list of variables provided, if such a set already
        exists

    Returns
    -------
    Set, Expression
        Set indexing the list of variables provided and an Expression,
        indexed by this set, containing the weighted penalty expressions

    """
    t = time_set.last()
    return get_penalty_at_time(
        variables,
        t,
        target_data,
        weight_data=weight_data,
        time_set=time_set,
        variable_set=variable_set,
    )
