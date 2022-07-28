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

from pyomo.contrib.mpc.dynamic_data.series_data import get_time_indexed_cuid
from pyomo.contrib.mpc.dynamic_data.scalar_data import ScalarData
from pyomo.contrib.mpc.dynamic_data.interval_data import (
    time_series_from_interval_data,
)


def get_tracking_cost_from_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    This function returns a tracking cost expression for the given time-indexed
    variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set by which to index the tracking expression
    setpoint_data: dict
        Maps variable names to setpoint values
    weight_data: dict
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by time, containing the sum of weighted
    squared difference between variables and setpoint values.

    """
    cuids = [
        get_time_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in cuids}
    elif isinstance(weight_data, ScalarData):
        weight_data = weight_data.get_data()
    else:
        weight_data = ScalarData(weight_data, time_set=time).get_data()
    # Note that if we used ScalarData everywhere, we wouldn't have to
    # process the incoming variables with get_time_indexed_cuid.
    # We would process them on lookup, which would be slightly more work,
    # but this function would be nicer. ScalarData would have to implement
    # __contains__ in that case.

    if isinstance(setpoint_data, ScalarData):
        setpoint_data = setpoint_data.get_data()
    else:
        setpoint_data = ScalarData(setpoint_data, time_set=time).get_data()

    # Note that at this point both weight_data and setpoint_data are dicts
    # mapping CUIDs to values.

    for i, cuid in enumerate(cuids):
        if cuid not in setpoint_data:
            raise KeyError(
                "Setpoint data dictionary does not contain a key for variable\n"
                "%s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Tracking weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (variables[i].name, cuid)
            )

    def tracking_rule(m, t):
        return sum(
            weight_data[cuid] * (var[t] - setpoint_data[cuid])**2
            for cuid, var in zip(cuids, variables)
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def _get_tracking_cost_from_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    Re-implementation of the above, trying to be simpler by leveraging
    ScalarData
    """
    # If provided, weight and setpoint data should be ScalarData.
    # ... it wouldn't be too hard to convert them if they are dicts/maps...
    if weight_data is None:
        weight_data = ScalarData(
            # var must be a variable because it needs to participate in an
            # expression below. (Reference is fine. We will immediately extract
            # the underlying slice.)
            ComponentMap((var, 1.0) for var in variables),
            # If time is not a set, it just won't get matched anywhere when
            # trying to replace indices with slices in a VarData.
            #time_set=time,
            # Should never be used; VarData should not be supported here.
        )
    # What if these were ScalarData at this point?
    # We could process variables at lookup time. This does extra processing
    # but makes this function simpler.
    # This defers the expense of generating cuids (from slices) to the calling
    # of this rule. Will this be too expensive?
    # The rule is called for every time point. This may get too expensive.
    def tracking_rule(m, t):
        return sum(
            (
                weight_data.get_data_from_key(var)
                * (var[t] - setpoint_data.get_data_from_key(var))**2
            ) for var in variables
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def _get_tracking_cost_from_constant_setpoint_2(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    Another re-implementation. I think I like this one.
    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, ScalarData):
        setpoint_data = ScalarData(setpoint_data)

    # TODO: Make sure data have keys for each var

    # Set up data structures so we don't have to re-process keys for each
    # time index in the rule.
    cuids = [get_time_indexed_cuid(var) for var in variables]
    setpoint_data = setpoint_data.get_data()
    weight_data = weight_data.get_data()
    def tracking_rule(m, t):
        return sum(
            weight_data[cuid] * (var[t] - setpoint_data[cuid])**2
            for cuid, var in zip(cuids, variables)
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def get_tracking_cost_from_piecewise_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    # - Setpoint data is in the form of "interval data"
    # - Need to convert to time series data 
    # - get_tracking_cost_from_time_varying_setpoint()
    setpoint_time_series = time_series_from_interval_data(setpoint_data, time)
    tracking_cost = get_tracking_cost_from_time_varying_setpoint(
        variables, time, setpoint_time_series, weight_data=weight_data
    )
    return tracking_cost


def get_quadratic_tracking_cost_at_time(var, t, setpoint, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - setpoint)**2


def get_tracking_cost_expressions_from_time_varying_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    cuids = [
        get_time_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    # TODO: Weight data (and setpoint data) are user-provided and don't
    # necessarily have CUIDs as keys. Should I processes the keys here
    # with get_time_indexed_cuid?
    if weight_data is None:
        #weight_data = {name: 1.0 for name in variable_names}
        weight_data = {cuid: 1.0 for cuid in cuids}

    # Here, setpoint_data is a TimeSeriesData object. Need to get
    # the actual dictionary that we can use for lookup.
    setpoint_dict = setpoint_data.get_data()

    for i, cuid in enumerate(cuids):
        if cuid not in setpoint_dict:
            raise KeyError(
                "Setpoint data dictionary does not contain a key for variable\n"
                "%s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Tracking weight dictionary does not contain a key for "
                "variable\n%s with ComponentUID %s" % (variables[i].name, cuid)
            )
    tracking_costs = [
        {
            t: get_quadratic_tracking_cost_at_time(
                # NOTE: Here I am assuming that the first n_t points in the
                # setpoint dict should be used...
                # What is the alternative?
                var, t, setpoint_dict[cuid][i], weight_data[cuid]
            ) for i, t in enumerate(time)
        } for var, cuid in zip(variables, cuids)
    ]
    return tracking_costs


def get_tracking_cost_from_time_varying_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    """
    # This is a list of dictionaries, one for each variable and each
    # mapping each time point to the quadratic weighted tracking cost term
    # at that time point.
    tracking_costs = get_tracking_cost_expressions_from_time_varying_setpoint(
        variables, time, setpoint_data, weight_data=weight_data
    )

    def tracking_rule(m, t):
        return sum(cost[t] for cost in tracking_costs)
    tracking_cost = Expression(time, rule=tracking_rule)
    return tracking_cost
