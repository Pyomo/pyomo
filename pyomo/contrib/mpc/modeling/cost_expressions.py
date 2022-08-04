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

from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression

from pyomo.contrib.mpc.data.series_data import get_time_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.interval_data import (
    time_series_from_interval_data,
)
from pyomo.contrib.mpc.data.convert import (
    interval_to_series,
)


def get_tracking_cost_from_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    This function returns a tracking cost expression for the given
    time-indexed variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set by which to index the tracking expression
    setpoint_data: ScalarData, dict, or ComponentMap
        Maps variable names to setpoint values
    weight_data: ScalarData, dict, or ComponentMap
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by time, containing the sum of weighted
    squared difference between variables and setpoint values.

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, ScalarData):
        setpoint_data = ScalarData(setpoint_data)

    # Make sure data have keys for each var
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError(
                "Setpoint data dictionary does not contain a"
                " key for variable %s" % var.name
            )
        if not weight_data.contains_key(var):
            raise KeyError(
                "Tracking weight dictionary does not contain a"
                " key for variable %s" % var.name
            )

    # Set up data structures so we don't have to re-process keys for each
    # time index in the rule.
    cuids = [get_time_indexed_cuid(var) for var in variables]
    setpoint_data = setpoint_data.get_data()
    weight_data = weight_data.get_data()
    def tracking_rule(m, t):
        return sum(
            get_quadratic_tracking_cost_at_time(
                var, t, setpoint_data[cuid], weight=weight_data[cuid]
            )
            for cuid, var in zip(cuids, variables)
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def get_tracking_cost_from_piecewise_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
    tolerance=0.0,
    prefer_left=True,
):
    # - Setpoint data is in the form of "interval data"
    # - Need to convert to time series data 
    # - get_tracking_cost_from_time_varying_setpoint()
    if isinstance(setpoint_data, IntervalData):
        setpoint_time_series = interval_to_series(
            setpoint_data,
            time_points=time,
            tolerance=tolerance,
            prefer_left=prefer_left,
        )
    else:
        setpoint_time_series = time_series_from_interval_data(
            setpoint_data, time
        )
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
    # NOTE: We may eventually want to support providing more time points
    # in the setpoint than are in the time set used as an indexing set.
    # We would just need the indexing points to exist in the time set.
    if list(time) != setpoint_data.get_time_points():
        raise ValueError(
            "Mismatch in time points between time set and points"
            " in the setpoint data structure"
        )

    for i, cuid in enumerate(cuids):
        if cuid not in setpoint_dict:
            raise KeyError(
                "Setpoint data dictionary does not contain a key for variable"
                " %s with ComponentUID %s" % (variables[i].name, cuid)
            )
        if cuid not in weight_data:
            raise KeyError(
                "Tracking weight dictionary does not contain a key for"
                " variable %s with ComponentUID %s"
                % (variables[i].name, cuid)
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
