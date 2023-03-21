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
from pyomo.core.base.set import Set

from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import interval_to_series, _process_to_dynamic_data


def get_penalty_from_constant_target(
    variables, time, setpoint_data, weight_data=None, variable_set=None
):
    """
    This function returns a tracking cost IndexedExpression for the given
    time-indexed variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set of variable indices for which a cost expression will be
        created
    setpoint_data: ScalarData, dict, or ComponentMap
        Maps variable names to setpoint values
    weight_data: ScalarData, dict, or ComponentMap
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.
    variable_set: Set
        Optional. A set of indices into the provided list of variables
        by which the cost expression will be indexed.

    Returns
    -------
    Set, Expression
        RangeSet that indexes the list of variables provided and an Expression
        indexed by the RangeSet and time containing the cost term for each
        variable at each point in time.

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, ScalarData):
        setpoint_data = ScalarData(setpoint_data)
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))

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
    cuids = [get_indexed_cuid(var) for var in variables]
    setpoint_data = setpoint_data.get_data()
    weight_data = weight_data.get_data()

    def tracking_rule(m, i, t):
        return get_quadratic_penalty_at_time(
            variables[i], t, setpoint_data[cuids[i]], weight=weight_data[cuids[i]]
        )

    tracking_expr = Expression(variable_set, time, rule=tracking_rule)
    return variable_set, tracking_expr


def get_penalty_from_piecewise_constant_target(
    variables,
    time,
    setpoint_data,
    weight_data=None,
    variable_set=None,
    tolerance=0.0,
    prefer_left=True,
):
    """Returns an IndexedExpression penalizing deviation between
    the specified variables and piecewise constant target data.

    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: IntervalData
        Holds the piecewise constant values that will be used as
        setpoints
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.
    tolerance: Float (optional)
        Tolerance used for determining whether a time point
        is within an interval. Default is zero.
    prefer_left: Bool (optional)
        If a time point lies at the boundary of two intervals, whether
        the value on the left will be chosen. Default is True.

    Returns
    -------
    Set, Expression
        Pyomo Expression, indexed by time, for the total weighted
        tracking cost with respect to the provided setpoint.

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))
    if isinstance(setpoint_data, IntervalData):
        setpoint_time_series = interval_to_series(
            setpoint_data,
            time_points=time,
            tolerance=tolerance,
            prefer_left=prefer_left,
        )
    else:
        setpoint_time_series = IntervalData(*setpoint_data)
    var_set, tracking_cost = get_penalty_from_time_varying_target(
        variables,
        time,
        setpoint_time_series,
        weight_data=weight_data,
        variable_set=variable_set,
    )
    return var_set, tracking_cost


def get_quadratic_penalty_at_time(var, t, setpoint, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - setpoint) ** 2


def _get_penalty_expressions_from_time_varying_target(
    variables, time, setpoint_data, weight_data=None
):
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, TimeSeriesData):
        setpoint_data = TimeSeriesData(*setpoint_data)

    # Validate incoming data
    if list(time) != setpoint_data.get_time_points():
        raise RuntimeError(
            "Mismatch in time points between time set and points"
            " in the setpoint data structure"
        )
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError("Setpoint data does not contain a key for variable %s" % var)
        if not weight_data.contains_key(var):
            raise KeyError(
                "Tracking weight does not contain a key for variable %s" % var
            )

    # Get lists of weights and setpoints so we don't have to process
    # the variables (to get CUIDs) and hash the CUIDs for every
    # time index.
    cuids = [get_indexed_cuid(var, sets=(time,)) for var in variables]
    weights = [weight_data.get_data_from_key(var) for var in variables]
    setpoints = [setpoint_data.get_data_from_key(var) for var in variables]
    tracking_costs = [
        {
            t: get_quadratic_penalty_at_time(var, t, setpoints[j][i], weights[j])
            for i, t in enumerate(time)
        }
        for j, var in enumerate(variables)
    ]
    return tracking_costs


def get_penalty_from_time_varying_target(
    variables, time, setpoint_data, weight_data=None, variable_set=None
):
    """Constructs a penalty expression for the specified variables and
    specified time-varying target data.

    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: TimeSeriesData
        Holds the trajectory values that will be used as a setpoint
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.
    variable_set: Set (optional)
        Set indexing the list of provided variables, if one exists already.

    Returns
    -------
    Set, Expression
        Set indexing the list of provided variables and Expression, indexed
        by the variable set and time, for the total weighted penalty with
        respect to the provided setpoint.

    """
    if variable_set is None:
        variable_set = Set(initialize=range(len(variables)))

    # This is a list of dictionaries, one for each variable and each
    # mapping each time point to the quadratic weighted tracking cost term
    # at that time point.
    tracking_costs = _get_penalty_expressions_from_time_varying_target(
        variables, time, setpoint_data, weight_data=weight_data
    )

    def tracking_rule(m, i, t):
        return tracking_costs[i][t]

    tracking_cost = Expression(variable_set, time, rule=tracking_rule)
    return variable_set, tracking_cost


def get_penalty_from_target(
    variables,
    time,
    setpoint_data,
    weight_data=None,
    variable_set=None,
    tolerance=None,
    prefer_left=None,
):
    """A function to get a penalty expression for specified variables from
    a target that is constant, piecewise constant, or time-varying.

    This function accepts ScalarData, IntervalData, or TimeSeriesData objects,
    or compatible mappings/tuples as the target, and builds the appropriate
    penalty expression for each. Mappings are converted to ScalarData, and
    tuples (of data dict, time list) are unpacked and converted to IntervalData
    or TimeSeriesData depending on the contents of the time list.

    Arguments
    ---------
    variables: List
        List of time-indexed variables to be penalized
    time: Set
        Set of time points at which to construct penalty expressions.
        Also indexes the returned Expression.
    setpoint_data: ScalarData, TimeSeriesData, or IntervalData
        Data structure representing the possibly time-varying or piecewise
        constant setpoint
    weight_data: ScalarData (optional)
        Data structure holding the weights to be applied to each variable
    variable_set: Set (optional)
        Set indexing the provided variables, if one already exists. Also
        indexes the returned Expression.
    tolerance: Float (optional)
        Tolerance for checking inclusion within an interval. Only may be
        provided if IntervalData is provided as the setpoint.
    prefer_left: Bool (optional)
        Flag indicating whether left endpoints of intervals should take
        precedence over right endpoints. Default is False. Only may be
        provided if IntervalData is provided as the setpoint.

    Returns
    -------
    Set, Expression
        Set indexing the list of provided variables and an Expression,
        indexed by this set and the provided time set, containing the
        penalties for each variable at each point in time.

    """
    setpoint_data = _process_to_dynamic_data(setpoint_data)
    args = (variables, time, setpoint_data)
    kwds = dict(weight_data=weight_data, variable_set=variable_set)

    def _error_if_used(tolerance, prefer_left, sp_type):
        if tolerance is not None or prefer_left is not None:
            raise RuntimeError(
                "tolerance and prefer_left arguments can only be used if"
                " IntervalData-compatible setpoint is provided. Got"
                " tolerance=%s, prefer_left=%s when using %s as a target."
                % (tolerance, prefer_left, sp_type)
            )

    if isinstance(setpoint_data, ScalarData):
        _error_if_used(tolerance, prefer_left, type(setpoint_data))
        return get_penalty_from_constant_target(*args, **kwds)
    elif isinstance(setpoint_data, TimeSeriesData):
        _error_if_used(tolerance, prefer_left, type(setpoint_data))
        return get_penalty_from_time_varying_target(*args, **kwds)
    elif isinstance(setpoint_data, IntervalData):
        tolerance = 0.0 if tolerance is None else tolerance
        prefer_left = True if prefer_left is None else prefer_left
        kwds.update(prefer_left=prefer_left, tolerance=tolerance)
        return get_penalty_from_piecewise_constant_target(*args, **kwds)
