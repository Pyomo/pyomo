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

from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
    find_nearest_interval_index,
)


def _raise_invalid_cuid(cuid, model):
    raise RuntimeError("Cannot find a component %s on block %s" % (cuid, model))


def load_data_from_scalar(data, model, time):
    """A function to load ScalarData into a model

    Arguments
    ---------
    data: ~scalar_data.ScalarData
    model: BlockData
    time: Iterable

    """
    data = data.get_data()
    t_iter = time if _is_iterable(time) else (time,)
    for cuid, val in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        # TODO: Time points should probably use find_nearest_index
        # This will have to happen in the calling function, as data
        # doesn't have a list of time points to check.
        if var.is_indexed():
            for t in t_iter:
                var[t].set_value(val)
        else:
            var.set_value(val)


def load_data_from_series(data, model, time, tolerance=0.0):
    """A function to load TimeSeriesData into a model

    Arguments
    ---------
    data: TimeSeriesData
    model: BlockData
    time: Iterable

    """
    time_list = list(time)
    time_indices = [
        find_nearest_index(time_list, t, tolerance=tolerance)
        for t in data.get_time_points()
    ]
    for idx, t in zip(time_indices, data.get_time_points()):
        if idx is None:
            raise RuntimeError("Time point %s not found time set" % t)
    if len(time_list) != len(data.get_time_points()):
        raise RuntimeError(
            "TimeSeriesData object and model must have same number"
            " of time points to load data from series"
        )
    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        for idx, val in zip(time_indices, vals):
            t = time_list[idx]
            var[t].set_value(val)


def load_data_from_interval(
    data,
    model,
    time,
    tolerance=0.0,
    prefer_left=True,
    exclude_left_endpoint=True,
    exclude_right_endpoint=False,
):
    """A function to load IntervalData into a model

    Loads values into specified variables at time points that are
    within the intervals specified. If a time point is on the boundary
    of two intervals, the default is to use the interval on the left.
    Often, intervals should be treated as half-open, i.e. one of the
    left or right endpoints should be excluded. This can be enforced
    with the corresponding optional arguments.

    Arguments
    ---------
    data: IntervalData
    model: BlockData
    time: Iterable
    tolerance: Float
    prefer_left: Bool
    exclude_left_endpoint: Bool
    exclude_right_endpoint: Bool

    """
    if prefer_left and exclude_right_endpoint and not exclude_left_endpoint:
        raise RuntimeError(
            "Cannot use prefer_left=True with exclude_left_endpoint=False"
            " and exclude_right_endpoint=True."
        )
    elif not prefer_left and exclude_left_endpoint and not exclude_right_endpoint:
        raise RuntimeError(
            "Cannot use prefer_left=False with exclude_left_endpoint=True"
            " and exclude_right_endpoint=False."
        )
    intervals = data.get_intervals()
    left_endpoints = [t for t, _ in intervals]
    right_endpoints = [t for _, t in intervals]
    # NOTE: O(len(time)*log(len(intervals)))
    idx_list = [
        find_nearest_interval_index(
            intervals, t, tolerance=tolerance, prefer_left=prefer_left
        )
        for t in time
    ]
    left_endpoint_indices = [
        # index of interval which t is the left endpoint of
        find_nearest_index(left_endpoints, t, tolerance=tolerance)
        for t in time
    ]
    right_endpoint_indices = [
        # index of interval which t is the right endpoint of
        find_nearest_index(right_endpoints, t, tolerance=tolerance)
        for t in time
    ]

    # Post-process indices to exclude endpoints
    for i, t in enumerate(time):
        if (
            exclude_left_endpoint
            and left_endpoint_indices[i] is not None
            and right_endpoint_indices[i] is None
        ):
            # If t is a left endpoint but not a right endpoint,
            # do not load a value at t.
            idx_list[i] = None
        elif (
            exclude_right_endpoint
            and right_endpoint_indices[i] is not None
            and left_endpoint_indices[i] is None
        ):
            # If t is a right endpoint but not a left endpoint,
            # do not load a value at t.
            idx_list[i] = None
        elif (
            exclude_left_endpoint
            and exclude_right_endpoint
            and right_endpoint_indices[i] is not None
            and left_endpoint_indices[i] is not None
        ):
            # t is both a left endpoint and a right endpoint
            idx_list[i] = None

    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        if var is None:
            _raise_invalid_cuid(cuid, model)
        for i, t in zip(idx_list, time):
            if i is None:
                # t could not be found in an interval. This is fine.
                # We don't necessarily require that the interval data
                # cover the entire time set.
                continue
            else:
                var[t].set_value(vals[i])
