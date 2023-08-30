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

from collections.abc import MutableMapping
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_interval_index


def _process_to_dynamic_data(data, time_set=None):
    """Processes a user's data to convert it to the appropriate type
    of dynamic data

    Mappings are converted to ScalarData, and length-two tuples are converted
    to TimeSeriesData or IntervalData, depending on the contents of the
    second item (the list of time points or intervals).

    Arguments
    ---------
    data: Dict, ComponentMap, or Tuple
        Data to convert to either ScalarData, TimeSeriesData, or
        IntervalData, depending on type.

    Returns
    -------
    ScalarData, TimeSeriesData, or IntervalData

    """
    if isinstance(data, _DynamicDataBase):
        return data
    if isinstance(data, MutableMapping):
        return ScalarData(data, time_set=time_set)
    elif isinstance(data, tuple):
        if len(data) != 2:
            raise TypeError(
                "_process_to_dynamic_data only accepts a mapping or a"
                " tuple of length two. Got tuple of length %s" % len(data)
            )
        if not isinstance(data[0], MutableMapping):
            raise TypeError(
                "First entry of data tuple must be instance of MutableMapping,"
                "e.g. dict or ComponentMap. Got %s" % type(data[0])
            )
        elif len(data[1]) == 0:
            raise ValueError(
                "Time sequence provided in data tuple is empty."
                " Cannot infer whether this is a list of points or intervals."
            )
        elif all(not _is_iterable(item) for item in data[1]):
            return TimeSeriesData(*data)
        elif all(_is_iterable(item) and len(item) == 2 for item in data[1]):
            return IntervalData(*data)
        else:
            raise TypeError(
                "Second entry of data tuple must be a non-empty iterable of"
                " scalars (time points) or length-two tuples (intervals)."
                " Got %s" % str(data[1])
            )


def interval_to_series(
    data, time_points=None, tolerance=0.0, use_left_endpoints=False, prefer_left=True
):
    """
    Arguments
    ---------
    data: IntervalData
        Data to convert to a TimeSeriesData object
    time_points: Iterable (optional)
        Points at which time series will be defined. Values are taken
        from the interval in which each point lives. The default is to
        use the right endpoint of each interval.
    tolerance: Float (optional)
        Tolerance within which time points are considered equal.
        Default is zero.
    use_left_endpoints: Bool (optional)
        Whether the left endpoints should be used in the case when
        time_points is not provided. Default is False, meaning that
        the right interval endpoints will be used. Should not be set
        if time points are provided.
    prefer_left: Bool (optional)
        If time_points is provided, and a time point is equal (within
        tolerance) to a boundary between two intervals, this flag
        controls which interval is used.

    Returns
    -------
    TimeSeriesData

    """
    if time_points is None:
        # TODO: Should first or last data points of first or last
        # intervals be included?
        if use_left_endpoints:
            time_points = [t for t, _ in data.get_intervals()]
        else:
            time_points = [t for _, t in data.get_intervals()]
        series_data = data.get_data()
        # TODO: Should TimeSeriesData be constructed with the original time set?
        return TimeSeriesData(series_data, time_points)
    if use_left_endpoints:
        raise RuntimeError("Cannot provide time_points with use_left_endpoints=True")

    intervals = data.get_intervals()
    data_dict = data.get_data()
    # NOTE: This implementation is O(len(time_points)*log(len(intervals))).
    # Could potentially do better with an O(len(time_points) + len(intervals))
    # implementation.
    idx_list = [
        find_nearest_interval_index(
            intervals, t, tolerance=tolerance, prefer_left=prefer_left
        )
        for t in time_points
    ]
    for i, t in enumerate(time_points):
        if idx_list[i] is None:
            raise RuntimeError(
                "Time point %s cannot be found in intervals within"
                " tolerance %s." % (t, tolerance)
            )
    new_data = {key: [vals[i] for i in idx_list] for key, vals in data_dict.items()}
    # TODO: Should TimeSeriesData be constructed with the original time set?
    return TimeSeriesData(new_data, time_points)


def series_to_interval(data, use_left_endpoints=False):
    """
    Arguments
    ---------
    data: TimeSeriesData
        Data that will be converted into an IntervalData object
    use_left_endpoints: Bool (optional)
        Flag indicating whether values on intervals should come
        from the values at the left or right endpoints of the
        intervals

    Returns
    -------
    IntervalData

    """
    time = data.get_time_points()
    data_dict = data.get_data()
    n_t = len(time)
    if n_t == 1:
        t0 = time[0]
        # TODO: Copy data dict?
        # TODO: Should we raise an error if time list has length one?
        return IntervalData(data_dict, [(t0, t0)])
    else:
        # This covers the case of n_t > 1 and n_t == 0
        new_data = {}
        intervals = [(time[i - 1], time[i]) for i in range(1, n_t)]
        for key, values in data_dict.items():
            interval_values = [
                values[i - 1] if use_left_endpoints else values[i]
                for i in range(1, n_t)
            ]
            new_data[key] = interval_values
        return IntervalData(new_data, intervals)
