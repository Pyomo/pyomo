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

from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_interval_index,
)

def load_data_from_scalar(data, model, time):
    """
    Arguments
    ---------
    data: ScalarData
    model: BlockData
    time: Iterable

    """
    data = data.get_data()
    for cuid, val in data.items():
        var = model.find_component(cuid)
        for t in time:
            var[t].set_value(val)


def load_data_from_series(data, model, time):
    """
    Arguments
    ---------
    data: TimeSeriesData
    model: BlockData
    time: Iterable

    """
    if list(time) != data.get_time_points():
        raise RuntimeError(
            "Cannot load time series data when time sets have"
            " different lengths"
        )
    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        for t, val in zip(time, vals):
            var[t].set_value(val)


def load_data_from_interval(
    data, model, time, tolerance=0.0, prefer_left=True
):
    intervals = data.get_intervals()
    # NOTE: O(len(time)*log(len(intervals)))
    idx_list = [
        find_nearest_interval_index(
            intervals, t, tolerance=tolerance, prefer_left=prefer_left
        ) for t in time
    ]
    data = data.get_data()
    for cuid, vals in data.items():
        var = model.find_component(cuid)
        for i, t in zip(idx_list, time):
            if i is None:
                # t could not be found in an interval. This is fine.
                # We don't necessarily require that the interval data
                # cover the entire time set.
                continue
            else:
                var[t].set_value(vals[i])
