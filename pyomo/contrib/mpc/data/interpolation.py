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

from bisect import bisect_right


def _get_time_index_vec(time_set, time_data):
    """Get the position index of time_data above and below the times in
    time_set. This can be used to find positions of points to interpolate
    between.

    Parameters
    ----------
    time_set: iterable
        Time points to locate
    time_data: iterable
        Sorted time points to locate time_set in

    Returns
    -------
    numpy.array
        Position index of the first time in time_data greater than the
        corresponding points time_set. If a time is less than all the times
        in time_data return 1. If a time is greater than all times time_data
        set return the last index of time_data.
    """
    pos = [None] * len(time_set)
    for i, t in enumerate(time_set):
        pos[i] = bisect_right(time_data, t)
        if pos[i] == 0:
            pos[i] = 1
        elif pos[i] == len(time_data):
            pos[i] = len(time_data) - 1
    return pos


def _get_interp_expr_vec(time_set, time_data, data, indexes=None):
    """Return an array of floats interpolated at the time points in time_set
    from data defined at time_data.

    Parameters
    ----------
    time_set: iterable
        Time points to locate
    time_data: iterable
        Sorted time points to locate time_set in
    data: iterable
        Data corresponding to times in time_data, must have the same
        length as time data.
    indexes: numpy.array
        Numpy array of position indexes of the time points to interpolate in the
        time data.  The format is the same as returned by ``_get_time_index_vec()``.
        If this is None, ``_get_time_index_vec()`` is called.  The reason to pass
        this is to avoid multiple position searches when interpolating multiple
        outputs with the same time points.

    Returns
    -------
    list
        If data are Pyomo components, this will return Pyomo expressions.
        If data are floats, this will return floats.
    """
    if indexes is None:
        indexes = _get_time_index_vec(time_set, time_data)
    expr = [None] * len(time_set)
    for i, (h, t) in enumerate(zip(indexes, time_set)):
        l = h - 1
        expr[i] = data[l] + (data[h] - data[l]) / (time_data[h] - time_data[l]) * (
            t - time_data[l]
        )
    return expr
