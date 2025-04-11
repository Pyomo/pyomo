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

from pyomo.common.dependencies import numpy as np


def get_time_index_vec(time_set, time_data):
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
    pos = np.searchsorted(time_data, time_set)
    pos[pos == 0] = 1
    pos[pos == len(time_data)] = len(time_data) - 1
    return pos


def get_interp_expr_vec(time_set, time_data, data, indexes=None):
    """Depending of the data types contained in data, return a list of 
    Pyomo expression or float interpolated data at times in time_set.

    Parameters
    ----------
    time_set: iterable
        Time points to locate
    time_data: iterable
        Sorted time points to locate time_set in
    data: iterable
        Data corresponding to times in time_data, must have the same
        length as time data.

    Returns
    -------
    list
        If data are Pyomo components, this will return Pyomo expressions
        interpolation if data are floats, this will return floats.
    """
    if indexes is None:
        indexes = get_surround_time_vec(time_set, time_data)
    expr = [None]*len(time_set)
    for i, (l, h, t) in enumerate(zip(indexes - 1, indexes, time_set)):
        expr[i] = (
            data[l] 
            + (data[h] - data[l]) 
            / (time_data[h] - time_data[l]) 
            * (t - time_data[l])
        )
    return expr