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

from collections import namedtuple

from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
)
from pyomo.contrib.mpc.data.get_cuid import (
    get_time_indexed_cuid,
)
from pyomo.contrib.mpc.data.dynamic_data_base import _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData


TimeSeriesTuple = namedtuple("TimeSeriesTuple", ["data", "time"])


class TimeSeriesData(_DynamicDataBase):
    """
    An object to store time series data associated with time-indexed
    variables.

    """

    def __init__(self, data, time, time_set=None, context=None):
        """
        Arguments:
        ----------
        data: dict or ComponentMap
            Maps variables, names, or CUIDs to lists of values
        time: list
            Contains the time points corresponding to variable data points.

        """
        self._time = list(time)

        # When looking up a value at a particular time point, we will use
        # this map to try and find the index of the time point. If this lookup
        # fails, we will use binary search-within-tolerance to attempt to find
        # a point that is close enough.
        self._time_idx_map = {t: idx for idx, t in enumerate(time)}

        # First make sure provided lists of variable data have the
        # same lengths as the provided time list.
        for key, data_list in data.items():
            if len(data_list) != len(time):
                raise ValueError(
                    "Data lists must have same length as time. "
                    "Length of time is %s while length of data for "
                    "key %s is %s."
                    % (len(time), key, len(data_list))
                )
        super().__init__(data, time_set=time_set, context=context)

    def get_time_points(self):
        """
        Get time points of the time series data
        """
        return self._time

    def get_data_at_time_indices(self, indices):
        """
        Returns data at the specified index or indices of this object's list
        of time points.
        """
        try:
            # Probably should raise an error if indices are not already sorted.
            index_list = list(sorted(indices))
            time_list = [self._time[i] for i in indices]
            data = {
                cuid: [values[idx] for idx in index_list]
                for cuid, values in self._data.items()
            }
            time_set = self._orig_time_set
            return TimeSeriesData(data, time_list, time_set=time_set)
        except TypeError:
            # indices is a scalar
            return ScalarData({
                cuid: values[indices] for cuid, values in self._data.items()
            })

    def get_data_at_time(self, time=None, tolerance=None):
        """
        Returns the data associated with the provided time point or points.
        This function attempts to map time points to indices, then uses
        get_data_at_time_indices to actually extract the data. If a provided
        time point does not exist in the time-index map, binary search is
        used to find the closest value within a tolerance.

        Parameters
        ----------
        time: Float or iterable
            The time point or points corresponding to returned data.
        tolerance: Float
            Tolerance within which we will search for a matching time point.
            The default is None, which corresponds to an infinite tolerance.

        Returns
        -------
        TimeSeriesData or dict
            TimeSeriesData containing only the specified time points
            or dict mapping CUIDs to values at the specified scalar time
            point.

        """
        if time is None:
            # If time is not specified, assume we want the entire time
            # set. Skip all the overhead, don't create a new object, and
            # return self.
            return self
        try:
            indices = [
                self._time_idx_map[t] if t in self._time_idx_map else
                find_nearest_index(self._time, t, tolerance=tolerance)
                for t in time
            ]
        except TypeError:
            # TODO: Probably shouldn't rely on TypeError here.
            # time is a scalar
            indices = (
                self._time_idx_map[time]
                if time in self._time_idx_map
                else find_nearest_index(self._time, time, tolerance=tolerance)
            )
        return self.get_data_at_time_indices(indices)

    def to_serializable(self):
        """
        Convert to json-serializable object.
        """
        time = self._time
        data = {str(cuid): values for cuid, values in self._data.items()}
        return TimeSeriesTuple(data, time)

    def concatenate(self, other):
        """
        Extend time list and variable data lists with the time points
        and variable values in the provided TimeSeriesData
        """
        # TODO: Potentially check here for "incompatible" time points,
        # i.e. violating sorted order. We don't assume that anywhere yet,
        # but it may be convenient to eventually.
        time = self._time.extend(other.get_time_points())

        data = self._data
        other_data = other.get_data()
        for cuid, values in data.items():
            # We assume that other contains all the cuids in self.
            # We make no assumption the other way around.
            values.extend(other_data[cuid])

    def shift_time_points(self, offset):
        """
        Apply an offset to stored time points.
        """
        # Note that this is different from what we are doing in
        # shift_values_by_time in the helper class.
        self._time = [t + offset for t in self._time]

    #def extract_variables(self, variables, context=None):
    #    """
    #    Only keep variables specified by the user.
    #    """
    #    data = {}
    #    for var in variables:        
    #        cuid = get_time_indexed_cuid(
    #            var, (self._orig_time_set,), context=context
    #        )
    #        data[cuid] = self._data[cuid]
    #    return TimeSeriesData(
    #        data,
    #        self._time,
    #        time_set=self._orig_time_set,
    #    )
