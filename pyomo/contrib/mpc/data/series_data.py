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

from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interpolation import (
    _get_time_index_vec,
    _get_interp_expr_vec,
)

TimeSeriesTuple = namedtuple("TimeSeriesTuple", ["data", "time"])


class TimeSeriesData(_DynamicDataBase):
    """
    An object to store time series data associated with time-indexed
    variables.

    Parameters
    ----------
    data : dict or ComponentMap
        Maps variables, names, or CUIDs to lists of values

    time : list
        Contains the time points corresponding to variable data points.

    time_set : ContinuousSetData

    context : BlockData
    """

    def __init__(self, data, time, time_set=None, context=None):
        """ """
        _time = list(time)
        if _time != list(sorted(time)):
            raise ValueError("Time points are not sorted in increasing order")
        self._time = _time

        # When looking up a value at a particular time point, we will use
        # this map to try and find the index of the time point. If this lookup
        # fails, we will use binary search-within-tolerance to attempt to find
        # a point that is close enough.
        #
        # WARNING: If the list of time points is updated, e.g. via
        # shift_time_points or concatenate, then this map needs to be
        # updated as well.
        self._time_idx_map = {t: idx for idx, t in enumerate(time)}

        # First make sure provided lists of variable data have the
        # same lengths as the provided time list.
        for key, data_list in data.items():
            if len(data_list) != len(time):
                raise ValueError(
                    "Data lists must have same length as time. "
                    "Length of time is %s while length of data for "
                    "key %s is %s." % (len(time), key, len(data_list))
                )
        super().__init__(data, time_set=time_set, context=context)

    def __eq__(self, other):
        if isinstance(other, TimeSeriesData):
            return self._data == other._data and self._time == other._time
        else:
            # Should this return False or raise TypeError?
            raise TypeError(
                "%s and %s are not comparable" % (self.__class__, other.__class__)
            )

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
        if _is_iterable(indices):
            # Raise error if indices not sorted?
            index_list = list(sorted(indices))
            time_list = [self._time[i] for i in indices]
            data = {
                cuid: [values[idx] for idx in index_list]
                for cuid, values in self._data.items()
            }
            time_set = self._orig_time_set
            return TimeSeriesData(data, time_list, time_set=time_set)
        else:
            # indices is a scalar
            return ScalarData(
                {cuid: values[indices] for cuid, values in self._data.items()}
            )

    def get_data_at_time(self, time=None, tolerance=0.0):
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
            The default is 0.0, meaning time points must be specified exactly.

        Returns
        -------
        TimeSeriesData or ~scalar_data.ScalarData
            TimeSeriesData containing only the specified time points
            or dict mapping CUIDs to values at the specified scalar time
            point.

        """
        if time is None:
            # If time is not specified, assume we want the entire time
            # set. Skip all the overhead, don't create a new object, and
            # return self.
            return self
        is_iterable = _is_iterable(time)
        time_iter = iter(time) if is_iterable else (time,)
        indices = []
        # Allocate indices list dynamically to support a general iterator
        # for time. Not sure if this will ever matter...
        for t in time_iter:
            if t in self._time_idx_map:
                idx = self._time_idx_map[t]
            else:
                idx = find_nearest_index(self._time, t, tolerance=tolerance)
            if idx is None:
                raise RuntimeError(
                    "Time point %s is invalid within tolerance %s" % (t, tolerance)
                )
            indices.append(idx)
        if not is_iterable:
            indices = indices[0]
        return self.get_data_at_time_indices(indices)

    def get_interpolated_data(self, time=None, tolerance=0.0):
        """
        Returns the data associated with the provided time point or points by
        linear interpolation.

        Parameters
        ----------
        time: Float or iterable
            The time point or points corresponding to returned data.
        tolerance: float
            Tolerance used when checking if time points are inside the data
            range.

        Returns
        -------
        TimeSeriesData or ~scalar_data.ScalarData
            TimeSeriesData containing only the specified time points
            or dict mapping CUIDs to values at the specified scalar time
            point.

        """
        if time is None:
            # If time is not specified, assume we want the entire time
            # set. Skip all the overhead, don't create a new object, and
            # return self.
            return self
        is_iterable = _is_iterable(time)
        if not is_iterable:
            time = [time]
        for t in time:
            if t > self._time[-1] + tolerance or t < self._time[0] - tolerance:
                raise RuntimeError("Requesting interpolation outside data range.")
        idxs = _get_time_index_vec(time, self._time)
        data = {}
        for cuid in self._data:
            v = _get_interp_expr_vec(time, self._time, self._data[cuid], idxs)
            data[cuid] = v
        if is_iterable:
            return TimeSeriesData(data, list(time))
        else:
            for cuid in self._data:
                data[cuid] = data[cuid][0]
            return ScalarData(data)

    def to_serializable(self):
        """
        Convert to json-serializable object.

        """
        time = self._time
        data = {
            str(cuid): [pyo_value(val) for val in values]
            for cuid, values in self._data.items()
        }
        return TimeSeriesTuple(data, time)

    def concatenate(self, other, tolerance=0.0):
        """
        Extend time list and variable data lists with the time points
        and variable values in the provided TimeSeriesData.
        The new time points must be strictly greater than the old time
        points.

        """
        other_time = other.get_time_points()
        time = self._time
        if other_time[0] < time[-1] + tolerance:
            raise ValueError(
                "Initial time point of target, %s, is not greater than"
                " final time point of source, %s, within tolerance %s."
                % (other_time[0], time[-1], tolerance)
            )
        self._time.extend(other.get_time_points())

        # Update _time_idx_map as we have altered the list of time points.
        n_time = len(time)
        for i, t in enumerate(other_time):
            self._time_idx_map[t] = n_time + i

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
        self._time_idx_map = {t: idx for idx, t in enumerate(self._time)}

    def extract_variables(self, variables, context=None, copy_values=False):
        """
        Only keep variables specified.

        """
        if copy_values:
            raise NotImplementedError(
                "extract_variables with copy_values=True has not been"
                " implemented by %s" % self.__class__
            )
        data = {}
        for var in variables:
            cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
            data[cuid] = self._data[cuid]
        return TimeSeriesData(data, self._time, time_set=self._orig_time_set)
