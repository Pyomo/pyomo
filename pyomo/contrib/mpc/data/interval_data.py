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

from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
    find_nearest_index,
    find_nearest_interval_index,
)


IntervalDataTuple = namedtuple("IntervalTuple", ["data", "intervals"])


def assert_disjoint_intervals(intervals):
    """
    This function takes intervals in the form of tuples and makes sure
    that they are disjoint.

    Arguments
    ---------
    intervals: iterable
        Iterable of tuples, each containing the low and high values of an
        interval.

    """
    intervals = list(sorted(intervals))
    for i, (lo, hi) in enumerate(intervals):
        if not lo <= hi:
            raise RuntimeError(
                "Lower endpoint of interval is higher than upper endpoint"
            )
        if i != 0:
            prev_lo, prev_hi = intervals[i - 1]
            if not prev_hi <= lo:
                raise RuntimeError(
                    "Intervals %s and %s are not disjoint"
                    % ((prev_lo, prev_hi), (lo, hi))
                )


class IntervalData(_DynamicDataBase):
    def __init__(self, data, intervals, time_set=None, context=None):
        intervals = list(intervals)
        if not intervals == list(sorted(intervals)):
            raise RuntimeError("Intervals are not sorted in increasing order.")
        assert_disjoint_intervals(intervals)
        self._intervals = intervals

        # First make sure provided lists of variable data have the
        # same lengths as the provided time list.
        for key, data_list in data.items():
            if len(data_list) != len(intervals):
                raise ValueError(
                    "Data lists must have same length as time. "
                    "Length of time is %s while length of data for "
                    "key %s is %s." % (len(intervals), key, len(data_list))
                )
        super().__init__(data, time_set=time_set, context=context)

    def __eq__(self, other):
        if isinstance(other, IntervalData):
            return (
                self._data == other.get_data()
                and self._intervals == other.get_intervals()
            )
        else:
            raise TypeError(
                "%s and %s are not comparable" % (self.__class__, other.__class__)
            )

    def get_intervals(self):
        return self._intervals

    def get_data_at_interval_indices(self, indices):
        # NOTE: Much of this code is repeated from TimeSeriesData.
        # TODO: Find some way to consolidate.
        if _is_iterable(indices):
            index_list = list(sorted(indices))
            interval_list = [self._intervals[i] for i in indices]
            data = {
                cuid: [values[idx] for idx in index_list]
                for cuid, values in self._data.items()
            }
            time_set = self._orig_time_set
            return IntervalData(data, interval_list, time_set=time_set)
        else:
            return ScalarData(
                {cuid: values[indices] for cuid, values in self._data.items()}
            )

    # TODO: get_data_at_interval, get_data_at_time
    def get_data_at_time(self, time, tolerance=None, prefer_left=True):
        if not _is_iterable(time):
            index = find_nearest_interval_index(
                self._intervals, time, tolerance=tolerance, prefer_left=prefer_left
            )
            if index is None:
                raise RuntimeError(
                    "Time point %s not found in an interval within"
                    " tolerance %s" % (time, tolerance)
                )
        else:
            raise RuntimeError(
                "get_data_at_time is not supported with multiple time points"
                " for IntervalData. To sample the piecewise-constant data at"
                " particular time points, please use interval_to_series from"
                " pyomo.contrib.mpc.data.convert"
            )
        return self.get_data_at_interval_indices(index)

    def to_serializable(self):
        """
        Convert to json-serializable object.

        """
        intervals = self._intervals
        data = {
            str(cuid): [pyo_value(val) for val in values]
            for cuid, values in self._data.items()
        }
        return IntervalDataTuple(data, intervals)

    def concatenate(self, other, tolerance=0.0):
        """
        Extend interval list and variable data lists with the intervals
        and variable values in the provided IntervalData

        """
        other_intervals = other.get_intervals()
        intervals = self._intervals
        if len(other_intervals) == 0:
            return
        if other_intervals[0][0] < intervals[-1][1] + tolerance:
            # First point of new intervals is less than (within
            # tolerance)
            raise ValueError(
                "Initial time point of target, %s, is not greater than"
                " final time point of source, %s, within tolerance %s."
                % (other_time[0][0], time[-1][1], tolerance)
            )
        self._intervals.extend(other_intervals)

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
        self._intervals = [(lo + offset, hi + offset) for lo, hi in self._intervals]

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
        if not isinstance(variables, (list, tuple)):
            # If variables is not a sequence and is instead a slice (or
            # indexed variable), we get either a confusing error message
            # or a lot of repeated work.
            raise TypeError("extract_values only accepts a list or tuple of variables")
        for var in variables:
            cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
            data[cuid] = self._data[cuid]
        return IntervalData(data, self._intervals, time_set=self._orig_time_set)
