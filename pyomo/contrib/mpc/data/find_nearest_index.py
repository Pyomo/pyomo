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

import bisect

def find_nearest_index(array, target, tolerance=None):
    # array needs to be sorted and we assume it is zero-indexed
    lo = 0
    hi = len(array)
    i = bisect.bisect_right(array, target, lo=lo, hi=hi)
    # i is the index at which target should be inserted if it is to be
    # right of any equal components. 

    if i == lo:
        # target is less than every entry of the set
        nearest_index = i
        delta = array[nearest_index] - target
    elif i == hi:
        # target is greater than or equal to every entry of the set
        nearest_index = i - 1
        delta = target - array[nearest_index]
    else:
        # p_le <= target < p_g
        # delta_left = target - p_le
        # delta_right = p_g - target
        # delta = min(delta_left, delta_right)
        # Tie goes to the index on the left.
        delta, nearest_index = min(
            (abs(target - array[j]), j) for j in [i-1, i]
        )

    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index


def _distance_from_interval(point, interval):
    lo, hi = interval
    if point < lo:
        return lo - point
    elif lo <= point and point <= hi:
        return 0.0
    elif point > hi:
        return point - hi


def find_nearest_interval_index(interval_array, target, tolerance=None):
    array_lo = 0
    array_hi = len(interval_array)
    target_tuple = (target,)
    i = bisect.bisect_right(
        interval_array, target_tuple, lo=array_lo, hi=array_hi
    )
    # Possible cases:
    # - target is less than everything
    # - target is within some interval (will appear to left of that interval)
    # - target is between two intervals
    # - target is greater than everything
    if i == array_lo:
        nearest_index = i
        delta = _distance_from_interval(target, interval_array[i])
    elif i == array_hi:
        nearest_index = i
        delta = _distance_from_interval(target, interval_array[i-1])
    else:
        delta, nearest_index = min(
            (_distance_from_interval(target, interval_array[j]), j)
            for j in [i-1, i]
            #(abs(target - array[j]), j) for j in [i-1, i]
        )

    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index
