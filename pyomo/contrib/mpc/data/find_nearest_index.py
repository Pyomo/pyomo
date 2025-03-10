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
        delta, nearest_index = min((abs(target - array[j]), j) for j in [i - 1, i])

    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index


def _distance_from_interval(point, interval, tolerance=None):
    lo, hi = interval
    if tolerance is None:
        tolerance = 0.0
    if point < lo - tolerance:
        return lo - point
    elif lo - tolerance <= point and point <= hi + tolerance:
        return 0.0
    elif point > hi + tolerance:
        return point - hi


def find_nearest_interval_index(
    interval_array, target, tolerance=None, prefer_left=True
):
    # NOTE: This function quickly begins to behave badly if tolerance
    # gets too large, e.g. greater than the width of the smallest
    # interval. For this reason, intervals that represent a single
    # point, e.g. (1.0, 1.0) should not be supported.
    array_lo = 0
    array_hi = len(interval_array)
    target_tuple = (target,)
    i = bisect.bisect_right(interval_array, target_tuple, lo=array_lo, hi=array_hi)
    distance_tol = 0.0 if tolerance is None else tolerance
    if i == array_lo:
        # We are at or to the left of the left endpoint of the
        # first interval.
        nearest_index = i
        delta = _distance_from_interval(
            target, interval_array[i], tolerance=distance_tol
        )
    elif i == array_hi:
        # We are within or to the right of the last interval.
        nearest_index = i - 1
        delta = _distance_from_interval(
            target, interval_array[i - 1], tolerance=distance_tol
        )
    else:
        # Find closest interval
        if prefer_left:
            # In the case of a tie, we return the left interval
            # by default.
            delta, nearest_index = min(
                (
                    _distance_from_interval(
                        target, interval_array[j], tolerance=distance_tol
                    ),
                    j,
                )
                for j in [i - 1, i]
            )
        else:
            # If prefer_left=False, we return the right interval.
            delta, neg_nearest_index = min(
                (
                    _distance_from_interval(
                        target, interval_array[j], tolerance=distance_tol
                    ),
                    -j,
                )
                for j in [i - 1, i]
            )
            nearest_index = -neg_nearest_index

    # If we have two adjacent intervals, e.g. [(0, 1), (1, 2)], and are just
    # to the right of the boundary, we will not check the left interval as
    # bisect places our tuple, e.g. (1.0+1e-10,), to the right of the right
    # interval.
    if prefer_left and nearest_index >= array_lo + 1:
        delta_left = _distance_from_interval(
            target, interval_array[nearest_index - 1], tolerance=distance_tol
        )
        if delta_left <= delta:
            nearest_index = nearest_index - 1
            delta = delta_left
    elif not prefer_left and nearest_index < array_hi - 1:
        delta_right = _distance_from_interval(
            target, interval_array[nearest_index + 1], tolerance=distance_tol
        )
        if delta_right <= delta:
            nearest_index = nearest_index + 1
            delta = delta_right

    if tolerance is not None:
        if delta > tolerance:
            return None
    return nearest_index
