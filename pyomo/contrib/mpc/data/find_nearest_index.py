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
