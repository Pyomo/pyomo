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
