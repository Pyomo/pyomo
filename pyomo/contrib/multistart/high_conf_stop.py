"""Utility functions for the high confidence stopping rule.

This stopping criterion operates by estimating the amount of missing optima,
and stops once the estimated mass of missing optima is within an acceptable
range, given some confidence.

"""
from __future__ import division

from collections import Counter
from math import log, sqrt


def num_one_occurrences(observed_obj_vals, tolerance):
    """
    Determines the number of optima that have only been observed once.
    Needed to estimate missing mass of optima.
    """
    obj_value_distribution = Counter(observed_obj_vals)
    sorted_histogram = list(sorted(obj_value_distribution.items()))
    if tolerance == 0:
        return sum(1 for _, count in sorted_histogram if count == 1)
    else:
        # Need to apply a tolerance to each value
        num_obj_vals_only_observed_once = 0
        for i, tup in enumerate(sorted_histogram):
            obj_val, count = tup
            if count == 1:
                # look at previous and next elements to make sure that they are
                # not within the tolerance
                if (i > 0 and
                        obj_val - sorted_histogram[i - 1][0] <= tolerance):
                    continue
                if (i < len(sorted_histogram) - 1 and
                        sorted_histogram[i + 1][0] - obj_val <= tolerance):
                    continue
                num_obj_vals_only_observed_once += 1
        return num_obj_vals_only_observed_once


def should_stop(solutions, stopping_mass, stopping_delta, tolerance):
    """
    Determines if the missing mass of unseen local optima is acceptable
    based on the High Confidence stopping rule.
    """
    f = num_one_occurrences(solutions, tolerance)
    n = len(solutions)
    if n == 0:
        return False  # Do not stop if no solutions have been found.
    d = stopping_delta
    c = stopping_mass
    confidence = f / n + (2 * sqrt(2) + sqrt(3)
                          ) * sqrt(log(3 / d) / n)
    return confidence < c
