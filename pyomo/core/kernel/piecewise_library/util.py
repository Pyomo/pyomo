#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import operator
import itertools

from six.moves import xrange
from six import advance_iterator

from pyomo.common.dependencies import (
    numpy, numpy_available, scipy, scipy_available
)

class PiecewiseValidationError(Exception):
    """An exception raised when validation of piecewise
    linear functions fail."""

def is_constant(vals):
    """Checks if a list of points is constant"""
    if len(vals) <= 1:
        return True
    it = iter(vals)
    advance_iterator(it)
    op = operator.eq
    return all(itertools.starmap(op, zip(it,vals)))

def is_nondecreasing(vals):
    """Checks if a list of points is nondecreasing"""
    if len(vals) <= 1:
        return True
    it = iter(vals)
    advance_iterator(it)
    op = operator.ge
    return all(itertools.starmap(op, zip(it,vals)))

def is_nonincreasing(vals):
    """Checks if a list of points is nonincreasing"""
    if len(vals) <= 1:
        return True
    it = iter(vals)
    advance_iterator(it)
    op = operator.le
    return all(itertools.starmap(op, zip(it,vals)))

def is_positive_power_of_two(x):
    """Checks if a number is a nonzero and positive power of 2"""
    if (x <= 0):
        return False
    else:
        return ( (x & (x - 1)) == 0 )

def log2floor(n):
    """Computes the exact value of floor(log2(n)) without
    using floating point calculations. Input argument must
    be a positive integer."""
    assert n > 0
    try:
        return n.bit_length() - 1
    except AttributeError:                        #pragma:nocover
        # int.bit_length() was introduced in Python 2.7.  Fallback to a
        # brute-force calculation if bit_length is not available.
        s = bin(n)         # binary representation:  bin(37) --> '0b100101'
        s = s.lstrip('0b') # remove leading zeros and 'b'
        return len(s) - 1

def generate_gray_code(nbits):
    """Generates a Gray code of nbits as list of lists"""
    bitset = [0 for i in xrange(nbits)]
    # important that we copy bitset each time
    graycode = [list(bitset)]

    for i in xrange(2,(1<<nbits)+1):
        if i%2:
            for j in xrange(-1,-nbits,-1):
                if bitset[j]:
                    bitset[j-1]=bitset[j-1]^1
                    break
        else:
            bitset[-1]=bitset[-1]^1
        # important that we copy bitset each time
        graycode.append(list(bitset))

    return graycode

def characterize_function(breakpoints, values):
    """
    Characterizes a piecewise linear function described by a
    list of breakpoints and function values.

    Args:
        breakpoints (list): The list of breakpoints of the
            piecewise linear function. It is assumed that
            the list of breakpoints is in non-decreasing
            order.
        values (list): The values of the piecewise linear
            function corresponding to the breakpoints.

    Returns:
        (int, list): a function characterization code and \
            the list of slopes.

    .. note::
        The function characterization codes are

          * 1: affine
          * 2: convex
          * 3: concave
          * 4: step
          * 5: other

        If the function has step points, some of the slopes
        may be :const:`None`.
    """
    if not is_nondecreasing(breakpoints):
        raise ValueError(
            "The list of breakpoints must be nondecreasing")

    step = False
    slopes = []
    for i in xrange(1, len(breakpoints)):
        if breakpoints[i] != breakpoints[i-1]:
            slope = float(values[i] - values[i-1]) / \
                    (breakpoints[i] - breakpoints[i-1])
        else:
            slope = None
            step = True
        slopes.append(slope)

    if step:
        return characterize_function.step, slopes
    elif is_constant(slopes):
        return characterize_function.affine, slopes
    elif is_nondecreasing(slopes):
        return characterize_function.convex, slopes
    elif is_nonincreasing(slopes):
        return characterize_function.concave, slopes
    else:
        return characterize_function.other, slopes
characterize_function.affine  = 1
characterize_function.convex  = 2
characterize_function.concave = 3
characterize_function.step    = 4
characterize_function.other   = 5

def generate_delaunay(variables, num=10, **kwds):
    """
    Generate a Delaunay triangulation of the D-dimensional
    bounded variable domain given a list of D variables.

    Requires numpy and scipy.

    Args:
        variables: A list of variables, each having a finite
            upper and lower bound.
        num (int): The number of grid points to generate for
            each variable (default=10).
        **kwds: All additional keywords are passed to the
          scipy.spatial.Delaunay constructor.

    Returns:
        A scipy.spatial.Delaunay object.
    """
    linegrids = []
    for v in variables:
        if v.has_lb() and v.has_ub():
            linegrids.append(numpy.linspace(v.lb, v.ub, num))
        else:
            raise ValueError(
                "Variable %s does not have a "
                "finite lower and upper bound.")
    # generates a meshgrid and then flattens and transposes
    # the meshgrid into an (npoints, D) shaped array of
    # coordinates
    points = numpy.vstack(numpy.meshgrid(*linegrids)).\
             reshape(len(variables),-1).T
    return scipy.spatial.Delaunay(points, **kwds)
