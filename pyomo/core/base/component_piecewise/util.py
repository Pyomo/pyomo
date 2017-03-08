#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import operator
import itertools

import six
from six.moves import xrange
from six import advance_iterator

numpy_available = False
try:
    import numpy
    numpy_available = True
except:             #pragma:nocover
    pass

scipy_available = False
try:
    import scipy
    import scipy.spatial
    scipy_available = True
except:             #pragma:nocover
    pass

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
    """
    Returns the exact value of floor(log2(n)).
    No floating point calculations are used.
    Requires positive integer type.
    """
    assert n > 0
    return n.bit_length() - 1

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
    Characterizes a piecewise linear function as affine (1),
    convex (2), concave (3), step (4), or None (5). Assumes
    breakpoints are in nondecreasing order. Returns an
    integer that signifies the function characterization and
    the slopes. If the function has step points, some of the
    slopes may be done.
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
        return 4, slopes # step
    elif is_constant(slopes):
        return 1, slopes # affine
    elif is_nondecreasing(slopes):
        return 2, slopes # convex
    elif is_nonincreasing(slopes):
        return 3, slopes # concave
    else:
        return 5, slopes # none of the above

def generate_delaunay(variables, num=10, **kwds):
    """
    Generate a Delaunay triangulation of the D-dimensional
    bounded variable domain given a list of D variables.

    Requires the numpy and scipy.spatial packages.
    Args:
        variables: A list of variables, each having a finite
            upper and lower bound.
        num (int): The number of grid points to generate for
            each variable (default=10).
        **kwds: All additional keywords are passed to the
          scipy.spatial.Delaunay constructor.

    Returns: A scipy.spatial.Delaunay object.
    """
    if not (numpy_available and scipy_available):             #pragma:nocover
        raise ImportError(
            "numpy and scipy are required")
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
