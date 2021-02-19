#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import math
import logging
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException, PyomoException

logger = logging.getLogger(__name__)
inf = float('inf')


class IntervalException(PyomoException):
    pass


def add(xl, xu, yl, yu):
    return xl + yl, xu + yu


def sub(xl, xu, yl, yu):
    return xl - yu, xu - yl


def mul(xl, xu, yl, yu):
    lb = min(xl*yl, xl*yu, xu*yl, xu*yu)
    ub = max(xl*yl, xl*yu, xu*yl, xu*yu)
    if math.isnan(lb):
        lb = -inf
    if math.isnan(ub):
        ub = inf
    return lb, ub


def inv(xl, xu, feasibility_tol):
    if xl <= feasibility_tol and xu >= -feasibility_tol:
        # if the denominator (x) can include 0, then 1/x is unbounded.
        lb = -inf
        ub = inf
    else:
        ub = 1.0 / xl
        lb = 1.0 / xu
    return lb, ub


def div(xl, xu, yl, yu, feasibility_tol):
    if yl <= feasibility_tol and yu >= -feasibility_tol:
        # if the denominator (y) can include 0, then x/y is unbounded.
        lb = -inf
        ub = inf
    else:
        lb, ub = mul(xl, xu, *inv(yl, yu, feasibility_tol))
    return lb, ub


def power(xl, xu, yl, yu):
    """
    Compute bounds on x**y.
    """
    if xl > 0:
        """
        If x is always positive, things are simple. We only need to worry about the sign of y.
        """
        if yl < 0 and yu > 0:
            lb = min(xu ** yl, xl ** yu)
            ub = max(xl ** yl, xu ** yu)
        elif yl >= 0:
            lb = min(xl**yl, xl**yu)
            ub = max(xu**yl, xu**yu)
        elif yu <= 0:
            lb = min(xu**yl, xu**yu)
            ub = max(xl**yl, xl**yu)
        else:
            raise DeveloperError()
    elif xl == 0:
        if yl >= 0:
            lb = min(xl ** yl, xl ** yu)
            ub = max(xu ** yl, xu ** yu)
        else:
            lb = -inf
            ub = inf
    elif yl == yu and yl == round(yl):
        # the exponent is an integer, so x can be negative
        """
        The logic here depends on several things:
        1) The sign of x
        2) The sign of y
        3) Whether y is even or odd.
        
        There are also special cases to avoid math domain errors.
        """
        y = yl
        if xu <= 0:
            if y < 0:
                if y % 2 == 0:
                    lb = xl ** y
                    if xu == 0:
                        ub = inf
                    else:
                        ub = xu ** y
                else:
                    if xu == 0:
                        lb = -inf
                        ub = inf
                    else:
                        lb = xu ** y
                        ub = xl ** y
            else:
                if y % 2 == 0:
                    lb = xu ** y
                    ub = xl ** y
                else:
                    lb = xl ** y
                    ub = xu ** y
        else:
            if y < 0:
                if y % 2 == 0:
                    lb = min(xl ** y, xu ** y)
                    ub = inf
                else:
                    lb = - inf
                    ub = inf
            else:
                if y % 2 == 0:
                    lb = 0
                    ub = max(xl ** y, xu ** y)
                else:
                    lb = xl ** y
                    ub = xu ** y
    elif yl == yu:
        # the exponent has to be fractional, so x must be positive
        if xu < 0:
            msg = 'Cannot raise a negative number to the power of {0}.\n'.format(yl)
            msg += 'The upper bound of a variable raised to the power of {0} is {1}'.format(yl, xu)
            raise InfeasibleConstraintException(msg)
        xl = 0
        lb, ub = power(xl, xu, yl, yu)
    else:
        lb = -inf
        ub = inf

    return lb, ub


def _inverse_power1(zl, zu, yl, yu, orig_xl, orig_xu, feasibility_tol):
    """
    z = x**y => compute bounds on x.

    First, start by computing bounds on x with

        x = exp(ln(z) / y)

    However, if y is an integer, then x can be negative, so there are several special cases. See the docs below.
    """
    xl, xu = log(zl, zu)
    xl, xu = div(xl, xu, yl, yu, feasibility_tol)
    xl, xu = exp(xl, xu)

    # if y is an integer, then x can be negative
    if yl == yu and yl == round(yl):  # y is a fixed integer
        y = yl
        if y == 0:
            # Anything to the power of 0 is 1, so if y is 0, then x can be anything
            # (assuming zl <= 1 <= zu, which is enforced when traversing the tree in the other direction)
            xl = -inf
            xu = inf
        elif y % 2 == 0:
            """
            if y is even, then there are two primary cases (note that it is much easier to walk through these
            while looking at plots):
            case 1: y is positive
                x**y is convex, positive, and symmetric. The bounds on x depend on the lower bound of z. If zl <= 0, 
                then xl should simply be -xu. However, if zl > 0, then we may be able to say something better. For 
                example, if the original lower bound on x is positive, then we can keep xl computed from 
                x = exp(ln(z) / y). Furthermore, if the original lower bound on x is larger than -xl computed from 
                x = exp(ln(z) / y), then we can still keep the xl computed from x = exp(ln(z) / y). Similar logic
                applies to the upper bound of x.
            case 2: y is negative
                The ideas are similar to case 1.
            """
            if zu + feasibility_tol < 0:
                raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
            if y > 0:
                if zu <= 0:
                    _xl = 0
                    _xu = 0
                elif zl <= 0:
                    _xl = -xu
                    _xu = xu
                else:
                    if orig_xl <= -xl + feasibility_tol:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl - feasibility_tol:
                        _xu = -xl
                    else:
                        _xu = xu
                xl = _xl
                xu = _xu
            else:
                if zu == 0:
                    raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
                elif zl <= 0:
                    _xl = -inf
                    _xu = inf
                else:
                    if orig_xl <= -xl + feasibility_tol:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl - feasibility_tol:
                        _xu = -xl
                    else:
                        _xu = xu
                xl = _xl
                xu = _xu
        else:  # y % 2 == 1
            """
            y is odd. 
            Case 1: y is positive
                x**y is monotonically increasing. If y is positive, then we can can compute the bounds on x using
                x = z**(1/y) and the signs on xl and xu depend on the signs of zl and zu.
            Case 2: y is negative
                Again, this is easier to visualize with a plot. x**y approaches zero when x approaches -inf or inf. 
                Thus, if zl < 0 < zu, then no bounds can be inferred for x. If z is positive (zl >=0 ) then we can
                use the bounds computed from x = exp(ln(z) / y). If z is negative (zu <= 0), then we live in the
                bottom left quadrant, xl depends on zu, and xu depends on zl.
            """
            if y > 0:
                xl = abs(zl)**(1.0/y)
                xl = math.copysign(xl, zl)
                xu = abs(zu)**(1.0/y)
                xu = math.copysign(xu, zu)
            else:
                if zl >= 0:
                    pass
                elif zu <= 0:
                    if zu == 0:
                        xl = -inf
                    else:
                        xl = -abs(zu)**(1.0/y)
                    if zl == 0:
                        xu = -inf
                    else:
                        xu = -abs(zl)**(1.0/y)
                else:
                    xl = -inf
                    xu = inf

    return xl, xu


def _inverse_power2(zl, zu, xl, xu, feasiblity_tol):
    """
    z = x**y => compute bounds on y
    y = ln(z) / ln(x)

    This function assumes the exponent can be fractional, so x must be positive. This method should not be called
    if the exponent is an integer.
    """
    if xu <= 0:
        raise IntervalException('Cannot raise a negative variable to a fractional power.')
    if (xl > 0 and zu <= 0) or (xl >= 0 and zu < 0):
        raise InfeasibleConstraintException('A positive variable raised to the power of anything must be positive.')
    lba, uba = log(zl, zu)
    lbb, ubb = log(xl, xu)
    yl, yu = div(lba, uba, lbb, ubb, feasiblity_tol)
    return yl, yu


def exp(xl, xu):
    try:
        lb = math.exp(xl)
    except OverflowError:
        lb = math.inf
    try:
        ub = math.exp(xu)
    except OverflowError:
        ub = math.inf
    return lb, ub


def log(xl, xu):
    if xl > 0:
        lb = math.log(xl)
    else:
        lb = -inf
    if xu > 0:
        ub = math.log(xu)
    else:
        ub = -inf
    return lb, ub


def log10(xl, xu):
    if xl > 0:
        lb = math.log10(xl)
    else:
        lb = -inf
    if xu > 0:
        ub = math.log10(xu)
    else:
        ub = -inf
    return lb, ub


def sin(xl, xu):
    """

    Parameters
    ----------
    xl: float
    xu: float

    Returns
    -------
    lb: float
    ub: float
    """

    # if there is a minimum between xl and xu, then the lower bound is -1. Minimums occur at 2*pi*n - pi/2
    # find the minimum value of i such that 2*pi*i - pi/2 >= xl. Then round i up. If 2*pi*i - pi/2 is still less
    # than or equal to xu, then there is a minimum between xl and xu. Thus the lb is -1. Otherwise, the minimum
    # occurs at either xl or xu
    if xl <= -inf or xu >= inf:
        return -1, 1
    pi = math.pi
    i = (xl + pi / 2) / (2 * pi)
    i = math.ceil(i)
    x_at_min = 2 * pi * i - pi / 2
    if x_at_min <= xu:
        lb = -1
    else:
        lb = min(math.sin(xl), math.sin(xu))

    # if there is a maximum between xl and xu, then the upper bound is 1. Maximums occur at 2*pi*n + pi/2
    i = (xu - pi / 2) / (2 * pi)
    i = math.floor(i)
    x_at_max = 2 * pi * i + pi / 2
    if x_at_max >= xl:
        ub = 1
    else:
        ub = max(math.sin(xl), math.sin(xu))

    return lb, ub


def cos(xl, xu):
    """

    Parameters
    ----------
    xl: float
    xu: float

    Returns
    -------
    lb: float
    ub: float
    """

    # if there is a minimum between xl and xu, then the lower bound is -1. Minimums occur at 2*pi*n - pi
    # find the minimum value of i such that 2*pi*i - pi >= xl. Then round i up. If 2*pi*i - pi/2 is still less
    # than or equal to xu, then there is a minimum between xl and xu. Thus the lb is -1. Otherwise, the minimum
    # occurs at either xl or xu
    if xl <= -inf or xu >= inf:
        return -1, 1
    pi = math.pi
    i = (xl + pi) / (2 * pi)
    i = math.ceil(i)
    x_at_min = 2 * pi * i - pi
    if x_at_min <= xu:
        lb = -1
    else:
        lb = min(math.cos(xl), math.cos(xu))

    # if there is a maximum between xl and xu, then the upper bound is 1. Maximums occur at 2*pi*n
    i = (xu) / (2 * pi)
    i = math.floor(i)
    x_at_max = 2 * pi * i
    if x_at_max >= xl:
        ub = 1
    else:
        ub = max(math.cos(xl), math.cos(xu))

    return lb, ub


def tan(xl, xu):
    """

    Parameters
    ----------
    xl: float
    xu: float

    Returns
    -------
    lb: float
    ub: float
    """

    # tan goes to -inf and inf at every pi*i + pi/2 (integer i). If one of these values is between xl and xu, then
    # the lb is -inf and the ub is inf. Otherwise the minimum occurs at xl and the maximum occurs at xu.
    # find the minimum value of i such that pi*i + pi/2 >= xl. Then round i up. If pi*i + pi/2 is still less
    # than or equal to xu, then there is an undefined point between xl and xu.
    if xl <= -inf or xu >= inf:
        return -inf, inf
    pi = math.pi
    i = (xl - pi / 2) / (pi)
    i = math.ceil(i)
    x_at_undefined = pi * i + pi / 2
    if x_at_undefined <= xu:
        lb = -inf
        ub = inf
    else:
        lb = math.tan(xl)
        ub = math.tan(xu)

    return lb, ub


def asin(xl, xu, yl, yu, feasibility_tol):
    """
    y = asin(x); propagate bounds from x to y
    x = sin(y)

    Parameters
    ----------
    xl: float
    xu: float
    yl: float
    yu: float

    Returns
    -------
    lb: float
    ub: float
    """
    if xl < -1:
        xl = -1
    if xu > 1:
        xu = 1

    pi = math.pi

    if yl <= -inf:
        lb = yl
    elif xl <= math.sin(yl) <= xu:
        # if sin(yl) >= xl then yl satisfies the bounds on x, and the lower bound of y cannot be improved
        lb = yl
    elif math.sin(yl) < xl:
        """
        we can only push yl up from its current value to the next lowest value such that xl = sin(y). In other words,
        we need to

        min y
        s.t.
            xl = sin(y)
            y >= yl

        globally. 
        """
        # first find the next minimum of x = sin(y). Minimums occur at y = 2*pi*n - pi/2 for integer n.
        i = (yl + pi / 2) / (2 * pi)
        i1 = math.floor(i)
        i2 = math.ceil(i)
        i1 = 2 * pi * i1 - pi / 2
        i2 = 2 * pi * i2 - pi / 2
        # now find the next value of y such that xl = sin(y). This can be computed by a distance from the minimum (i).
        y_tmp = math.asin(xl)  # this will give me a value between -pi/2 and pi/2
        dist = y_tmp - (-pi / 2)  # this is the distance between the minimum of the sin function and a value that
        # satisfies xl = sin(y)
        lb1 = i1 + dist
        lb2 = i2 + dist
        if lb1 >= yl - feasibility_tol:
            lb = lb1
        else:
            lb = lb2
    else:
        # sin(yl) > xu
        i = (yl - pi / 2) / (2 * pi)
        i1 = math.floor(i)
        i2 = math.ceil(i)
        i1 = 2 * pi * i1 + pi / 2
        i2 = 2 * pi * i2 + pi / 2
        y_tmp = math.asin(xu)
        dist = pi / 2 - y_tmp
        lb1 = i1 + dist
        lb2 = i2 + dist
        if lb1 >= yl - feasibility_tol:
            lb = lb1
        else:
            lb = lb2

    # use the same logic for the maximum
    if yu >= inf:
        ub = yu
    elif xl <= math.sin(yu) <= xu:
        ub = yu
    elif math.sin(yu) > xu:
        i = (yu - pi / 2) / (2 * pi)
        i1 = math.ceil(i)
        i2 = math.floor(i)
        i1 = 2 * pi * i1 + pi / 2
        i2 = 2 * pi * i2 + pi / 2
        y_tmp = math.asin(xu)
        dist = pi / 2 - y_tmp
        ub1 = i1 - dist
        ub2 = i2 - dist
        if ub1 <= yu + feasibility_tol:
            ub = ub1
        else:
            ub = ub2
    else:
        # math.sin(yu) < xl
        i = (yu + pi / 2) / (2 * pi)
        i1 = math.ceil(i)
        i2 = math.floor(i)
        i1 = 2 * pi * i1 - pi / 2
        i2 = 2 * pi * i2 - pi / 2
        y_tmp = math.asin(xl)
        dist = y_tmp - (-pi / 2)
        ub1 = i1 - dist
        ub2 = i2 - dist
        if ub1 <= yu + feasibility_tol:
            ub = ub1
        else:
            ub = ub2

    return lb, ub


def acos(xl, xu, yl, yu, feasibility_tol):
    """
    y = acos(x); propagate bounds from x to y
    x = cos(y)

    Parameters
    ----------
    xl: float
    xu: float
    yl: float
    yu: float

    Returns
    -------
    lb: float
    ub: float
    """
    if xl < -1:
        xl = -1
    if xu > 1:
        xu = 1

    pi = math.pi

    if yl <= -inf:
        lb = yl
    elif xl <= math.cos(yl) <= xu:
        # if xl <= cos(yl) <= xu then yl satisfies the bounds on x, and the lower bound of y cannot be improved
        lb = yl
    elif math.cos(yl) < xl:
        """
        we can only push yl up from its current value to the next lowest value such that xl = cos(y). In other words,
        we need to

        min y
        s.t.
            xl = cos(y)
            y >= yl

        globally. 
        """
        # first find the next minimum of x = cos(y). Minimums occur at y = 2*pi*n - pi for integer n.
        i = (yl + pi) / (2 * pi)
        i1 = math.floor(i)
        i2 = math.ceil(i)
        i1 = 2 * pi * i1 - pi
        i2 = 2 * pi * i2 - pi
        # now find the next value of y such that xl = cos(y). This can be computed by a distance from the minimum (i).
        y_tmp = math.acos(xl)  # this will give me a value between 0 and pi
        dist = pi - y_tmp  # this is the distance between the minimum of the sin function and a value that
        # satisfies xl = sin(y)
        lb1 = i1 + dist
        lb2 = i2 + dist
        if lb1 >= yl - feasibility_tol:
            lb = lb1
        else:
            lb = lb2
    else:
        # cos(yl) > xu
        # first find the next maximum of x = cos(y).
        i = yl / (2 * pi)
        i1 = math.floor(i)
        i2 = math.ceil(i)
        i1 = 2 * pi * i1
        i2 = 2 * pi * i2
        y_tmp = math.acos(xu)
        dist = y_tmp
        lb1 = i1 + dist
        lb2 = i2 + dist
        if lb1 >= yl - feasibility_tol:
            lb = lb1
        else:
            lb = lb2

    # use the same logic for the maximum
    if yu >= inf:
        ub = yu
    elif xl <= math.cos(yu) <= xu:
        ub = yu
    elif math.cos(yu) > xu:
        i = yu / (2 * pi)
        i1 = math.ceil(i)
        i2 = math.floor(i)
        i1 = 2 * pi * i1
        i2 = 2 * pi * i2
        y_tmp = math.acos(xu)
        dist = y_tmp
        ub1 = i1 - dist
        ub2 = i2 - dist
        if ub1 <= yu + feasibility_tol:
            ub = ub1
        else:
            ub = ub2
    else:
        # math.cos(yu) < xl
        i = (yu + pi) / (2 * pi)
        i1 = math.ceil(i)
        i2 = math.floor(i)
        i1 = 2 * pi * i1 - pi
        i2 = 2 * pi * i2 - pi
        y_tmp = math.acos(xl)
        dist = pi - y_tmp
        ub1 = i1 - dist
        ub2 = i2 - dist
        if ub1 <= yu + feasibility_tol:
            ub = ub1
        else:
            ub = ub2

    return lb, ub


def atan(xl, xu, yl, yu):
    """
    y = atan(x); propagate bounds from x to y
    x = tan(y)

    Parameters
    ----------
    xl: float
    xu: float
    yl: float
    yu: float

    Returns
    -------
    lb: float
    ub: float
    """

    pi = math.pi

    # tan goes to -inf and inf at every pi*i + pi/2 (integer i).
    if xl <= -inf or yl <= -inf:
        lb = yl
    else:
        i = (yl - pi / 2) / pi
        i = math.floor(i)
        i = pi * i + pi / 2
        y_tmp = math.atan(xl)
        dist = y_tmp - (-pi/2)
        lb = i + dist

    if xu >= inf or yu >= inf:
        ub = yu
    else:
        i = (yu - pi / 2) / pi
        i = math.ceil(i)
        i = pi * i + pi / 2
        y_tmp = math.atan(xu)
        dist = pi / 2 - y_tmp
        ub = i - dist

    if yl > lb:
        lb = yl
    if yu < ub:
        ub = yu

    return lb, ub
