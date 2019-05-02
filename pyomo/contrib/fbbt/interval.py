import math
import warnings
import logging
from pyomo.common.errors import DeveloperError, InfeasibleConstraintException

logger = logging.getLogger(__name__)


def add(xl, xu, yl, yu):
    return xl + yl, xu + yu


def sub(xl, xu, yl, yu):
    return xl - yu, xu - yl


def mul(xl, xu, yl, yu):
    return min(xl*yl, xl*yu, xu*yl, xu*yu), max(xl*yl, xl*yu, xu*yl, xu*yu)


def inv(xl, xu):
    if xl <= 0 and xu >= 0:
        return -math.inf, math.inf
    return 1.0/xu, 1.0/xl


def div(xl, xu, yl, yu):
    return mul(xl, xu, *inv(yl, yu))


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
            lb = xl ** yl
            ub = xu ** yu
        elif yu <= 0:
            lb = xu ** yl
            ub = xl ** yu
        else:
            raise DeveloperError()
    elif xl == 0:
        # this section is only needed so we do not encounter math domain errors;
        # The logic is essentially the same as above (xl > 0)
        if xu == 0 and yl < 0:
            _lba = math.inf
        else:
            _lba = xu ** yl
        if yu < 0:
            _lbb = math.inf
        else:
            _lbb = xl ** yu
        lb = min(_lba, _lbb)

        if yl < 0:
            _uba = math.inf
        else:
            _uba = xl ** yl
        if xu == 0 and yu < 0:
            _ubb = math.inf
        else:
            _ubb = xu ** yu
        ub = max(_uba, _ubb)
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
                        ub = math.inf
                    else:
                        ub = xu ** y
                else:
                    if xu == 0:
                        lb = -math.inf
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
                    ub = math.inf
                else:
                    lb = - math.inf
                    ub = math.inf
            else:
                if y % 2 == 0:
                    lb = 0
                    ub = max(xl ** y, xu ** y)
                else:
                    lb = xl ** y
                    ub = xu ** y
    elif yl == yu:
        # the exponent is allowed to be fractional, so x must be positive
        xl = 0
        lb, ub = power(xl, xu, yl, yu)
    else:
        msg = 'encountered an exponent where the base is allowed to be negative '
        msg += 'and the exponent is allowed to be fractional and is not fixed. '
        msg += 'Assuming the lower bound of the base to be 0.'
        warnings.warn(msg)
        logger.warning(msg)
        xl = 0
        lb, ub = power(xl, xu, yl, yu)

    return lb, ub


def _inverse_power1(zl, zu, yl, yu, orig_xl, orig_xu):
    """
    z = x**y => compute bounds on x.

    First, start by computing bounds on x with

        x = exp(ln(z) / y)

    However, if y is an integer, then x can be negative, so there are several special cases. See the docs below.
    """
    xl, xu = log(zl, zu)
    xl, xu = div(xl, xu, yl, yu)
    xl, xu = exp(xl, xu)

    # if y is an integer, then x can be negative
    if yl == yu and yl == round(yl):  # y is a fixed integer
        y = yl
        if y == 0:
            # Anything to the power of 0 is 1, so if y is 0, then x can be anything
            # (assuming zl <= 1 <= zu, which is enforced when traversing the tree in the other direction)
            xl = -math.inf
            xu = math.inf
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
            if zu < 0:
                raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
            if y > 0:
                if zu == 0:
                    _xl = 0
                    _xu = 0
                elif zl <= 0:
                    _xl = -xu
                    _xu = xu
                else:
                    if orig_xl <= -xl:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl:
                        _xu = -xl
                    else:
                        _xu = xu
                xl = _xl
                xu = _xu
            else:
                if zu == 0:
                    raise InfeasibleConstraintException('Infeasible. Anything to the power of {0} must be positive.'.format(y))
                elif zl <= 0:
                    _xl = -math.inf
                    _xu = math.inf
                else:
                    if orig_xl <= -xl:
                        _xl = -xu
                    else:
                        _xl = xl
                    if orig_xu < xl:
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
                        xl = -math.inf
                    else:
                        xl = -abs(zu)**(1.0/y)
                    if zl == 0:
                        xu = -math.inf
                    else:
                        xu = -abs(zl)**(1.0/y)
                else:
                    xl = -math.inf
                    xu = math.inf

    return xl, xu


def _inverse_power2(zl, zu, xl, xu):
    """
    z = x**y => compute bounds on y
    y = ln(z) / ln(x)
    """
    lba, uba = log(zl, zu)
    lbb, ubb = log(xl, xu)
    yl, yu = div(lba, uba, lbb, ubb)
    return yl, yu


def exp(xl, xu):
    return math.exp(xl), math.exp(xu)


def log(xl, xu):
    if xl > 0:
        return math.log(xl), math.log(xu)
    elif xl == 0:
        if xu > 0:
            return -math.inf, math.log(xu)
        else:
            return -math.inf, -math.inf
    else:
        return -math.inf, math.inf


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
    if xl <= -math.inf or xu >= math.inf:
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
    if xl <= -math.inf or xu >= math.inf:
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
    if xl <= -math.inf or xu >= math.inf:
        return -math.inf, math.inf
    pi = math.pi
    i = (xl - pi / 2) / (pi)
    i = math.ceil(i)
    x_at_undefined = pi * i + pi / 2
    if x_at_undefined <= xu:
        lb = -math.inf
        ub = math.inf
    else:
        lb = math.tan(xl)
        ub = math.tan(xu)

    return lb, ub


def asin(xl, xu, yl, yu):
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

    if yl <= -math.inf:
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
        if lb1 >= yl:
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
        if lb1 >= yl:
            lb = lb1
        else:
            lb = lb2

    # use the same logic for the maximum
    if yu >= math.inf:
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
        if ub1 <= yu:
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
        if ub1 <= yu:
            ub = ub1
        else:
            ub = ub2

    return lb, ub


def acos(xl, xu, yl, yu):
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

    if yl <= -math.inf:
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
        if lb1 >= yl:
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
        if lb1 >= yl:
            lb = lb1
        else:
            lb = lb2

    # use the same logic for the maximum
    if yu >= math.inf:
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
        if ub1 <= yu:
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
        if ub1 <= yu:
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
    if xl <= -math.inf or yl <= -math.inf:
        lb = yl
    else:
        i = (yl - pi / 2) / pi
        i = math.floor(i)
        i = pi * i + pi / 2
        y_tmp = math.atan(xl)
        dist = y_tmp - (-pi/2)
        lb = i + dist

    if xu >= math.inf or yu >= math.inf:
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
