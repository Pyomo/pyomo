import math


def add(xl, xu, yl, yu):
    return xl + yl, xu + yu


def sub(xl, xu, yl, yu):
    return xl - yu, xu - yl


def mul(xl, xu, yl, yu):
    return min(xl*yl, xl*yu, xu*yl, xu*yu), max(xl*yl, xl*yu, xu*yl, xu*yu)


def inv(xl, xu):
    if xl <= 0 and xu >= 0:
        return -math.inf, math.inf
    return 1/xu, 1/xl


def div(xl, xu, yl, yu):
    return mul(xl, xu, *inv(yl, yu))


def power(xl, xu, yl, yu):
    if xl > 0:
        if yl < 0 and yu > 0:
            lb = min(xu ** yl, xl ** yu)
            ub = max(xl ** yl, xu ** yu)
        elif yl >= 0:
            lb = xl ** yl
            ub = xu ** yu
        elif yu <= 0:
            lb = xu ** yl
            ub = xl ** yu
    elif xl == 0:
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
    else:
        lb = -math.inf
        ub = math.inf

    return lb, ub


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
