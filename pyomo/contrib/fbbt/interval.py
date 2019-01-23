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


