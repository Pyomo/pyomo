import logging
import pyomo.environ as pe
import warnings
import math

logger = logging.getLogger(__name__)
pyo = pe


def _get_bnds_tuple(v):
    lb = pe.value(v.lb)
    ub = pe.value(v.ub)
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf

    return lb, ub


def _get_bnds_list(v):
    return list(_get_bnds_tuple(v))


def var_info_str(v):
    s = '\tVar: {0}\n'.format(v)
    return s


def bnds_info_str(vlb, vub):
    s = '\tLB: {0}\n'.format(vlb)
    s += '\tUB: {0}\n'.format(vub)
    return s


def x_pts_info_str(_x_pts):
    s = '\tx_pts: {0}\n'.format(_x_pts)
    return s


def check_var_pts(x, x_pts=None):
    xlb = pe.value(x.lb)
    xub = pe.value(x.ub)

    if xlb is None:
        xlb = -math.inf
    if xub is None:
        xub = math.inf

    raise_error = False
    raise_warning = False
    msg = None

    if xub < xlb:
        msg = 'Lower bound is larger than upper bound:\n' + var_info_str(x) + bnds_info_str(xlb, xub)
        raise_error = True

    if x_pts is not None:
        ordered = all(x_pts[i] <= x_pts[i+1] for i in range(len(x_pts)-1))
        if not ordered:
            msg = 'x_pts must be ordered:\n' + var_info_str(x) + bnds_info_str(xlb, xub) + x_pts_info_str(x_pts)
            raise_error = True

        if xlb != x_pts[0] or xub != x_pts[-1]:
            msg = ('end points of the x_pts list must be equal to the bounds on the x variable:\n' + var_info_str(x) +
                   bnds_info_str(xlb, xub) + x_pts_info_str(x_pts))
            raise_error = True

    if raise_error:
        logger.error(msg)
        raise ValueError(msg)

    if raise_warning:
        logger.warning(msg)
        warnings.warn(msg)
