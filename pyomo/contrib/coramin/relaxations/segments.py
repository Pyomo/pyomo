import pyomo.environ as pyo
import warnings
import logging

logger = logging.getLogger(__name__)


def compute_k_segment_points(v, k):
    """
    Return a list of points that generats k segments between v.lb and v.ub

    Parameters
    ----------
    v: pyo.Var
    k: int

    Returns
    -------
    pts: list of float
    """
    delta = (pyo.value(v.ub) - pyo.value(v.lb)) / k
    pts = [pyo.value(v.lb) + i * delta for i in range(k)]
    pts.append(pyo.value(v.ub))
    return pts
