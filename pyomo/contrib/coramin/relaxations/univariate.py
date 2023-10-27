import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide, FunctionShape
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef, _check_cut
from .custom_block import declare_custom_block
import numpy as np
import math
import scipy.optimize
from ._utils import check_var_pts, _get_bnds_list, _get_bnds_tuple
from pyomo.core.base.param import ScalarParam, IndexedParam
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.core.expr.numeric_expr import LinearExpression
import logging
from typing import Optional, Union, Sequence
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
logger = logging.getLogger(__name__)
pe = pyo


def _sin_overestimator_fn(x, LB):
    return np.sin(x) + np.cos(x) * (LB - x) - np.sin(LB)


def _sin_underestimator_fn(x, UB):
    return np.sin(x) + np.cos(-x) * (UB - x) - np.sin(UB)


def _compute_sine_overestimator_tangent_point(vlb):
    assert vlb < 0
    tangent_point, res = scipy.optimize.bisect(f=_sin_overestimator_fn, a=0, b=math.pi / 2, args=(vlb,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = float(np.cos(tangent_point))
        intercept = float(np.sin(vlb) - slope * vlb)
        return tangent_point, slope, intercept
    else:
        raise RuntimeError('Unable to build relaxation for sin(x)\nBisect info: ' + str(res))


def _compute_sine_underestimator_tangent_point(vub):
    assert vub > 0
    tangent_point, res = scipy.optimize.bisect(f=_sin_underestimator_fn, a=-math.pi / 2, b=0, args=(vub,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = float(np.cos(-tangent_point))
        intercept = float(np.sin(vub) - slope * vub)
        return tangent_point, slope, intercept
    else:
        raise RuntimeError('Unable to build relaxation for sin(x)\nBisect info: ' + str(res))


def _atan_overestimator_fn(x, LB):
    return (1 + x**2) * (np.arctan(x) - np.arctan(LB)) - x + LB


def _atan_underestimator_fn(x, UB):
    return (1 + x**2) * (np.arctan(x) - np.arctan(UB)) - x + UB


def _compute_arctan_overestimator_tangent_point(vlb):
    assert vlb < 0
    tangent_point, res = scipy.optimize.bisect(f=_atan_overestimator_fn, a=0, b=abs(vlb), args=(vlb,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = 1/(1 + tangent_point**2)
        intercept = float(np.arctan(vlb) - slope * vlb)
        return tangent_point, slope, intercept
    else:
        raise RuntimeError('Unable to build relaxation for arctan(x)\nBisect info: ' + str(res))


def _compute_arctan_underestimator_tangent_point(vub):
    assert vub > 0
    tangent_point, res = scipy.optimize.bisect(f=_atan_underestimator_fn, a=-vub, b=0, args=(vub,),
                                               full_output=True, disp=False)
    if res.converged:
        tangent_point = float(tangent_point)
        slope = 1/(1 + tangent_point**2)
        intercept = float(np.arctan(vub) - slope * vub)
        return tangent_point, slope, intercept
    else:
        raise RuntimeError('Unable to build relaxation for arctan(x)\nBisect info: ' + str(res))


class _FxExpr(object):
    def __init__(self, expr, x):
        self._expr = expr
        self._x = x
        self._deriv = reverse_sd(expr)[x]

    def eval(self, _xval):
        _xval = pyo.value(_xval)
        orig_xval = self._x.value
        self._x.value = _xval
        res = pyo.value(self._expr)
        self._x.set_value(orig_xval, skip_validation=True)
        return res

    def deriv(self, _xval):
        _xval = pyo.value(_xval)
        orig_xval = self._x.value
        self._x.value = _xval
        res = pyo.value(self._deriv)
        self._x.set_value(orig_xval, skip_validation=True)
        return res

    def __call__(self, _xval):
        return self.eval(_xval)


def _func_wrapper(obj):
    def _func(m, val):
        return obj(val)
    return _func


def _pw_univariate_relaxation(b, x, w, x_pts, f_x_expr, pw_repn='INC', shape=FunctionShape.UNKNOWN,
                              relaxation_side=RelaxationSide.BOTH, large_eval_tol=math.inf,
                              safety_tol=0):
    """
    This function creates piecewise envelopes to relax "w=f(x)" where f(x) is univariate and either convex over the
    entire domain of x or concave over the entire domain of x.

    Parameters
    ----------
    b: pyo.Block
    x: pyo.Var
        The "x" variable in f(x)
    w: pyo.Var
        The "w" variable that is replacing f(x)
    x_pts: Sequence[float]
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the last
        point (x_pts[-1]) is equal to x.ub
    f_x_expr: pyomo expression
        An expression for f(x)
    pw_repn: str
        This must be one of the valid strings for the peicewise representation to use (directly from the Piecewise
        component). Use help(Piecewise) to learn more.
    shape: FunctionShape
        Specify the shape of the function. Valid values are minlp.FunctionShape.CONVEX or minlp.FunctionShape.CONCAVE
    relaxation_side: RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    large_eval_tol: float
        To avoid numerical problems, if f_x_expr or its derivative evaluates to a value larger than large_eval_tol, 
        at a point in x_pts, then that point is skipped.
    """
    assert shape in {FunctionShape.CONCAVE, FunctionShape.CONVEX}
    assert relaxation_side in {RelaxationSide.UNDER, RelaxationSide.OVER}
    if relaxation_side == RelaxationSide.UNDER:
        assert shape == FunctionShape.CONCAVE
    else:
        assert shape == FunctionShape.CONVEX

    _eval = _FxExpr(expr=f_x_expr, x=x)
    xlb = x_pts[0]
    xub = x_pts[-1]

    check_var_pts(x, x_pts)

    if x.is_fixed():
        b.x_fixed_con = pyo.Constraint(expr=w == _eval(x.value))
    elif xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == _eval(x.lb))
    else:
        # Do the non-convex piecewise portion if shape=CONCAVE and relaxation_side=Under/BOTH
        # or if shape=CONVEX and relaxation_side=Over/BOTH
        pw_constr_type = None
        if shape == FunctionShape.CONVEX and relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            pw_constr_type = 'UB'
            _eval = _FxExpr(expr=f_x_expr + safety_tol, x=x)
        if shape == FunctionShape.CONCAVE and relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            pw_constr_type = 'LB'
            _eval = _FxExpr(expr=f_x_expr - safety_tol, x=x)

        if pw_constr_type is not None:
            # Build the piecewise side of the envelope
            if x_pts[0] > -math.inf and x_pts[-1] < math.inf:
                tmp_pts = list()
                for _pt in x_pts:
                    try:
                        f = _eval(_pt)
                        if abs(f) >= large_eval_tol:
                            logger.warning(f'Skipping pt {_pt} for var {str(x)} because |{str(f_x_expr)}| '
                                           f'evaluated at {_pt} is larger than {large_eval_tol}')
                            continue
                        tmp_pts.append(_pt)
                    except (ZeroDivisionError, ValueError, OverflowError):
                        pass
                if len(tmp_pts) >= 2 and tmp_pts[0] == x_pts[0] and tmp_pts[-1] == x_pts[-1]:
                    b.pw_linear_under_over = pyo.Piecewise(w, x,
                                                           pw_pts=tmp_pts,
                                                           pw_repn=pw_repn,
                                                           pw_constr_type=pw_constr_type,
                                                           f_rule=_func_wrapper(_eval)
                                                           )


def pw_sin_relaxation(b, x, w, x_pts, relaxation_side=RelaxationSide.BOTH, safety_tol=1e-10):
    """
    This function creates piecewise relaxations to relax "w=sin(x)" for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    b: pyo.Block
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in sin(x). The lower bound on x must greater than or equal to
        -pi/2 and the upper bound on x must be less than or equal to pi/2.
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing sin(x)
    x_pts: Sequence[float]
        A list of floating point numbers to define the points over which the piecewise
        representation will be generated. This list must be ordered, and it is expected
        that the first point (x_pts[0]) is equal to x.lb and the last point (x_pts[-1])
        is equal to x.ub
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    safety_tol: float
        amount to lift the overestimator or drop the underestimator. This is used to ensure none of the feasible
        region is cut off by error in computing the over and under estimators.
    """
    check_var_pts(x, x_pts)
    expr = pyo.sin(x)

    xlb = x_pts[0]
    xub = x_pts[-1]

    if x.is_fixed() or xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == (pyo.value(expr)))
        return

    if xlb < -np.pi / 2.0:
        return

    if xub > np.pi / 2.0:
        return

    OE_tangent_x, OE_tangent_slope, OE_tangent_intercept = _compute_sine_overestimator_tangent_point(xlb)
    UE_tangent_x, UE_tangent_slope, UE_tangent_intercept = _compute_sine_underestimator_tangent_point(xub)
    non_piecewise_overestimators_pts = []
    non_piecewise_underestimator_pts = []

    if relaxation_side == RelaxationSide.OVER:
        if OE_tangent_x < xub:
            new_x_pts = [i for i in x_pts if i < OE_tangent_x]
            new_x_pts.append(xub)
            non_piecewise_overestimators_pts = [OE_tangent_x]
            non_piecewise_overestimators_pts.extend(i for i in x_pts if i > OE_tangent_x)
            x_pts = new_x_pts
    elif relaxation_side == RelaxationSide.UNDER:
        if UE_tangent_x > xlb:
            new_x_pts = [xlb]
            new_x_pts.extend(i for i in x_pts if i > UE_tangent_x)
            non_piecewise_underestimator_pts = [i for i in x_pts if i < UE_tangent_x]
            non_piecewise_underestimator_pts.append(UE_tangent_x)
            x_pts = new_x_pts

    b.non_piecewise_overestimators = pyo.ConstraintList()
    b.non_piecewise_underestimators = pyo.ConstraintList()
    for pt in non_piecewise_overestimators_pts:
        b.non_piecewise_overestimators.add(w <= math.sin(pt) + safety_tol + (x - pt) * math.cos(pt))
    for pt in non_piecewise_underestimator_pts:
        b.non_piecewise_underestimators.add(w >= math.sin(pt) - safety_tol + (x - pt) * math.cos(pt))

    intervals = []
    for i in range(len(x_pts)-1):
        intervals.append((x_pts[i], x_pts[i+1]))

    b.interval_set = pyo.Set(initialize=range(len(intervals)), ordered=True)
    b.x = pyo.Var(b.interval_set)
    b.w = pyo.Var(b.interval_set)
    if len(intervals) == 1:
        b.lam = pyo.Param(b.interval_set, mutable=True)
        b.lam[0].value = 1.0
    else:
        b.lam = pyo.Var(b.interval_set, within=pyo.Binary)
    b.x_lb = pyo.ConstraintList()
    b.x_ub = pyo.ConstraintList()
    b.x_sum = pyo.Constraint(expr=x == sum(b.x[i] for i in b.interval_set))
    b.w_sum = pyo.Constraint(expr=w == sum(b.w[i] for i in b.interval_set))
    b.lam_sum = pyo.Constraint(expr=sum(b.lam[i] for i in b.interval_set) == 1)
    b.overestimators = pyo.ConstraintList()
    b.underestimators = pyo.ConstraintList()

    for i, tup in enumerate(intervals):
        x0 = tup[0]
        x1 = tup[1]

        b.x_lb.add(x0 * b.lam[i] <= b.x[i])
        b.x_ub.add(b.x[i] <= x1 * b.lam[i])

        # Overestimators
        if relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            if x0 < 0 and x1 <= 0:
                slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                intercept = math.sin(x0) - slope * x0
                b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            elif (x0 < 0) and (x1 > 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_sine_overestimator_tangent_point(x0)
                if tangent_x <= x1:
                    b.overestimators.add(b.w[i] <= tangent_slope * b.x[i] + (tangent_intercept + safety_tol) * b.lam[i])
                    b.overestimators.add(b.w[i] <= math.cos(x1) * b.x[i] +
                                         (math.sin(x1) - x1 * math.cos(x1) + safety_tol) * b.lam[i])
                else:
                    slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                    intercept = math.sin(x0) - slope * x0
                    b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            else:
                b.overestimators.add(b.w[i] <= math.cos(x0)*b.x[i] +
                                     (math.sin(x0) - x0*math.cos(x0) + safety_tol)*b.lam[i])
                b.overestimators.add(b.w[i] <= math.cos(x1)*b.x[i] +
                                     (math.sin(x1) - x1*math.cos(x1) + safety_tol)*b.lam[i])

        # Underestimators
        if relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            if x0 >= 0 and x1 > 0:
                slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                intercept = math.sin(x0) - slope * x0
                b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            elif (x1 > 0) and (x0 < 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_sine_underestimator_tangent_point(x1)
                if tangent_x >= x0:
                    b.underestimators.add(b.w[i] >= tangent_slope*b.x[i] + (tangent_intercept - safety_tol)*b.lam[i])
                    b.underestimators.add(b.w[i] >= math.cos(x0)*b.x[i] +
                                          (math.sin(x0) - x0 * math.cos(x0) - safety_tol)*b.lam[i])
                else:
                    slope = (math.sin(x1) - math.sin(x0)) / (x1 - x0)
                    intercept = math.sin(x0) - slope * x0
                    b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            else:
                b.underestimators.add(b.w[i] >= math.cos(x0)*b.x[i] +
                                      (math.sin(x0) - x0 * math.cos(x0) - safety_tol)*b.lam[i])
                b.underestimators.add(b.w[i] >= math.cos(x1)*b.x[i] +
                                      (math.sin(x1) - x1 * math.cos(x1) - safety_tol)*b.lam[i])

    return x_pts


def pw_arctan_relaxation(b, x, w, x_pts, relaxation_side=RelaxationSide.BOTH, safety_tol=1e-10):
    """
    This function creates piecewise relaxations to relax "w=sin(x)" for -pi/2 <= x <= pi/2.

    Parameters
    ----------
    b: pyo.Block
    x: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The "x" variable in sin(x). The lower bound on x must greater than or equal to
        -pi/2 and the upper bound on x must be less than or equal to pi/2.
    w: pyomo.core.base.var.SimpleVar or pyomo.core.base.var._GeneralVarData
        The auxillary variable replacing sin(x)
    x_pts: Sequence[float]
        A list of floating point numbers to define the points over which the piecewise
        representation will be generated. This list must be ordered, and it is expected
        that the first point (x_pts[0]) is equal to x.lb and the last point (x_pts[-1])
        is equal to x.ub
    relaxation_side: minlp.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    safety_tol: float
        amount to lift the overestimator or drop the underestimator. This is used to ensure none of the feasible
        region is cut off by error in computing the over and under estimators.
    """
    check_var_pts(x, x_pts)
    expr = pyo.atan(x)
    _eval = _FxExpr(expr, x)

    xlb = x_pts[0]
    xub = x_pts[-1]

    if x.is_fixed() or xlb == xub:
        b.x_fixed_con = pyo.Constraint(expr=w == pyo.value(expr))
        return

    if xlb == -math.inf or xub == math.inf:
        return

    OE_tangent_x, OE_tangent_slope, OE_tangent_intercept = _compute_arctan_overestimator_tangent_point(xlb)
    UE_tangent_x, UE_tangent_slope, UE_tangent_intercept = _compute_arctan_underestimator_tangent_point(xub)
    non_piecewise_overestimators_pts = []
    non_piecewise_underestimator_pts = []

    if relaxation_side == RelaxationSide.OVER:
        if OE_tangent_x < xub:
            new_x_pts = [i for i in x_pts if i < OE_tangent_x]
            new_x_pts.append(xub)
            non_piecewise_overestimators_pts = [OE_tangent_x]
            non_piecewise_overestimators_pts.extend(i for i in x_pts if i > OE_tangent_x)
            x_pts = new_x_pts
    elif relaxation_side == RelaxationSide.UNDER:
        if UE_tangent_x > xlb:
            new_x_pts = [xlb]
            new_x_pts.extend(i for i in x_pts if i > UE_tangent_x)
            non_piecewise_underestimator_pts = [i for i in x_pts if i < UE_tangent_x]
            non_piecewise_underestimator_pts.append(UE_tangent_x)
            x_pts = new_x_pts

    b.non_piecewise_overestimators = pyo.ConstraintList()
    b.non_piecewise_underestimators = pyo.ConstraintList()
    for pt in non_piecewise_overestimators_pts:
        b.non_piecewise_overestimators.add(w <= math.atan(pt) + safety_tol + (x - pt) * _eval.deriv(pt))
    for pt in non_piecewise_underestimator_pts:
        b.non_piecewise_underestimators.add(w >= math.atan(pt) - safety_tol + (x - pt) * _eval.deriv(pt))

    intervals = []
    for i in range(len(x_pts)-1):
        intervals.append((x_pts[i], x_pts[i+1]))

    b.interval_set = pyo.Set(initialize=range(len(intervals)))
    b.x = pyo.Var(b.interval_set)
    b.w = pyo.Var(b.interval_set)
    if len(intervals) == 1:
        b.lam = pyo.Param(b.interval_set, mutable=True)
        b.lam[0].value = 1.0
    else:
        b.lam = pyo.Var(b.interval_set, within=pyo.Binary)
    b.x_lb = pyo.ConstraintList()
    b.x_ub = pyo.ConstraintList()
    b.x_sum = pyo.Constraint(expr=x == sum(b.x[i] for i in b.interval_set))
    b.w_sum = pyo.Constraint(expr=w == sum(b.w[i] for i in b.interval_set))
    b.lam_sum = pyo.Constraint(expr=sum(b.lam[i] for i in b.interval_set) == 1)
    b.overestimators = pyo.ConstraintList()
    b.underestimators = pyo.ConstraintList()

    for i, tup in enumerate(intervals):
        x0 = tup[0]
        x1 = tup[1]

        b.x_lb.add(x0 * b.lam[i] <= b.x[i])
        b.x_ub.add(b.x[i] <= x1 * b.lam[i])

        # Overestimators
        if relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            if x0 < 0 and x1 <= 0:
                slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                intercept = math.atan(x0) - slope * x0
                b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            elif (x0 < 0) and (x1 > 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_arctan_overestimator_tangent_point(x0)
                if tangent_x <= x1:
                    b.overestimators.add(b.w[i] <= tangent_slope * b.x[i] + (tangent_intercept + safety_tol) * b.lam[i])
                    b.overestimators.add(b.w[i] <= _eval.deriv(x1)*b.x[i] +
                                         (math.atan(x1) - x1*_eval.deriv(x1) + safety_tol)*b.lam[i])
                else:
                    slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                    intercept = math.atan(x0) - slope * x0
                    b.overestimators.add(b.w[i] <= slope * b.x[i] + (intercept + safety_tol) * b.lam[i])
            else:
                b.overestimators.add(b.w[i] <= _eval.deriv(x0)*b.x[i] +
                                     (math.atan(x0) - x0*_eval.deriv(x0) + safety_tol)*b.lam[i])
                b.overestimators.add(b.w[i] <= _eval.deriv(x1)*b.x[i] +
                                     (math.atan(x1) - x1*_eval.deriv(x1) + safety_tol)*b.lam[i])

        # Underestimators
        if relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            if x0 >= 0 and x1 > 0:
                slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                intercept = math.atan(x0) - slope * x0
                b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            elif (x1 > 0) and (x0 < 0):
                tangent_x, tangent_slope, tangent_intercept = _compute_arctan_underestimator_tangent_point(x1)
                if tangent_x >= x0:
                    b.underestimators.add(b.w[i] >= tangent_slope*b.x[i] + (tangent_intercept - safety_tol)*b.lam[i])
                    b.underestimators.add(b.w[i] >= _eval.deriv(x0)*b.x[i] +
                                          (math.atan(x0) - x0*_eval.deriv(x0) - safety_tol)*b.lam[i])
                else:
                    slope = (math.atan(x1) - math.atan(x0)) / (x1 - x0)
                    intercept = math.atan(x0) - slope * x0
                    b.underestimators.add(b.w[i] >= slope * b.x[i] + (intercept - safety_tol) * b.lam[i])
            else:
                b.underestimators.add(b.w[i] >= _eval.deriv(x0)*b.x[i] +
                                      (math.atan(x0) - x0*_eval.deriv(x0) - safety_tol)*b.lam[i])
                b.underestimators.add(b.w[i] >= _eval.deriv(x1)*b.x[i] +
                                      (math.atan(x1) - x1*_eval.deriv(x1) - safety_tol)*b.lam[i])

    return x_pts


@declare_custom_block(name='PWUnivariateRelaxation')
class PWUnivariateRelaxationData(BasePWRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of aux_var = f(x) where f(x) is either convex
    or concave.
    """

    def __init__(self, component):
        super().__init__(component)
        self._xref = ComponentWeakRef(None)
        self._aux_var_ref = ComponentWeakRef(None)
        self._pw_repn = 'INC'
        self._function_shape = FunctionShape.UNKNOWN
        self._f_x_expr = None
        self._secant: Optional[Union[ScalarConstraint, IndexedConstraint]] = None
        self._secant_expr: Optional[LinearExpression] = None
        self._secant_slope: Optional[Union[ScalarParam, IndexedParam]] = None
        self._secant_intercept: Optional[Union[ScalarParam, IndexedParam]] = None
        self._pw_secant = None

    @property
    def _x(self):
        return self._xref.get_component()

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return self._x,

    def get_rhs_expr(self):
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        res = list()
        if self.relaxation_side == RelaxationSide.BOTH:
            res.append(self._x)
        elif self.relaxation_side == RelaxationSide.UNDER and not self.is_rhs_convex():
            res.append(self._x)
        elif self.relaxation_side == RelaxationSide.OVER and not self.is_rhs_concave():
            res.append(self._x)
        return res

    def set_input(self, x, aux_var, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  use_linear_relaxation=True, large_coef=1e5, small_coef=1e-10, safety_tol=1e-10):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        shape: FunctionShape
            Options are FunctionShape.CONVEX and FunctionShape.CONCAVE
        f_x_expr: pyomo expression
            The pyomo expression representing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        super().set_input(relaxation_side=relaxation_side,
                          use_linear_relaxation=use_linear_relaxation,
                          large_coef=large_coef, small_coef=small_coef,
                          safety_tol=safety_tol)
        self._pw_repn = pw_repn
        self._function_shape = shape
        self._f_x_expr = f_x_expr

        self._xref.set_component(x)
        self._aux_var_ref.set_component(aux_var)
        bnds_list = _get_bnds_list(self._x)
        self._partitions[self._x] = bnds_list

    def build(self, x, aux_var, shape, f_x_expr, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              use_linear_relaxation=True, large_coef=1e5, small_coef=1e-10, safety_tol=1e-10):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        shape: FunctionShape
            Options are FunctionShape.CONVEX and FunctionShape.CONCAVE
        f_x_expr: pyomo expression
            The pyomo expression representing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        self.set_input(x=x, aux_var=aux_var, shape=shape, f_x_expr=f_x_expr, pw_repn=pw_repn,
                       relaxation_side=relaxation_side, use_linear_relaxation=use_linear_relaxation,
                       large_coef=large_coef, small_coef=small_coef, safety_tol=safety_tol)
        self.rebuild()

    def _remove_relaxation(self):
        del self._secant, self._secant_slope, self._secant_intercept, self._pw_secant
        self._secant = None
        self._secant_expr = None
        self._secant_slope = None
        self._secant_intercept = None
        self._pw_secant = None

    def remove_relaxation(self):
        super().remove_relaxation()
        self._remove_relaxation()

    def _needs_secant(self):
        if self.relaxation_side == RelaxationSide.BOTH and (self.is_rhs_convex() or self.is_rhs_concave()):
            return True
        elif self.relaxation_side == RelaxationSide.UNDER and self.is_rhs_concave():
            return True
        elif self.relaxation_side == RelaxationSide.OVER and self.is_rhs_convex():
            return True
        else:
            return False

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        super().rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                        ensure_oa_at_vertices=ensure_oa_at_vertices)
        if not build_nonlinear_constraint:
            if self._check_valid_domain_for_relaxation():
                if self._needs_secant():
                    if len(self._partitions[self._x]) == 2:
                        if self._secant is None:
                            self._remove_relaxation()
                            self._build_secant()
                        self._update_secant()
                    else:
                        self._remove_relaxation()
                        self._build_pw_secant()
                else:
                    self._remove_relaxation()
            else:
                self._remove_relaxation()

    def _build_secant(self):
        del self._secant_slope
        del self._secant_intercept
        del self._secant
        del self._secant_expr
        self._secant_slope = ScalarParam(mutable=True)
        self._secant_intercept = ScalarParam(mutable=True)
        e = LinearExpression(constant=self._secant_intercept, linear_coefs=[self._secant_slope], linear_vars=[self._x])
        self._secant_expr = e
        if self.is_rhs_concave():
            self._secant = ScalarConstraint(expr=self._aux_var >= e)
        elif self.is_rhs_convex():
            self._secant = ScalarConstraint(expr=self._aux_var <= e)
        else:
            raise RuntimeError('Function should be either convex or concave in order to build the secant')

    def _update_secant(self):
        _eval = _FxExpr(self._f_x_expr, self._x)
        assert len(self._partitions[self._x]) == 2

        try:
            x1 = self._partitions[self._x][0]
            x2 = self._partitions[self._x][1]
            if x1 == x2:
                slope = 0
                intercept = _eval(x1)
            else:
                y1 = _eval(x1)
                y2 = _eval(x2)
                slope = (y2 - y1) / (x2 - x1)
                intercept = y2 - slope*x2
            err_message = None
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            slope = None
            intercept = None
            err_message = str(e)
        if err_message is not None:
            logger.debug(f'Encountered exception when adding secant for "{self._get_pprint_string()}"; Error message: {err_message}')
            self._remove_relaxation()
        else:
            self._secant_slope._value = slope
            self._secant_intercept._value = intercept
            if self.is_rhs_concave():
                rel_side = RelaxationSide.UNDER
            else:
                rel_side = RelaxationSide.OVER
            success, bad_var, bad_coef, err_msg = _check_cut(self._secant_expr, too_small=self.small_coef,
                                                             too_large=self.large_coef, relaxation_side=rel_side,
                                                             safety_tol=self.safety_tol)
            if not success:
                self._log_bad_cut(bad_var, bad_coef, err_msg)
                self._secant.deactivate()
            else:
                self._secant.activate()

    def _build_pw_secant(self):
        del self._pw_secant
        self._pw_secant = pe.Block(concrete=True)
        if self.is_rhs_convex():
            _pw_univariate_relaxation(b=self._pw_secant, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                      f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONVEX,
                                      relaxation_side=RelaxationSide.OVER, large_eval_tol=self.large_coef,
                                      safety_tol=self.safety_tol)
        else:
            _pw_univariate_relaxation(b=self._pw_secant, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                      f_x_expr=self._f_x_expr, pw_repn=self._pw_repn, shape=FunctionShape.CONCAVE,
                                      relaxation_side=RelaxationSide.UNDER, large_eval_tol=self.large_coef,
                                      safety_tol=self.safety_tol)

    def add_partition_point(self, value=None):
        """
        This method adds one point to the partitioning of x. If value is not
        specified, a single point will be added to the partitioning of x at the current value of x. If value is
        specified, then value is added to the partitioning of x.

        Parameters
        ----------
        value: float
            The point to be added to the partitioning of x.
        """
        self._add_partition_point(self._x, value)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONVEX

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return self._function_shape == FunctionShape.CONCAVE

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val):
        self._use_linear_relaxation = val


@declare_custom_block(name='CustomUnivariateBaseRelaxation')
class CustomUnivariateBaseRelaxationData(PWUnivariateRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of aux_var = x**2.
    """

    def _rhs_func(self, x):
        raise NotImplementedError('This should be implemented by a derived class')

    def set_input(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
                  use_linear_relaxation=True, large_coef=1e5, small_coef=1e-10, safety_tol=1e-10):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        super().set_input(x=x, aux_var=aux_var, shape=FunctionShape.UNKNOWN,
                          f_x_expr=self._rhs_func(x), pw_repn=pw_repn,
                          relaxation_side=relaxation_side,
                          use_linear_relaxation=use_linear_relaxation,
                          large_coef=large_coef, small_coef=small_coef,
                          safety_tol=safety_tol)

    def build(self, x, aux_var, pw_repn='INC', relaxation_side=RelaxationSide.BOTH,
              use_linear_relaxation=True, large_coef=1e5, small_coef=1e-10, safety_tol=1e-10):
        """
        Parameters
        ----------
        x: pyomo.core.base.var._GeneralVarData
            The "x" variable in aux_var = f(x).
        aux_var: pyomo.core.base.var._GeneralVarData
            The auxillary variable replacing f(x)
        pw_repn: str
            This must be one of the valid strings for the piecewise representation to use (directly from the Piecewise
            component). Use help(Piecewise) to learn more.
        relaxation_side: RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        use_linear_relaxation: bool
            Specifies whether a linear or nonlinear relaxation should be used
        """
        self.set_input(x=x, aux_var=aux_var, pw_repn=pw_repn, relaxation_side=relaxation_side,
                       use_linear_relaxation=use_linear_relaxation, large_coef=large_coef, small_coef=small_coef,
                       safety_tol=safety_tol)
        self.rebuild()


@declare_custom_block(name='PWXSquaredRelaxation')
class PWXSquaredRelaxationData(CustomUnivariateBaseRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of aux_var = x**2.
    """
    def _rhs_func(self, x):
        return x**2

    def is_rhs_convex(self):
        return True


@declare_custom_block(name='PWCosRelaxation')
class PWCosRelaxationData(CustomUnivariateBaseRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = cos(x) for -pi/2 <= x <= pi/2.
    """

    def __init__(self, component):
        super().__init__(component)
        self._last_concave = None

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        current_concave = self.is_rhs_concave()
        if current_concave != self._last_concave:
            self._needs_rebuilt = True
        self._last_concave = current_concave
        super().rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                        ensure_oa_at_vertices=ensure_oa_at_vertices)

    def _rhs_func(self, x):
        return pe.cos(x)

    def is_rhs_concave(self):
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= -math.pi/2 and ub <= math.pi/2:
            return True
        else:
            return False


@declare_custom_block(name='SinArctanBaseRelaxation')
class SinArctanBaseRelaxationData(CustomUnivariateBaseRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = sin(x) for -pi/2 <= x <= pi/2.
    """

    def _rhs_func(self, x):
        raise NotImplementedError('This should be implemented by a derived class')

    def __init__(self, component):
        super().__init__(component)
        self._secant_index = None
        self._secant_exprs = None
        self._last_convex = None
        self._last_concave = None

    def _remove_relaxation(self):
        super()._remove_relaxation()
        del self._secant_index
        del self._secant_exprs
        self._secant_index = None
        self._secant_exprs = None

    def _pw_func(self):
        raise NotImplementedError('This should be implemented by a derived class')

    def _underestimator_func(self):
        raise NotImplementedError('This should be implemented by a derived class')

    def _overestimator_func(self):
        raise NotImplementedError('This should be implemented by a derived class')

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        current_convex = self.is_rhs_convex()
        current_concave = self.is_rhs_concave()
        if current_convex != self._last_convex or current_concave != self._last_concave:
            self._needs_rebuilt = True
        self._last_convex = current_convex
        self._last_concave = current_concave
        super().rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                        ensure_oa_at_vertices=ensure_oa_at_vertices)
        if not build_nonlinear_constraint:
            if self._check_valid_domain_for_relaxation():
                if (not self.is_rhs_convex()) and (not self.is_rhs_concave()):
                    if len(self._partitions[self._x]) == 2:
                        if self._secant is None:
                            self._remove_relaxation()
                            self._build_relaxation()
                        self._update_relaxation()
                    else:
                        self._remove_relaxation()
                        del self._pw_secant
                        self._pw_secant = pe.Block(concrete=True)
                        self._pw_func()(b=self._pw_secant, x=self._x, w=self._aux_var, x_pts=self._partitions[self._x],
                                        relaxation_side=self.relaxation_side,
                                        safety_tol=self.safety_tol)
            else:
                self._remove_relaxation()

    def _build_relaxation(self):
        del self._secant_index, self._secant_slope, self._secant_intercept, self._secant
        self._secant_index = pe.Set(initialize=[0, 1, 2, 3])
        self._secant_exprs = dict()
        self._secant_slope = IndexedParam(self._secant_index, mutable=True)
        self._secant_intercept = IndexedParam(self._secant_index, mutable=True)
        self._secant = IndexedConstraint(self._secant_index)
        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}:
            for ndx in [0, 1]:
                e = LinearExpression(constant=self._secant_intercept[ndx],
                                     linear_coefs=[self._secant_slope[ndx]], linear_vars=[self._x])
                self._secant_exprs[ndx] = e
                self._secant[ndx] = self._aux_var >= e
        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            for ndx in [2, 3]:
                e = LinearExpression(constant=self._secant_intercept[ndx],
                                     linear_coefs=[self._secant_slope[ndx]], linear_vars=[self._x])
                self._secant_exprs[ndx] = e
                self._secant[ndx] = self._aux_var <= e

    def _check_expr(self, ndx):
        if ndx in {0, 1}:
            rel_side = RelaxationSide.UNDER
        else:
            rel_side = RelaxationSide.OVER
        success, bad_var, bad_coef, err_msg = _check_cut(self._secant_exprs[ndx], too_small=self.small_coef,
                                                         too_large=self.large_coef, relaxation_side=rel_side,
                                                         safety_tol=self.safety_tol)
        if not success:
            self._log_bad_cut(bad_var, bad_coef, err_msg)
            self._secant[ndx].deactivate()
        else:
            self._secant[ndx].activate()

    def _update_relaxation(self):
        xlb, xub = _get_bnds_tuple(self._x)
        _eval = _FxExpr(self.get_rhs_expr(), self._x)
        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}:
            tangent_x, tangent_slope, tangent_int = self._underestimator_func()(xub)
            if tangent_x >= xlb:
                self._secant_slope[0]._value = tangent_slope
                self._secant_intercept[0]._value = tangent_int
                self._secant_slope[1]._value = _eval.deriv(xlb)
                self._secant_intercept[1]._value = _eval(xlb) - xlb * _eval.deriv(xlb)
                self._check_expr(0)
                self._check_expr(1)
            else:
                y1 = _eval(xlb)
                y2 = _eval(xub)
                slope = (y2 - y1) / (xub - xlb)
                intercept = y2 - slope * xub
                self._secant_slope[0]._value = slope
                self._secant_intercept[0]._value = intercept
                self._check_expr(0)
                self._secant[1].deactivate()

        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            tangent_x, tangent_slope, tangent_int = self._overestimator_func()(xlb)
            if tangent_x <= xub:
                self._secant_slope[2]._value = tangent_slope
                self._secant_intercept[2]._value = tangent_int
                self._secant_slope[3]._value = _eval.deriv(xub)
                self._secant_intercept[3]._value = _eval(xub) - xub * _eval.deriv(xub)
                self._check_expr(2)
                self._check_expr(3)
            else:
                y1 = _eval(xlb)
                y2 = _eval(xub)
                slope = (y2 - y1) / (xub - xlb)
                intercept = y2 - slope * xub
                self._secant_slope[2]._value = slope
                self._secant_intercept[2]._value = intercept
                self._check_expr(2)
                self._secant[3].deactivate()


@declare_custom_block(name='PWSinRelaxation')
class PWSinRelaxationData(SinArctanBaseRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = sin(x) for -pi/2 <= x <= pi/2.
    """

    def _rhs_func(self, x):
        return pe.sin(x)

    def _check_valid_domain_for_relaxation(self) -> bool:
        lb, ub = _get_bnds_tuple(self._x)
        if lb >= -math.pi / 2 and ub <= math.pi / 2:
            return True
        return False

    def _pw_func(self):
        return pw_sin_relaxation

    def _underestimator_func(self):
        return _compute_sine_underestimator_tangent_point

    def _overestimator_func(self):
        return _compute_sine_overestimator_tangent_point

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= -math.pi / 2 and ub <= 0:
            return True
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= 0 and ub <= math.pi / 2:
            return True
        return False


@declare_custom_block(name='PWArctanRelaxation')
class PWArctanRelaxationData(SinArctanBaseRelaxationData):
    """
    A helper class for building and modifying piecewise relaxations of w = arctan(x).
    """

    def _rhs_func(self, x):
        return pe.atan(x)

    def _pw_func(self):
        return pw_arctan_relaxation

    def _underestimator_func(self):
        return _compute_arctan_underestimator_tangent_point

    def _overestimator_func(self):
        return _compute_arctan_overestimator_tangent_point

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if ub <= 0:
            return True
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        lb, ub = tuple(_get_bnds_list(self._x))
        if lb >= 0:
            return True
        return False
