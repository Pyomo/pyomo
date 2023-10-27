import logging
import pyomo.environ as pyo
from coramin.utils.coramin_enums import RelaxationSide
from .custom_block import declare_custom_block
from .relaxations_base import BasePWRelaxationData, ComponentWeakRef, _check_cut
import math
from ._utils import check_var_pts, _get_bnds_list, _get_bnds_tuple
from pyomo.core.base.param import IndexedParam
from pyomo.core.base.constraint import IndexedConstraint
from pyomo.core.expr.numeric_expr import LinearExpression
from typing import Optional, Dict, Sequence
pe = pyo

logger = logging.getLogger(__name__)


def _build_pw_mccormick_relaxation(b, x1, x2, aux_var, x1_pts, relaxation_side=RelaxationSide.BOTH, safety_tol=1e-10):
    """
    This function creates piecewise envelopes to relax "aux_var = x1*x2". Note that the partitioning is done on "x1" only.
    This is the "nf4r" from Gounaris, Misener, and Floudas (2009).

    Parameters
    ----------
    b: pyo.ConcreteModel or pyo.Block
    x1: pyomo.core.base.var._GeneralVarData
        The "x1" variable in x1*x2
    x2: pyomo.core.base.var._GeneralVarData
        The "x2" variable in x1*x2
    aux_var: pyomo.core.base.var._GeneralVarData
        The "aux_var" variable that is replacing x*y
    x1_pts: Sequence[float]
        A list of floating point numbers to define the points over which the piecewise representation will generated.
        This list must be ordered, and it is expected that the first point (x_pts[0]) is equal to x.lb and the
        last point (x_pts[-1]) is equal to x.ub
    relaxation_side : minlp.minlp_defn.RelaxationSide
        Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
    """
    assert len(x1_pts) > 2

    x1_lb = x1_pts[0]
    x1_ub = x1_pts[-1]
    x2_lb, x2_ub = tuple(_get_bnds_list(x2))

    check_var_pts(x1, x_pts=x1_pts)
    check_var_pts(x2)

    if x1.is_fixed() and x2.is_fixed():
        b.x1_x2_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1) * pyo.value(x2))
    elif x1.is_fixed():
        b.x1_fixed_eq = pyo.Constraint(expr= aux_var == pyo.value(x1) * x2)
    elif x2.is_fixed():
        b.x2_fixed_eq = pyo.Constraint(expr= aux_var == x1 * pyo.value(x2))
    else:
        # create the lambda_ variables (binaries for the pw representation)
        b.interval_set = pyo.Set(initialize=range(1, len(x1_pts)))
        b.lambda_ = pyo.Var(b.interval_set, within=pyo.Binary)

        # create the delta x2 variables
        b.delta_x2 = pyo.Var(b.interval_set, bounds=(0, None))

        # create the "sos1" constraint
        b.lambda_sos1 = pyo.Constraint(expr=sum(b.lambda_[n] for n in b.interval_set) == 1.0)

        # create the x1 interval constraints
        b.x1_interval_lb = pyo.Constraint(expr=sum(x1_pts[n - 1] * b.lambda_[n] for n in b.interval_set) <= x1)
        b.x1_interval_ub = pyo.Constraint(expr=x1 <= sum(x1_pts[n] * b.lambda_[n] for n in b.interval_set))

        # create the x2 constraints
        b.x2_con = pyo.Constraint(expr=x2 == x2_lb + sum(b.delta_x2[n] for n in b.interval_set))

        def delta_x2n_ub_rule(m, n):
            return b.delta_x2[n] <= (x2_ub - x2_lb) * b.lambda_[n]

        b.delta_x2n_ub = pyo.Constraint(b.interval_set, rule=delta_x2n_ub_rule)

        # create the relaxation constraints
        if relaxation_side == RelaxationSide.UNDER or relaxation_side == RelaxationSide.BOTH:
            b.aux_var_lb1 = pyo.Constraint(expr=(aux_var >= x2_ub * x1 + sum(x1_pts[n] * b.delta_x2[n] for n in b.interval_set) -
                                                 (x2_ub - x2_lb) * sum(x1_pts[n] * b.lambda_[n] for n in b.interval_set) - safety_tol))
            b.aux_var_lb2 = pyo.Constraint(expr=aux_var >= x2_lb * x1 + sum(x1_pts[n - 1] * b.delta_x2[n] for n in b.interval_set) - safety_tol)

        if relaxation_side == RelaxationSide.OVER or relaxation_side == RelaxationSide.BOTH:
            b.aux_var_ub1 = pyo.Constraint(expr=(aux_var <= x2_ub * x1 + sum(x1_pts[n - 1] * b.delta_x2[n] for n in b.interval_set) -
                                                 (x2_ub - x2_lb) * sum(x1_pts[n - 1] * b.lambda_[n] for n in b.interval_set) + safety_tol))
            b.aux_var_ub2 = pyo.Constraint(expr=aux_var <= x2_lb * x1 + sum(x1_pts[n] * b.delta_x2[n] for n in b.interval_set) + safety_tol)


@declare_custom_block(name='PWMcCormickRelaxation')
class PWMcCormickRelaxationData(BasePWRelaxationData):
    """
    A class for managing McCormick relaxations of bilinear terms (aux_var = x1 * x2).
    """

    def __init__(self, component):
        BasePWRelaxationData.__init__(self, component)
        self._x1ref = ComponentWeakRef(None)
        self._x2ref = ComponentWeakRef(None)
        self._aux_var_ref = ComponentWeakRef(None)
        self._f_x_expr = None
        self._mc_index = None
        self._slopes_index = None
        self._v_index = None
        self._slopes: Optional[IndexedParam] = None
        self._intercepts: Optional[IndexedParam] = None
        self._mccormicks: Optional[IndexedConstraint] = None
        self._mc_exprs: Dict[int, LinearExpression] = dict()
        self._pw = None

    @property
    def _x1(self):
        return self._x1ref.get_component()

    @property
    def _x2(self):
        return self._x2ref.get_component()

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return self._x1, self._x2

    def get_rhs_expr(self):
        return self._f_x_expr

    def vars_with_bounds_in_relaxation(self):
        return [self._x1, self._x2]

    def _remove_relaxation(self):
        del self._slopes, self._intercepts, self._mccormicks, self._pw, \
            self._mc_index, self._v_index, self._slopes_index
        self._mc_index = None
        self._v_index = None
        self._slopes_index = None
        self._slopes = None
        self._intercepts = None
        self._mccormicks = None
        self._mc_exprs = dict()
        self._pw = None

    def set_input(self, x1, x2, aux_var, relaxation_side=RelaxationSide.BOTH, large_coef=1e5, small_coef=1e-10,
                  safety_tol=1e-10):
        """
        Parameters
        ----------
        x1 : pyomo.core.base.var._GeneralVarData
            The "x1" variable in x1*x2
        x2 : pyomo.core.base.var._GeneralVarData
            The "x2" variable in x1*x2
        aux_var : pyomo.core.base.var._GeneralVarData
            The "aux_var" auxillary variable that is replacing x1*x2
        relaxation_side : minlp.minlp_defn.RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        """
        super(PWMcCormickRelaxationData, self).set_input(relaxation_side=relaxation_side,
                                                         use_linear_relaxation=True,
                                                         large_coef=large_coef, small_coef=small_coef,
                                                         safety_tol=safety_tol)
        self._x1ref.set_component(x1)
        self._x2ref.set_component(x2)
        self._aux_var_ref.set_component(aux_var)
        self._partitions[self._x1] = _get_bnds_list(self._x1)
        self._f_x_expr = x1 * x2

    def build(self, x1, x2, aux_var, relaxation_side=RelaxationSide.BOTH, large_coef=1e5, small_coef=1e-10,
              safety_tol=1e-10):
        """
        Parameters
        ----------
        x1 : pyomo.core.base.var._GeneralVarData
            The "x1" variable in x1*x2
        x2 : pyomo.core.base.var._GeneralVarData
            The "x2" variable in x1*x2
        aux_var : pyomo.core.base.var._GeneralVarData
            The "aux_var" auxillary variable that is replacing x1*x2
        relaxation_side : minlp.minlp_defn.RelaxationSide
            Provide the desired side for the relaxation (OVER, UNDER, or BOTH)
        """
        self.set_input(x1=x1, x2=x2, aux_var=aux_var, relaxation_side=relaxation_side,
                       large_coef=large_coef, small_coef=small_coef, safety_tol=safety_tol)
        self.rebuild()

    def remove_relaxation(self):
        super(PWMcCormickRelaxationData, self).remove_relaxation()
        self._remove_relaxation()

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        super(PWMcCormickRelaxationData, self).rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                                                       ensure_oa_at_vertices=ensure_oa_at_vertices)
        if not build_nonlinear_constraint:
            if self._check_valid_domain_for_relaxation():
                if len(self._partitions[self._x1]) == 2:
                    if self._mccormicks is None:
                        self._remove_relaxation()
                        self._build_mccormicks()
                    self._update_mccormicks()
                else:
                    self._remove_relaxation()
                    del self._pw
                    self._pw = pe.Block(concrete=True)
                    _build_pw_mccormick_relaxation(b=self._pw, x1=self._x1, x2=self._x2, aux_var=self._aux_var,
                                                   x1_pts=self._partitions[self._x1],
                                                   relaxation_side=self.relaxation_side, safety_tol=self.safety_tol)
            else:
                self._remove_relaxation()

    def _build_mccormicks(self):
        del self._mc_index, self._v_index, self._slopes_index, self._slopes, self._intercepts, self._mccormicks
        self._mc_exprs = dict()
        self._mc_index = pe.Set(initialize=[0, 1, 2, 3])
        self._v_index = pe.Set(initialize=[1, 2])
        self._slopes_index = pe.Set(initialize=self._mc_index * self._v_index)
        self._slopes = IndexedParam(self._slopes_index, mutable=True)
        self._intercepts = IndexedParam(self._mc_index, mutable=True)
        self._mccormicks = IndexedConstraint(self._mc_index)

        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}:
            for ndx in [0, 1]:
                e = LinearExpression(constant=self._intercepts[ndx],
                                     linear_coefs=[self._slopes[ndx, 1], self._slopes[ndx, 2]],
                                     linear_vars=[self._x1, self._x2])
                self._mc_exprs[ndx] = e
                self._mccormicks[ndx] = self._aux_var >= e

        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            for ndx in [2, 3]:
                e = LinearExpression(constant=self._intercepts[ndx],
                                     linear_coefs=[self._slopes[ndx, 1], self._slopes[ndx, 2]],
                                     linear_vars=[self._x1, self._x2])
                self._mc_exprs[ndx] = e
                self._mccormicks[ndx] = self._aux_var <= e

    def _check_expr(self, ndx):
        if ndx in {0, 1}:
            rel_side = RelaxationSide.UNDER
        else:
            rel_side = RelaxationSide.OVER
        success, bad_var, bad_coef, err_msg = _check_cut(self._mc_exprs[ndx], too_small=self.small_coef,
                                                         too_large=self.large_coef, relaxation_side=rel_side,
                                                         safety_tol=self.safety_tol)
        if not success:
            self._log_bad_cut(bad_var, bad_coef, err_msg)
            self._mccormicks[ndx].deactivate()
        else:
            self._mccormicks[ndx].activate()

    def _update_mccormicks(self):
        x1_lb, x1_ub = _get_bnds_tuple(self._x1)
        x2_lb, x2_ub = _get_bnds_tuple(self._x2)

        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}:
            self._slopes[0, 1]._value = x2_lb
            self._slopes[0, 2]._value = x1_lb
            self._intercepts[0]._value = -x1_lb * x2_lb

            self._slopes[1, 1]._value = x2_ub
            self._slopes[1, 2]._value = x1_ub
            self._intercepts[1]._value = -x1_ub * x2_ub

            self._check_expr(0)
            self._check_expr(1)

        if self.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}:
            self._slopes[2, 1]._value = x2_lb
            self._slopes[2, 2]._value = x1_ub
            self._intercepts[2]._value = -x1_ub * x2_lb

            self._slopes[3, 1]._value = x2_ub
            self._slopes[3, 2]._value = x1_lb
            self._intercepts[3]._value = -x1_lb * x2_ub

            self._check_expr(2)
            self._check_expr(3)

    def add_partition_point(self, value=None):
        """
        This method adds one point to the partitioning of x1. If value is not
        specified, a single point will be added to the partitioning of x1 at the current value of x1. If value is
        specified, then value is added to the partitioning of x1.

        Parameters
        ----------
        value: float
            The point to be added to the partitioning of x1.
        """
        self._add_partition_point(self._x1, value)

    def is_rhs_convex(self):
        """
        Returns True if linear underestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False

    def is_rhs_concave(self):
        """
        Returns True if linear overestimators do not need binaries. Otherwise, returns False.

        Returns
        -------
        bool
        """
        return False
