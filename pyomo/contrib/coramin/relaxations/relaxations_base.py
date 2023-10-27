from pyomo.core.base.block import _BlockData, Block
from .custom_block import declare_custom_block
import weakref
import pyomo.environ as pe
from collections.abc import Iterable
from pyomo.common.collections import ComponentSet, ComponentMap
from coramin.utils.coramin_enums import FunctionShape, RelaxationSide
import warnings
import logging
import math
from ._utils import _get_bnds_list, _get_bnds_tuple
import sys
from pyomo.core.expr import taylor_series_expansion
from typing import Sequence, Dict, Tuple, Optional, Union, Mapping, MutableMapping, List
from pyomo.core.base.param import IndexedParam, ScalarParam, _ParamData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.numeric_expr import LinearExpression, ExpressionBase
from pyomo.core.base.constraint import IndexedConstraint, ScalarConstraint, _GeneralConstraintData
from pyomo.contrib.fbbt import interval

pyo = pe
logger = logging.getLogger(__name__)

"""
Base classes for relaxations
"""


class _OACut(object):
    def __init__(self,
                 nonlin_expr,
                 expr_vars: Sequence[_GeneralVarData],
                 coefficients: Sequence[_ParamData],
                 offset: _ParamData):
        self.expr_vars = expr_vars
        self.nonlin_expr = nonlin_expr
        self.coefficients = coefficients
        self.offset = offset
        derivs = reverse_sd(self.nonlin_expr)
        self.derivs = [derivs[i] for i in self.expr_vars]
        self.cut_expr = LinearExpression(constant=self.offset,
                                         linear_coefs=self.coefficients,
                                         linear_vars=self.expr_vars)
        self.current_pt = None

    def update(self,
               var_vals: Sequence[float],
               relaxation_side: RelaxationSide,
               too_small: float,
               too_large: float,
               safety_tol: float) -> Tuple[bool, Optional[_GeneralVarData], Optional[float], Optional[str]]:
        res = (True, None, None, None)
        self.current_pt = var_vals
        orig_values = [i.value for i in self.expr_vars]
        for v, val in zip(self.expr_vars, var_vals):
            v.set_value(val, skip_validation=True)
        try:
            offset_val = pe.value(self.nonlin_expr)
            for ndx, v in enumerate(self.expr_vars):
                der = pe.value(self.derivs[ndx])
                offset_val -= der * v.value
                self.coefficients[ndx]._value = der
            self.offset._value = offset_val
        except (OverflowError, ValueError, ZeroDivisionError) as e:
            res = (False, None, None, str(e))
        finally:
            for v, val in zip(self.expr_vars, orig_values):
                v.set_value(val, skip_validation=True)
        if res[0]:
            res = _check_cut(self.cut_expr, too_small=too_small, too_large=too_large, relaxation_side=relaxation_side,
                             safety_tol=safety_tol)
        return res

    def __repr__(self):
        pt_str = {str(v): p for v, p in zip(self.expr_vars, self.current_pt)}
        pt_str = str(pt_str)
        s = f'OA Cut at {pt_str}'
        return s

    def __str__(self):
        return self.__repr__()


def _check_cut(cut: LinearExpression, too_small, too_large, relaxation_side, safety_tol):
    res = (True, None, None, None)
    for coef_p, v in zip(cut.linear_coefs, cut.linear_vars):
        coef = coef_p.value
        if not math.isfinite(coef) or abs(coef) >= too_large:
            res = (False, v, coef, None)
        elif 0 < abs(coef) <= too_small and v.has_lb() and v.has_ub():
            coef_p._value = 0
            if relaxation_side == RelaxationSide.UNDER:
                cut.constant._value = interval.add(cut.constant.value, cut.constant.value,
                                                   *interval.mul(v.lb, v.ub, coef, coef))[0]
            elif relaxation_side == RelaxationSide.OVER:
                cut.constant._value = interval.add(cut.constant.value, cut.constant.value,
                                                   *interval.mul(v.lb, v.ub, coef, coef))[1]
            else:
                raise ValueError('relaxation_side should be either UNDER or OVER')
    if relaxation_side == RelaxationSide.UNDER:
        cut.constant._value -= safety_tol
    else:
        cut.constant._value += safety_tol
    if not math.isfinite(cut.constant.value) or abs(cut.constant.value) >= too_large:
        res = (False, None, cut.constant.value, None)
    return res


@declare_custom_block(name='BaseRelaxation')
class BaseRelaxationData(_BlockData):
    def __init__(self, component):
        _BlockData.__init__(self, component)
        self._relaxation_side = RelaxationSide.BOTH
        self._use_linear_relaxation = True
        self._large_coef = 1e5
        self._small_coef = 1e-10
        self._needs_rebuilt = True
        self.safety_tol = 1e-10

        self._oa_points: Dict[Tuple[float, ...], _OACut] = dict()
        self._oa_param_indices: MutableMapping[_ParamData, int] = pe.ComponentMap()
        self._current_param_index = 0
        self._oa_params: Optional[IndexedParam] = None
        self._cuts: Optional[IndexedConstraint] = None

        self._saved_oa_points = list()
        self._oa_stack_map = dict()

        self._original_constraint: Optional[ScalarConstraint] = None
        self._nonlinear: Optional[ScalarConstraint] = None

    def set_input(self, relaxation_side=RelaxationSide.BOTH,
                  use_linear_relaxation=True, large_coef=1e5,
                  small_coef=1e-10, safety_tol=1e-10):
        self.relaxation_side = relaxation_side
        self.use_linear_relaxation = use_linear_relaxation
        self._large_coef = large_coef
        self._small_coef = small_coef
        self.safety_tol = safety_tol
        self._needs_rebuilt = True

        self.clear_oa_points()
        self._saved_oa_points = list()
        self._oa_stack_map = dict()

    def get_aux_var(self) -> _GeneralVarData:
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns w

        Returns
        -------
        aux_var: pyomo.core.base.var._GeneralVarData
            The variable representing w in w = f(x) (which is the constraint being relaxed).
        """
        return self._aux_var

    def get_rhs_vars(self) -> Tuple[_GeneralVarData, ...]:
        raise NotImplementedError('This method should be implemented by subclasses')

    def get_rhs_expr(self) -> ExpressionBase:
        raise NotImplementedError('This method should be implemented by subclasses')

    def _get_expr_for_oa(self):
        return self.get_rhs_expr()

    @property
    def small_coef(self):
        return self._small_coef

    @small_coef.setter
    def small_coef(self, val):
        self._small_coef = val

    @property
    def large_coef(self):
        return self._large_coef

    @large_coef.setter
    def large_coef(self, val):
        self._large_coef = val

    @property
    def use_linear_relaxation(self) -> bool:
        """
        If this is True, the relaxation will use a linear relaxation. If False, then a nonlinear relaxation may be used.
        Take x^2 for example, the underestimator can be quadratic.

        Returns
        -------
        bool
        """
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, val: bool):
        if not val:
            raise ValueError('Relaxations of type {0} do not support relaxations that are not linear.'.format(type(self)))

    def remove_relaxation(self):
        """
        Remove any auto-created vars/constraints from the relaxation block
        """
        del self._cuts
        self._cuts = None
        del self._original_constraint
        self._original_constraint = None
        del self._nonlinear
        self._nonlinear = None

    def _has_a_convex_side(self):
        if self.has_convex_underestimator() and self.relaxation_side in {RelaxationSide.UNDER, RelaxationSide.BOTH}:
            return True
        if self.has_concave_overestimator() and self.relaxation_side in {RelaxationSide.OVER, RelaxationSide.BOTH}:
            return True
        return False

    def _check_valid_domain_for_relaxation(self) -> bool:
        for v in self.get_rhs_vars():
            lb, ub = _get_bnds_tuple(v)
            if not math.isfinite(lb) or not math.isfinite(ub):
                return False
        return True

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        # we have to ensure only one of
        #    - self._cuts
        #    - self._nonlinear
        #    - self._original_constraint
        # is ever not None at one time
        needs_rebuilt = self._needs_rebuilt
        if build_nonlinear_constraint:
            if self._original_constraint is None:
                needs_rebuilt = True
        else:
            if self.use_linear_relaxation:
                if self._nonlinear is not None or self._original_constraint is not None:
                    needs_rebuilt = True
            else:
                if self._cuts is not None or self._original_constraint is not None:
                    needs_rebuilt = True

        if needs_rebuilt:
            self.remove_relaxation()

        self._needs_rebuilt = False

        if build_nonlinear_constraint and self._original_constraint is None:
            del self._original_constraint
            if self.relaxation_side == RelaxationSide.BOTH:
                self._original_constraint = pe.Constraint(expr=self.get_aux_var() == self.get_rhs_expr())
            elif self.relaxation_side == RelaxationSide.UNDER:
                self._original_constraint = pe.Constraint(expr=self.get_aux_var() >= self.get_rhs_expr())
            else:
                self._original_constraint = pe.Constraint(expr=self.get_aux_var() <= self.get_rhs_expr())
        else:
            if self._has_a_convex_side():
                if self.use_linear_relaxation:
                    if self._cuts is None:
                        del self._cuts
                        self._cuts = IndexedConstraint(pe.Any)
                    if self._oa_params is None:
                        del self._oa_params
                        self._oa_params = IndexedParam(pe.Any, mutable=True)
                    self.clean_oa_points(ensure_oa_at_vertices=ensure_oa_at_vertices)
                    self._update_oa_cuts()
                else:
                    if self._nonlinear is None:
                        del self._nonlinear
                        if self.has_convex_underestimator():
                            self._nonlinear = pe.Constraint(expr=self.get_aux_var() >= self._get_expr_for_oa() - self.safety_tol)
                        else:
                            assert self.has_concave_overestimator()
                            self._nonlinear = pe.Constraint(expr=self.get_aux_var() <= self._get_expr_for_oa() + self.safety_tol)

    def vars_with_bounds_in_relaxation(self):
        """
        This method returns a list of variables whose bounds appear in the constraints defining the relaxation.
        Take the McCormick relaxation of a bilinear term (w = x * y) for example. The McCormick relaxation is

        w >= xl * y + x * yl - xl * yl
        w >= xu * y + x * yu - xu * yu
        w <= xu * y + x * yl - xu * yl
        w <= x * yu + xl * y - xl * yu

        where xl and xu are the lower and upper bounds for x, respectively, and yl and yu are the lower and upper
        bounds for y, respectively. Because xl, xu, yl, and yu appear in the constraints, this method would return

        [x, y]

        As another example, take w >= x**2. A linear relaxation of this constraint just involves linear underestimators,
        which do not depend on the bounds of x or w. Therefore, this method would return an empty list.
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def get_deviation(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns

        max{f(x) - w, 0} if relaxation_side is RelaxationSide.UNDER
        max{w - f(x), 0} if relaxation_side is RelaxationSide.OVER
        abs(w - f(x)) if relaxation_side is RelaxationSide.BOTH

        Returns
        -------
        float
        """
        dev = self.get_aux_var().value - pe.value(self.get_rhs_expr())
        if self.relaxation_side is RelaxationSide.BOTH:
            dev = abs(dev)
        elif self.relaxation_side is RelaxationSide.UNDER:
            dev = max(-dev, 0)
        else:
            dev = max(dev, 0)
        return dev

    def is_rhs_convex(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns True if f(x)
        is convex and False otherwise.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def is_rhs_concave(self):
        """
        All Coramin relaxations are relaxations of constraints of the form w <=/=/>= f(x). This method returns True if f(x)
        is concave and False otherwise.

        Returns
        -------
        bool
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def has_convex_underestimator(self):
        return self.is_rhs_convex()

    def has_concave_overestimator(self):
        return self.is_rhs_concave()

    @property
    def relaxation_side(self):
        return self._relaxation_side

    @relaxation_side.setter
    def relaxation_side(self, val):
        if val not in RelaxationSide:
            raise ValueError('{0} is not a valid member of RelaxationSide'.format(val))
        if val != self._relaxation_side:
            self._needs_rebuilt = True
        self._relaxation_side = val

    def _get_pprint_string(self):
        if self.relaxation_side == RelaxationSide.BOTH:
            relational_operator_string = '=='
        elif self.relaxation_side == RelaxationSide.UNDER:
            relational_operator_string = '>='
        elif self.relaxation_side == RelaxationSide.OVER:
            relational_operator_string = '<='
        else:
            raise ValueError('Unexpected relaxation side')
        return f'Relaxation for {self.get_aux_var().name} {relational_operator_string} {str(self.get_rhs_expr())}'

    def pprint(self, ostream=None, verbose=False, prefix=""):
        if ostream is None:
            ostream = sys.stdout

        ostream.write('{0}{1}: {2}\n'.format(prefix, self.name, self._get_pprint_string()))

        if verbose:
            super(BaseRelaxationData, self).pprint(ostream=ostream,
                                                   verbose=verbose, prefix=(prefix + '  '))

    def _get_oa_cut(self) -> _OACut:
        rhs_vars = self.get_rhs_vars()
        coef_params = list()
        for v in rhs_vars:
            p = self._oa_params[self._current_param_index]
            self._oa_param_indices[p] = self._current_param_index
            coef_params.append(p)
            self._current_param_index += 1
        offset_param = self._oa_params[self._current_param_index]
        self._oa_param_indices[offset_param] = self._current_param_index
        self._current_param_index += 1
        oa_cut = _OACut(self._get_expr_for_oa(), rhs_vars, coef_params, offset_param)
        return oa_cut

    def _remove_oa_cut(self, oa_cut: _OACut):
        for p in oa_cut.coefficients:
            del self._oa_params[self._oa_param_indices[p]]
            del self._oa_param_indices[p]
        del self._oa_params[self._oa_param_indices[oa_cut.offset]]
        del self._oa_param_indices[oa_cut.offset]
        if oa_cut in self._cuts:  # if the cut did not pass _check_cut, it won't be in self._cuts
            del self._cuts[oa_cut]

    def _log_bad_cut(self, fail_var, fail_coef, err_msg):
        if fail_var is None and fail_coef is None:
            logger.debug(f'Encountered exception when adding OA cut '
                         f'for "{self._get_pprint_string()}"; Error message: {err_msg}')
        elif fail_var is None:
            logger.debug(f'Skipped OA cut for "{self._get_pprint_string()}" due to a '
                         f'large constant value: {fail_coef}')
        else:
            logger.debug(f'Skipped OA cut for "{self._get_pprint_string()}" due to a '
                         f'small or large coefficient for {str(fail_var)}: {fail_coef}')

    def _add_oa_cut(self, pt_tuple: Tuple[float, ...], oa_cut: _OACut) -> Optional[_GeneralConstraintData]:
        if self._nonlinear is not None or self._original_constraint is not None:
            raise ValueError('Can only add an OA cut when using a linear relaxation')
        if self.has_convex_underestimator():
            rel_side = RelaxationSide.UNDER
        else:
            assert self.has_concave_overestimator()
            rel_side = RelaxationSide.OVER
        cut_info = oa_cut.update(var_vals=pt_tuple, relaxation_side=rel_side,
                                 too_small=self.small_coef, too_large=self.large_coef,
                                 safety_tol=self.safety_tol)
        success, fail_var, fail_coef, err_msg = cut_info
        if not success:
            self._log_bad_cut(fail_var, fail_coef, err_msg)
            if oa_cut in self._cuts:
                del self._cuts[oa_cut]
        else:
            if oa_cut not in self._cuts:
                if self.has_convex_underestimator():
                    self._cuts[oa_cut] = self.get_aux_var() >= oa_cut.cut_expr
                else:
                    self._cuts[oa_cut] = self.get_aux_var() <= oa_cut.cut_expr
                return self._cuts[oa_cut]
        return None

    def _update_oa_cuts(self):
        for pt_tuple, oa_cut in self._oa_points.items():
            self._add_oa_cut(pt_tuple, oa_cut)

        # remove any cuts that may have been added with add_cut(keep_cut=False)
        all_oa_cuts = set(self._oa_points.values())
        for oa_cut in list(self._cuts):
            if oa_cut not in all_oa_cuts:
                self._remove_oa_cut(oa_cut)

    def _add_oa_point(self, pt_tuple: Tuple[float, ...]):
        if pt_tuple not in self._oa_points:
            self._oa_points[pt_tuple] = self._get_oa_cut()

    def add_oa_point(self, var_values: Optional[Union[Tuple[float, ...], Mapping[_GeneralVarData, float]]] = None):
        """
        Add a point at which an outer-approximation cut for a convex constraint should be added. This does not
        rebuild the relaxation. You must call rebuild() for the constraint to get added.

        Parameters
        ----------
        var_values: Optional[Union[Tuple[float, ...], Mapping[_GeneralVarData, float]]]
        """
        if self._has_a_convex_side():
            if var_values is None:
                var_values = tuple(v.value for v in self.get_rhs_vars())
            elif type(var_values) is tuple:
                pass
            else:
                var_values = tuple(var_values[v] for v in self.get_rhs_vars())
            self._add_oa_point(var_values)

    def push_oa_points(self, key=None):
        """
        Save the current list of OA points for later use through pop_oa_points().
        """
        to_save = [i for i in self._oa_points.keys()]
        if key is not None:
            self._oa_stack_map[key] = to_save
        else:
            self._saved_oa_points.append(to_save)

    def clear_oa_points(self):
        """
        Delete any existing OA points.
        """
        self._oa_points = dict()
        self._oa_param_indices = pe.ComponentMap()
        self._current_param_index = 0
        if self._oa_params is not None:
            del self._oa_params
            self._oa_params = pe.Param(pe.Any, mutable=True)
        if self._cuts is not None:
            del self._cuts
            self._cuts = pe.Constraint(pe.Any)

    def pop_oa_points(self, key=None):
        """
        Use the most recently saved list of OA points
        """
        self.clear_oa_points()
        if key is None:
            list_of_points = self._saved_oa_points.pop(-1)
        else:
            list_of_points = self._oa_stack_map.pop(key)
        for pt_tuple in list_of_points:
            self._add_oa_point(pt_tuple)

    def add_cut(self, keep_cut=True, check_violation=True, feasibility_tol=1e-8) -> Optional[_GeneralConstraintData]:
        """
        This function will add a linear cut to the relaxation. Cuts are only generated for the convex side of the
        constraint (if the constraint has a convex side). For example, if the relaxation is a PWXSquaredRelaxationData
        for y = x**2, the add_cut will add an underestimator at x.value (but only if y.value < x.value**2). If
        relaxation is a PWXSquaredRelaxationData for y < x**2, then no cut will be added. If relaxation is is a
        PWMcCormickRelaxationData, then no cut will be added.

        Parameters
        ----------
        keep_cut: bool
            If keep_cut is True, then add_oa_point will also be called. Be careful if the relaxation object is relaxing
            the nonconvex side of the constraint. Thus, the cut will be reconstructed when rebuild is called. If
            keep_cut is False, then the cut will be discarded when rebuild is called.
        check_violation: bool
            If True, then a cut is only added if the cut generated would cut off the current point (current values
            of the variables) by more than feasibility_tol.
        feasibility_tol: float
            Only used if check_violation is True

        Returns
        -------
        new_con: pyomo.core.base.constraint._GeneralConstraintData
        """
        rhs_vars = self.get_rhs_vars()
        var_vals = tuple(v.value for v in rhs_vars)

        if var_vals in self._oa_points:
            return None

        new_con = None
        if self._has_a_convex_side():
            if check_violation:
                needs_cut = False
                try:
                    rhs_val = pe.value(self._get_expr_for_oa())
                except (OverflowError, ZeroDivisionError, ValueError):
                    rhs_val = None
                if rhs_val is not None:
                    if self.has_convex_underestimator():
                        viol = rhs_val - self.get_aux_var().value
                    else:
                        viol = self.get_aux_var().value - rhs_val
                    if viol > feasibility_tol:
                        needs_cut = True
            else:
                needs_cut = True
            if needs_cut:
                oa_cut = self._get_oa_cut()
                new_con = self._add_oa_cut(pt_tuple=var_vals, oa_cut=oa_cut)
                if keep_cut:
                    self._oa_points[var_vals] = oa_cut

        return new_con

    def clean_oa_points(self, ensure_oa_at_vertices=True):
        if not self._has_a_convex_side():
            return

        rhs_vars = self.get_rhs_vars()
        bnds_list: List[Tuple[float, float]] = list()
        for v in rhs_vars:
            bnds_list.append(_get_bnds_tuple(v))

        for pt_tuple, oa_cut in list(self._oa_points.items()):
            new_pt_list = list()
            for (v_lb, v_ub), pt in zip(bnds_list, pt_tuple):
                if pt < v_lb:
                    new_pt_list.append(v_lb)
                elif pt > v_ub:
                    new_pt_list.append(v_ub)
                else:
                    new_pt_list.append(pt)
            new_pt_tuple = tuple(new_pt_list)
            del self._oa_points[pt_tuple]
            if new_pt_tuple in self._oa_points:
                self._remove_oa_cut(oa_cut)
            else:
                self._oa_points[new_pt_tuple] = oa_cut
        if ensure_oa_at_vertices:
            lb_list = list()
            ub_list = list()
            for lb, ub in bnds_list:
                if math.isfinite(lb) and math.isfinite(ub):
                    lb_list.append(lb)
                    ub_list.append(ub)
                elif math.isfinite(lb):
                    lb_list.append(lb)
                    ub_list.append(max(lb + 1, 1))
                elif math.isfinite(ub):
                    lb_list.append(min(ub - 1, -1))
                    ub_list.append(ub)
                else:
                    lb_list.append(-1)
                    ub_list.append(1)
            lb_tuple = tuple(lb_list)
            ub_tuple = tuple(ub_list)
            if lb_tuple not in self._oa_points:
                if len(self._oa_points) <= 1:
                    self._add_oa_point(lb_tuple)
                else:  # move the smallest point to lb_tuple
                    min_pt = min(self._oa_points.keys())
                    min_oa_cut = self._oa_points[min_pt]
                    del self._oa_points[min_pt]
                    self._oa_points[lb_tuple] = min_oa_cut
            if ub_tuple not in self._oa_points:
                if len(self._oa_points) <= 1:
                    self._add_oa_point(ub_tuple)
                else: # move the largest point to ub_tuple
                    max_pt = max(self._oa_points.keys())
                    max_oa_cut = self._oa_points[max_pt]
                    del self._oa_points[max_pt]
                    self._oa_points[ub_tuple] = max_oa_cut


@declare_custom_block(name='BasePWRelaxation')
class BasePWRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        BaseRelaxationData.__init__(self, component)

        self._partitions = ComponentMap()  # ComponentMap: var: list of float
        self._saved_partitions = list()  # list of CompnentMap

    def rebuild(self, build_nonlinear_constraint=False, ensure_oa_at_vertices=True):
        """
        Remove any auto-created vars/constraints from the relaxation block and recreate it
        """
        super(BasePWRelaxationData, self).rebuild(build_nonlinear_constraint=build_nonlinear_constraint,
                                                  ensure_oa_at_vertices=ensure_oa_at_vertices)
        self.clean_partitions()

    def set_input(self, relaxation_side=RelaxationSide.BOTH, use_linear_relaxation=True, large_coef=1e5,
                  small_coef=1e-10, safety_tol=1e-10):
        super(BasePWRelaxationData, self).set_input(relaxation_side=relaxation_side,
                                                    use_linear_relaxation=use_linear_relaxation, large_coef=large_coef,
                                                    small_coef=small_coef, safety_tol=safety_tol)
        self._partitions = ComponentMap()
        self._saved_partitions = list()

    def add_partition_point(self):
        """
        Add a point to the current partitioning. This does not rebuild the relaxation. You must call rebuild()
        to rebuild the relaxation.
        """
        raise NotImplementedError('This method should be implemented in the derived class.')

    def _add_partition_point(self, var, value=None):
        if value is None:
            value = pe.value(var)
        # if the point is outside the variable's bounds, then it will simply get removed when clean_partitions
        # gets called.
        self._partitions[var].append(value)

    def push_partitions(self):
        """
        Save the current partitioning for later use through pop_partitions().
        """
        self._saved_partitions.append(pe.ComponentMap((k, list(v)) for k, v in self._partitions.items()))

    def clear_partitions(self):
        """
        Delete any existing partitioning scheme.
        """
        tmp = ComponentMap()
        for var, pts in self._partitions.items():
            tmp[var] = [pe.value(var.lb), pe.value(var.ub)]
        self._partitions = tmp

    def pop_partitions(self):
        """
        Use the most recently saved partitioning.
        """
        self._partitions = self._saved_partitions.pop(-1)

    def clean_partitions(self):
        # discard any points in the partitioning that are not within the variable bounds
        for var, pts in list(self._partitions.items()):
            pts = list(set(pts))
            pts.sort()
            self._partitions[var] = pts

        for var, pts in self._partitions.items():
            lb, ub = tuple(_get_bnds_list(var))

            new_pts = list()
            new_pts.append(lb)
            for val in pts[1:-1]:
                if lb < val < ub:
                    new_pts.append(val)
            new_pts.append(ub)
            self._partitions[var] = new_pts

    def get_active_partitions(self):
        ans = ComponentMap()
        for var, pts in self._partitions.items():
            val = pyo.value(var)
            lower = None
            upper = None
            if not (pts[0] - 1e-6 <= val <= pts[-1] + 1e-6):
                raise ValueError('The variable value must be within the variable bounds')
            if val < pts[0]:
                lower = pts[0]
                upper = pts[1]
            elif val > pts[-1]:
                lower = pts[-2]
                upper = pts[-1]
            else:
                for p1, p2 in zip(pts[0:-1], pts[1:]):
                    if p1 <= val <= p2:
                        lower = p1
                        upper = p2
                        break
            assert lower is not None
            assert upper is not None
            ans[var] = lower, upper
        return ans


class ComponentWeakRef(object):
    """
    This object is used to reference components from a block that are not owned by that block.
    """
    # ToDo: Example in the documentation
    def __init__(self, comp):
        self.compref = None
        self.set_component(comp)

    def get_component(self):
        if self.compref is None:
            return None
        return self.compref()

    def set_component(self, comp):
        self.compref = None
        if comp is not None:
            self.compref = weakref.ref(comp)

    def __setstate__(self, state):
        self.set_component(state['compref'])

    def __getstate__(self):
        return {'compref': self.get_component()}
