from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import enum
import pyomo.environ as pe
from pyomo.core.expr.numvalue import is_fixed
import math
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import numpy as np
from coramin.utils.coramin_enums import EigenValueBounder
from pyomo.core.base.block import _BlockData
from typing import Optional, MutableMapping
from pyomo.core.expr.numeric_expr import ExpressionBase
from pyomo.contrib import appsi
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.var import _GeneralVarData
from coramin.utils.pyomo_utils import simplify_expr


def _2d_determinant(mat: np.ndarray):
    return mat[0, 0] * mat[1, 1] - mat[1, 0] * mat[0, 1]


def _determinant(mat):
    nrows, ncols = mat.shape
    assert nrows == ncols
    if nrows == 1:
        det = mat[0, 0]
    elif nrows == 2:
        det = _2d_determinant(mat)
    else:
        i = 0
        det = 0
        next_rows = np.array(list(range(i+1, nrows)), dtype=int)
        for j in range(nrows):
            next_cols = [k for k in range(j)]
            next_cols.extend(k for k in range(j+1, nrows))
            next_cols = np.array(next_cols, dtype=int)
            next_mat = mat[next_rows, :]
            next_mat = next_mat[:, next_cols]
            det += (-1)**(i + j) * mat[i, j] * _determinant(next_mat)
    return simplify_expr(det)


class Hessian(object):
    def __init__(
        self,
        expr: ExpressionBase,
        opt: Optional[appsi.base.Solver],
        method: EigenValueBounder = EigenValueBounder.LinearProgram,
    ):
        self.method = EigenValueBounder(method)
        self.opt = opt
        self._expr = expr
        self._var_list = list(identify_variables(expr=expr, include_fixed=False))
        self._ndx_map = pe.ComponentMap(
            (v, ndx) for ndx, v in enumerate(self._var_list)
        )
        self._hessian = self.compute_symbolic_hessian()
        self._eigenvalue_problem: Optional[_BlockData] = None
        self._eigenvalue_relaxation: Optional[_BlockData] = None
        self._orig_to_relaxation_vars: Optional[
            MutableMapping[_GeneralVarData, _GeneralVarData]
        ] = None

    def variables(self):
        return tuple(self._var_list)

    def formulate_eigenvalue_problem(self, sense=pe.minimize):
        if self._eigenvalue_problem is not None:
            min_eig, max_eig = self.bound_eigenvalues_from_interval_hessian()
            if min_eig > self._eigenvalue_problem.eig.lb:
                self._eigenvalue_problem.eig.setlb(min_eig)
            if max_eig < self._eigenvalue_problem.eig.ub:
                self._eigenvalue_problem.eig.setub(max_eig)
            self._eigenvalue_problem.obj.sense = sense
            return self._eigenvalue_problem
        min_eig, max_eig = self.bound_eigenvalues_from_interval_hessian()
        m = pe.ConcreteModel()
        m.eig = pe.Var(bounds=(min_eig, max_eig))
        m.obj = pe.Objective(expr=m.eig, sense=sense)
        for v in self._var_list:
            m.add_component(v.name, pe.Reference(v))

        n = len(self._var_list)
        np_hess = np.empty((n, n), dtype=object)
        for ndx1, v1 in enumerate(self._var_list):
            hess_v1 = self._hessian[v1]
            for ndx2 in range(ndx1, n):
                v2 = self._var_list[ndx2]
                if v2 in hess_v1:
                    np_hess[ndx1, ndx2] = hess_v1[v2]
                else:
                    np_hess[ndx1, ndx2] = 0
                if v1 is v2:
                    np_hess[ndx1, ndx2] -= m.eig
                else:
                    np_hess[ndx2, ndx1] = np_hess[ndx1, ndx2]
        m.det_con = pe.Constraint(expr=_determinant(np_hess) == 0)
        self._eigenvalue_problem = m
        return m

    def formulate_eigenvalue_relaxation(self, sense=pe.minimize):
        if self._eigenvalue_relaxation is not None:
            for orig_v, rel_v in self._orig_to_relaxation_vars.items():
                orig_lb, orig_ub = orig_v.bounds
                rel_lb, rel_ub = rel_v.bounds
                if orig_lb is not None:
                    if rel_lb is None or orig_lb > rel_lb:
                        rel_v.setlb(orig_lb)
                if orig_ub is not None:
                    if rel_ub is None or orig_ub < rel_ub:
                        rel_v.setub(orig_ub)
            from .iterators import relaxation_data_objects
            for b in relaxation_data_objects(self._eigenvalue_relaxation, descend_into=True, active=True):
                b.rebuild()
            self._eigenvalue_relaxation.obj.sense = sense
            return self._eigenvalue_relaxation
        m = self.formulate_eigenvalue_problem(sense=sense)
        all_vars = list(
            ComponentSet(
                m.component_data_objects(pe.Var, descend_into=True)
            )
        )
        tmp_name = unique_component_name(m, "all_vars")
        setattr(m, tmp_name, all_vars)
        from .auto_relax import relax
        relaxation = relax(m, in_place=False)
        new_vars = getattr(relaxation, "all_vars")
        self._orig_to_relaxation_vars = pe.ComponentMap(zip(all_vars, new_vars))
        delattr(m, tmp_name)
        delattr(relaxation, tmp_name)
        self._eigenvalue_relaxation = relaxation
        return relaxation

    def get_minimum_eigenvalue(self):
        if self.method <= EigenValueBounder.GershgorinWithSimplification:
            res = self.bound_eigenvalues_from_interval_hessian()[0]
        elif self.method == EigenValueBounder.LinearProgram:
            m = self.formulate_eigenvalue_relaxation()
            res = self.opt.solve(m).best_objective_bound
        else:
            m = self.formulate_eigenvalue_problem()
            res = self.opt.solve(m).best_objective_bound
        return res

    def get_maximum_eigenvalue(self):
        if self.method <= EigenValueBounder.GershgorinWithSimplification:
            res = self.bound_eigenvalues_from_interval_hessian()[1]
        elif self.method == EigenValueBounder.LinearProgram:
            m = self.formulate_eigenvalue_relaxation(sense=pe.maximize)
            res = self.opt.solve(m).best_objective_bound
        else:
            m = self.formulate_eigenvalue_problem(sense=pe.maximize)
            res = self.opt.solve(m).best_objective_bound
        return res

    def bound_eigenvalues_from_interval_hessian(self):
        ih = self.compute_interval_hessian()
        min_eig = math.inf
        max_eig = -math.inf
        for v1 in self._var_list:
            h = ih[v1]
            if v1 in h:
                row_min = h[v1][0]
                row_max = h[v1][1]
            else:
                row_min = 0
                row_max = 0
            for v2, (lb, ub) in h.items():
                if v2 is v1:
                    continue
                row_min -= max(abs(lb), abs(ub))
                row_max += max(abs(lb), abs(ub))
            min_eig = min(min_eig, row_min)
            max_eig = max(max_eig, row_max)
        return min_eig, max_eig

    def compute_symbolic_hessian(self):
        ders = reverse_sd(self._expr)
        ders2 = pe.ComponentMap()
        for v in self._var_list:
            ders2[v] = reverse_sd(ders[v])

        res = pe.ComponentMap()
        for v in self._var_list:
            res[v] = pe.ComponentMap()

        n = len(self._var_list)
        for v1 in self._var_list:
            ndx1 = self._ndx_map[v1]
            for ndx2 in range(ndx1, n):
                v2 = self._var_list[ndx2]
                if v2 not in ders2[v1]:
                    continue
                der = ders2[v1][v2]
                if is_fixed(der):
                    val = pe.value(der)
                    res[v1][v2] = val
                else:
                    if self.method >= EigenValueBounder.GershgorinWithSimplification:
                        _der = simplify_expr(der)
                    else:
                        _der = der
                    res[v1][v2] = _der
                res[v2][v1] = res[v1][v2]

        return res

    def compute_interval_hessian(self):
        res = pe.ComponentMap()
        for v in self._var_list:
            res[v] = pe.ComponentMap()

        n = len(self._var_list)
        for ndx1, v1 in enumerate(self._var_list):
            for ndx2 in range(ndx1, n):
                v2 = self._var_list[ndx2]
                if v2 not in self._hessian[v1]:
                    continue
                lb, ub = compute_bounds_on_expr(self._hessian[v1][v2])
                if lb is None:
                    lb = -math.inf
                if ub is None:
                    ub = math.inf
                res[v1][v2] = (lb, ub)
                res[v2][v1] = (lb, ub)
        return res

    def pprint(self, intervals=False):
        if intervals:
            ih = self.compute_interval_hessian()
        else:
            ih = self._hessian
        n = len(self._var_list)
        lens = np.ones((n, n), dtype=int)
        strs = dict()
        for ndx in range(n):
            strs[ndx] = dict()

        for v1 in self._var_list:
            ndx1 = self._ndx_map[v1]
            for ndx2 in range(ndx1, n):
                v2 = self._var_list[ndx2]
                if v2 in ih[v1]:
                    der = ih[v1][v2]
                else:
                    if intervals:
                        der = (0, 0)
                    else:
                        der = 0
                if intervals:
                    lb, ub = der
                    der_str = f"({lb:<.3f}, {ub:<.3f})"
                else:
                    der_str = str(der)
                strs[ndx1][ndx2] = der_str
                strs[ndx2][ndx1] = der_str
                lens[ndx1, ndx2] = len(der_str)
                lens[ndx2, ndx1] = len(der_str)

        col_lens = np.max(lens, axis=0)
        row_string = ""
        for ndx, cl in enumerate(col_lens):
            row_string += f"{{{ndx}:<{cl+2}}}"

        res = ""
        for row_ndx in range(n):
            row_entries = tuple(strs[row_ndx][i] for i in range(n))
            res += row_string.format(*row_entries)
            res += '\n'

        print(res)
