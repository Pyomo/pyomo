# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Sequence, Mapping
from collections.abc import Iterable

from pyomo.common.collections import ComponentSet, OrderedSet, ComponentMap
from pyomo.common.errors import PyomoException
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.objective import ObjectiveData
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.sos import SOSConstraintData, SOSConstraint
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.contrib.solver.common.results import Results
from pyomo.contrib.solver.common.util import IncompatibleModelError
from pyomo.contrib.solver.common.base import PersistentSolverBase
from pyomo.core.staleflag import StaleFlagManager
from .gurobi_direct_base import (
    GurobiDirectBase,
    gurobipy,
    GurobiConfig,
    GurobiDirectSolutionLoaderBase,
)
from .gurobi_direct import GurobiDirectSolutionLoader
from pyomo.contrib.solver.common.util import get_objective
from pyomo.contrib.observer.model_observer import (
    Observer,
    ModelChangeDetector,
    AutoUpdateConfig,
    Reason,
)

logger = logging.getLogger(__name__)


class GurobiPersistentSolutionLoader(GurobiDirectSolutionLoaderBase):
    def __init__(self, solver_model, var_map, con_map) -> None:
        super().__init__(solver_model)
        self._var_map = var_map
        self._con_map = con_map
        self._valid = True

    def _get_var_lists(self):
        return list(self._var_map.keys()), list(self._var_map.values())

    def _get_var_map(self):
        return self._var_map

    def _get_con_map(self):
        return self._con_map

    def invalidate(self):
        self._valid = False

    def _assert_solution_still_valid(self):
        if not self._valid:
            raise RuntimeError('The results in the solver are no longer valid.')

    def load_vars(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=0
    ) -> None:
        self._assert_solution_still_valid()
        return super().load_vars(vars_to_load, solution_id)

    def get_primals(
        self, vars_to_load: Sequence[VarData] | None = None, solution_id=0
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return super().get_primals(vars_to_load, solution_id)

    def get_duals(
        self, cons_to_load: Sequence[ConstraintData] | None = None
    ) -> Dict[ConstraintData, float]:
        self._assert_solution_still_valid()
        return super().get_duals(cons_to_load)

    def get_reduced_costs(
        self, vars_to_load: Sequence[VarData] | None = None
    ) -> Mapping[VarData, float]:
        self._assert_solution_still_valid()
        return super().get_reduced_costs(vars_to_load)


class _MutableLowerBound:

    __slots__ = ("var", "expr", "var_map")

    def __init__(self, var, expr, var_map):
        self.var = var
        self.expr = expr
        self.var_map = var_map

    def update(self):
        self.var_map[self.var].setAttr('lb', value(self.expr))


class _MutableUpperBound:

    __slots__ = ("var", "expr", "var_map")

    def __init__(self, var, expr, var_map):
        self.var = var
        self.expr = expr
        self.var_map = var_map

    def update(self):
        self.var_map[self.var].setAttr('ub', value(self.expr))


class _MutableLinearCoefficient:

    __slots__ = ("expr", "pyomo_con", "pyomo_var", "con_map", "var_map", "gurobi_model")

    def __init__(self, expr, pyomo_con, con_map, pyomo_var, var_map, gurobi_model):
        self.expr = expr
        self.pyomo_con = pyomo_con
        self.pyomo_var = pyomo_var
        self.con_map = con_map
        self.var_map = var_map
        self.gurobi_model = gurobi_model

    @property
    def gurobi_var(self):
        return self.var_map[self.pyomo_var]

    @property
    def gurobi_con(self):
        return self.con_map[self.pyomo_con]

    def update(self):
        self.gurobi_model.chgCoeff(self.gurobi_con, self.gurobi_var, value(self.expr))


class _MutableRangeConstant:

    __slots__ = (
        "lhs_expr",
        "rhs_expr",
        "pyomo_con",
        "con_map",
        "slack_name",
        "gurobi_model",
    )

    def __init__(
        self, lhs_expr, rhs_expr, pyomo_con, con_map, slack_name, gurobi_model
    ):
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr
        self.pyomo_con = pyomo_con
        self.con_map = con_map
        self.slack_name = slack_name
        self.gurobi_model = gurobi_model

    def update(self):
        rhs_val = value(self.rhs_expr)
        lhs_val = value(self.lhs_expr)
        con = self.con_map[self.pyomo_con]
        con.rhs = rhs_val
        slack = self.gurobi_model.getVarByName(self.slack_name)
        slack.ub = rhs_val - lhs_val


class _MutableConstant:

    __slots__ = ("expr", "pyomo_con", "con_map")

    def __init__(self, expr, pyomo_con, con_map):
        self.expr = expr
        self.pyomo_con = pyomo_con
        self.con_map = con_map

    def update(self):
        con = self.con_map[self.pyomo_con]
        con.rhs = value(self.expr)


class _MutableQuadraticConstraint:

    __slots__ = (
        "pyomo_con",
        "con_map",
        "gurobi_model",
        "constant",
        "last_constant_value",
        "linear_coefs",
        "last_linear_coef_values",
        "quadratic_coefs",
        "last_quadratic_coef_values",
    )

    def __init__(
        self, gurobi_model, pyomo_con, con_map, constant, linear_coefs, quadratic_coefs
    ):
        self.pyomo_con = pyomo_con
        self.con_map = con_map
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.last_constant_value = value(self.constant.expr)
        self.linear_coefs = linear_coefs
        self.last_linear_coef_values = [value(i.expr) for i in self.linear_coefs]
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    @property
    def gurobi_con(self):
        return self.con_map[self.pyomo_con]

    def get_updated_expression(self):
        gurobi_expr = self.gurobi_model.getQCRow(self.gurobi_con)
        for ndx, coef in enumerate(self.linear_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_linear_coef_values[ndx]
            )
            gurobi_expr += incremental_coef_value * coef.gurobi_var
            self.last_linear_coef_values[ndx] = current_coef_value
        for ndx, coef in enumerate(self.quadratic_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_quadratic_coef_values[ndx]
            )
            gurobi_expr += incremental_coef_value * coef.var1 * coef.var2
            self.last_quadratic_coef_values[ndx] = current_coef_value
        return gurobi_expr

    def get_updated_rhs(self):
        return value(self.constant.expr)


class _MutableObjective:

    __slots__ = (
        "gurobi_model",
        "constant",
        "linear_coefs",
        "quadratic_coefs",
        "last_quadratic_coef_values",
    )

    def __init__(self, gurobi_model, constant, linear_coefs, quadratic_coefs):
        self.gurobi_model = gurobi_model
        self.constant: _MutableConstant = constant
        self.linear_coefs: List[_MutableLinearCoefficient] = linear_coefs
        self.quadratic_coefs: List[_MutableQuadraticCoefficient] = quadratic_coefs
        self.last_quadratic_coef_values: List[float] = [
            value(i.expr) for i in self.quadratic_coefs
        ]

    def get_updated_expression(self):
        for ndx, coef in enumerate(self.linear_coefs):
            coef.gurobi_var.obj = value(coef.expr)
        self.gurobi_model.ObjCon = value(self.constant.expr)

        gurobi_expr = None
        for ndx, coef in enumerate(self.quadratic_coefs):
            if value(coef.expr) != self.last_quadratic_coef_values[ndx]:
                if gurobi_expr is None:
                    self.gurobi_model.update()
                    gurobi_expr = self.gurobi_model.getObjective()
                current_coef_value = value(coef.expr)
                incremental_coef_value = (
                    current_coef_value - self.last_quadratic_coef_values[ndx]
                )
                gurobi_expr += incremental_coef_value * coef.var1 * coef.var2
                self.last_quadratic_coef_values[ndx] = current_coef_value
        return gurobi_expr


class _MutableQuadraticCoefficient:

    __slots__ = ("expr", "var_map", "v1", "v2")

    def __init__(self, expr, v1, v2, var_map):
        self.expr = expr
        self.var_map = var_map
        self.v1 = v1
        self.v2 = v2

    @property
    def var1(self):
        return self.var_map[self.v1]

    @property
    def var2(self):
        return self.var_map[self.v2]


class GurobiPersistentConfig(GurobiConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        GurobiConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.auto_updates: bool = self.declare('auto_updates', AutoUpdateConfig())


class GurobiPersistent(GurobiDirectBase, PersistentSolverBase, Observer):
    _minimum_version = (7, 0, 0)
    CONFIG = GurobiPersistentConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # we actually want to only grab the license when
        # set_instance is called
        self._release_env_client()
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_sos_to_solver_sos_map = {}
        self._pyomo_model = None
        self._objective = None
        self._mutable_helpers = {}
        self._mutable_bounds = {}
        self._mutable_quadratic_helpers = {}
        self._mutable_objective = None
        self._needs_updated = True
        self._callback_func = None
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._last_results_object: Optional[Results] = None
        self._change_detector = None
        self._constraint_ndx = 0
        self._disallow_set_var_attr = {'lb', 'ub', 'vtype', 'varname'}
        self._disallow_linear_constraint_attr = {'sense', 'rhs', 'constrname'}

    def _clear(self):
        release = False
        if self._solver_model is not None:
            release = True
        self._solver_model = None
        if release:
            self._release_env_client()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_sos_to_solver_sos_map = {}
        self._pyomo_model = None
        self._objective = None
        self._mutable_helpers = {}
        self._mutable_bounds = {}
        self._mutable_quadratic_helpers = {}
        self._mutable_objective = None
        self._needs_updated = True
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._last_results_object = None
        self._change_detector = None
        self._constraint_ndx = 0

    def _create_solver_model(self, pyomo_model, config):
        if pyomo_model is self._pyomo_model:
            self.update(**config)
        else:
            self.set_instance(pyomo_model, **config)

        solution_loader = GurobiPersistentSolutionLoader(
            solver_model=self._solver_model,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
        )
        has_obj = self._objective is not None
        return self._solver_model, solution_loader, has_obj

    def _pyomo_gurobi_var_iter(self):
        return self._pyomo_var_to_solver_var_map.items()

    def release_license(self):
        self._clear()
        super().release_license()

    def solve(self, model, **kwds) -> Results:
        res = super().solve(model, **kwds)
        self._needs_updated = False
        return res

    def _process_domain_and_bounds(self, var):
        lb, ub, step = var.domain.get_interval()
        if lb is None:
            lb = -gurobipy.GRB.INFINITY
        if ub is None:
            ub = gurobipy.GRB.INFINITY
        if step == 0:
            vtype = gurobipy.GRB.CONTINUOUS
        elif step == 1:
            if lb == 0 and ub == 1:
                vtype = gurobipy.GRB.BINARY
            else:
                vtype = gurobipy.GRB.INTEGER
        else:
            raise ValueError(f'Unrecognized domain: {var.domain}')
        if var.fixed:
            lb = var.value
            ub = lb
        else:
            if var._lb is not None:
                lb = max(lb, value(var._lb))
            if var._ub is not None:
                ub = min(ub, value(var._ub))
        if not is_constant(var._lb):
            mutable_lb = _MutableLowerBound(
                var, var.lower, self._pyomo_var_to_solver_var_map
            )
            self._mutable_bounds[id(var), 'lb'] = (var, mutable_lb)
        if not is_constant(var._ub):
            mutable_ub = _MutableUpperBound(
                var, var.upper, self._pyomo_var_to_solver_var_map
            )
            self._mutable_bounds[id(var), 'ub'] = (var, mutable_ub)
        return lb, ub, vtype

    def _add_variables(self, variables: List[VarData]):
        self._invalidate_last_results()
        vtypes = []
        lbs = []
        ubs = []
        for ndx, var in enumerate(variables):
            lb, ub, vtype = self._process_domain_and_bounds(var)
            vtypes.append(vtype)
            lbs.append(lb)
            ubs.append(ub)

        gurobi_vars = self._solver_model.addVars(
            len(variables), lb=lbs, ub=ubs, vtype=vtypes
        ).values()

        for pyomo_var, gurobi_var in zip(variables, gurobi_vars):
            self._pyomo_var_to_solver_var_map[pyomo_var] = gurobi_var
        self._vars_added_since_update.update(variables)
        self._needs_updated = True

    def set_instance(self, pyomo_model, **kwds):
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        self._clear()
        self._register_env_client()
        self._pyomo_model = pyomo_model
        self._solver_model = gurobipy.Model(env=self.env())
        timer.start('set_instance')
        self._change_detector = ModelChangeDetector(
            model=self._pyomo_model, observers=[self], **config.auto_updates
        )
        timer.stop('set_instance')

    def update(self, **kwds):
        config = self.config(value=kwds, preserve_implicit=True)
        if config.timer is None:
            timer = HierarchicalTimer()
        else:
            timer = config.timer
        if self._pyomo_model is None:
            raise RuntimeError('must call set_instance or solve before update')
        timer.start('update')
        if self._needs_updated:
            self._update_gurobi_model()
        self._change_detector.update(timer=timer, **config.auto_updates)
        timer.stop('update')

    def _get_expr_from_pyomo_repn(self, repn):
        if repn.nonlinear_expr is not None:
            raise NotImplementedError(
                f'GurobiPersistent only supports linear and quadratic expressions for now: {repn}.'
            )

        if len(repn.linear_vars) > 0:
            coef_list = [value(i) for i in repn.linear_coefs]
            vlist = [self._pyomo_var_to_solver_var_map[v] for v in repn.linear_vars]
            new_expr = gurobipy.LinExpr(coef_list, vlist)
        else:
            new_expr = 0.0

        if len(repn.quadratic_vars) > 0:
            for coef, (x, y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
                gurobi_x = self._pyomo_var_to_solver_var_map[x]
                gurobi_y = self._pyomo_var_to_solver_var_map[y]
                new_expr += value(coef) * gurobi_x * gurobi_y

        return new_expr

    def _add_constraints(self, cons: List[ConstraintData]):
        self._invalidate_last_results()
        gurobi_expr_list = []
        for ndx, con in enumerate(cons):
            lb, body, ub = con.to_bounded_expression(evaluate_bounds=False)
            repn = generate_standard_repn(body, quadratic=True, compute_values=False)
            gurobi_expr = self._get_expr_from_pyomo_repn(repn)
            mutable_constant = None
            if lb is None:
                if ub is None:
                    raise ValueError(
                        f"Constraint does not have a lower or an upper bound: {con}"
                    )
                rhs_expr = ub - repn.constant
                gurobi_expr_list.append(gurobi_expr <= float(value(rhs_expr)))
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant(
                        rhs_expr, con, self._pyomo_con_to_solver_con_map
                    )
            elif ub is None:
                rhs_expr = lb - repn.constant
                gurobi_expr_list.append(float(value(rhs_expr)) <= gurobi_expr)
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant(
                        rhs_expr, con, self._pyomo_con_to_solver_con_map
                    )
            elif con.equality:
                rhs_expr = lb - repn.constant
                gurobi_expr_list.append(gurobi_expr == float(value(rhs_expr)))
                if not is_constant(rhs_expr):
                    mutable_constant = _MutableConstant(
                        rhs_expr, con, self._pyomo_con_to_solver_con_map
                    )
            else:
                assert (
                    len(repn.quadratic_vars) == 0
                ), "Quadratic range constraints are not supported"
                lhs_expr = lb - repn.constant
                rhs_expr = ub - repn.constant
                gurobi_expr_list.append(
                    gurobi_expr == [float(value(lhs_expr)), float(value(rhs_expr))]
                )
                if not is_constant(lhs_expr) or not is_constant(rhs_expr):
                    conname = f'c{self._constraint_ndx}[{ndx}]'
                    mutable_constant = _MutableRangeConstant(
                        lhs_expr,
                        rhs_expr,
                        con,
                        self._pyomo_con_to_solver_con_map,
                        'Rg' + conname,
                        self._solver_model,
                    )

            mlc_list = []
            for c, v in zip(repn.linear_coefs, repn.linear_vars):
                if not is_constant(c):
                    mlc = _MutableLinearCoefficient(
                        c,
                        con,
                        self._pyomo_con_to_solver_con_map,
                        v,
                        self._pyomo_var_to_solver_var_map,
                        self._solver_model,
                    )
                    mlc_list.append(mlc)

            if len(repn.quadratic_vars) == 0:
                if len(mlc_list) > 0:
                    self._mutable_helpers[con] = mlc_list
                if mutable_constant is not None:
                    if con not in self._mutable_helpers:
                        self._mutable_helpers[con] = []
                    self._mutable_helpers[con].append(mutable_constant)
            else:
                if mutable_constant is None:
                    mutable_constant = _MutableConstant(
                        rhs_expr, con, self._pyomo_con_to_solver_con_map
                    )
                mqc_list = []
                for coef, (x, y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
                    if not is_constant(coef):
                        mqc = _MutableQuadraticCoefficient(
                            coef, x, y, self._pyomo_var_to_solver_var_map
                        )
                        mqc_list.append(mqc)
                mqc = _MutableQuadraticConstraint(
                    self._solver_model,
                    con,
                    self._pyomo_con_to_solver_con_map,
                    mutable_constant,
                    mlc_list,
                    mqc_list,
                )
                self._mutable_quadratic_helpers[con] = mqc

        gurobi_cons = list(
            self._solver_model.addConstrs(
                (gurobi_expr_list[i] for i in range(len(gurobi_expr_list))),
                name=f'c{self._constraint_ndx}',
            ).values()
        )
        self._constraint_ndx += 1
        self._pyomo_con_to_solver_con_map.update(zip(cons, gurobi_cons))
        self._constraints_added_since_update.update(cons)
        self._needs_updated = True

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        self._invalidate_last_results()
        for con in cons:
            level = con.level
            if level == 1:
                sos_type = gurobipy.GRB.SOS_TYPE1
            elif level == 2:
                sos_type = gurobipy.GRB.SOS_TYPE2
            else:
                raise ValueError(
                    f"Solver does not support SOS level {level} constraints"
                )

            gurobi_vars = []
            weights = []

            for v, w in con.get_items():
                gurobi_vars.append(self._pyomo_var_to_solver_var_map[v])
                weights.append(w)

            gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
            self._pyomo_sos_to_solver_sos_map[con] = gurobipy_con
        self._constraints_added_since_update.update(cons)
        self._needs_updated = True

    def _remove_objectives(self, objs: List[ObjectiveData]):
        for obj in objs:
            if obj is not self._objective:
                raise RuntimeError(
                    'tried to remove an objective that has not been added: '
                    f'{str(obj)}'
                )
            else:
                self._invalidate_last_results()
                self._solver_model.setObjective(0, sense=gurobipy.GRB.MINIMIZE)
                # see PR #2454
                self._solver_model.update()
                self._objective = None
                self._needs_updated = False

    def _add_objectives(self, objs: List[ObjectiveData]):
        if len(objs) > 1:
            raise NotImplementedError(
                'the persistent interface to gurobi currently '
                f'only supports single-objective problems; got {len(objs)}: '
                f'{[str(i) for i in objs]}'
            )

        if len(objs) == 0:
            return

        obj = objs[0]

        if self._objective is not None:
            raise NotImplementedError(
                'the persistent interface to gurobi currently '
                'only supports single-objective problems; tried to add '
                f'an objective ({str(obj)}), but there is already an '
                f'active objective ({str(self._objective)})'
            )

        self._invalidate_last_results()

        if obj.sense == minimize:
            sense = gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError(f'Objective sense is not recognized: {obj.sense}')

        repn = generate_standard_repn(obj.expr, quadratic=True, compute_values=False)
        repn_constant = value(repn.constant)
        gurobi_expr = self._get_expr_from_pyomo_repn(repn)

        mutable_constant = _MutableConstant(repn.constant, None, None)

        mlc_list = []
        for c, v in zip(repn.linear_coefs, repn.linear_vars):
            if not is_constant(c):
                mlc = _MutableLinearCoefficient(
                    c,
                    None,
                    None,
                    v,
                    self._pyomo_var_to_solver_var_map,
                    self._solver_model,
                )
                mlc_list.append(mlc)

        mqc_list = []
        for coef, (x, y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
            if not is_constant(coef):
                mqc = _MutableQuadraticCoefficient(
                    coef, x, y, self._pyomo_var_to_solver_var_map
                )
                mqc_list.append(mqc)

        self._mutable_objective = _MutableObjective(
            self._solver_model, mutable_constant, mlc_list, mqc_list
        )

        self._solver_model.setObjective(gurobi_expr + repn_constant, sense=sense)
        self._objective = obj
        self._needs_updated = True

    def _update_gurobi_model(self):
        self._solver_model.update()
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def _remove_constraints(self, cons: List[ConstraintData]):
        self._invalidate_last_results()
        for con in cons:
            if con in self._constraints_added_since_update:
                self._update_gurobi_model()
            solver_con = self._pyomo_con_to_solver_con_map[con]
            self._solver_model.remove(solver_con)
            del self._pyomo_con_to_solver_con_map[con]
            self._mutable_helpers.pop(con, None)
            self._mutable_quadratic_helpers.pop(con, None)
        self._needs_updated = True

    def _remove_sos_constraints(self, cons: List[SOSConstraintData]):
        self._invalidate_last_results()
        for con in cons:
            if con in self._constraints_added_since_update:
                self._update_gurobi_model()
            solver_sos_con = self._pyomo_sos_to_solver_sos_map[con]
            self._solver_model.remove(solver_sos_con)
            del self._pyomo_sos_to_solver_sos_map[con]
        self._needs_updated = True

    def _remove_variables(self, variables: List[VarData]):
        self._invalidate_last_results()
        for var in variables:
            v_id = id(var)
            if var in self._vars_added_since_update:
                self._update_gurobi_model()
            solver_var = self._pyomo_var_to_solver_var_map.pop(var)
            self._solver_model.remove(solver_var)
            self._mutable_bounds.pop((v_id, 'lb'), None)
            self._mutable_bounds.pop((v_id, 'ub'), None)
        self._needs_updated = True

    def _update_variables(self, variables: Mapping[VarData, Reason]):
        self._invalidate_last_results()
        new_vars = []
        old_vars = []
        mod_vars = []
        for v, reason in variables.items():
            if reason & Reason.added:
                new_vars.append(v)
            elif reason & Reason.removed:
                old_vars.append(v)
            else:
                mod_vars.append(v)

        if new_vars:
            self._add_variables(new_vars)
        if old_vars:
            self._remove_variables(old_vars)

        cons_to_reprocess = OrderedSet()
        cons_to_update = OrderedSet()
        reprocess_obj = False
        update_obj = False

        for v in mod_vars:
            reason = variables[v]
            if reason & (Reason.bounds | Reason.domain | Reason.fixed | Reason.value):
                var_id = id(v)
                self._mutable_bounds.pop((var_id, 'lb'), None)
                self._mutable_bounds.pop((var_id, 'ub'), None)
                gurobipy_var = self._pyomo_var_to_solver_var_map[v]
                lb, ub, vtype = self._process_domain_and_bounds(v)
                gurobipy_var.setAttr('lb', lb)
                gurobipy_var.setAttr('ub', ub)
                gurobipy_var.setAttr('vtype', vtype)
            if reason & Reason.fixed:
                cons_to_reprocess.update(
                    self._change_detector.get_constraints_impacted_by_var(v)
                )
                objs = self._change_detector.get_objectives_impacted_by_var(v)
                if objs:
                    assert len(objs) == 1
                    assert objs[0] is self._objective
                    reprocess_obj = True
            elif (reason & Reason.value) and v.fixed:
                cons_to_update.update(
                    self._change_detector.get_constraints_impacted_by_var(v)
                )
                objs = self._change_detector.get_objectives_impacted_by_var(v)
                if objs:
                    assert len(objs) == 1
                    assert objs[0] is self._objective
                    update_obj = True

        self._remove_constraints(cons_to_reprocess)
        self._add_constraints(cons_to_reprocess)
        cons_to_update -= cons_to_reprocess
        for c in cons_to_update:
            if c in self._mutable_helpers:
                for i in self._mutable_helpers[c]:
                    i.update()
            self._update_quadratic_constraint(c)

        if reprocess_obj:
            obj = self._objective
            self._remove_objectives([obj])
            self._add_objectives([obj])
        elif update_obj:
            self._mutable_objective_update()

        self._needs_updated = True

    def _update_constraints(self, cons: Mapping[ConstraintData, Reason]):
        self._invalidate_last_results()
        new_cons = []
        old_cons = []
        for c, reason in cons.items():
            if reason & Reason.added:
                new_cons.append(c)
            elif reason & Reason.removed:
                old_cons.append(c)
            elif reason & Reason.expr:
                old_cons.append(c)
                new_cons.append(c)

        if old_cons:
            self._remove_constraints(old_cons)
        if new_cons:
            self._add_constraints(new_cons)
        self._needs_updated = True

    def _update_sos_constraints(self, cons: Mapping[SOSConstraintData, Reason]):
        self._invalidate_last_results()
        new_cons = []
        old_cons = []
        for c, reason in cons.items():
            if reason & Reason.added:
                new_cons.append(c)
            elif reason & Reason.removed:
                old_cons.append(c)
            elif reason & Reason.sos_items:
                old_cons.append(c)
                new_cons.append(c)

        if old_cons:
            self._remove_sos_constraints(old_cons)
        if new_cons:
            self._add_sos_constraints(new_cons)
        self._needs_updated = True

    def _update_objectives(self, objs: Mapping[ObjectiveData, Reason]):
        self._invalidate_last_results()
        new_objs = []
        old_objs = []
        new_sense = []
        for obj, reason in objs.items():
            if reason & Reason.added:
                new_objs.append(obj)
            elif reason & Reason.removed:
                old_objs.append(obj)
            elif reason & Reason.expr:
                old_objs.append(obj)
                new_objs.append(obj)
            elif reason & Reason.sense:
                new_sense.append(obj)

        if old_objs:
            self._remove_objectives(old_objs)
        if new_objs:
            self._add_objectives(new_objs)
        if new_sense:
            assert len(new_sense) == 1
            obj = new_sense[0]
            assert obj is self._objective
            if obj.sense == minimize:
                sense = gurobipy.GRB.MINIMIZE
            elif obj.sense == maximize:
                sense = gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError(f'Objective sense is not recognized: {obj.sense}')
            self._solver_model.ModelSense = sense

    def _update_quadratic_constraint(self, c: ConstraintData):
        if c in self._mutable_quadratic_helpers:
            if c in self._constraints_added_since_update:
                self._update_gurobi_model()
            helper = self._mutable_quadratic_helpers[c]
            gurobi_con = helper.gurobi_con
            new_gurobi_expr = helper.get_updated_expression()
            new_rhs = helper.get_updated_rhs()
            new_sense = gurobi_con.qcsense
            self._solver_model.remove(gurobi_con)
            new_con = self._solver_model.addQConstr(new_gurobi_expr, new_sense, new_rhs)
            self._pyomo_con_to_solver_con_map[c] = new_con
            assert helper.pyomo_con is c
            self._constraints_added_since_update.add(c)

    def _mutable_objective_update(self):
        if self._mutable_objective is not None:
            new_gurobi_expr = self._mutable_objective.get_updated_expression()
            if new_gurobi_expr is not None:
                if self._objective.sense == minimize:
                    sense = gurobipy.GRB.MINIMIZE
                else:
                    sense = gurobipy.GRB.MAXIMIZE
                # TODO: need a test for when part of the object is linear
                #       and part of the objective is quadratic, but both
                #       parts have mutable coefficients
                self._solver_model.setObjective(new_gurobi_expr, sense=sense)

    def _update_parameters(self, params: Mapping[ParamData, Reason]):
        self._invalidate_last_results()

        cons_to_update = OrderedSet()
        update_obj = False
        vars_to_update = ComponentSet()
        for p, reason in params.items():
            if reason & Reason.added:
                continue
            if reason & Reason.removed:
                continue
            if reason & Reason.value:
                cons_to_update.update(
                    self._change_detector.get_constraints_impacted_by_param(p)
                )
                objs = self._change_detector.get_objectives_impacted_by_param(p)
                if objs:
                    assert len(objs) == 1
                    assert objs[0] is self._objective
                    update_obj = True
                vars_to_update.update(
                    self._change_detector.get_variables_impacted_by_param(p)
                )

        for c in cons_to_update:
            if c in self._mutable_helpers:
                for i in self._mutable_helpers[c]:
                    i.update()
            self._update_quadratic_constraint(c)

        if update_obj:
            self._mutable_objective_update()

        for v in vars_to_update:
            vid = id(v)
            if (vid, 'lb') in self._mutable_bounds:
                self._mutable_bounds[(vid, 'lb')][1].update()
            if (vid, 'ub') in self._mutable_bounds:
                self._mutable_bounds[(vid, 'ub')][1].update()

        self._needs_updated = True

    def _invalidate_last_results(self):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()

    def get_model_attr(self, attr):
        """
        Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for descriptions of the attributes.
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._solver_model.getAttr(attr)

    def write(self, filename):
        """
        Write the model to a file (e.g., and lp file).

        Parameters
        ----------
        filename: str
            Name of the file to which the model should be written.
        """
        self._solver_model.write(filename)
        self._constraints_added_since_update = OrderedSet()
        self._vars_added_since_update = ComponentSet()
        self._needs_updated = False

    def set_linear_constraint_attr(self, con, attr, val):
        """
        Set the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                CBasis
                DStart
                Lazy
        val: any
            See gurobi documentation for acceptable values.
        """
        attr = attr.lower()
        if attr in self._disallow_linear_constraint_attr:
            raise ValueError(
                f'Linear constraint attr {attr} cannot be set with'
                ' the set_linear_constraint_attr method. Please use'
                ' the remove_constraint and add_constraint methods.'
            )
        self._pyomo_con_to_solver_con_map[con].setAttr(attr, val)
        self._needs_updated = True

    def set_var_attr(self, var, attr, val):
        """
        Set the value of an attribute on a gurobi variable.

        Parameters
        ----------
        var: pyomo.core.base.var.VarData
            The pyomo var for which the corresponding gurobi var attribute
            should be modified.
        attr: str
            The attribute to be modified. Options are:
                Start
                VarHintVal
                VarHintPri
                BranchPriority
                VBasis
                PStart
        val: any
            See gurobi documentation for acceptable values.
        """
        attr = attr.lower()
        if attr in self._disallow_set_var_attr:
            raise ValueError(
                f'Var attr {attr} cannot be set with'
                ' the set_var_attr method. Please use'
                ' the update_var method.'
            )
        if attr == 'obj':
            raise ValueError(
                'Var attr Obj cannot be set with'
                ' the set_var_attr method. Please use'
                ' the set_objective method.'
            )
        self._pyomo_var_to_solver_var_map[var].setAttr(attr, val)
        self._needs_updated = True

    def get_var_attr(self, var, attr):
        """
        Get the value of an attribute on a gurobi var.

        Parameters
        ----------
        var: pyomo.core.base.var.VarData
            The pyomo var for which the corresponding gurobi var attribute
            should be retrieved.
        attr: str
            The attribute to get. See gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_var_to_solver_var_map[var].getAttr(attr)

    def get_linear_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi linear constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def get_sos_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos.SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_sos_to_solver_sos_map[con].getAttr(attr)

    def get_quadratic_constraint_attr(self, con, attr):
        """
        Get the value of an attribute on a gurobi quadratic constraint.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The pyomo constraint for which the corresponding gurobi constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. See the Gurobi documentation
        """
        if self._needs_updated:
            self._update_gurobi_model()
        return self._pyomo_con_to_solver_con_map[con].getAttr(attr)

    def set_gurobi_param(self, param, val):
        """
        Set a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to set. Options include any gurobi parameter.
            Please see the Gurobi documentation for options.
        val: any
            The value to set the parameter to. See Gurobi documentation for possible values.
        """
        self._solver_model.setParam(param, val)

    def get_gurobi_param_info(self, param):
        """
        Get information about a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to get info for. See Gurobi documentation for possible options.

        Returns
        -------
        six-tuple containing the parameter name, type, value, minimum value, maximum value, and default value.
        """
        return self._solver_model.getParamInfo(param)

    def _intermediate_callback(self):
        def f(gurobi_model, where):
            self._callback_func(self._pyomo_model, self, where)

        return f

    def set_callback(self, func=None):
        """
        Specify a callback for gurobi to use.

        Parameters
        ----------
        func: function
            The function to call. The function should have three arguments. The first will be the pyomo model being
            solved. The second will be the GurobiPersistent instance. The third will be an enum member of
            gurobipy.GRB.Callback. This will indicate where in the branch and bound algorithm gurobi is at. For
            example, suppose we want to solve

            .. math::

                min 2*x + y

                s.t.

                    y >= (x-2)**2

                    0 <= x <= 4

                    y >= 0

                    y integer

            as an MILP using extended cutting planes in callbacks.

                >>> from gurobipy import GRB # doctest:+SKIP
                >>> import pyomo.environ as pyo
                >>> from pyomo.core.expr.taylor_series import taylor_series_expansion
                >>> from pyomo.contrib import appsi
                >>>
                >>> m = pyo.ConcreteModel()
                >>> m.x = pyo.Var(bounds=(0, 4))
                >>> m.y = pyo.Var(within=pyo.Integers, bounds=(0, None))
                >>> m.obj = pyo.Objective(expr=2*m.x + m.y)
                >>> m.cons = pyo.ConstraintList()  # for the cutting planes
                >>>
                >>> def _add_cut(xval):
                ...     # a function to generate the cut
                ...     m.x.value = xval
                ...     return m.cons.add(m.y >= taylor_series_expansion((m.x - 2)**2))
                ...
                >>> _c = _add_cut(0)  # start with 2 cuts at the bounds of x
                >>> _c = _add_cut(4)  # this is an arbitrary choice
                >>>
                >>> opt = appsi.solvers.Gurobi()
                >>> opt.config.stream_solver = True
                >>> opt.set_instance(m) # doctest:+SKIP
                >>> opt.gurobi_options['PreCrush'] = 1
                >>> opt.gurobi_options['LazyConstraints'] = 1
                >>>
                >>> def my_callback(cb_m, cb_opt, cb_where):
                ...     if cb_where == GRB.Callback.MIPSOL:
                ...         cb_opt.cbGetSolution(variables=[m.x, m.y])
                ...         if m.y.value < (m.x.value - 2)**2 - 1e-6:
                ...             cb_opt.cbLazy(_add_cut(m.x.value))
                ...
                >>> opt.set_callback(my_callback)
                >>> res = opt.solve(m) # doctest:+SKIP

        """
        if func is not None:
            self._callback_func = func
            self._callback = self._intermediate_callback()
        else:
            self._callback = None
            self._callback_func = None

    def cbCut(self, con):
        """
        Add a cut within a callback.

        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The cut to add
        """
        if not con.active:
            raise ValueError('cbCut expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbCut expected a non-trivial constraint')

        repn = generate_standard_repn(con.body, quadratic=True, compute_values=True)
        gurobi_expr = self._get_expr_from_pyomo_repn(repn)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbCut.')
            if not is_fixed(con.lower):
                raise ValueError(f'Lower bound of constraint {con} is not constant.')
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(f'Upper bound of constraint {con} is not constant.')

        if con.equality:
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.EQUAL,
                rhs=value(con.lower - repn.constant),
            )
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=value(con.lower - repn.constant),
            )
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbCut(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.LESS_EQUAL,
                rhs=value(con.upper - repn.constant),
            )
        else:
            raise ValueError(
                f'Constraint does not have a lower or an upper bound {con} \n'
            )

    def cbGet(self, what):
        return self._solver_model.cbGet(what)

    def cbGetNodeRel(self, variables):
        """
        Parameters
        ----------
        variables: Var or iterable of Var
        """
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in variables]
        var_values = self._solver_model.cbGetNodeRel(gurobi_vars)
        for i, v in enumerate(variables):
            v.set_value(var_values[i], skip_validation=True)

    def cbGetSolution(self, variables):
        """
        Parameters
        ----------
        variables: iterable of vars
        """
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in variables]
        var_values = self._solver_model.cbGetSolution(gurobi_vars)
        for i, v in enumerate(variables):
            v.set_value(var_values[i], skip_validation=True)

    def cbLazy(self, con):
        """
        Parameters
        ----------
        con: pyomo.core.base.constraint.ConstraintData
            The lazy constraint to add
        """
        if not con.active:
            raise ValueError('cbLazy expected an active constraint.')

        if is_fixed(con.body):
            raise ValueError('cbLazy expected a non-trivial constraint')

        repn = generate_standard_repn(con.body, quadratic=True, compute_values=True)
        gurobi_expr = self._get_expr_from_pyomo_repn(repn)

        if con.has_lb():
            if con.has_ub():
                raise ValueError('Range constraints are not supported in cbLazy.')
            if not is_fixed(con.lower):
                raise ValueError(f'Lower bound of constraint {con} is not constant.')
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(f'Upper bound of constraint {con} is not constant.')

        if con.equality:
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.EQUAL,
                rhs=value(con.lower - repn.constant),
            )
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=value(con.lower - repn.constant),
            )
        elif con.has_ub() and (value(con.upper) < float('inf')):
            self._solver_model.cbLazy(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.LESS_EQUAL,
                rhs=value(con.upper - repn.constant),
            )
        else:
            raise ValueError(
                f'Constraint does not have a lower or an upper bound {con} \n'
            )

    def cbSetSolution(self, variables, solution):
        if not isinstance(variables, Iterable):
            variables = [variables]
        gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in variables]
        self._solver_model.cbSetSolution(gurobi_vars, solution)

    def cbUseSolution(self):
        return self._solver_model.cbUseSolution()

    def reset(self):
        self._solver_model.reset()

    def add_constraints(self, cons):
        self._change_detector.add_constraints(cons)

    def add_sos_constraints(self, cons):
        self._change_detector.add_sos_constraints(cons)

    def set_objective(self, obj: ObjectiveData):
        self._change_detector.add_objectives([obj])

    def remove_constraints(self, cons):
        self._change_detector.remove_constraints(cons)

    def remove_sos_constraints(self, cons):
        self._change_detector.remove_sos_constraints(cons)

    def update_variables(self, variables):
        self._change_detector.update_variables(variables)

    def update_parameters(self, params):
        self._change_detector.update_parameters(params)
