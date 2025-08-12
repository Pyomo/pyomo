#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import io
import logging
import math
from typing import Dict, List, NoReturn, Optional, Sequence, Mapping
from collections.abc import Iterable

from pyomo.common.collections import ComponentSet, ComponentMap, OrderedSet
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import ConstraintData, Constraint
from pyomo.core.base.sos import SOSConstraintData, SOSConstraint
from pyomo.core.base.param import ParamData
from pyomo.core.expr.numvalue import value, is_constant, is_fixed, native_numeric_types
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.contrib.solver.common.base import PersistentSolverBase, Availability
from pyomo.contrib.solver.common.results import (
    Results,
    TerminationCondition,
    SolutionStatus,
)
from pyomo.contrib.solver.common.config import PersistentBranchAndBoundConfig
from pyomo.contrib.solver.solvers.gurobi_direct import (
    GurobiConfigMixin,
    GurobiSolverMixin,
)
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.persistent import (
    PersistentSolverUtils,
    PersistentSolverMixin,
)
from pyomo.contrib.solver.common.solution_loader import PersistentSolutionLoader, SolutionLoaderBase
from pyomo.core.staleflag import StaleFlagManager
from .gurobi_direct_base import (
    GurobiConfig, 
    GurobiDirectBase, 
    gurobipy, 
    _load_suboptimal_mip_solution,
    _load_vars,
    _get_primals,
    _get_duals,
    _get_reduced_costs,
)
from pyomo.contrib.solver.common.util import get_objective
from pyomo.repn.quadratic import QuadraticRepn, QuadraticRepnVisitor


logger = logging.getLogger(__name__)


class GurobiDirectQuadraticSolutionLoader(SolutionLoaderBase):
    def __init__(
        self,
        solver_model,
        var_id_map,
        var_map,
        con_map,
        linear_cons,
        quadratic_cons,
    ) -> None:
        super().__init__()
        self._solver_model = solver_model
        self._vars = var_id_map
        self._var_map = var_map
        self._con_map = con_map
        self._linear_cons = linear_cons
        self._quadratic_cons = quadratic_cons

    def load_vars(
        self, 
        vars_to_load: Optional[Sequence[VarData]] = None,
        solution_id=0,
    ) -> None:
        if vars_to_load is None:
            vars_to_load = list(self._vars.values())
        _load_vars(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
            solution_number=solution_id,
        )

    def get_primals(
        self, 
        vars_to_load: Optional[Sequence[VarData]] = None,
        solution_id=0,
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = list(self._vars.values())
        return _get_primals(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
            solution_number=solution_id,
        )
    
    def get_reduced_costs(
        self, 
        vars_to_load: Optional[Sequence[VarData]] = None,
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = list(self._vars.values())
        return _get_reduced_costs(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
        )
    
    def get_duals(
        self, 
        cons_to_load: Optional[Sequence[ConstraintData]] = None,
    ) -> Dict[ConstraintData, float]:
        if cons_to_load is None:
            cons_to_load = list(self._con_map.keys())
        linear_cons_to_load = []
        quadratic_cons_to_load = []
        for c in cons_to_load:
            if c in self._linear_cons:
                linear_cons_to_load.append(c)
            else:
                assert c in self._quadratic_cons
                quadratic_cons_to_load.append(c)
        return _get_duals(
            solver_model=self._solver_model,
            con_map=self._con_map,
            linear_cons_to_load=linear_cons_to_load,
            quadratic_cons_to_load=quadratic_cons_to_load,
        )


class GurobiPersistentSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        self._solver._load_vars(
            vars_to_load=vars_to_load, solution_number=solution_number
        )

    def get_primals(self, vars_to_load=None, solution_number=0):
        self._assert_solution_still_valid()
        return self._solver._get_primals(
            vars_to_load=vars_to_load, solution_number=solution_number
        )


class _MutableLowerBound:
    def __init__(self, var_id, expr, var_map):
        self.var_id = var_id
        self.expr = expr
        self.var_map = var_map

    def update(self):
        self.var_map[self.var_id].setAttr('lb', value(self.expr))


class _MutableUpperBound:
    def __init__(self, var_id, expr, var_map):
        self.var_id = var_id
        self.expr = expr
        self.var_map = var_map

    def update(self):
        self.var_map[self.var_id].setAttr('ub', value(self.expr))


class _MutableLinearCoefficient:
    def __init__(self):
        self.expr = None
        self.var = None
        self.con = None
        self.gurobi_model = None

    def update(self):
        self.gurobi_model.chgCoeff(self.con, self.var, value(self.expr))


class _MutableRangeConstant:
    def __init__(self):
        self.lhs_expr = None
        self.rhs_expr = None
        self.con = None
        self.slack_name = None
        self.gurobi_model = None

    def update(self):
        rhs_val = value(self.rhs_expr)
        lhs_val = value(self.lhs_expr)
        self.con.rhs = rhs_val
        slack = self.gurobi_model.getVarByName(self.slack_name)
        slack.ub = rhs_val - lhs_val


class _MutableConstant:
    def __init__(self):
        self.expr = None
        self.con = None

    def update(self):
        self.con.rhs = value(self.expr)


class _MutableQuadraticConstraint:
    def __init__(
        self, gurobi_model, gurobi_con, constant, linear_coefs, quadratic_coefs
    ):
        self.con = gurobi_con
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.last_constant_value = value(self.constant.expr)
        self.linear_coefs = linear_coefs
        self.last_linear_coef_values = [value(i.expr) for i in self.linear_coefs]
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        gurobi_expr = self.gurobi_model.getQCRow(self.con)
        for ndx, coef in enumerate(self.linear_coefs):
            current_coef_value = value(coef.expr)
            incremental_coef_value = (
                current_coef_value - self.last_linear_coef_values[ndx]
            )
            gurobi_expr += incremental_coef_value * coef.var
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
    def __init__(self, gurobi_model, constant, linear_coefs, quadratic_coefs):
        self.gurobi_model = gurobi_model
        self.constant = constant
        self.linear_coefs = linear_coefs
        self.quadratic_coefs = quadratic_coefs
        self.last_quadratic_coef_values = [value(i.expr) for i in self.quadratic_coefs]

    def get_updated_expression(self):
        for ndx, coef in enumerate(self.linear_coefs):
            coef.var.obj = value(coef.expr)
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
    def __init__(self):
        self.expr = None
        self.var1 = None
        self.var2 = None


class GurobiDirectQuadratic(GurobiDirectBase):
    _minimum_version = (7, 0, 0)

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._solver_model = None
        self._vars = {}  # from id(v) to v
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        self._linear_cons = set()
        self._quadratic_cons = set()
        self._pyomo_sos_to_solver_sos_map = {}

    def _create_solver_model(self, pyomo_model):
        timer = self.config.timer
        timer.start('create gurobipy model')
        self._clear()
        self._solver_model = gurobipy.Model(env=self.env())
        timer.start('collect constraints')
        cons = list(pyomo_model.component_data_objects(Constraint, descend_into=True, active=True))
        timer.stop('collect constraints')
        timer.start('translate constraints')
        self._add_constraints(cons)
        timer.stop('translate constraints')
        timer.start('sos')
        sos = list(pyomo_model.component_data_objects(SOSConstraint, descend_into=True, active=True))
        self._add_sos_constraints(sos)
        timer.stop('sos')
        timer.start('get objective')
        obj = get_objective(pyomo_model)
        timer.stop('get objective')
        timer.start('translate objective')
        self._set_objective(obj)
        timer.stop('translate objective')
        has_obj = obj is not None
        solution_loader = GurobiDirectQuadraticSolutionLoader(
            solver_model=self._solver_model,
            var_id_map=self._vars,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
            linear_cons=self._linear_cons,
            quadratic_cons=self._quadratic_cons,
        )
        timer.stop('create gurobipy model')
        return self._solver_model, solution_loader, has_obj
    
    def _clear(self):
        self._solver_model = None
        self._vars = {}
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
        self._linear_cons = set()
        self._quadratic_cons = set()
        self._pyomo_sos_to_solver_sos_map = {}

    def _pyomo_gurobi_var_iter(self):
        for vid, v in self._vars.items():
            yield v, self._pyomo_var_to_solver_var_map[vid]

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
            raise ValueError(
                f'Unrecognized domain: {var.domain}'
            )
        if var.fixed:
            lb = var.value
            ub = lb
        else:
            if var._lb is not None:
                lb = max(lb, value(var._lb))
            if var._ub is not None:
                ub = min(ub, value(var._ub))
        return lb, ub, vtype

    def _add_variables(self, variables: List[VarData]):
        vtypes = []
        lbs = []
        ubs = []
        for ndx, var in enumerate(variables):
            self._vars[id(var)] = var
            lb, ub, vtype = self._process_domain_and_bounds(var)
            vtypes.append(vtype)
            lbs.append(lb)
            ubs.append(ub)

        gurobi_vars = self._solver_model.addVars(
            len(variables), lb=lbs, ub=ubs, vtype=vtypes
        ).values()

        for pyomo_var, gurobi_var in zip(variables, gurobi_vars):
            self._pyomo_var_to_solver_var_map[id(pyomo_var)] = gurobi_var

    def _get_expr_from_pyomo_repn(self, repn):
        if repn.nonlinear_expr is not None:
            raise IncompatibleModelError(
                f'GurobiDirectQuadratic only supports linear and quadratic expressions: {expr}.'
            )
        
        if len(repn.linear_vars) > 0:
            missing_vars = [v for v in repn.linear_vars if id(v) not in self._vars]
            self._add_variables(missing_vars)
            vlist = [self._pyomo_var_to_solver_var_map[id(v)] for v in repn.linear_vars]
            new_expr = gurobipy.LinExpr(
                repn.linear_coefs,
                vlist,
            )
        else:
            new_expr = 0.0

        if len(repn.quadratic_vars) > 0:
            missing_vars = {}
            for x, y in repn.quadratic_vars:
                for v in [x, y]:
                    vid = id(v)
                    if vid not in self._vars:
                        missing_vars[vid] = v
            self._add_variables(list(missing_vars.values()))
            for coef, (x, y) in zip(repn.quadratic_coefs, repn.quadratic_vars):
                gurobi_x = self._pyomo_var_to_solver_var_map[id(x)]
                gurobi_y = self._pyomo_var_to_solver_var_map[id(y)]
                new_expr += coef * gurobi_x * gurobi_y

        return new_expr, repn.constant

    def _add_constraints(self, cons: List[ConstraintData]):
        gurobi_expr_list = []
        for con in cons:
            lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
            repn = generate_standard_repn(body, quadratic=True, compute_values=True)
            if len(repn.quadratic_vars) > 0:
                self._quadratic_cons.add(con)
            else:
                self._linear_cons.add(con)
            gurobi_expr, repn_constant = self._get_expr_from_pyomo_repn(repn)
            if lb is None and ub is None:
                raise ValueError(
                    "Constraint does not have a lower "
                    f"or an upper bound: {con} \n"
                )
            elif lb is None:
                gurobi_expr_list.append(gurobi_expr <= float(ub - repn_constant))
            elif ub is None:
                gurobi_expr_list.append(float(lb - repn_constant) <= gurobi_expr)
            elif lb == ub:
                gurobi_expr_list.append(gurobi_expr == float(lb - repn_constant))
            else:
                gurobi_expr_list.append(gurobi_expr == [float(lb-repn_constant), float(ub-repn_constant)])

        gurobi_cons = self._solver_model.addConstrs((gurobi_expr_list[i] for i in range(len(gurobi_expr_list)))).values()
        self._pyomo_con_to_solver_con_map.update(zip(cons, gurobi_cons))

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
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

            missing_vars = {id(v): v for v, w in con.get_items() if id(v) not in self._vars}
            self._add_variables(list(missing_vars.values()))

            for v, w in con.get_items():
                v_id = id(v)
                gurobi_vars.append(self._pyomo_var_to_solver_var_map[v_id])
                weights.append(w)

            gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
            self._pyomo_sos_to_solver_sos_map[con] = gurobipy_con

    def _set_objective(self, obj):
        if obj is None:
            sense = gurobipy.GRB.MINIMIZE
            gurobi_expr = 0
            repn_constant = 0
        else:
            if obj.sense == minimize:
                sense = gurobipy.GRB.MINIMIZE
            elif obj.sense == maximize:
                sense = gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError(f'Objective sense is not recognized: {obj.sense}')

            repn = generate_standard_repn(obj.expr, quadratic=True, compute_values=True)
            gurobi_expr, repn_constant = self._get_expr_from_pyomo_repn(repn)

        self._solver_model.setObjective(gurobi_expr + repn_constant, sense=sense)
        self._needs_updated = True


class GurobiPersistentQuadratic(GurobiDirectQuadratic):
    _minimum_version = (7, 0, 0)
