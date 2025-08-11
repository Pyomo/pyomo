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
from typing import List, Optional
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
from pyomo.contrib.solver.common.solution_loader import PersistentSolutionLoader
from pyomo.core.staleflag import StaleFlagManager
from .gurobi_direct_base import GurobiConfig, GurobiDirectBase, gurobipy
from pyomo.contrib.solver.common.util import get_objective
from pyomo.repn.quadratic import QuadraticRepn, QuadraticRepnVisitor


logger = logging.getLogger(__name__)


class GurobiSolutionLoader(PersistentSolutionLoader):
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
        self._pyomo_sos_to_solver_sos_map = {}

    def _create_solver_model(self, pyomo_model):
        self._clear()
        self._solver_model = gurobipy.Model(env=self.env())
        cons = list(pyomo_model.component_data_objects(Constraint, descend_into=True, active=True))
        self._add_constraints(cons)
        sos = list(pyomo_model.component_data_objects(SOSConstraint, descend_into=True, active=True))
        self._add_sos_constraints(sos)
        obj = get_objective(pyomo_model)
        self._set_objective(obj)
    
    def _clear(self):
        self._solver_model = None
        self._vars = {}
        self._pyomo_var_to_solver_var_map = {}
        self._pyomo_con_to_solver_con_map = {}
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
            lb = max(lb, value(var._lb))
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
        )

        for pyomo_var, gurobi_var in zip(variables, gurobi_vars):
            self._pyomo_var_to_solver_var_map[id(pyomo_var)] = gurobi_var

    def _get_expr_from_pyomo_expr(self, expr):
        repn = generate_standard_repn(expr, quadratic=True, compute_values=True)

        if repn.nonlinear_expr is not None:
            raise IncompatibleModelError(
                f'GurobiDirectQuadratic only supports linear and quadratic expressions: {expr}.'
            )
        
        if len(repn.linear_vars) > 0:
            missing_vars = [v for v in repn.linear_vars if id(v) not in self._vars]
            self._add_variables(missing_vars)
            new_expr = gurobipy.LinExpr(
                repn.linear_coefs,
                [self._pyomo_var_to_solver_var_map[id(v)] for v in repn.linear_vars],
            )
        else:
            new_expr = 0.0

        for coef, v in zip(repn.quadratic_coefs, repn.quadratic_vars):
            x, y = v
            gurobi_x = self._pyomo_var_to_solver_var_map[id(x)]
            gurobi_y = self._pyomo_var_to_solver_var_map[id(y)]
            new_expr += coef * gurobi_x * gurobi_y

        return new_expr, repn.constant

    def _add_constraints(self, cons: List[ConstraintData]):
        gurobi_expr_list = []
        for con in cons:
            lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
            gurobi_expr, repn_constant = self._get_expr_from_pyomo_expr(body)
            if lb is None and ub is None:
                raise ValueError(
                    "Constraint does not have a lower "
                    f"or an upper bound: {con} \n"
                )
            elif lb is None:
                gurobi_expr_list.append(gurobi_expr <= ub - repn_constant)
            elif ub is None:
                gurobi_expr_list.append(lb - repn_constant <= gurobi_expr)
            elif lb == ub:
                gurobi_expr_list.append(gurobi_expr == lb - repn_constant)
            else:
                gurobi_expr_list.append(gurobi_expr == [lb-repn_constant, ub-repn_constant])

        gurobi_cons = self._solver_model.addConstrs(gurobi_expr_list)
        self._pyomo_con_to_solver_con_map.update(zip(cons, gurobi_cons))
            self._pyomo_con_to_solver_con_map[con] = gurobipy_con
            self._solver_con_to_pyomo_con_map[id(gurobipy_con)] = con
        self._constraints_added_since_update.update(cons)
        self._needs_updated = True


class GurobiPersistentQuadratic(GurobiDirectQuadratic):
    _minimum_version = (7, 0, 0)
