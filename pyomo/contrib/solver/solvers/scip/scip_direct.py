# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from __future__ import annotations

import datetime
import io
import logging
import math
from typing import Tuple, List

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.tee import capture_output, TeeStream

from pyomo.core.base.block import BlockData
from pyomo.core.base.constraint import Constraint, ConstraintData
from pyomo.core.base.sos import SOSConstraint, SOSConstraintData
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.staleflag import StaleFlagManager

from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.solution_loader import NoSolutionSolutionLoader
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    get_objective,
)

from pyomo.contrib.solver.solvers.scip.base import (
    scip,
    scip_available,
    ScipConfig,
    _PyomoToScipVisitor,
    ScipSolutionLoader,
)

logger = logging.getLogger(__name__)


class ScipDirect(SolverBase):

    _available = None
    _tc_map = None
    _minimum_version = (5, 5, 0)  # this is probably conservative

    CONFIG = ScipConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_param_to_solver_param_map = (
            ComponentMap()
        )  # param to scip var with equal bounds
        self._pyomo_sos_to_solver_sos_map = {}
        self._expr_visitor = _PyomoToScipVisitor(self)
        self._objective = None  # pyomo objective
        self._obj_var = (
            None  # a scip variable because the objective cannot be nonlinear
        )
        self._obj_con = None  # a scip constraint (obj_var >= obj_expr)

    def _clear(self):
        self._solver_model = None
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = {}
        self._pyomo_param_to_solver_param_map = ComponentMap()
        self._pyomo_sos_to_solver_sos_map = {}
        self._objective = None
        self._obj_var = None
        self._obj_con = None

    def available(self) -> Availability:
        if self._available is not None:
            return self._available

        if not scip_available:
            ScipDirect._available = Availability.NotFound
        elif self.version() < self._minimum_version:
            ScipDirect._available = Availability.BadVersion
        else:
            ScipDirect._available = Availability.FullLicense

        return self._available

    def version(self) -> Tuple:
        return tuple(int(i) for i in scip.__version__.split('.'))

    def solve(self, model: BlockData, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        config = self.config(value=kwds, preserve_implicit=True)

        StaleFlagManager.mark_all_as_stale()

        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        ostreams = [io.StringIO()] + config.tee

        scip_model, solution_loader, has_obj = self._create_solver_model(model, config)

        scip_model.hideOutput(quiet=False)
        if config.threads is not None:
            scip_model.setParam('lp/threads', config.threads)
        if config.time_limit is not None:
            scip_model.setParam('limits/time', config.time_limit)
        if config.rel_gap is not None:
            scip_model.setParam('limits/gap', config.rel_gap)
        if config.abs_gap is not None:
            scip_model.setParam('limits/absgap', config.abs_gap)

        if config.warmstart_discrete_vars:
            self._mipstart()

        for key, option in config.solver_options.items():
            scip_model.setParam(key, option)

        timer.start('optimize')
        with capture_output(TeeStream(*ostreams), capture_fd=True):
            scip_model.optimize()
        timer.stop('optimize')

        results = self._populate_results(scip_model, solution_loader, has_obj, config)

        results.solver_log = ostreams[0].getvalue()
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        results.timing_info.start_timestamp = start_timestamp
        results.timing_info.wall_time = (
            end_timestamp - start_timestamp
        ).total_seconds()
        results.timing_info.timer = timer
        return results

    def _get_tc_map(self):
        if ScipDirect._tc_map is None:
            tc = TerminationCondition
            ScipDirect._tc_map = {
                "unknown": tc.unknown,
                "userinterrupt": tc.interrupted,
                "nodelimit": tc.iterationLimit,
                "totalnodelimit": tc.iterationLimit,
                "stallnodelimit": tc.iterationLimit,
                "timelimit": tc.maxTimeLimit,
                "memlimit": tc.unknown,
                "gaplimit": tc.convergenceCriteriaSatisfied,  # TODO: check this
                "primallimit": tc.objectiveLimit,
                "duallimit": tc.objectiveLimit,
                "sollimit": tc.unknown,
                "bestsollimit": tc.unknown,
                "restartlimit": tc.unknown,
                "optimal": tc.convergenceCriteriaSatisfied,
                "infeasible": tc.provenInfeasible,
                "unbounded": tc.unbounded,
                "inforunbd": tc.infeasibleOrUnbounded,
                "terminate": tc.unknown,
            }
        return ScipDirect._tc_map

    def _scip_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val

        lb, ub = var.bounds

        if lb is None:
            lb = -self._solver_model.infinity()
        if ub is None:
            ub = self._solver_model.infinity()

        return lb, ub

    def _add_var(self, var):
        vtype = self._scip_vtype_from_var(var)
        lb, ub = self._scip_lb_ub_from_var(var)

        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype)

        self._pyomo_var_to_solver_var_map[var] = scip_var
        return scip_var

    def _add_param(self, p):
        vtype = "C"
        lb = ub = p.value
        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype)
        self._pyomo_param_to_solver_param_map[p] = scip_var
        return scip_var

    def _add_constraints(self, cons: List[ConstraintData]):
        for con in cons:
            self._add_constraint(con)

    def _add_sos_constraints(self, cons: List[SOSConstraintData]):
        for con in cons:
            self._add_sos_constraint(con)

    def _create_solver_model(self, model, config):
        timer = config.timer
        timer.start('create scip model')
        self._clear()
        self._solver_model = scip.Model()
        timer.start('collect constraints')
        cons = list(
            model.component_data_objects(Constraint, descend_into=True, active=True)
        )
        timer.stop('collect constraints')
        timer.start('translate constraints')
        self._add_constraints(cons)
        timer.stop('translate constraints')
        timer.start('sos')
        sos = list(
            model.component_data_objects(SOSConstraint, descend_into=True, active=True)
        )
        self._add_sos_constraints(sos)
        timer.stop('sos')
        timer.start('get objective')
        obj = get_objective(model)
        timer.stop('get objective')
        timer.start('translate objective')
        self._set_objective(obj)
        timer.stop('translate objective')
        has_obj = obj is not None
        solution_loader = ScipSolutionLoader(
            solver_model=self._solver_model,
            var_map=self._pyomo_var_to_solver_var_map,
            con_map=self._pyomo_con_to_solver_con_map,
            pyomo_model=model,
            opt=self,
        )
        timer.stop('create scip model')
        return self._solver_model, solution_loader, has_obj

    def _add_constraint(self, con):
        scip_expr = self._expr_visitor.walk_expression(con.expr)
        scip_con = self._solver_model.addCons(scip_expr)
        self._pyomo_con_to_solver_con_map[con] = scip_con

    def _add_sos_constraint(self, con):
        level = con.level
        if level not in [1, 2]:
            raise ValueError(
                f"{self.name} does not support SOS level {level} constraints"
            )

        scip_vars = []
        weights = []

        for v, w in con.get_items():
            if v not in self._pyomo_var_to_solver_var_map:
                self._add_var(v)
            scip_vars.append(self._pyomo_var_to_solver_var_map[v])
            weights.append(w)

        if level == 1:
            scip_cons = self._solver_model.addConsSOS1(scip_vars, weights=weights)
        else:
            scip_cons = self._solver_model.addConsSOS2(scip_vars, weights=weights)
        self._pyomo_con_to_solver_con_map[con] = scip_cons

    def _scip_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate SCIP variable type

        Parameters
        ----------
        var: pyomo.core.base.var.Var
            The pyomo variable that we want to retrieve the SCIP vtype of

        Returns
        -------
        vtype: str
            B for Binary, I for Integer, or C for Continuous
        """
        if var.is_binary():
            vtype = "B"
        elif var.is_integer():
            vtype = "I"
        elif var.is_continuous():
            vtype = "C"
        else:
            raise ValueError(f"Variable domain type is not recognized for {var.domain}")
        return vtype

    def _set_objective(self, obj):
        if self._obj_var is None:
            self._obj_var = self._solver_model.addVar(
                lb=-self._solver_model.infinity(),
                ub=self._solver_model.infinity(),
                vtype="C",
            )

        if self._obj_con is not None:
            self._solver_model.delCons(self._obj_con)

        if obj is None:
            scip_expr = 0
            sense = "minimize"
        else:
            scip_expr = self._expr_visitor.walk_expression(obj.expr)
            if obj.sense == minimize:
                sense = "minimize"
            elif obj.sense == maximize:
                sense = "maximize"
            else:
                raise ValueError(f"Objective sense is not recognized: {obj.sense}")

        if sense == "minimize":
            self._obj_con = self._solver_model.addCons(self._obj_var >= scip_expr)
        else:
            self._obj_con = self._solver_model.addCons(self._obj_var <= scip_expr)

        self._solver_model.setObjective(self._obj_var, sense=sense)
        self._objective = obj

    def _populate_results(self, scip_model, solution_loader, has_obj, config):
        results = Results()
        results.solution_loader = solution_loader
        results.timing_info.scip_time = scip_model.getSolvingTime()
        results.termination_condition = self._get_tc_map().get(
            scip_model.getStatus(), TerminationCondition.unknown
        )

        if solution_loader.get_number_of_solutions() > 0:
            if (
                results.termination_condition
                == TerminationCondition.convergenceCriteriaSatisfied
            ):
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if has_obj:
            try:
                if (
                    scip_model.getNSols() > 0
                    and scip_model.getObjVal() < scip_model.infinity()
                ):
                    results.incumbent_objective = scip_model.getObjVal()
                else:
                    results.incumbent_objective = None
            except:
                results.incumbent_objective = None
            try:
                results.objective_bound = scip_model.getDualbound()
                if results.objective_bound <= -scip_model.infinity():
                    results.objective_bound = -math.inf
                if results.objective_bound >= scip_model.infinity():
                    results.objective_bound = math.inf
            except:
                if self._objective.sense == minimize:
                    results.objective_bound = -math.inf
                else:
                    results.objective_bound = math.inf
        else:
            results.incumbent_objective = None
            results.objective_bound = None

        config.timer.start('load solution')
        if config.load_solutions:
            if solution_loader.get_number_of_solutions() > 0:
                solution_loader.load_solution()
            else:
                raise NoFeasibleSolutionError()
        config.timer.stop('load solution')

        results.extra_info['NNodes'] = scip_model.getNNodes()
        results.solver_config = config
        results.solver_name = self.name
        results.solver_version = self.version()

        return results

    def _mipstart(self):
        # TODO: it is also possible to specify continuous variables, but
        #       I think we should have a different option for that
        sol = self._solver_model.createPartialSol()
        for pyomo_var, scip_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.is_integer():
                sol[scip_var] = pyomo_var.value
        self._solver_model.addSol(sol)
