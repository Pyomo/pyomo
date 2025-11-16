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

import datetime
import io
import math
import os
import logging
from typing import Mapping, Optional, Sequence, Dict

from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.common.enums import ObjectiveSense
from pyomo.common.errors import ApplicationError
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.base import VarData, ConstraintData

from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase


logger = logging.getLogger(__name__)

gurobipy, gurobipy_available = attempt_import('gurobipy')


class GurobiConfig(BranchAndBoundConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        BranchAndBoundConfig.__init__(
            self,
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.warmstart_discrete_vars: bool = self.declare(
            'warmstart_discrete_vars',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Gurobi.",
            ),
        )


def _load_suboptimal_mip_solution(solver_model, var_map, vars_to_load, solution_number):
    """
    solver_model: gurobipy.Model
    var_map: Mapping[VarData, gurobipy.Var]
        Maps the pyomo variable to the gurobipy variable
    vars_to_load: List[VarData]
    solution_number: int
    """
    if (
        solver_model.getAttr('NumIntVars') == 0
        and solver_model.getAttr('NumBinVars') == 0
    ):
        raise ValueError('Cannot obtain suboptimal solutions for a continuous model')
    original_solution_number = solver_model.getParamInfo('SolutionNumber')[2]
    solver_model.setParam('SolutionNumber', solution_number)
    gurobi_vars_to_load = [var_map[v] for v in vars_to_load]
    vals = solver_model.getAttr("Xn", gurobi_vars_to_load)
    res = ComponentMap(zip(vars_to_load, vals))
    solver_model.setParam('SolutionNumber', original_solution_number)
    return res


def _load_vars(solver_model, var_map, vars_to_load, solution_number=0):
    """
    solver_model: gurobipy.Model
    var_map: Mapping[VarData, gurobipy.Var]
        Maps the pyomo variable to the gurobipy variable
    vars_to_load: List[VarData]
    solution_number: int
    """
    for v, val in _get_primals(
        solver_model=solver_model,
        var_map=var_map,
        vars_to_load=vars_to_load,
        solution_number=solution_number,
    ).items():
        v.set_value(val, skip_validation=True)
    StaleFlagManager.mark_all_as_stale(delayed=True)


def _get_primals(solver_model, var_map, vars_to_load, solution_number=0):
    """
    solver_model: gurobipy.Model
    var_map: Mapping[Vardata, gurobipy.Var]
        Maps the pyomo variable to the gurobipy variable
    vars_to_load: List[VarData]
    solution_number: int
    """
    if solver_model.SolCount == 0:
        raise NoSolutionError()

    if solution_number != 0:
        return _load_suboptimal_mip_solution(
            solver_model=solver_model,
            var_map=var_map,
            vars_to_load=vars_to_load,
            solution_number=solution_number,
        )

    gurobi_vars_to_load = [var_map[v] for v in vars_to_load]
    vals = solver_model.getAttr("X", gurobi_vars_to_load)

    res = ComponentMap(zip(vars_to_load, vals))
    return res


def _get_reduced_costs(solver_model, var_map, vars_to_load):
    """
    solver_model: gurobipy.Model
    var_map: Mapping[VarData, gurobipy.Var]
        Maps the pyomo variable to the gurobipy variable
    vars_to_load: List[VarData]
    """
    if solver_model.Status != gurobipy.GRB.OPTIMAL:
        raise NoReducedCostsError()
    if solver_model.IsMIP:
        # this will also return True for continuous, nonconvex models
        raise NoReducedCostsError()

    gurobi_vars_to_load = [var_map[v] for v in vars_to_load]
    vals = solver_model.getAttr("Rc", gurobi_vars_to_load)

    res = ComponentMap(zip(vars_to_load, vals))
    return res


def _get_duals(solver_model, con_map, cons_to_load):
    """
    solver_model: gurobipy.Model
    con_map: Dict[ConstraintData, gurobipy.Constr]
        Maps the pyomo constraint to the gurobipy constraint
    cons_to_load: List[ConstraintData]
    """
    if solver_model.Status != gurobipy.GRB.OPTIMAL:
        raise NoDualsError()
    if solver_model.IsMIP:
        # this will also return True for continuous, nonconvex models
        raise NoDualsError()

    qcons = set(solver_model.getQConstrs())

    duals = {}
    for c in cons_to_load:
        gurobi_con = con_map[c]
        if type(gurobi_con) is tuple:
            # only linear range constraints are supported
            gc1, gc2 = gurobi_con
            d1 = gc1.Pi
            d2 = gc2.Pi
            if abs(d1) > abs(d2):
                duals[c] = d1
            else:
                duals[c] = d2
        else:
            if gurobi_con in qcons:
                duals[c] = gurobi_con.QCPi
            else:
                duals[c] = gurobi_con.Pi

    return duals


class GurobiDirectSolutionLoaderBase(SolutionLoaderBase):
    def __init__(self, solver_model, var_map, con_map) -> None:
        super().__init__()
        self._solver_model = solver_model
        self._var_map = var_map
        self._con_map = con_map
        GurobiDirectBase._register_env_client()

    def __del__(self):
        # Release the gurobi license if this is the last reference to
        # the environment (either through a results object or solver
        # interface)
        GurobiDirectBase._release_env_client()

    def load_vars(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=0
    ) -> None:
        if vars_to_load is None:
            vars_to_load = self._var_map
        _load_vars(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
            solution_number=solution_id,
        )

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None, solution_id=0
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = self._var_map
        return _get_primals(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
            solution_number=solution_id,
        )

    def get_reduced_costs(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if vars_to_load is None:
            vars_to_load = self._var_map
        return _get_reduced_costs(
            solver_model=self._solver_model,
            var_map=self._var_map,
            vars_to_load=vars_to_load,
        )

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        if cons_to_load is None:
            cons_to_load = self._con_map
        return _get_duals(
            solver_model=self._solver_model,
            con_map=self._con_map,
            cons_to_load=cons_to_load,
        )


class GurobiDirectBase(SolverBase):

    _num_gurobipy_env_clients = 0
    _gurobipy_env = None
    _available = None
    _gurobipy_available = gurobipy_available
    _tc_map = None
    _minimum_version = (0, 0, 0)

    CONFIG = GurobiConfig()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._register_env_client()
        self._callback = None

    def __del__(self):
        if not python_is_shutting_down():
            self._release_env_client()

    def available(self):
        if self._available is None:
            # this triggers the deferred import, and for the persistent
            # interface, may update the _available flag
            #
            # Note that we set the _available flag on the *most derived
            # class* and not on the instance, or on the base class.  That
            # allows different derived interfaces to have different
            # availability (e.g., persistent has a minimum version
            # requirement that the direct interface doesn't - is that true?)
            if not self._gurobipy_available:
                if self._available is None:
                    self.__class__._available = Availability.NotFound
            else:
                self.__class__._available = self._check_license()
                if self.version() < self._minimum_version:
                    self.__class__._available = Availability.BadVersion
        return self._available

    @staticmethod
    def release_license():
        if GurobiDirectBase._gurobipy_env is None:
            return
        if GurobiDirectBase._num_gurobipy_env_clients:
            logger.warning(
                "Call to GurobiDirectBase.release_license() with %s remaining "
                "environment clients." % (GurobiDirectBase._num_gurobipy_env_clients,)
            )
        GurobiDirectBase._gurobipy_env.close()
        GurobiDirectBase._gurobipy_env = None

    @staticmethod
    def env():
        if GurobiDirectBase._gurobipy_env is None:
            with capture_output(capture_fd=True):
                GurobiDirectBase._gurobipy_env = gurobipy.Env()
        return GurobiDirectBase._gurobipy_env

    @staticmethod
    def _register_env_client():
        GurobiDirectBase._num_gurobipy_env_clients += 1

    @staticmethod
    def _release_env_client():
        GurobiDirectBase._num_gurobipy_env_clients -= 1
        if GurobiDirectBase._num_gurobipy_env_clients <= 0:
            # Note that _num_gurobipy_env_clients should never be <0,
            # but if it is, release_license will issue a warning (that
            # we want to know about)
            GurobiDirectBase.release_license()

    def _check_license(self):
        try:
            model = gurobipy.Model(env=self.env())
        except gurobipy.GurobiError:
            return Availability.BadLicense

        model.setParam('OutputFlag', 0)
        try:
            model.addVars(range(2001))
            model.optimize()
            return Availability.FullLicense
        except gurobipy.GurobiError:
            return Availability.LimitedLicense
        finally:
            model.dispose()

    def version(self):
        if not gurobipy_available:
            return None
        version = (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )
        return version

    def _create_solver_model(self, pyomo_model, config):
        # should return gurobi_model, solution_loader, has_objective
        raise NotImplementedError('should be implemented by derived classes')

    def _pyomo_gurobi_var_iter(self):
        # generator of tuples (pyomo_var, gurobi_var)
        raise NotImplementedError('should be implemented by derived classes')

    def _mipstart(self):
        for pyomo_var, gurobi_var in self._pyomo_gurobi_var_iter():
            if pyomo_var.is_integer() and pyomo_var.value is not None:
                gurobi_var.setAttr('Start', pyomo_var.value)

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        orig_cwd = os.getcwd()
        try:
            config = self.config(value=kwds, preserve_implicit=True)

            if config.timer is None:
                config.timer = HierarchicalTimer()
            timer = config.timer

            StaleFlagManager.mark_all_as_stale()
            ostreams = [io.StringIO()] + config.tee

            if config.working_dir:
                os.chdir(config.working_dir)
            with capture_output(TeeStream(*ostreams), capture_fd=False):
                gurobi_model, solution_loader, has_obj = self._create_solver_model(
                    model,
                    config,
                )
                options = config.solver_options

                gurobi_model.setParam('LogToConsole', 1)

                if config.threads is not None:
                    gurobi_model.setParam('Threads', config.threads)
                if config.time_limit is not None:
                    gurobi_model.setParam('TimeLimit', config.time_limit)
                if config.rel_gap is not None:
                    gurobi_model.setParam('MIPGap', config.rel_gap)
                if config.abs_gap is not None:
                    gurobi_model.setParam('MIPGapAbs', config.abs_gap)

                if config.warmstart_discrete_vars:
                    self._mipstart()

                for key, option in options.items():
                    gurobi_model.setParam(key, option)

                timer.start('optimize')
                gurobi_model.optimize(self._callback)
                timer.stop('optimize')

            res = self._populate_results(
                grb_model=gurobi_model, solution_loader=solution_loader, has_obj=has_obj, config=config
            )
        finally:
            os.chdir(orig_cwd)

        res.solver_log = ostreams[0].getvalue()
        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        return res

    def _get_tc_map(self):
        if GurobiDirectBase._tc_map is None:
            grb = gurobipy.GRB
            tc = TerminationCondition
            GurobiDirectBase._tc_map = {
                grb.LOADED: tc.unknown,  # problem is loaded, but no solution
                grb.OPTIMAL: tc.convergenceCriteriaSatisfied,
                grb.INFEASIBLE: tc.provenInfeasible,
                grb.INF_OR_UNBD: tc.infeasibleOrUnbounded,
                grb.UNBOUNDED: tc.unbounded,
                grb.CUTOFF: tc.objectiveLimit,
                grb.ITERATION_LIMIT: tc.iterationLimit,
                grb.NODE_LIMIT: tc.iterationLimit,
                grb.TIME_LIMIT: tc.maxTimeLimit,
                grb.SOLUTION_LIMIT: tc.unknown,
                grb.INTERRUPTED: tc.interrupted,
                grb.NUMERIC: tc.unknown,
                grb.SUBOPTIMAL: tc.unknown,
                grb.USER_OBJ_LIMIT: tc.objectiveLimit,
            }
        return GurobiDirectBase._tc_map

    def _populate_results(self, grb_model, solution_loader, has_obj, config):
        status = grb_model.Status

        results = Results()
        results.solution_loader = solution_loader
        results.timing_info.gurobi_time = grb_model.Runtime

        if grb_model.SolCount > 0:
            if status == gurobipy.GRB.OPTIMAL:
                results.solution_status = SolutionStatus.optimal
            else:
                results.solution_status = SolutionStatus.feasible
        else:
            results.solution_status = SolutionStatus.noSolution

        results.termination_condition = self._get_tc_map().get(
            status, TerminationCondition.unknown
        )

        if (
            results.termination_condition
            != TerminationCondition.convergenceCriteriaSatisfied
            and config.raise_exception_on_nonoptimal_result
        ):
            raise NoOptimalSolutionError()

        if has_obj:
            try:
                if math.isfinite(grb_model.ObjVal):
                    results.incumbent_objective = grb_model.ObjVal
                else:
                    results.incumbent_objective = None
            except (gurobipy.GurobiError, AttributeError):
                results.incumbent_objective = None
            try:
                results.objective_bound = grb_model.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                if grb_model.ModelSense == ObjectiveSense.minimize:
                    results.objective_bound = -math.inf
                else:
                    results.objective_bound = math.inf
        else:
            results.incumbent_objective = None
            results.objective_bound = None

        results.extra_info.IterCount = grb_model.getAttr('IterCount')
        results.extra_info.BarIterCount = grb_model.getAttr('BarIterCount')
        results.extra_info.NodeCount = grb_model.getAttr('NodeCount')

        config.timer.start('load solution')
        if config.load_solutions:
            if grb_model.SolCount > 0:
                results.solution_loader.load_vars()
            else:
                raise NoFeasibleSolutionError()
        config.timer.stop('load solution')

        results.solver_config = config
        results.solver_name = self.name
        results.solver_version = self.version()

        return results
