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
import operator
import os

from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigValue
from pyomo.common.dependencies import attempt_import
from pyomo.common.enums import ObjectiveSense
from pyomo.common.errors import MouseTrap, ApplicationError
from pyomo.common.shutdown import python_is_shutting_down
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

from pyomo.contrib.solver.common.base import SolverBase, Availability
from pyomo.contrib.solver.common.config import BranchAndBoundConfig
from pyomo.contrib.solver.common.util import (
    NoFeasibleSolutionError,
    NoOptimalSolutionError,
    NoDualsError,
    NoReducedCostsError,
    NoSolutionError,
    IncompatibleModelError,
)
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase


gurobipy, gurobipy_available = attempt_import('gurobipy')


class GurobiConfigMixin:
    """
    Mixin class for Gurobi-specific configurations
    """

    def __init__(self):
        self.use_mipstart: bool = self.declare(
            'use_mipstart',
            ConfigValue(
                default=False,
                domain=bool,
                description="If True, the current values of the integer variables "
                "will be passed to Gurobi.",
            ),
        )


class GurobiConfig(BranchAndBoundConfig, GurobiConfigMixin):
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
        GurobiConfigMixin.__init__(self)


class GurobiDirectSolutionLoader(SolutionLoaderBase):
    def __init__(self, grb_model, grb_cons, grb_vars, pyo_cons, pyo_vars, pyo_obj):
        self._grb_model = grb_model
        self._grb_cons = grb_cons
        self._grb_vars = grb_vars
        self._pyo_cons = pyo_cons
        self._pyo_vars = pyo_vars
        self._pyo_obj = pyo_obj
        GurobiDirect._register_env_client()

    def __del__(self):
        if python_is_shutting_down():
            return
        # Free the associated model
        if self._grb_model is not None:
            self._grb_cons = None
            self._grb_vars = None
            self._pyo_cons = None
            self._pyo_vars = None
            self._pyo_obj = None
            # explicitly release the model
            self._grb_model.dispose()
            self._grb_model = None
        # Release the gurobi license if this is the last reference to
        # the environment (either through a results object or solver
        # interface)
        GurobiDirect._release_env_client()

    def load_vars(self, vars_to_load=None, solution_number=0):
        assert solution_number == 0
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, self._grb_vars.x.tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        for p_var, g_var in iterator:
            p_var.set_value(g_var, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None, solution_number=0):
        assert solution_number == 0
        if self._grb_model.SolCount == 0:
            raise NoSolutionError()

        iterator = zip(self._pyo_vars, self._grb_vars.x.tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_val: var_val[0] in vars_to_load, iterator)
        return ComponentMap(iterator)

    def get_duals(self, cons_to_load=None):
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoDualsError()

        def dedup(_iter):
            last = None
            for con_info_dual in _iter:
                if not con_info_dual[1] and con_info_dual[0][0] is last:
                    continue
                last = con_info_dual[0][0]
                yield con_info_dual

        iterator = dedup(zip(self._pyo_cons, self._grb_cons.getAttr('Pi').tolist()))
        if cons_to_load:
            cons_to_load = set(cons_to_load)
            iterator = filter(
                lambda con_info_dual: con_info_dual[0][0] in cons_to_load, iterator
            )
        return {con_info[0]: dual for con_info, dual in iterator}

    def get_reduced_costs(self, vars_to_load=None):
        if self._grb_model.Status != gurobipy.GRB.OPTIMAL:
            raise NoReducedCostsError()

        iterator = zip(self._pyo_vars, self._grb_vars.getAttr('Rc').tolist())
        if vars_to_load:
            vars_to_load = ComponentSet(vars_to_load)
            iterator = filter(lambda var_rc: var_rc[0] in vars_to_load, iterator)
        return ComponentMap(iterator)


class GurobiSolverMixin:
    """
    gurobi_direct and gurobi_persistent check availability and set versions
    in the same way. This moves the logic to a central location to reduce
    duplicate code.
    """

    _num_gurobipy_env_clients = 0
    _gurobipy_env = None
    _available = None
    _gurobipy_available = gurobipy_available

    def available(self):
        if self._available is None:
            # this triggers the deferred import, and for the persistent
            # interface, may update the _available flag
            #
            # Note that we set the _available flag on the *most derived
            # class* and not on the instance, or on the base class.  That
            # allows different derived interfaces to have different
            # availability (e.g., persistent has a minimum version
            # requirement that the direct interface doesn't)
            if not self._gurobipy_available:
                if self._available is None:
                    self.__class__._available = Availability.NotFound
            else:
                self.__class__._available = self._check_license()
        return self._available

    @staticmethod
    def release_license():
        if GurobiSolverMixin._gurobipy_env is None:
            return
        if GurobiSolverMixin._num_gurobipy_env_clients:
            logger.warning(
                "Call to GurobiSolverMixin.release_license() with %s remaining "
                "environment clients." % (GurobiSolverMixin._num_gurobipy_env_clients,)
            )
        GurobiSolverMixin._gurobipy_env.close()
        GurobiSolverMixin._gurobipy_env = None

    @staticmethod
    def env():
        if GurobiSolverMixin._gurobipy_env is None:
            with capture_output(capture_fd=True):
                GurobiSolverMixin._gurobipy_env = gurobipy.Env()
        return GurobiSolverMixin._gurobipy_env

    @staticmethod
    def _register_env_client():
        GurobiSolverMixin._num_gurobipy_env_clients += 1

    @staticmethod
    def _release_env_client():
        GurobiSolverMixin._num_gurobipy_env_clients -= 1
        if GurobiSolverMixin._num_gurobipy_env_clients <= 0:
            # Note that _num_gurobipy_env_clients should never be <0,
            # but if it is, release_license will issue a warning (that
            # we want to know about)
            GurobiSolverMixin.release_license()

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
        version = (
            gurobipy.GRB.VERSION_MAJOR,
            gurobipy.GRB.VERSION_MINOR,
            gurobipy.GRB.VERSION_TECHNICAL,
        )
        return version


class GurobiDirect(GurobiSolverMixin, SolverBase):
    """
    Interface to Gurobi using gurobipy
    """

    CONFIG = GurobiConfig()

    _tc_map = None

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._register_env_client()

    def __del__(self):
        if not python_is_shutting_down():
            self._release_env_client()

    def solve(self, model, **kwds) -> Results:
        start_timestamp = datetime.datetime.now(datetime.timezone.utc)
        config = self.config(value=kwds, preserve_implicit=True)
        if not self.available():
            c = self.__class__
            raise ApplicationError(
                f'Solver {c.__module__}.{c.__qualname__} is not available '
                f'({self.available()}).'
            )
        if config.timer is None:
            config.timer = HierarchicalTimer()
        timer = config.timer

        StaleFlagManager.mark_all_as_stale()

        timer.start('compile_model')
        repn = LinearStandardFormCompiler().write(
            model, mixed_form=True, set_sense=None
        )
        timer.stop('compile_model')

        if len(repn.objectives) > 1:
            raise IncompatibleModelError(
                f"The {self.__class__.__name__} solver only supports models "
                f"with zero or one objectives (received {len(repn.objectives)})."
            )

        timer.start('prepare_matrices')
        inf = float('inf')
        ninf = -inf
        bounds = list(map(operator.attrgetter('bounds'), repn.columns))
        lb = [ninf if _b is None else _b for _b in map(operator.itemgetter(0), bounds)]
        ub = [inf if _b is None else _b for _b in map(operator.itemgetter(1), bounds)]
        CON = gurobipy.GRB.CONTINUOUS
        BIN = gurobipy.GRB.BINARY
        INT = gurobipy.GRB.INTEGER
        vtype = [
            (
                CON
                if v.is_continuous()
                else BIN if v.is_binary() else INT if v.is_integer() else '?'
            )
            for v in repn.columns
        ]
        sense_type = list('=<>')  # Note: ordering matches 0, 1, -1
        sense = [sense_type[r[1]] for r in repn.rows]
        timer.stop('prepare_matrices')

        ostreams = [io.StringIO()] + config.tee
        res = Results()

        orig_cwd = os.getcwd()
        try:
            if config.working_dir:
                os.chdir(config.working_dir)
            with capture_output(TeeStream(*ostreams), capture_fd=False):
                gurobi_model = gurobipy.Model(env=self.env())

                timer.start('transfer_model')
                x = gurobi_model.addMVar(
                    len(repn.columns),
                    lb=lb,
                    ub=ub,
                    obj=repn.c.todense()[0] if repn.c.shape[0] else 0,
                    vtype=vtype,
                )
                A = gurobi_model.addMConstr(repn.A, x, sense, repn.rhs)
                if repn.c.shape[0]:
                    gurobi_model.setAttr('ObjCon', repn.c_offset[0])
                    gurobi_model.setAttr('ModelSense', int(repn.objectives[0].sense))
                # Note: calling gurobi_model.update() here is not
                # necessary (it will happen as part of optimize()):
                # gurobi_model.update()
                timer.stop('transfer_model')

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

                if config.use_mipstart:
                    raise MouseTrap("MIPSTART not yet supported")

                for key, option in options.items():
                    gurobi_model.setParam(key, option)

                timer.start('optimize')
                gurobi_model.optimize()
                timer.stop('optimize')
        finally:
            os.chdir(orig_cwd)

        res = self._postsolve(
            timer,
            config,
            GurobiDirectSolutionLoader(
                gurobi_model, A, x, repn.rows, repn.columns, repn.objectives
            ),
        )

        res.solver_config = config
        res.solver_name = 'Gurobi'
        res.solver_version = self.version()
        res.solver_log = ostreams[0].getvalue()

        end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        res.timing_info.start_timestamp = start_timestamp
        res.timing_info.wall_time = (end_timestamp - start_timestamp).total_seconds()
        res.timing_info.timer = timer
        return res

    def _postsolve(self, timer: HierarchicalTimer, config, loader):
        grb_model = loader._grb_model
        status = grb_model.Status

        results = Results()
        results.solution_loader = loader
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

        if loader._pyo_obj:
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

        results.iteration_count = grb_model.getAttr('IterCount')

        timer.start('load solution')
        if config.load_solutions:
            if grb_model.SolCount > 0:
                results.solution_loader.load_vars()
            else:
                raise NoFeasibleSolutionError()
        timer.stop('load solution')

        return results

    def _get_tc_map(self):
        if GurobiDirect._tc_map is None:
            grb = gurobipy.GRB
            tc = TerminationCondition
            GurobiDirect._tc_map = {
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
        return GurobiDirect._tc_map
