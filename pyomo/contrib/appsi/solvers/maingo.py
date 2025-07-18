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

from collections import namedtuple
import logging
import math
import sys
from typing import Optional, List, Dict

from pyomo.contrib.appsi.base import (
    PersistentSolver,
    Results,
    TerminationCondition,
    MIPSolverConfig,
    PersistentBase,
    PersistentSolutionLoader,
)
from pyomo.contrib.appsi.cmodel import cmodel, cmodel_available
from pyomo.common.collections import ComponentMap
from pyomo.common.config import (
    ConfigValue,
    ConfigDict,
    NonNegativeInt,
    NonNegativeFloat,
)
from pyomo.common.dependencies import attempt_import
from pyomo.common.errors import PyomoException
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output, TeeStream
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base import SymbolMap, NumericLabeler, TextLabeler
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.expression import ScalarExpression
from pyomo.core.base.param import _ParamData
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.core.base.var import Var, ScalarVar, _GeneralVarData
import pyomo.core.expr.expr_common as common
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
    value,
    is_constant,
    is_fixed,
    native_numeric_types,
    native_types,
    nonpyomo_leaf_types,
)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.util import valid_expr_ctypes_minlp


logger = logging.getLogger(__name__)
MaingoVar = namedtuple("MaingoVar", "type name lb ub init")
maingopy, maingopy_available = attempt_import("maingopy")
# Note that importing maingo_solvermodel will trigger the import of
# maingopy, so we defer that import using attempt_import (which will
# always succeed, even if maingopy is not available)
maingo_solvermodel = attempt_import("pyomo.contrib.appsi.solvers.maingo_solvermodel")[0]


class MAiNGOConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(MAiNGOConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )
        self.tolerances: ConfigDict = self.declare(
            'tolerances', ConfigDict(implicit=True)
        )

        self.tolerances.epsilonA: Optional[float] = self.tolerances.declare(
            'epsilonA',
            ConfigValue(
                domain=NonNegativeFloat,
                default=1e-5,
                description="Absolute optimality tolerance",
            ),
        )
        self.tolerances.epsilonR: Optional[float] = self.tolerances.declare(
            'epsilonR',
            ConfigValue(
                domain=NonNegativeFloat,
                default=1e-5,
                description="Relative optimality tolerance",
            ),
        )
        self.tolerances.deltaEq: Optional[float] = self.tolerances.declare(
            'deltaEq',
            ConfigValue(
                domain=NonNegativeFloat, default=1e-6, description="Equality tolerance"
            ),
        )

        self.tolerances.deltaIneq: Optional[float] = self.tolerances.declare(
            'deltaIneq',
            ConfigValue(
                domain=NonNegativeFloat,
                default=1e-6,
                description="Inequality tolerance",
            ),
        )
        self.declare("logfile", ConfigValue(domain=str, default=""))
        self.declare("solver_output_logger", ConfigValue(default=logger))
        self.declare(
            "log_level", ConfigValue(domain=NonNegativeInt, default=logging.INFO)
        )


class MAiNGOSolutionLoader(PersistentSolutionLoader):
    def load_vars(self, vars_to_load=None):
        self._assert_solution_still_valid()
        self._solver.load_vars(vars_to_load=vars_to_load)

    def get_primals(self, vars_to_load=None):
        self._assert_solution_still_valid()
        return self._solver.get_primals(vars_to_load=vars_to_load)


class MAiNGOResults(Results):
    def __init__(self, solver):
        super(MAiNGOResults, self).__init__()
        self.wallclock_time = None
        self.cpu_time = None
        self.globally_optimal = None
        self.solution_loader = MAiNGOSolutionLoader(solver=solver)


class MAiNGO(PersistentBase, PersistentSolver):
    """
    Interface to MAiNGO
    """

    _available = None

    def __init__(self, only_child_vars=False):
        super(MAiNGO, self).__init__(only_child_vars=only_child_vars)
        self._config = MAiNGOConfig()
        self._solver_options = dict()
        self._solver_model = None
        self._mymaingo = None
        self._symbol_map = SymbolMap()
        self._labeler = None
        self._maingo_vars = []
        self._objective = None
        self._cons = []
        self._pyomo_var_to_solver_var_id_map = dict()
        self._last_results_object: Optional[MAiNGOResults] = None

    def available(self):
        if self._available is None:
            if maingopy_available:
                MAiNGO._available = True
            else:
                MAiNGO._available = MAiNGO.Availability.NotFound
        return self._available

    def version(self):
        import importlib.metadata

        try:
            version = importlib.metadata.version('maingopy').split('.')
        except ImportError:
            return None
        for i, n in enumerate(version):
            try:
                version[i] = int(version[i])
            except:
                pass
        return tuple(version)

    @property
    def config(self) -> MAiNGOConfig:
        return self._config

    @config.setter
    def config(self, val: MAiNGOConfig):
        self._config = val

    @property
    def maingo_options(self):
        """
        A dictionary mapping solver options to values for those options. These
        are solver specific.

        Returns
        -------
        dict
            A dictionary mapping solver options to values for those options
        """
        return self._solver_options

    @maingo_options.setter
    def maingo_options(self, val: Dict):
        self._solver_options = val

    @property
    def symbol_map(self):
        return self._symbol_map

    def _solve(self, timer: HierarchicalTimer):
        ostreams = [
            LogStream(
                level=self.config.log_level, logger=self.config.solver_output_logger
            )
        ]
        if self.config.stream_solver:
            ostreams.append(sys.stdout)

        with capture_output(output=TeeStream(*ostreams), capture_fd=False):
            config = self.config
            options = self.maingo_options

            self._mymaingo = maingopy.MAiNGO(self._solver_model)

            self._mymaingo.set_option("loggingDestination", 2)
            self._mymaingo.set_log_file_name(config.logfile)
            self._mymaingo.set_option("epsilonA", config.tolerances.epsilonA)
            self._mymaingo.set_option("epsilonR", config.tolerances.epsilonR)
            self._mymaingo.set_option("deltaEq", config.tolerances.deltaEq)
            self._mymaingo.set_option("deltaIneq", config.tolerances.deltaIneq)

            if config.time_limit is not None:
                self._mymaingo.set_option("maxTime", config.time_limit)
            if config.mip_gap is not None:
                self._mymaingo.set_option("epsilonR", config.mip_gap)
            for key, option in options.items():
                self._mymaingo.set_option(key, option)

            timer.start("MAiNGO solve")
            self._mymaingo.solve()
            timer.stop("MAiNGO solve")

        return self._postsolve(timer)

    def solve(self, model, timer: HierarchicalTimer = None):
        StaleFlagManager.mark_all_as_stale()

        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if timer is None:
            timer = HierarchicalTimer()
        if model is not self._model:
            timer.start("set_instance")
            self.set_instance(model)
            timer.stop("set_instance")
        else:
            timer.start("Update")
            self.update(timer=timer)
            timer.stop("Update")
        res = self._solve(timer)
        self._last_results_object = res
        if self.config.report_timing:
            logger.info("\n" + str(timer))
        return res

    def _process_domain_and_bounds(self, var):
        _v, _lb, _ub, _fixed, _domain_interval, _value = self._vars[id(var)]
        lb, ub, step = _domain_interval

        if _fixed:
            lb = _value
            ub = _value
        else:
            if lb is None and _lb is None:
                logger.warning(
                    "No lower bound for variable "
                    + var.getname()
                    + " set. Using -1e10 instead. Please consider setting a valid lower bound."
                )
            if ub is None and _ub is None:
                logger.warning(
                    "No upper bound for variable "
                    + var.getname()
                    + " set. Using +1e10 instead. Please consider setting a valid upper bound."
                )

            if _lb is None:
                _lb = -1e10
            if _ub is None:
                _ub = 1e10
            if lb is None:
                lb = -1e10
            if ub is None:
                ub = 1e10

            lb = max(value(_lb), lb)
            ub = min(value(_ub), ub)

        if step == 0:
            vtype = maingopy.VT_CONTINUOUS
        elif step == 1:
            if lb == 0 and ub == 1:
                vtype = maingopy.VT_BINARY
            else:
                vtype = maingopy.VT_INTEGER
        else:
            raise ValueError(
                f"Unrecognized domain step: {step} (should be either 0 or 1)"
            )

        return lb, ub, vtype

    def _add_variables(self, variables: List[_GeneralVarData]):
        for var in variables:
            varname = self._symbol_map.getSymbol(var, self._labeler)
            lb, ub, vtype = self._process_domain_and_bounds(var)
            self._maingo_vars.append(
                MaingoVar(name=varname, type=vtype, lb=lb, ub=ub, init=var.value)
            )
            self._pyomo_var_to_solver_var_id_map[id(var)] = len(self._maingo_vars) - 1

    def _add_params(self, params: List[_ParamData]):
        pass

    def _reinit(self):
        saved_config = self.config
        saved_options = self.maingo_options
        saved_update_config = self.update_config
        self.__init__(only_child_vars=self._only_child_vars)
        self.config = saved_config
        self.maingo_options = saved_options
        self.update_config = saved_update_config

    def set_instance(self, model):
        if self._last_results_object is not None:
            self._last_results_object.solution_loader.invalidate()
        if not self.available():
            c = self.__class__
            raise PyomoException(
                f"Solver {c.__module__}.{c.__qualname__} is not available "
                f"({self.available()})."
            )
        self._reinit()
        self._model = model
        if self.use_extensions and cmodel_available:
            self._expr_types = cmodel.PyomoExprTypes()

        if self.config.symbolic_solver_labels:
            self._labeler = TextLabeler()
        else:
            self._labeler = NumericLabeler("x")

        self.add_block(model)

        self._solver_model = maingo_solvermodel.SolverModel(
            var_list=self._maingo_vars,
            con_list=self._cons,
            objective=self._objective,
            idmap=self._pyomo_var_to_solver_var_id_map,
            logger=logger,
        )

    def _add_constraints(self, cons: List[_GeneralConstraintData]):
        self._cons += cons

    def _add_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) >= 1:
            raise NotImplementedError(
                "MAiNGO does not currently support SOS constraints."
            )
        pass

    def _remove_constraints(self, cons: List[_GeneralConstraintData]):
        for con in cons:
            self._cons.remove(con)

    def _remove_sos_constraints(self, cons: List[_SOSConstraintData]):
        if len(cons) >= 1:
            raise NotImplementedError(
                "MAiNGO does not currently support SOS constraints."
            )
        pass

    def _remove_variables(self, variables: List[_GeneralVarData]):
        removed_maingo_vars = []
        for var in variables:
            varname = self._symbol_map.getSymbol(var, self._labeler)
            del self._maingo_vars[self._pyomo_var_to_solver_var_id_map[id(var)]]
            removed_maingo_vars += [self._pyomo_var_to_solver_var_id_map[id(var)]]
            del self._pyomo_var_to_solver_var_id_map[id(var)]

        # Update _pyomo_var_to_solver_var_id_map to account for removed variables
        for pyomo_var, maingo_var_id in self._pyomo_var_to_solver_var_id_map.items():
            num_removed = 0
            for removed_var in removed_maingo_vars:
                if removed_var <= maingo_var_id:
                    num_removed += 1
            self._pyomo_var_to_solver_var_id_map[pyomo_var] = (
                maingo_var_id - num_removed
            )

    def _remove_params(self, params: List[_ParamData]):
        pass

    def _update_variables(self, variables: List[_GeneralVarData]):
        for var in variables:
            if id(var) not in self._pyomo_var_to_solver_var_id_map:
                raise ValueError(
                    'The Var provided to update_var needs to be added first: {0}'.format(
                        var
                    )
                )
            lb, ub, vtype = self._process_domain_and_bounds(var)
            self._maingo_vars[self._pyomo_var_to_solver_var_id_map[id(var)]] = (
                MaingoVar(name=var.name, type=vtype, lb=lb, ub=ub, init=var.value)
            )

    def update_params(self):
        vars = [var[0] for var in self._vars.values()]
        self._update_variables(vars)

    def _set_objective(self, obj):

        if not obj.sense in {minimize, maximize}:
            raise ValueError("Objective sense is not recognized: {0}".format(obj.sense))
        self._objective = obj

    def _postsolve(self, timer: HierarchicalTimer):
        config = self.config

        mprob = self._mymaingo
        status = mprob.get_status()
        results = MAiNGOResults(solver=self)
        results.wallclock_time = mprob.get_wallclock_solution_time()
        results.cpu_time = mprob.get_cpu_solution_time()

        if status in {maingopy.GLOBALLY_OPTIMAL, maingopy.FEASIBLE_POINT}:
            results.termination_condition = TerminationCondition.optimal
            results.globally_optimal = True
            if status == maingopy.FEASIBLE_POINT:
                results.globally_optimal = False
                logger.warning(
                    "MAiNGO found a feasible solution but did not prove its global optimality."
                )
        elif status == maingopy.INFEASIBLE:
            results.termination_condition = TerminationCondition.infeasible
        else:
            results.termination_condition = TerminationCondition.unknown

        results.best_feasible_objective = None
        results.best_objective_bound = None
        if self._objective is not None:
            try:
                if self._objective.sense == maximize:
                    results.best_feasible_objective = -mprob.get_objective_value()
                else:
                    results.best_feasible_objective = mprob.get_objective_value()
            except:
                results.best_feasible_objective = None
            try:
                if self._objective.sense == maximize:
                    results.best_objective_bound = -mprob.get_final_LBD()
                else:
                    results.best_objective_bound = mprob.get_final_LBD()
            except:
                if self._objective.sense == maximize:
                    results.best_objective_bound = math.inf
                else:
                    results.best_objective_bound = -math.inf

            if results.best_feasible_objective is not None and not math.isfinite(
                results.best_feasible_objective
            ):
                results.best_feasible_objective = None

        timer.start("load solution")
        if config.load_solution:
            if results.termination_condition is TerminationCondition.optimal:
                if not results.globally_optimal:
                    logger.warning(
                        "Loading a feasible but suboptimal solution. "
                        "Please set load_solution=False and check "
                        "results.termination_condition and "
                        "results.found_feasible_solution() before loading a solution."
                    )
                self.load_vars()
            else:
                raise RuntimeError(
                    "A feasible solution was not found, so no solution can be loaded."
                    "Please set opt.config.load_solution=False and check "
                    "results.termination_condition and "
                    "results.best_feasible_objective before loading a solution."
                )
        timer.stop("load solution")

        return results

    def load_vars(self, vars_to_load=None):
        for v, val in self.get_primals(vars_to_load=vars_to_load).items():
            v.set_value(val, skip_validation=True)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(self, vars_to_load=None):
        if not self._mymaingo.get_status() in {
            maingopy.GLOBALLY_OPTIMAL,
            maingopy.FEASIBLE_POINT,
        }:
            raise RuntimeError(
                "Solver does not currently have a valid solution."
                "Please check the termination condition."
            )

        var_id_map = self._pyomo_var_to_solver_var_id_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_id_map.keys()
        else:
            vars_to_load = [id(v) for v in vars_to_load]

        maingo_var_ids_to_load = [
            var_id_map[pyomo_var_id] for pyomo_var_id in vars_to_load
        ]

        solution_point = self._mymaingo.get_solution_point()
        vals = [solution_point[var_id] for var_id in maingo_var_ids_to_load]

        res = ComponentMap()
        for var_id, val in zip(vars_to_load, vals):
            using_cons, using_sos, using_obj = ref_vars[var_id]
            if using_cons or using_sos or (using_obj is not None):
                res[self._vars[var_id][0]] = val
        return res

    def get_reduced_costs(self, vars_to_load=None):
        raise ValueError("MAiNGO does not support returning Reduced Costs")

    def get_duals(self, cons_to_load=None):
        raise ValueError("MAiNGO does not support returning Duals")

    def update(self, timer: HierarchicalTimer = None):
        super(MAiNGO, self).update(timer=timer)
        self._solver_model = maingo_solvermodel.SolverModel(
            var_list=self._maingo_vars,
            con_list=self._cons,
            objective=self._objective,
            idmap=self._pyomo_var_to_solver_var_id_map,
            logger=logger,
        )
