from pyomo.contrib import appsi
from pyomo.common.collections import ComponentSet
from coramin.utils.coramin_enums import RelaxationSide
import pyomo.environ as pe
import time
from pyomo.common.config import ConfigValue, NonNegativeInt, NonNegativeFloat, In
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.staleflag import StaleFlagManager
from coramin.utils import get_objective
from coramin.relaxations import relaxation_data_objects
from typing import Optional
from typing import Sequence, Mapping, MutableMapping, Tuple, List
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.core.base.param import _ParamData


import logging

logger = logging.getLogger(__name__)


class ECPConfig(appsi.base.SolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(ECPConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.feasibility_tol = self.declare(
            "feasibility_tol",
            ConfigValue(
                default=1e-6,
                domain=NonNegativeFloat,
                doc="Tolerance below which cuts will not be added",
            ),
        )
        self.max_iter = self.declare(
            "max_iter",
            ConfigValue(
                default=30, domain=NonNegativeInt, doc="Maximum number of iterations"
            ),
        )
        self.keep_cuts = self.declare(
            "keep_cuts",
            ConfigValue(
                default=False,
                domain=In([True, False]),
                doc="Whether or not to keep the cuts generated after the solve",
            ),
        )

        self.time_limit = 600


class ECPSolutionLoader(appsi.base.SolutionLoaderBase):
    def __init__(self, primals: MutableMapping[_GeneralVarData, float]):
        self._primals = primals

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        if vars_to_load is None:
            primals = pe.ComponentMap(self._primals)
        else:
            primals = pe.ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[v]
        return primals


class ECPResults(appsi.base.Results):
    def __init__(self):
        super(ECPResults, self).__init__()
        self.wallclock_time = None


class ECPBounder(appsi.base.PersistentSolver):
    """
    A solver designed for use inside of OBBT. This solver is a persistent solver for
    efficient changes to the objective. Additionally, it provides a mechanism for
    refining convex nonlinear constraints during OBBT.
    """

    def __init__(self, subproblem_solver: appsi.base.PersistentSolver):
        super(ECPBounder, self).__init__()
        self._subproblem_solver = subproblem_solver
        self._relaxations = ComponentSet()
        self._relaxations_with_added_cuts = ComponentSet()
        self._pyomo_model = None
        self._config = ECPConfig()
        self._start_time: Optional[float] = None
        self._update_config = appsi.base.UpdateConfig()

    def available(self):
        return self._subproblem_solver.available()

    def version(self) -> Tuple:
        return 0, 1, 0

    @property
    def symbol_map(self):
        raise NotImplementedError("ECPBounder does not use a symbol map")

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val: ECPConfig):
        self._config = val

    @property
    def update_config(self) -> appsi.base.UpdateConfig:
        return self._update_config

    @update_config.setter
    def update_config(self, val: appsi.base.UpdateConfig):
        self._update_config = val

    @property
    def _elapsed_time(self):
        return time.time() - self._start_time

    @property
    def _remaining_time(self):
        return max(0.0, self.config.time_limit - self._elapsed_time)

    def solve(self, model, timer: HierarchicalTimer = None) -> ECPResults:
        self._start_time = time.time()
        if timer is None:
            timer = HierarchicalTimer()
        timer.start("ECP Solve")
        StaleFlagManager.mark_all_as_stale()
        logger.info(
            "{0:<10}{1:<12}{2:<12}{3:<12}{4:<12}".format(
                "Iter", "objective", "max_viol", "time", "# cuts"
            )
        )
        self._pyomo_model = model

        obj = get_objective(model)
        if obj is None:
            raise ValueError("Could not find any active objectives")

        final_res = ECPResults()
        self._relaxations = ComponentSet()
        self._relaxations_with_added_cuts = ComponentSet()
        for b in relaxation_data_objects(
            self._pyomo_model, descend_into=True, active=True
        ):
            self._relaxations.add(b)

        self._subproblem_solver.config.load_solution = False
        orig_var_vals = pe.ComponentMap()
        for v in self._pyomo_model.component_data_objects(pe.Var, descend_into=True):
            orig_var_vals[v] = v.value

        all_added_cons = list()
        for _iter in range(self.config.max_iter):
            if self._elapsed_time >= self.config.time_limit:
                final_res.termination_condition = (
                    appsi.base.TerminationCondition.maxTimeLimit
                )
                logger.warning("ECPBounder: time limit reached.")
                break
            self._subproblem_solver.config.time_limit = self._remaining_time
            res = self._subproblem_solver.solve(self._pyomo_model, timer=timer)
            if res.termination_condition == appsi.base.TerminationCondition.optimal:
                res.solution_loader.load_vars()
            else:
                final_res.termination_condition = res.termination_condition
                logger.warning("ECPBounder: subproblem did not terminate optimally")
                break

            new_con_list = list()
            max_viol = 0
            for b in self._relaxations:
                viol = None
                try:
                    if b.is_rhs_convex() and b.relaxation_side in {
                        RelaxationSide.BOTH,
                        RelaxationSide.UNDER,
                    }:
                        viol = pe.value(b.get_rhs_expr()) - b.get_aux_var().value
                    elif b.is_rhs_concave() and b.relaxation_side in {
                        RelaxationSide.BOTH,
                        RelaxationSide.OVER,
                    }:
                        viol = b.get_aux_var().value - pe.value(b.get_rhs_expr())
                except (OverflowError, ZeroDivisionError, ValueError) as err:
                    logger.warning("could not generate ECP cut due to " + str(err))
                if viol is not None:
                    if viol > max_viol:
                        max_viol = viol
                    if viol > self.config.feasibility_tol:
                        new_con = b.add_cut(
                            keep_cut=self.config.keep_cuts,
                            check_violation=True,
                            feasibility_tol=self.config.feasibility_tol,
                        )
                        if new_con is not None:
                            self._relaxations_with_added_cuts.add(b)
                            new_con_list.append(new_con)
            self._subproblem_solver.add_constraints(new_con_list)

            final_res.best_objective_bound = res.best_objective_bound
            logger.info(
                "{0:<10d}{1:<12.3e}{2:<12.3e}{3:<12.3e}{4:<12d}".format(
                    _iter,
                    final_res.best_objective_bound,
                    max_viol,
                    self._elapsed_time,
                    len(new_con_list),
                )
            )

            all_added_cons.extend(new_con_list)

            if len(new_con_list) == 0:
                # The goal of the ECPBounder is not to find the optimal solution.
                # Rather, the goal is just to get a decent bound quickly.
                # However, if the problem is convex, we may still be able to declare
                # optimality
                final_res.termination_condition = (
                    appsi.base.TerminationCondition.unknown
                )
                logger.info("ECPBounder: converged!")

                found_feasible_solution = True
                for b in self._relaxations:
                    deviation = b.get_deviation()
                    if deviation > self.config.feasibility_tol:
                        found_feasible_solution = False

                if found_feasible_solution:
                    final_res.termination_condition = (
                        appsi.base.TerminationCondition.optimal
                    )
                    final_res.best_feasible_objective = final_res.best_objective_bound
                    primal_sol = res.solution_loader.get_primals()
                    final_res.solution_loader = ECPSolutionLoader(primal_sol)

                break

            if _iter == self.config.max_iter - 1:
                final_res.termination_condition = (
                    appsi.base.TerminationCondition.maxIterations
                )
                logger.warning("ECPBounder: reached maximum number of iterations")

        if not self.config.keep_cuts:
            self._subproblem_solver.remove_constraints(all_added_cons)
            for b in self._relaxations_with_added_cuts:
                b.rebuild()

        if final_res.termination_condition == appsi.base.TerminationCondition.optimal:
            if not self.config.load_solution:
                for v, val in orig_var_vals.items():
                    v.value = val
        else:
            if self.config.load_solution:
                raise RuntimeError(
                    "A feasible solution was not found, so no solution can be loaded. "
                    "Please set opt.config.load_solution=False and check "
                    "results.termination_condition and results.best_feasible_objective "
                    "before loading a solution."
                )
            for v, val in orig_var_vals.items():
                v.value = val

        final_res.wallclock_time = self._elapsed_time
        timer.stop("ECP Solve")
        return final_res

    def set_instance(self, model):
        saved_update_config = self.update_config
        saved_config = self.config
        self.__init__(self._subproblem_solver)
        self.config = saved_config
        self.update_config = saved_update_config
        self._pyomo_model = model
        self._subproblem_solver.set_instance(model)

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> Mapping[_GeneralVarData, float]:
        return self._subproblem_solver.get_primals(vars_to_load=vars_to_load)

    def add_block(self, block):
        self._subproblem_solver.add_block(block)

    def add_constraints(self, cons: List[_GeneralConstraintData]):
        self._subproblem_solver.add_constraints(cons)

    def add_variables(self, variables: List[_GeneralVarData]):
        self._subproblem_solver.add_variables(variables=variables)

    def add_params(self, params: List[_ParamData]):
        self._subproblem_solver.add_params(params=params)

    def remove_block(self, block):
        self._subproblem_solver.remove_block(block)

    def remove_constraints(self, cons: List[_GeneralConstraintData]):
        self._subproblem_solver.remove_constraints(cons=cons)

    def remove_variables(self, variables: List[_GeneralVarData]):
        self._subproblem_solver.remove_variables(variables=variables)

    def remove_params(self, params: List[_ParamData]):
        self._subproblem_solver.remove_params(params=params)

    def set_objective(self, obj):
        self._subproblem_solver.set_objective(obj)

    def update_variables(self, variables: List[_GeneralVarData]):
        self._subproblem_solver.update_variables(variables=variables)

    def update_params(self):
        return self._subproblem_solver.update_params()
