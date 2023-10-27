import math
from coramin.relaxations.relaxations_base import (
    BaseRelaxationData,
    BasePWRelaxationData,
)
import pyomo.environ as pe
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import _BlockData
from pyomo.contrib.appsi.base import (
    Results,
    PersistentSolver,
    Solver,
    MIPSolverConfig,
    TerminationCondition,
    SolutionLoaderBase,
    UpdateConfig,
)
from pyomo.contrib import appsi
from typing import Tuple, Optional, MutableMapping, Sequence
from pyomo.common.config import (
    ConfigValue, NonNegativeInt, PositiveFloat, PositiveInt, NonNegativeFloat, InEnum
)
import logging
from coramin.relaxations.auto_relax import relax
from coramin.relaxations.iterators import relaxation_data_objects
from coramin.utils.coramin_enums import RelaxationSide, Effort, EigenValueBounder
from coramin.domain_reduction.dbt import push_integers, pop_integers, collect_vars_to_tighten
from coramin.domain_reduction.obbt import perform_obbt
import time
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.objective import _GeneralObjectiveData
from coramin.utils.pyomo_utils import get_objective, active_vars
from pyomo.common.collections.component_set import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import BoundsManager
import numpy as np
from pyomo.core.expr.visitor import identify_variables
from coramin.clone import clone_active_flat


logger = logging.getLogger(__name__)


class MultiTreeConfig(MIPSolverConfig):
    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super(MultiTreeConfig, self).__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.declare("solver_output_logger", ConfigValue())
        self.declare("log_level", ConfigValue(domain=NonNegativeInt))
        self.declare("feasibility_tolerance", ConfigValue(domain=PositiveFloat))
        self.declare("abs_gap", ConfigValue(domain=PositiveFloat))
        self.declare("max_partitions_per_iter", ConfigValue(domain=PositiveInt))
        self.declare("max_iter", ConfigValue(domain=NonNegativeInt))
        self.declare("root_obbt_max_iter", ConfigValue(domain=NonNegativeInt))
        self.declare("show_obbt_progress_bar", ConfigValue(domain=bool))
        self.declare("integer_tolerance", ConfigValue(domain=PositiveFloat))
        self.declare("small_coef", ConfigValue(domain=NonNegativeFloat))
        self.declare("large_coef", ConfigValue(domain=NonNegativeFloat))
        self.declare("safety_tol", ConfigValue(domain=NonNegativeFloat))
        self.declare("convexity_effort", ConfigValue(domain=InEnum(Effort)))
        self.declare("obbt_at_new_incumbents", ConfigValue(domain=bool))
        self.declare("relax_integers_for_obbt", ConfigValue(domain=bool))

        self.solver_output_logger = logger
        self.log_level = logging.INFO
        self.feasibility_tolerance = 1e-6
        self.integer_tolerance = 1e-4
        self.time_limit = 600
        self.abs_gap = 1e-4
        self.mip_gap = 0.001
        self.max_partitions_per_iter = 5
        self.max_iter = 1000
        self.root_obbt_max_iter = 1000
        self.show_obbt_progress_bar = False
        self.small_coef = 1e-10
        self.large_coef = 1e5
        self.safety_tol = 1e-10
        self.convexity_effort = Effort.high
        self.obbt_at_new_incumbents: bool = True
        self.relax_integers_for_obbt: bool = True


def _is_problem_definitely_convex(m: _BlockData) -> bool:
    res = True
    for r in relaxation_data_objects(m, descend_into=True, active=True):
        if r.relaxation_side == RelaxationSide.BOTH:
            res = False
            break
        elif r.relaxation_side == RelaxationSide.UNDER and not r.is_rhs_convex():
            res = False
            break
        elif r.relaxation_side == RelaxationSide.OVER and not r.is_rhs_concave():
            res = False
            break
    return res


class MultiTreeResults(Results):
    def __init__(self):
        super().__init__()
        self.wallclock_time = None


class MultiTreeSolutionLoader(SolutionLoaderBase):
    def __init__(self, primals: MutableMapping):
        self._primals = primals

    def get_primals(
        self, vars_to_load: Optional[Sequence[_GeneralVarData]] = None
    ) -> MutableMapping[_GeneralVarData, float]:
        if vars_to_load is None:
            return pe.ComponentMap(self._primals.items())
        else:
            primals = pe.ComponentMap()
            for v in vars_to_load:
                primals[v] = self._primals[v]
        return primals


class MultiTree(Solver):
    def __init__(self, mip_solver: PersistentSolver, nlp_solver: PersistentSolver):
        super(MultiTree, self).__init__()
        self._config = MultiTreeConfig()
        self.mip_solver: PersistentSolver = mip_solver
        self.nlp_solver: PersistentSolver = nlp_solver
        self._original_model: Optional[_BlockData] = None
        self._relaxation: Optional[_BlockData] = None
        self._nlp: Optional[_BlockData] = None
        self._start_time: Optional[float] = None
        self._incumbent: Optional[pe.ComponentMap] = None
        self._best_feasible_objective: Optional[float] = None
        self._best_objective_bound: Optional[float] = None
        self._objective: Optional[_GeneralObjectiveData] = None
        self._relaxation_objects: Optional[Sequence[BaseRelaxationData]] = None
        self._stop: Optional[TerminationCondition] = None
        self._discrete_vars: Optional[Sequence[_GeneralVarData]] = None
        self._rel_to_nlp_map: Optional[MutableMapping] = None
        self._nlp_to_orig_map: Optional[MutableMapping] = None
        self._nlp_tightener: Optional[appsi.fbbt.IntervalTightener] = None
        self._iter: int = 0

    def _re_init(self):
        self._original_model: Optional[_BlockData] = None
        self._relaxation: Optional[_BlockData] = None
        self._nlp: Optional[_BlockData] = None
        self._start_time: Optional[float] = None
        self._incumbent: Optional[pe.ComponentMap] = None
        self._best_feasible_objective: Optional[float] = None
        self._best_objective_bound: Optional[float] = None
        self._objective: Optional[_GeneralObjectiveData] = None
        self._relaxation_objects: Optional[Sequence[BaseRelaxationData]] = None
        self._stop: Optional[TerminationCondition] = None
        self._discrete_vars: Optional[Sequence[_GeneralVarData]] = None
        self._rel_to_nlp_map: Optional[MutableMapping] = None
        self._nlp_to_orig_map: Optional[MutableMapping] = None
        self._nlp_tightener: Optional[appsi.fbbt.IntervalTightener] = None
        self._iter: int = 0

    def available(self):
        if (
            self.mip_solver.available() == Solver.Availability.FullLicense
            and self.nlp_solver.available() == Solver.Availability.FullLicense
        ):
            return Solver.Availability.FullLicense
        elif self.mip_solver.available() == Solver.Availability.FullLicense:
            return self.nlp_solver.available()
        else:
            return self.mip_solver.available()

    def version(self) -> Tuple:
        return 0, 1, 0

    @property
    def config(self) -> MultiTreeConfig:
        return self._config

    @config.setter
    def config(self, val: MultiTreeConfig):
        self._config = val

    @property
    def symbol_map(self):
        raise NotImplementedError("This solver does not have a symbol map")

    def _should_terminate(self) -> Tuple[bool, Optional[TerminationCondition]]:
        if self._elapsed_time >= self.config.time_limit:
            return True, TerminationCondition.maxTimeLimit
        if self._iter >= self.config.max_iter:
            return True, TerminationCondition.maxIterations
        if self._stop is not None:
            return True, self._stop
        primal_bound = self._get_primal_bound()
        dual_bound = self._get_dual_bound()
        if self._objective.sense == pe.minimize:
            assert primal_bound >= dual_bound - 1e-6*max(abs(primal_bound), abs(dual_bound)) - 1e-6
        else:
            assert primal_bound <= dual_bound + 1e-6*max(abs(primal_bound), abs(dual_bound)) + 1e-6
        abs_gap, rel_gap = self._get_abs_and_rel_gap()
        if abs_gap <= self.config.abs_gap:
            return True, TerminationCondition.optimal
        if rel_gap <= self.config.mip_gap:
            return True, TerminationCondition.optimal
        return False, TerminationCondition.unknown

    def _get_results(self, termination_condition: TerminationCondition) -> MultiTreeResults:
        res = MultiTreeResults()
        res.termination_condition = termination_condition
        res.best_feasible_objective = self._best_feasible_objective
        res.best_objective_bound = self._best_objective_bound
        if self._best_feasible_objective is not None:
            res.solution_loader = MultiTreeSolutionLoader(self._incumbent)
        res.wallclock_time = self._elapsed_time

        if self.config.load_solution:
            if res.best_feasible_objective is not None:
                if res.termination_condition != TerminationCondition.optimal:
                    logger.warning('Loading a feasible but potentially sub-optimal '
                                   'solution. Please check the termination condition.')
                res.solution_loader.load_vars()
            else:
                raise RuntimeError('No feasible solution was found. Please '
                                   'set opt.config.load_solution=False and check the '
                                   'termination condition before loading a solution.')

        return res

    def _get_primal_bound(self) -> float:
        if self._best_feasible_objective is None:
            if self._objective.sense == pe.minimize:
                primal_bound = math.inf
            else:
                primal_bound = -math.inf
        else:
            primal_bound = self._best_feasible_objective
        return primal_bound

    def _get_dual_bound(self) -> float:
        if self._best_objective_bound is None:
            if self._objective.sense == pe.minimize:
                dual_bound = -math.inf
            else:
                dual_bound = math.inf
        else:
            dual_bound = self._best_objective_bound
        return dual_bound

    def _get_abs_and_rel_gap(self):
        primal_bound = self._get_primal_bound()
        dual_bound = self._get_dual_bound()
        abs_gap = abs(primal_bound - dual_bound)
        if abs_gap == 0:
            rel_gap = 0
        elif primal_bound == 0:
            rel_gap = math.inf
        elif math.isinf(abs_gap):
            rel_gap = math.inf
        else:
            rel_gap = abs_gap / abs(primal_bound)
        return abs_gap, rel_gap

    def _get_constr_violation(self):
        viol_list = list()
        if len(self._relaxation_objects) == 0:
            return 0
        for b in self._relaxation_objects:
            any_none = False
            for v in b.get_rhs_vars():
                if v.value is None:
                    any_none = True
                    break
            if any_none:
                viol_list.append(math.inf)
                break
            else:
                viol_list.append(b.get_deviation())
        return max(viol_list)

    def _log(self, header=False, num_lb_improved=0, num_ub_improved=0,
             avg_lb_improvement=0, avg_ub_improvement=0, rel_termination=None,
             nlp_termination=None, constr_viol=None):
        logger = self.config.solver_output_logger
        log_level = self.config.log_level
        if header:
            msg = (
                f"{'Iter':<6}{'Primal Bd':<12}{'Dual Bd':<12}{'Abs Gap':<9}"
                f"{'% Gap':<7}{'CnstrVio':<10}{'Time':<6}{'NLP Term':<10}"
                f"{'Rel Term':<10}{'#LBs':<6}{'#UBs':<6}{'Avg LB':<9}"
                f"{'Avg UB':<9}"
            )
            logger.log(log_level, msg)
            if self.config.stream_solver:
                print(msg)
        else:
            if rel_termination is None:
                rel_termination = '-'
            else:
                rel_termination = str(rel_termination.name)
            if nlp_termination is None:
                nlp_termination = '-'
            else:
                nlp_termination = str(nlp_termination.name)
            rel_termination = rel_termination[:9]
            nlp_termination = nlp_termination[:9]
            primal_bound = self._get_primal_bound()
            dual_bound = self._get_dual_bound()
            abs_gap, rel_gap = self._get_abs_and_rel_gap()
            if constr_viol is None:
                constr_viol = '-'
            else:
                constr_viol = f'{constr_viol:<10.1e}'
            elapsed_time = self._elapsed_time
            if elapsed_time < 100:
                elapsed_time_str = f'{elapsed_time:<6.2f}'
            else:
                elapsed_time_str = f'{round(elapsed_time):<6d}'
            percent_gap = rel_gap*100
            if math.isinf(percent_gap):
                percent_gap_str = f'{percent_gap:<7.2f}'
            elif percent_gap >= 100:
                percent_gap_str = f'{round(percent_gap):<7d}'
            else:
                percent_gap_str = f'{percent_gap:<7.2f}'
            msg = (
                f"{self._iter:<6}{primal_bound:<12.3e}{dual_bound:<12.3e}"
                f"{abs_gap:<9.1e}{percent_gap_str:<7}{constr_viol:<10}"
                f"{elapsed_time_str:<6}{nlp_termination:<10}"
                f"{rel_termination:<10}{num_lb_improved:<6}"
                f"{num_ub_improved:<6}{avg_lb_improvement:<9.1e}"
                f"{avg_ub_improvement:<9.1e}"
            )
            logger.log(log_level, msg)
            if self.config.stream_solver:
                print(msg)

    def _update_dual_bound(self, res: Results):
        if res.best_objective_bound is not None:
            if self._objective.sense == pe.minimize:
                if (
                    self._best_objective_bound is None
                    or res.best_objective_bound > self._best_objective_bound
                ):
                    self._best_objective_bound = res.best_objective_bound
            else:
                if (
                    self._best_objective_bound is None
                    or res.best_objective_bound < self._best_objective_bound
                ):
                    self._best_objective_bound = res.best_objective_bound

        if res.best_feasible_objective is not None:
            max_viol = self._get_constr_violation()
            if max_viol > self.config.feasibility_tolerance:
                all_cons_satisfied = False
            else:
                all_cons_satisfied = True
            if all_cons_satisfied:
                for v in self._discrete_vars:
                    if v.value is None:
                        assert v.stale
                        continue
                    if not math.isclose(v.value, round(v.value), rel_tol=self.config.integer_tolerance, abs_tol=self.config.integer_tolerance):
                        all_cons_satisfied = False
                        break
            if all_cons_satisfied:
                for rel_v, nlp_v in self._rel_to_nlp_map.items():
                    if rel_v.value is None:
                        assert rel_v.stale
                        if rel_v.has_lb() and rel_v.has_ub() and math.isclose(rel_v.lb, rel_v.ub, rel_tol=self.config.feasibility_tolerance, abs_tol=self.config.feasibility_tolerance):
                            nlp_v.value = 0.5*(rel_v.lb + rel_v.ub)
                        else:
                            nlp_v.value = None
                    else:
                        nlp_v.set_value(rel_v.value, skip_validation=True)
                self._update_primal_bound(res)

    def _update_primal_bound(self, res: Results):
        should_update = False
        if res.best_feasible_objective is not None:
            if self._objective.sense == pe.minimize:
                if (
                    self._best_feasible_objective is None
                    or res.best_feasible_objective < self._best_feasible_objective
                ):
                    should_update = True
            else:
                if (
                    self._best_feasible_objective is None
                    or res.best_feasible_objective > self._best_feasible_objective
                ):
                    should_update = True

        if should_update:
            self._best_feasible_objective = res.best_feasible_objective
            self._incumbent = pe.ComponentMap()
            for nlp_v, orig_v in self._nlp_to_orig_map.items():
                self._incumbent[orig_v] = nlp_v.value

    def _solve_nlp_with_fixed_vars(
        self,
        integer_var_values: MutableMapping[_GeneralVarData, float],
        rhs_var_bounds: MutableMapping[_GeneralVarData, Tuple[float, float]],
    ) -> Results:
        self._iter += 1

        bm = BoundsManager(self._nlp)
        bm.save_bounds()

        fixed_vars = list()
        for v in self._discrete_vars:
            if v.fixed:
                continue
            val = integer_var_values[v]
            assert math.isclose(val, round(val), rel_tol=self.config.integer_tolerance, abs_tol=self.config.integer_tolerance)
            val = round(val)
            nlp_v = self._rel_to_nlp_map[v]
            orig_v = self._nlp_to_orig_map[nlp_v]
            nlp_v.fix(val)
            orig_v.fix(val)
            fixed_vars.append(nlp_v)
            fixed_vars.append(orig_v)

        for v, (v_lb, v_ub) in rhs_var_bounds.items():
            if v.fixed:
                continue
            nlp_v = self._rel_to_nlp_map[v]
            nlp_v.setlb(v_lb)
            nlp_v.setub(v_ub)

        nlp_res = Results()

        active_constraints = list()
        for c in ComponentSet(
            self._nlp.component_data_objects(
                pe.Constraint, active=True, descend_into=True
            )
        ):
            active_constraints.append(c)

        try:
            self._nlp_tightener.perform_fbbt(self._nlp)
            proven_infeasible = False
        except InfeasibleConstraintException:
            # the original NLP may still be feasible
            proven_infeasible = True

        if proven_infeasible:
            any_unfixed_vars = False
            for v in self._original_model.component_data_objects(
                pe.Var, descend_into=True
            ):
                if not v.fixed:
                    any_unfixed_vars = True
                    break
            if any_unfixed_vars:
                self.nlp_solver.config.time_limit = self._remaining_time
                nlp_res = self.nlp_solver.solve(self._original_model)
                if nlp_res.best_feasible_objective is not None:
                    nlp_res.solution_loader.load_vars()
                    for nlp_v, orig_v in self._nlp_to_orig_map.items():
                        nlp_v.set_value(orig_v.value, skip_validation=True)
                else:
                    nlp_res = Results()
                    nlp_res.termination_condition = TerminationCondition.infeasible
        else:
            for v in ComponentSet(
                self._nlp.component_data_objects(pe.Var, descend_into=True)
            ):
                if v.fixed:
                    continue
                if v.has_lb() and v.has_ub():
                    if math.isclose(v.lb, v.ub, rel_tol=self.config.feasibility_tolerance, abs_tol=self.config.feasibility_tolerance):
                        v.fix(0.5 * (v.lb + v.ub))
                        fixed_vars.append(v)
                    else:
                        v.value = 0.5 * (v.lb + v.ub)

            any_unfixed_vars = False
            for c in self._nlp.component_data_objects(
                pe.Constraint, active=True, descend_into=True
            ):
                for v in identify_variables(c.body, include_fixed=False):
                    any_unfixed_vars = True
                    break
            if not any_unfixed_vars:
                for obj in self._nlp.component_data_objects(
                    pe.Objective, active=True, descend_into=True
                ):
                    for v in identify_variables(obj.expr, include_fixed=False):
                        any_unfixed_vars = True
                        break

            if any_unfixed_vars:
                self.nlp_solver.config.time_limit = self._remaining_time
                self.nlp_solver.config.load_solution = False
                try:
                    nlp_res = self.nlp_solver.solve(self._nlp)
                    solve_error = False
                except Exception:
                    solve_error = True
                if not solve_error and nlp_res.best_feasible_objective is not None:
                    nlp_res.solution_loader.load_vars()
                else:
                    self.nlp_solver.config.time_limit = self._remaining_time
                    try:
                        nlp_res = self.nlp_solver.solve(self._original_model)
                        solve_error = False
                    except Exception:
                        solve_error = True
                    if not solve_error and nlp_res.best_feasible_objective is not None:
                        nlp_res.solution_loader.load_vars()
                        for nlp_v, orig_v in self._nlp_to_orig_map.items():
                            nlp_v.value = orig_v.value
            else:
                nlp_obj = get_objective(self._nlp)
                # there should not be any active constraints
                # they should all have been deactivated by FBBT
                for c in active_constraints:
                    assert not c.active
                nlp_res.termination_condition = TerminationCondition.optimal
                nlp_res.best_feasible_objective = pe.value(nlp_obj)
                nlp_res.best_objective_bound = nlp_res.best_feasible_objective
                nlp_res.solution_loader = MultiTreeSolutionLoader(pe.ComponentMap((v, v.value) for v in self._nlp.component_data_objects(pe.Var, descend_into=True)))

        self._update_primal_bound(nlp_res)
        self._log(header=False, nlp_termination=nlp_res.termination_condition)

        for v in fixed_vars:
            v.unfix()

        bm.pop_bounds()

        for c in active_constraints:
            c.activate()

        return nlp_res

    def _solve_relaxation(self) -> Results:
        self._iter += 1
        self.mip_solver.config.time_limit = self._remaining_time
        self.mip_solver.config.load_solution = False
        rel_res = self.mip_solver.solve(self._relaxation)

        if rel_res.best_feasible_objective is not None:
            rel_res.solution_loader.load_vars()

        self._update_dual_bound(rel_res)
        self._log(header=False, rel_termination=rel_res.termination_condition, constr_viol=self._get_constr_violation())
        if rel_res.termination_condition not in {
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.objectiveLimit,
            TerminationCondition.interrupted,
        }:
            self._stop = rel_res.termination_condition
        return rel_res

    def _partition_helper(self):
        dev_list = list()

        err = False

        for b in self._relaxation_objects:
            for v in b.get_rhs_vars():
                if not v.has_lb() or not v.has_ub():
                    logger.error(
                        'The multitree algorithm is not guaranteed to converge '
                        'for problems with unbounded variables. Please bound all '
                        'variables.')
                    self._stop = TerminationCondition.error
                    err = True
                    break
            if err:
                break

            aux_val = b.get_aux_var().value
            rhs_val = pe.value(b.get_rhs_expr())
            if (
                aux_val > rhs_val + self.config.feasibility_tolerance
                and b.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.OVER}
                and not b.is_rhs_concave()
            ):
                dev_list.append((b, aux_val - rhs_val))
            elif (
                aux_val < rhs_val - self.config.feasibility_tolerance
                and b.relaxation_side in {RelaxationSide.BOTH, RelaxationSide.UNDER}
                and not b.is_rhs_convex()
            ):
                dev_list.append((b, rhs_val - aux_val))

        if not err:
            dev_list.sort(key=lambda x: x[1], reverse=True)

            for b, dev in dev_list[: self.config.max_partitions_per_iter]:
                b.add_partition_point()
                b.rebuild()

    def _oa_cut_helper(self, tol):
        new_con_list = list()
        for b in self._relaxation_objects:
            new_con = b.add_cut(
                keep_cut=True, check_violation=True, feasibility_tol=tol
            )
            if new_con is not None:
                new_con_list.append(new_con)
        self.mip_solver.add_constraints(new_con_list)
        return new_con_list

    def _add_oa_cuts(self, tol, max_iter) -> Results:
        original_update_config: UpdateConfig = self.mip_solver.update_config()

        self.mip_solver.update()

        self.mip_solver.update_config.update_params = False
        self.mip_solver.update_config.update_vars = False
        self.mip_solver.update_config.update_objective = False
        self.mip_solver.update_config.update_constraints = False
        self.mip_solver.update_config.check_for_new_objective = False
        self.mip_solver.update_config.check_for_new_or_removed_constraints = False
        self.mip_solver.update_config.check_for_new_or_removed_vars = False
        self.mip_solver.update_config.check_for_new_or_removed_params = True
        self.mip_solver.update_config.treat_fixed_vars_as_params = True
        self.mip_solver.update_config.update_named_expressions = False

        last_res = None

        for _iter in range(max_iter):
            if self._should_terminate()[0]:
                break

            rel_res = self._solve_relaxation()
            if rel_res.best_feasible_objective is not None:
                last_res = Results()
                last_res.best_feasible_objective = rel_res.best_feasible_objective
                last_res.best_objective_bound = rel_res.best_objective_bound
                last_res.termination_condition = rel_res.termination_condition
                last_res.solution_loader = MultiTreeSolutionLoader(
                    rel_res.solution_loader.get_primals(
                        vars_to_load=self._discrete_vars
                    )
                )

            if self._should_terminate()[0]:
                break

            new_con_list = self._oa_cut_helper(tol=tol)
            if len(new_con_list) == 0:
                break

        self.mip_solver.update_config.update_params = (
            original_update_config.update_params
        )
        self.mip_solver.update_config.update_vars = original_update_config.update_vars
        self.mip_solver.update_config.update_objective = (
            original_update_config.update_objective
        )
        self.mip_solver.update_config.update_constraints = (
            original_update_config.update_constraints
        )
        self.mip_solver.update_config.check_for_new_objective = (
            original_update_config.check_for_new_objective
        )
        self.mip_solver.update_config.check_for_new_or_removed_constraints = (
            original_update_config.check_for_new_or_removed_constraints
        )
        self.mip_solver.update_config.check_for_new_or_removed_vars = (
            original_update_config.check_for_new_or_removed_vars
        )
        self.mip_solver.update_config.check_for_new_or_removed_params = (
            original_update_config.check_for_new_or_removed_params
        )
        self.mip_solver.update_config.treat_fixed_vars_as_params = (
            original_update_config.treat_fixed_vars_as_params
        )
        self.mip_solver.update_config.update_named_expressions = (
            original_update_config.update_named_expressions
        )

        if last_res is None:
            last_res = Results()

        return last_res

    def _construct_nlp(self):
        all_vars = list(
            ComponentSet(
                self._original_model.component_data_objects(pe.Var, descend_into=True)
            )
        )
        tmp_name = unique_component_name(self._original_model, "all_vars")
        setattr(self._original_model, tmp_name, all_vars)

        # this has to be 0 because the Multitree solver cannot use alpha-bb relaxations
        max_vars_per_alpha_bb = 0
        max_eigenvalue_for_alpha_bb = 0

        if self.config.convexity_effort == Effort.none:
            perform_expression_simplification = False
            use_alpha_bb = False
            eigenvalue_bounder = EigenValueBounder.Gershgorin
            eigenvalue_opt = None
        elif self.config.convexity_effort <= Effort.low:
            perform_expression_simplification = False
            use_alpha_bb = True
            eigenvalue_bounder = EigenValueBounder.Gershgorin
            eigenvalue_opt = None
        elif self.config.convexity_effort <= Effort.medium:
            perform_expression_simplification = True
            use_alpha_bb = True
            eigenvalue_bounder = EigenValueBounder.GershgorinWithSimplification
            eigenvalue_opt = None
        elif self.config.convexity_effort <= Effort.high:
            perform_expression_simplification = True
            use_alpha_bb = True
            eigenvalue_bounder = EigenValueBounder.LinearProgram
            eigenvalue_opt = self.mip_solver.__class__()
            eigenvalue_opt.config = self.mip_solver.config()
            # TODO: need to update the solver options
        else:
            perform_expression_simplification = True
            use_alpha_bb = True
            eigenvalue_bounder = EigenValueBounder.Global
            mip_solver = self.mip_solver.__class__()
            mip_solver.config = self.mip_solver.config()
            nlp_solver = self.nlp_solver.__class__()
            nlp_solver.config = self.nlp_solver.config()
            eigenvalue_opt = MultiTree(mip_solver=mip_solver, nlp_solver=nlp_solver)
            eigenvalue_opt.config = self.config()
            eigenvalue_opt.config.convexity_effort = min(self.config.convexity_effort, Effort.medium)

        self._nlp = relax(
            model=self._original_model,
            in_place=False,
            use_fbbt=True,
            fbbt_options={"deactivate_satisfied_constraints": True, "max_iter": 5},
            perform_expression_simplification=perform_expression_simplification,
            use_alpha_bb=use_alpha_bb,
            eigenvalue_bounder=eigenvalue_bounder,
            eigenvalue_opt=eigenvalue_opt,
            max_vars_per_alpha_bb=max_vars_per_alpha_bb,
            max_eigenvalue_for_alpha_bb=max_eigenvalue_for_alpha_bb,
        )
        new_vars = getattr(self._nlp, tmp_name)
        self._nlp_to_orig_map = pe.ComponentMap(zip(new_vars, all_vars))
        delattr(self._original_model, tmp_name)
        delattr(self._nlp, tmp_name)

        for b in relaxation_data_objects(self._nlp, descend_into=True, active=True):
            b.rebuild(build_nonlinear_constraint=True)

    def _construct_relaxation(self):
        all_vars = list(
            ComponentSet(
                self._nlp.component_data_objects(pe.Var, descend_into=True)
            )
        )
        tmp_name = unique_component_name(self._nlp, "all_vars")
        setattr(self._nlp, tmp_name, all_vars)
        self._relaxation = self._nlp.clone()
        new_vars = getattr(self._relaxation, tmp_name)
        self._rel_to_nlp_map = pe.ComponentMap(zip(new_vars, all_vars))
        delattr(self._nlp, tmp_name)
        delattr(self._relaxation, tmp_name)

        for b in relaxation_data_objects(self._relaxation, descend_into=True, active=True):
            b.small_coef = self.config.small_coef
            b.large_coef = self.config.large_coef
            b.safety_tol = self.config.safety_tol
            b.rebuild()

    def _get_nlp_specs_from_rel(self):
        integer_var_values = pe.ComponentMap()
        for v in self._discrete_vars:
            integer_var_values[v] = v.value
        rhs_var_bounds = pe.ComponentMap()
        for r in self._relaxation_objects:
            if not isinstance(r, BasePWRelaxationData):
                continue
            any_unbounded_vars = False
            for v in r.get_rhs_vars():
                if not v.has_lb() or not v.has_ub():
                    any_unbounded_vars = True
                    break
            if any_unbounded_vars:
                continue
            active_parts = r.get_active_partitions()
            assert len(active_parts) == 1
            v, bnds = list(active_parts.items())[0]
            if v in rhs_var_bounds:
                existing_bnds = rhs_var_bounds[v]
                bnds = (max(bnds[0], existing_bnds[0]), min(bnds[1], existing_bnds[1]))
            assert bnds[0] <= bnds[1]
            rhs_var_bounds[v] = bnds
        return integer_var_values, rhs_var_bounds

    @property
    def _elapsed_time(self):
        return time.time() - self._start_time

    @property
    def _remaining_time(self):
        return max(0.0, self.config.time_limit - self._elapsed_time)

    def _perform_obbt(self, vars_to_tighten):
        safety_tol = 1e-4
        self._iter += 1
        orig_lbs = list()
        orig_ubs = list()
        for v in vars_to_tighten:
            v_lb, v_ub = v.bounds
            if v_lb is None:
                v_lb = -math.inf
            if v_ub is None:
                v_ub = math.inf
            orig_lbs.append(v_lb)
            orig_ubs.append(v_ub)
        orig_lbs = np.array(orig_lbs)
        orig_ubs = np.array(orig_ubs)
        perform_obbt(self._relaxation, solver=self.mip_solver,
                     varlist=list(vars_to_tighten),
                     objective_bound=self._best_feasible_objective,
                     with_progress_bar=self.config.show_obbt_progress_bar,
                     time_limit=self._remaining_time)
        new_lbs = list()
        new_ubs = list()
        for ndx, v in enumerate(vars_to_tighten):
            v_lb, v_ub = v.bounds
            if v_lb is None:
                v_lb = -math.inf
            if v_ub is None:
                v_ub = math.inf
            v_lb -= safety_tol
            v_ub += safety_tol
            if v_lb < orig_lbs[ndx]:
                v_lb = orig_lbs[ndx]
            if v_ub > orig_ubs[ndx]:
                v_ub = orig_ubs[ndx]
            v.setlb(v_lb)
            v.setub(v_ub)
            new_lbs.append(v_lb)
            new_ubs.append(v_ub)
        for r in self._relaxation_objects:
            r.rebuild()
        new_lbs = np.array(new_lbs)
        new_ubs = np.array(new_ubs)
        lb_diff = new_lbs - orig_lbs
        ub_diff = orig_ubs - new_ubs
        lb_improved = lb_diff > 1e-3
        ub_improved = ub_diff > 1e-3
        lb_improved_indices = lb_improved.nonzero()[0]
        ub_improved_indices = ub_improved.nonzero()[0]
        num_lb_improved = len(lb_improved_indices)
        num_ub_improved = len(ub_improved_indices)
        if num_lb_improved > 0:
            avg_lb_improvement = np.mean(lb_diff[lb_improved_indices])
        else:
            avg_lb_improvement = 0
        if num_ub_improved > 0:
            avg_ub_improvement = np.mean(ub_diff[ub_improved_indices])
        else:
            avg_ub_improvement = 0
        self._log(header=False, num_lb_improved=num_lb_improved,
                  num_ub_improved=num_ub_improved,
                  avg_lb_improvement=avg_lb_improvement,
                  avg_ub_improvement=avg_ub_improvement)

        return num_lb_improved, num_ub_improved, avg_lb_improvement, avg_ub_improvement

    def solve(self, model: _BlockData, timer: HierarchicalTimer = None) -> MultiTreeResults:
        model = clone_active_flat(model)
        self._re_init()

        self._start_time = time.time()
        if timer is None:
            timer = HierarchicalTimer()
        timer.start("solve")

        self._original_model = model

        self._log(header=True)

        timer.start("construct relaxation")
        self._construct_nlp()
        self._construct_relaxation()
        timer.stop("construct relaxation")

        self._objective = get_objective(self._relaxation)
        self._relaxation_objects = list()
        for r in relaxation_data_objects(
            self._relaxation, descend_into=True, active=True
        ):
            self._relaxation_objects.append(r)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        self._log(header=False)

        self.mip_solver.set_instance(self._relaxation)
        self._nlp_tightener = appsi.fbbt.IntervalTightener()
        self._nlp_tightener.config.deactivate_satisfied_constraints = True
        self._nlp_tightener.config.feasibility_tol = self.config.feasibility_tolerance
        self._nlp_tightener.set_instance(self._nlp, symbolic_solver_labels=False)

        relaxed_binaries, relaxed_integers = push_integers(self._relaxation)
        self._discrete_vars = list(relaxed_binaries) + list(relaxed_integers)
        oa_results = self._add_oa_cuts(self.config.feasibility_tolerance * 100, 100)
        pop_integers(relaxed_binaries, relaxed_integers)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        if _is_problem_definitely_convex(self._relaxation):
            oa_results = self._add_oa_cuts(self.config.feasibility_tolerance, 100)
        else:
            oa_results = self._add_oa_cuts(self.config.feasibility_tolerance * 1e3, 3)

        should_terminate, reason = self._should_terminate()
        if should_terminate:
            return self._get_results(reason)

        if oa_results.best_feasible_objective is not None:
            integer_var_values, rhs_var_bounds = self._get_nlp_specs_from_rel()
            nlp_res = self._solve_nlp_with_fixed_vars(
                integer_var_values, rhs_var_bounds
            )

        vars_to_tighten = collect_vars_to_tighten(self._relaxation)
        for obbt_iter in range(self.config.root_obbt_max_iter):
            should_terminate, reason = self._should_terminate()
            if should_terminate:
                return self._get_results(reason)
            relaxed_binaries, relaxed_integers = push_integers(self._relaxation)
            num_lb, num_ub, avg_lb, avg_ub = self._perform_obbt(vars_to_tighten)
            pop_integers(relaxed_binaries, relaxed_integers)
            should_terminate, reason = self._should_terminate()
            if (num_lb + num_ub) < 1 or (avg_lb < 1e-3 and avg_ub < 1e-3):
                break
            if should_terminate:
                return self._get_results(reason)
            self._solve_relaxation()

        while True:
            should_terminate, reason = self._should_terminate()
            if should_terminate:
                break

            rel_res = self._solve_relaxation()

            should_terminate, reason = self._should_terminate()
            if should_terminate:
                break

            if rel_res.best_feasible_objective is not None:
                self._oa_cut_helper(self.config.feasibility_tolerance)
                self._partition_helper()

                integer_var_values, rhs_var_bounds = self._get_nlp_specs_from_rel()
                start_primal_bound = self._get_primal_bound()
                nlp_res = self._solve_nlp_with_fixed_vars(
                    integer_var_values, rhs_var_bounds
                )
                end_primal_bound = self._get_primal_bound()

                should_terminate, reason = self._should_terminate()
                if should_terminate:
                    break

                if self.config.obbt_at_new_incumbents and not math.isclose(start_primal_bound, end_primal_bound, rel_tol=1e-4, abs_tol=1e-4):
                    if self.config.relax_integers_for_obbt:
                        relaxed_binaries, relaxed_integers = push_integers(self._relaxation)
                    num_lb, num_ub, avg_lb, avg_ub = self._perform_obbt(vars_to_tighten)
                    if self.config.relax_integers_for_obbt:
                        pop_integers(relaxed_binaries, relaxed_integers)
            else:
                self.config.solver_output_logger.warning(
                    f"relaxation did not find a feasible solution: "
                    f"{rel_res.termination_condition}"
                )

        res = self._get_results(reason)

        timer.stop("solve")

        return res
