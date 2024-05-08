import pybnb
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import _BlockData
import pyomo.environ as pe
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib import appsi
from pyomo.common.config import ConfigValue, PositiveFloat
from pyomo.contrib.coramin.clone import clone_shallow_active_flat, get_clone_and_var_map
from pyomo.contrib.coramin.relaxations.auto_relax import _relax_cloned_model
from pyomo.contrib.coramin.relaxations import iterators
from pyomo.contrib.coramin.utils.pyomo_utils import get_objective
from pyomo.core.base.block import _BlockData
from pyomo.contrib.coramin.heuristics.diving import run_diving_heuristic
from pyomo.contrib.coramin.domain_reduction.obbt import perform_obbt
from pyomo.contrib.coramin.cutting_planes.base import CutGenerator
from typing import Tuple, List, Optional
import math
from pyomo.common.dependencies import numpy as np
import logging
from pyomo.contrib.appsi.base import (
    Solver,
    MIPSolverConfig,
    Results,
    TerminationCondition,
    SolutionLoader,
    SolverFactory,
)
from pyomo.core.staleflag import StaleFlagManager
from pyomo.contrib.coramin.algorithms.alg_utils import (
    impose_structure,
    collect_vars,
    relax_integers,
)
from pyomo.contrib.coramin.algorithms.cut_gen import find_cut_generators, AlphaBBConfig


logger = logging.getLogger(__name__)


class BnBConfig(MIPSolverConfig):
    def __init__(self):
        super().__init__(None, None, False, None, 0)
        self.feasibility_tol = self.declare(
            "feasibility_tol", ConfigValue(domain=PositiveFloat, default=1e-7)
        )
        self.lp_solver = self.declare("lp_solver", ConfigValue())
        self.nlp_solver = self.declare("nlp_solver", ConfigValue())
        self.abs_gap = self.declare("abs_gap", ConfigValue(default=1e-4))
        self.integer_tol = self.declare("integer_tol", ConfigValue(default=1e-4))
        self.node_limit = self.declare("node_limit", ConfigValue(default=1000000000))
        self.mip_gap = 1e-3
        self.num_root_obbt_iters = self.declare(
            "num_root_obbt_iters", ConfigValue(default=3)
        )
        self.node_obbt_frequency = self.declare(
            "node_obbt_frequency", ConfigValue(default=2)
        )
        self.alphabb = self.declare("alphabb", AlphaBBConfig())


class NodeState(object):
    def __init__(
        self,
        lbs: np.ndarray,
        ubs: np.ndarray,
        parent: Optional[pybnb.Node],
        sol: Optional[np.ndarray] = None,
        obj: Optional[float] = None,
    ) -> None:
        self.lbs: np.ndarray = lbs
        self.ubs: np.ndarray = ubs
        self.parent: Optional[pybnb.Node] = parent
        self.sol: Optional[np.ndarray] = sol
        self.obj: Optional[float] = obj
        self.valid_cut_indices: List[int] = list()
        self.active_cut_indices: List[int] = list()


def _fix_vars_with_close_bounds(varlist, tol=1e-12):
    for v in varlist:
        if v.is_fixed():
            v.setlb(v.value)
            v.setub(v.value)
        lb, ub = v.bounds
        if lb is None or ub is None:
            continue
        if abs(ub - lb) <= tol * min(abs(lb), abs(ub)) + tol:
            v.fix(0.5 * (lb + ub))


class _BnB(pybnb.Problem):
    def __init__(self, model: _BlockData, config: BnBConfig, feasible_objective=None):
        # remove all parameters, fixed variables, etc.
        nlp, relaxation = clone_shallow_active_flat(model, 2)
        self.nlp: _BlockData = nlp
        self.relaxation: _BlockData = relaxation
        self.config = config
        self.config.lp_solver.config.load_solution = False
        self.relaxation_solution = None

        self.obj = obj = get_objective(nlp)
        if obj.sense == pe.minimize:
            self._sense = pybnb.minimize
        else:
            self._sense = pybnb.maximize

        # perform fbbt before constructing relaxations in case
        # we can identify things like x**3 is convex because
        # x >= 0
        self.interval_tightener = it = appsi.fbbt.IntervalTightener()
        it.config.deactivate_satisfied_constraints = False
        it.config.feasibility_tol = config.feasibility_tol
        if feasible_objective is not None:
            if obj.sense == pe.minimize:
                relaxation.obj_ineq = pe.Constraint(expr=obj.expr <= feasible_objective)
            else:
                relaxation.obj_ineq = pe.Constraint(expr=obj.expr >= feasible_objective)
        it.perform_fbbt(relaxation)
        if feasible_objective is not None:
            del relaxation.obj_ineq
        _fix_vars_with_close_bounds(relaxation.vars)

        impose_structure(relaxation)
        self.cut_generators: List[CutGenerator] = find_cut_generators(
            relaxation, self.config.alphabb
        )
        _relax_cloned_model(relaxation)
        relaxation.cuts = pe.ConstraintList()
        self.relaxation_objects = list()
        for r in iterators.relaxation_data_objects(
            relaxation, descend_into=True, active=True
        ):
            self.relaxation_objects.append(r)
            r.rebuild(build_nonlinear_constraint=True)
        self.interval_tightener.perform_fbbt(self.relaxation)

        binary_vars, integer_vars = collect_vars(nlp)
        relax_integers(binary_vars, integer_vars)
        self.binary_vars = binary_vars
        self.integer_vars = integer_vars
        self.bin_and_int_vars = list(binary_vars) + list(integer_vars)
        int_var_set = ComponentSet(self.bin_and_int_vars)

        self.rhs_vars = list()
        for r in self.relaxation_objects:
            self.rhs_vars.extend(i for i in r.get_rhs_vars() if not i.is_fixed())
        self.rhs_vars = list(ComponentSet(self.rhs_vars) - int_var_set)

        var_set = ComponentSet(self.binary_vars + self.integer_vars + self.rhs_vars)
        other_vars = ComponentSet(i for i in relaxation.vars if i not in var_set)
        self.other_vars = other_vars = list(other_vars)

        self.all_branching_vars = (
            list(binary_vars) + list(integer_vars) + list(self.rhs_vars)
        )
        self.all_vars = self.all_branching_vars + self.other_vars
        self.var_to_ndx_map = ComponentMap(
            (v, ndx) for ndx, v in enumerate(self.all_vars)
        )

        self.current_node: Optional[pybnb.Node] = None
        self.feasible_objective = feasible_objective

        if self._sense == pybnb.minimize:
            if self.feasible_objective is None:
                feasible_objective = math.inf
            else:
                feasible_objective = self.feasible_objective
            feasible_objective += abs(feasible_objective) * 1e-3 + 1e-3
        else:
            if self.feasible_objective is None:
                feasible_objective = -math.inf
            else:
                feasible_objective = self.feasible_objective
            feasible_objective -= abs(feasible_objective) * 1e-3 + 1e-3

        for _ in range(self.config.num_root_obbt_iters):
            for r in self.relaxation_objects:
                r.rebuild()
            perform_obbt(
                relaxation,
                solver=self.config.lp_solver,
                varlist=list(self.rhs_vars),
                objective_bound=feasible_objective,
                parallel=False,
            )
            for r in self.relaxation_objects:
                r.rebuild(build_nonlinear_constraint=True)
            self.interval_tightener.perform_fbbt(self.relaxation)
        for r in self.relaxation_objects:
            r.rebuild()

    def sense(self):
        return self._sense

    def bound(self):
        # Do FBBT
        for r in self.relaxation_objects:
            r.rebuild(build_nonlinear_constraint=True)
            for v in self.binary_vars:
                v.domain = pe.Binary
            for v in self.integer_vars:
                v.domain = pe.Integers
        try:
            self.interval_tightener.perform_fbbt(self.relaxation)
        except InfeasibleConstraintException:
            return self.infeasible_objective()
        finally:
            for r in self.relaxation_objects:
                r.rebuild()
            for v in self.bin_and_int_vars:
                v.domain = pe.Reals

        # solve the relaxation
        res = self.config.lp_solver.solve(self.relaxation)
        if res.termination_condition == appsi.base.TerminationCondition.infeasible:
            return self.infeasible_objective()
        if res.termination_condition != appsi.base.TerminationCondition.optimal:
            raise RuntimeError(
                f"Cannot handle termination condition {res.termination_condition} when solving relaxation"
            )
        res.solution_loader.load_vars()

        # add OA cuts for convex constraints
        while True:
            added_cuts = False
            for r in self.relaxation_objects:
                new_con = r.add_cut(
                    keep_cut=True,
                    check_violation=True,
                    feasibility_tol=self.config.feasibility_tol,
                )
                if new_con is not None:
                    added_cuts = True
            if added_cuts:
                res = self.config.lp_solver.solve(self.relaxation)
                if (
                    res.termination_condition
                    == appsi.base.TerminationCondition.infeasible
                ):
                    return self.infeasible_objective()
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"Cannot handle termination condition {res.termination_condition} when solving relaxation"
                    )
                res.solution_loader.load_vars()
            else:
                break

        # add all other types of cuts
        while True:
            added_cuts = False
            for cg in self.cut_generators:
                cut_expr = cg.generate(self.current_node)
                if cut_expr is not None:
                    new_con = self.relaxation.cuts.add(cut_expr)
                    new_con_index = new_con.index()
                    self.current_node.state.valid_cut_indices.append(new_con_index)
                    self.current_node.state.active_cut_indices.append(new_con_index)
            if added_cuts:
                res = self.config.lp_solver.solve(self.relaxation)
                if (
                    res.termination_condition
                    == appsi.base.TerminationCondition.infeasible
                ):
                    return self.infeasible_objective()
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"Cannot handle termination condition {res.termination_condition} when solving relaxation"
                    )
                res.solution_loader.load_vars()
            else:
                break

        # save the variable values to reload later
        self.relaxation_solution = res.solution_loader.get_primals()

        # if the solution is feasible, we are done
        is_feasible = True
        for v in self.bin_and_int_vars:
            err = abs(v.value - round(v.value))
            if err > self.config.integer_tol:
                is_feasible = False
                break
        if is_feasible:
            for r in self.relaxation_objects:
                err = r.get_deviation()
                if err > self.config.feasibility_tol:
                    is_feasible = False
                    break
        if is_feasible:
            sol = np.array([v.value for v in self.all_vars], dtype=float)
            self.current_node.state.sol = sol
            self.current_node.state.obj = res.best_feasible_objective
            ret = res.best_feasible_objective
            if self.sense() == pybnb.minimize:
                ret -= min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
            else:
                ret += min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
            return ret

        # maybe do OBBT
        if (
            self.current_node.tree_depth % self.config.node_obbt_frequency == 0
            and self.current_node.tree_depth != 0
        ):
            should_obbt = True
            if self._sense == pybnb.minimize:
                if self.feasible_objective is None:
                    feasible_objective = math.inf
                else:
                    feasible_objective = self.feasible_objective
                feasible_objective += abs(feasible_objective) * 1e-3 + 1e-3
                if (
                    feasible_objective - res.best_objective_bound
                    <= self.config.mip_gap * feasible_objective + self.config.abs_gap
                ):
                    should_obbt = False
            else:
                if self.feasible_objective is None:
                    feasible_objective = -math.inf
                else:
                    feasible_objective = self.feasible_objective
                feasible_objective -= abs(feasible_objective) * 1e-3 + 1e-3
                if (
                    res.best_objective_bound - feasible_objective
                    <= self.config.mip_gap * feasible_objective + self.config.abs_gap
                ):
                    should_obbt = False
            if not math.isfinite(feasible_objective):
                feasible_objective = None
            if should_obbt:
                perform_obbt(
                    self.relaxation,
                    solver=self.config.lp_solver,
                    varlist=list(self.rhs_vars),
                    objective_bound=feasible_objective,
                    parallel=False,
                )
                for r in self.relaxation_objects:
                    r.rebuild()
                res = self.config.lp_solver.solve(self.relaxation)
                if (
                    res.termination_condition
                    == appsi.base.TerminationCondition.infeasible
                ):
                    return self.infeasible_objective()
                res.solution_loader.load_vars()
                self.relaxation_solution = res.solution_loader.get_primals()

        ret = res.best_objective_bound
        if self.sense() == pybnb.minimize:
            ret -= min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
        else:
            ret += min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
        return ret

    def objective(self):
        if self.current_node.state.sol is not None:
            return self.current_node.state.obj
        if self.current_node.tree_depth % 10 != 0:
            return self.infeasible_objective()
        unfixed_vars = [v for v in self.bin_and_int_vars if not v.is_fixed()]
        for v in unfixed_vars:
            val = round(v.value)
            if val < v.lb:
                val += 1
            if val > v.ub:
                val -= 1
            assert v.lb <= val <= v.ub
            v.fix(val)
        try:
            res = self.config.nlp_solver.solve(
                self.nlp, load_solutions=False, skip_trivial_constraints=True, tee=False
            )
            success = True
        except:
            success = False
        if not success or not pe.check_optimal_termination(res):
            ret = self.infeasible_objective()
        else:
            self.nlp.solutions.load_from(res)
            ret = pe.value(self.obj.expr)
            if self.sense == pybnb.minimize:
                if self.feasible_objective is None or ret < self.feasible_objective:
                    self.feasible_objective = ret
            else:
                if self.feasible_objective is None or ret > self.feasible_objective:
                    self.feasible_objective = ret
            sol = np.array([v.value for v in self.all_vars], dtype=float)
            self.current_node.state.sol = sol
            self.current_node.state.obj = ret
        for v in unfixed_vars:
            v.unfix()
        if self.sense() == pybnb.minimize:
            ret += min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
        else:
            ret -= min(abs(ret) * 0.001 * self.config.mip_gap, 0.01)
        return ret

    def get_state(self) -> NodeState:
        xl = list()
        xu = list()

        for v in self.bin_and_int_vars:
            xl.append(math.ceil(v.lb - self.config.integer_tol))
            xu.append(math.floor(v.ub + self.config.integer_tol))

        for v in self.rhs_vars + self.other_vars:
            lb, ub = v.bounds
            if lb is None:
                xl.append(-math.inf)
            else:
                xl.append(v.lb)
            if xu is None:
                xu.append(math.inf)
            else:
                xu.append(v.ub)

        xl = np.array(xl, dtype=float)
        xu = np.array(xu, dtype=float)

        return NodeState(xl, xu, None, None, None)

    def save_state(self, node):
        node.state = self.get_state()

    def load_state(self, node):
        self.current_node = node
        xl = node.state.lbs
        xu = node.state.ubs

        xl = [float(i) for i in xl]
        xu = [float(i) for i in xu]

        for v, lb, ub in zip(self.all_vars, xl, xu):
            if math.isfinite(lb):
                v.setlb(lb)
            else:
                v.setlb(None)
            if math.isfinite(ub):
                v.setub(ub)
            else:
                v.setub(None)

            v.unfix()

        _fix_vars_with_close_bounds(self.all_vars)

        for r in self.relaxation_objects:
            r.rebuild()

        for c in self.relaxation.cuts.values():
            c.deactivate()

        for ndx in node.state.active_cut_indices:
            self.relaxation.cuts[ndx].activate()

    def branch(self):
        ns = self.get_state()
        xl = ns.lbs
        xu = ns.ubs

        # reload the solution to the relaxation to make sure branching happens correctly
        for v, val in self.relaxation_solution.items():
            v.set_value(val, skip_validation=True)

        int_var_to_branch_on = None
        max_viol = 0
        for v in self.bin_and_int_vars:
            err = abs(v.value - round(v.value))
            if err > max_viol and err > self.config.integer_tol:
                int_var_to_branch_on = v
                max_viol = err

        max_viol = 0
        nl_var_to_branch_on = None
        for r in self.relaxation_objects:
            err = r.get_deviation()
            if err > max_viol and err > self.config.feasibility_tol:
                nl_var_to_branch_on = r.get_rhs_vars()[0]
                max_viol = err

        if self.current_node.tree_depth % 2 == 0:
            if int_var_to_branch_on is not None:
                var_to_branch_on = int_var_to_branch_on
            else:
                var_to_branch_on = nl_var_to_branch_on
        else:
            if nl_var_to_branch_on is not None:
                var_to_branch_on = nl_var_to_branch_on
            else:
                var_to_branch_on = int_var_to_branch_on

        if var_to_branch_on is None:
            # the relaxation was feasible
            # no nodes in this part of the tree need explored
            return []

        xl1 = xl.copy()
        xu1 = xu.copy()
        xl2 = xl.copy()
        xu2 = xu.copy()
        child1 = pybnb.Node()
        child2 = pybnb.Node()

        ndx_to_branch_on = self.var_to_ndx_map[var_to_branch_on]
        new_lb = new_ub = var_to_branch_on.value
        if ndx_to_branch_on < len(self.bin_and_int_vars):
            new_ub = math.floor(new_ub)
            new_lb = math.ceil(new_lb)
        xu1[ndx_to_branch_on] = new_ub
        xl2[ndx_to_branch_on] = new_lb

        child1.state = NodeState(xl1, xu1, self.current_node, None, None)
        child2.state = NodeState(xl2, xu2, self.current_node, None, None)

        child1.state.valid_cut_indices = list(self.current_node.state.valid_cut_indices)
        child2.state.valid_cut_indices = list(self.current_node.state.valid_cut_indices)
        child1.state.active_cut_indices = list(
            self.current_node.state.active_cut_indices
        )
        child2.state.active_cut_indices = list(
            self.current_node.state.active_cut_indices
        )

        yield child1
        yield child2


def solve_with_bnb(model: _BlockData, config: BnBConfig):
    # we don't want to modify the original model
    model, orig_var_map = get_clone_and_var_map(model)
    diving_obj, diving_sol = run_diving_heuristic(
        model, config.feasibility_tol, config.integer_tol, node_limit=100
    )
    prob = _BnB(model, config, feasible_objective=diving_obj)
    res: pybnb.SolverResults = pybnb.solve(
        prob,
        best_objective=diving_obj,
        queue_strategy=pybnb.QueueStrategy.bound,
        absolute_gap=config.abs_gap,
        relative_gap=config.mip_gap,
        comparison_tolerance=1e-4,
        comm=None,
        time_limit=config.time_limit,
        node_limit=config.node_limit,
        # log=logger,
    )
    ret = Results()
    ret.best_feasible_objective = res.objective
    ret.best_objective_bound = res.bound
    ss = pybnb.SolutionStatus
    tc = pybnb.TerminationCondition
    if res.solution_status == ss.optimal:
        ret.termination_condition = TerminationCondition.optimal
    elif res.solution_status == ss.infeasible:
        ret.termination_condition = TerminationCondition.infeasible
    elif res.solution_status == ss.unbounded:
        ret.termination_condition = TerminationCondition.unbounded
    elif res.termination_condition == tc.time_limit:
        ret.termination_condition = TerminationCondition.maxTimeLimit
    elif res.termination_condition == tc.objective_limit:
        ret.termination_condition = TerminationCondition.objectiveLimit
    elif res.termination_condition == tc.node_limit:
        ret.termination_condition = TerminationCondition.maxIterations
    elif res.termination_condition == tc.interrupted:
        ret.termination_condition = TerminationCondition.interrupted
    else:
        ret.termination_condition = TerminationCondition.unknown
    best_node = res.best_node
    if best_node is None:
        if diving_obj is not None:
            ret.solution_loader = SolutionLoader(
                primals={
                    id(orig_var_map[v]): (orig_var_map[v], val)
                    for v, val in diving_sol.items()
                },
                duals=None,
                slacks=None,
                reduced_costs=None,
            )
    else:
        vals = best_node.state.sol
        primals = dict()
        orig_vars = ComponentSet(prob.nlp.vars)
        for v, val in zip(prob.all_vars, vals):
            if v in orig_vars:
                ov = orig_var_map[v]
                primals[id(ov)] = (ov, val)
        ret.solution_loader = SolutionLoader(
            primals=primals, duals=None, slacks=None, reduced_costs=None
        )
    return ret


class BnBSolver(Solver):
    def __init__(self) -> None:
        super().__init__()
        self._config = BnBConfig()

    def available(self):
        return self.Availability.FullLicense

    def version(self) -> Tuple:
        return (1, 0, 0)

    @property
    def config(self):
        return self._config

    @property
    def symbol_map(self):
        raise NotImplementedError('do this')

    def solve(self, model: _BlockData, timer: HierarchicalTimer = None) -> Results:
        StaleFlagManager.mark_all_as_stale()
        res = solve_with_bnb(model, self.config)
        if self.config.load_solution:
            res.solution_loader.load_vars()
        return res


SolverFactory.register(name="coramin_bnb", doc="Coramin Branch and Bound Solver")(
    BnBSolver
)
