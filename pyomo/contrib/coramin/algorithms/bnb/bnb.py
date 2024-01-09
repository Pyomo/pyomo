import pybnb
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.block import _BlockData
from pyomo.common.modeling import unique_component_name
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib import appsi
from pyomo.common.config import ConfigDict, ConfigValue, PositiveFloat
from pyomo.contrib.coramin.clone import clone_active_flat
from pyomo.contrib.coramin.relaxations.auto_relax import relax
from pyomo.repn.standard_repn import generate_standard_repn, StandardRepn
from pyomo.contrib.coramin.relaxations.split_expr import split_expr
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.contrib.coramin.relaxations import iterators
from pyomo.contrib.coramin.utils.pyomo_utils import get_objective
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.coramin.heuristics.diving import run_diving_heuristic
from pyomo.contrib.coramin.domain_reduction.obbt import perform_obbt
from typing import Tuple, List, Sequence
import math
import numpy as np
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


logger = logging.getLogger(__name__)


def _get_clone_and_var_map(m1: _BlockData):
    orig_vars = list()
    for c in iterators.nonrelaxation_component_data_objects(
        m1, pe.Constraint, active=True, descend_into=True
    ):
        for v in identify_variables(c.body, include_fixed=False):
            orig_vars.append(v)
    obj = get_objective(m1)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=False):
            orig_vars.append(v)
    for r in iterators.relaxation_data_objects(m1, descend_into=True, active=True):
        orig_vars.extend(r.get_rhs_vars())
        orig_vars.append(r.get_aux_var())
    orig_vars = list(ComponentSet(orig_vars))
    tmp_name = unique_component_name(m1, "active_vars")
    setattr(m1, tmp_name, orig_vars)
    m2 = m1.clone()
    new_vars = getattr(m2, tmp_name)
    var_map = ComponentMap(zip(new_vars, orig_vars))
    delattr(m1, tmp_name)
    delattr(m2, tmp_name)
    return m2, var_map


class BnBConfig(MIPSolverConfig):
    def __init__(self):
        super().__init__(None, None, False, None, 0)
        self.feasibility_tol = self.declare(
            "feasibility_tol", ConfigValue(domain=PositiveFloat, default=1e-8)
        )
        self.lp_solver = self.declare("lp_solver", ConfigValue())
        self.nlp_solver = self.declare("nlp_solver", ConfigValue())
        self.abs_gap = self.declare("abs_gap", ConfigValue(default=1e-4))
        self.integer_tol = self.declare("integer_tol", ConfigValue(default=1e-4))
        self.node_limit = self.declare("node_limit", ConfigValue(default=1000000000))
        self.mip_gap = 1e-3


def collect_vars(m: _BlockData) -> Tuple[List[_GeneralVarData], List[_GeneralVarData]]:
    binary_vars = ComponentSet()
    integer_vars = ComponentSet()
    for v in m.vars:
        if v.is_binary():
            binary_vars.add(v)
        elif v.is_integer():
            integer_vars.add(v)
    return list(binary_vars), list(integer_vars)


def relax_integers(binary_vars: Sequence[_GeneralVarData], integer_vars: Sequence[_GeneralVarData]):
    for v in list(binary_vars) + list(integer_vars):
        lb, ub = v.bounds
        v.domain = pe.Reals
        v.setlb(lb)
        v.setub(ub)


def impose_structure(m):
    m.aux_vars = pe.VarList()

    for key, c in list(m.nonlinear.cons.items()):
        repn: StandardRepn = generate_standard_repn(c.body, quadratic=False, compute_values=True)
        expr_list = split_expr(repn.nonlinear_expr)
        if len(expr_list) == 1:
            continue

        linear_coefs = list(repn.linear_coefs)
        linear_vars = list(repn.linear_vars)
        for term in expr_list:
            v = m.aux_vars.add()
            linear_coefs.append(1)
            linear_vars.append(v)
            m.vars.append(v)
            m.nonlinear.cons.add(v == term)
        new_expr = LinearExpression(constant=repn.constant, linear_coefs=linear_coefs, linear_vars=linear_vars)
        m.linear.cons.add((c.lb, new_expr, c.ub))
        del m.nonlinear.cons[key]


def _fix_vars_with_close_bounds(m, tol=1e-12):
    for v in m.vars:
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
        nlp, relaxation = clone_active_flat(model, 2)
        self.nlp: _BlockData = nlp
        relaxation: _BlockData = relaxation
        self.config = config
        self.config.lp_solver.config.load_solution = False
        self.config.lp_solver.update_config.treat_fixed_vars_as_params = True
        self.config.nlp_solver.config.load_solution = False

        obj = get_objective(nlp)
        if obj.sense == pe.minimize:
            self._sense = pybnb.minimize
        else:
            self._sense = pybnb.maximize

        # perform fbbt before constructing relaxations in case
        # we can identify things like x**3 is convex because
        # x >= 0
        self.interval_tightener = it = appsi.fbbt.IntervalTightener()
        it.config.deactivate_satisfied_constraints = True
        it.config.feasibility_tol = config.feasibility_tol
        if feasible_objective is not None:
            if obj.sense == pe.minimize:
                relaxation.obj_ineq = pe.Constraint(expr=obj.expr <= feasible_objective)
            else:
                relaxation.obj_ineq = pe.Constraint(expr=obj.expr >= feasible_objective)
        it.perform_fbbt(relaxation)
        del relaxation.obj_ineq
        _fix_vars_with_close_bounds(relaxation)

        impose_structure(relaxation)
        #find_cut_generators(relaxation)
        self.relaxation = relaxation = relax(relaxation)
        self.relaxation_objects = list()
        for r in iterators.relaxation_data_objects(relaxation, descend_into=True, active=True):
            self.relaxation_objects.append(r)

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
        other_vars = ComponentSet(i for i in nlp.vars if i not in var_set)
        other_vars.update(i for i in relaxation.aux_vars.values() if i not in var_set)
        self.other_vars = other_vars = list(other_vars)

        self.all_branching_vars = list(binary_vars) + list(integer_vars) + list(self.rhs_vars)
        self.all_vars = self.all_branching_vars + self.other_vars
        self.var_to_ndx_map = ComponentMap((v, ndx) for ndx, v in enumerate(self.all_vars))

        self.current_node: Optional[pybnb.Node] = None
        self.feasible_objective = feasible_objective

        for iter in range(3):
            perform_obbt(
                relaxation,
                solver=self.config.lp_solver,
                varlist=list(self.rhs_vars),
                objective_bound=feasible_objective,
            )
            for r in self.relaxation_objects:
                r.rebuild()

    def sense(self):
        return self._sense

    def bound(self):
        res = self.config.lp_solver.solve(self.relaxation)
        if res.termination_condition == appsi.base.TerminationCondition.infeasible:
            return self.infeasible_objective()
        if res.termination_condition != appsi.base.TerminationCondition.optimal:
            raise RuntimeError(f"Cannot handle termination condition {res.termination_condition} when solving relaxation")
        res.solution_loader.load_vars()

        while True:
            added_cuts = False
            for r in self.relaxation_objects:
                new_con = r.add_cut(keep_cut=True, check_violation=True, feasibility_tol=self.config.feasibility_tol)
                if new_con is not None:
                    added_cuts = True
            if added_cuts:
                res = self.config.lp_solver.solve(self.relaxation)
                if res.termination_condition == appsi.base.TerminationCondition.infeasible:
                    return self.infeasible_objective()
                if res.termination_condition != appsi.base.TerminationCondition.optimal:
                    raise RuntimeError(f"Cannot handle termination condition {res.termination_condition} when solving relaxation")
                res.solution_loader.load_vars()
            else:
                break

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
            return res.best_feasible_objective

        # maybe do OBBT
        if self.current_node.tree_depth % 2 == 0 and self.current_node.tree_depth != 0:
            should_obbt = True
            if self._sense == pybnb.minimize:
                if self.feasible_objective is None:
                    feasible_objective = math.inf
                else:
                    feasible_objective = self.feasible_objective
                feasible_objective += abs(feasible_objective) * 1e-3 + 1e-3
                if feasible_objective - res.best_objective_bound <= self.config.mip_gap * feasible_objective + self.config.abs_gap:
                    should_obbt = False
            else:
                if self.feasible_objective is None:
                    feasible_objective = -math.inf
                else:
                    feasible_objective = self.feasible_objective
                feasible_objective -= abs(feasible_objective) * 1e-3 + 1e-3
                if res.best_objective_bound - feasible_objective <= self.config.mip_gap * feasible_objective + self.config.abs_gap:
                    should_obbt = False
            if not math.isfinite(feasible_objective):
                feasible_objective = None
            if should_obbt:
                perform_obbt(
                    self.relaxation,
                    solver=self.config.lp_solver,
                    varlist=list(self.rhs_vars),
                    objective_bound=feasible_objective,
                )
                for r in self.relaxation_objects:
                    r.rebuild()

        return res.best_objective_bound

    def objective(self):
        if self.current_node.state[2] is not None:
            return self.current_node.state[3]
        if self.current_node.tree_depth % 10 != 0:
            return self.infeasible_objective()
        unfixed_vars = [v for v in self.bin_and_int_vars if not v.is_fixed()]
        vals = [v.value for v in unfixed_vars]
        for v in unfixed_vars:
            v.fix(round(v.value))
        res = self.config.nlp_solver.solve(self.nlp)
        if res.best_feasible_objective is None:
            ret = self.infeasible_objective()
        else:
            ret = res.best_feasible_objective
            if self.sense == pybnb.minimize:
                if ret < self.feasible_objective:
                    self.feasible_objective = ret
            else:
                if ret > self.feasible_objective:
                    self.feasible_objective = ret
            res.solution_loader.load_vars()
            orig_vars = ComponentSet(self.nlp.vars)
            sol = np.array([v.value for v in self.all_vars], dtype=float)
            xl, xu, _, _ = self.current_node.state
            self.current_node.state = (xl, xu, sol, ret)
        for v, val in zip(unfixed_vars, vals):
            v.unfix()
            # we have to restore the values so that branch() works properly
            v.set_value(val, skip_validation=True)
        return ret

    def get_state(self):
        xl = list()
        xu = list()

        for v in self.bin_and_int_vars:
            xl.append(math.ceil(v.lb))
            xu.append(math.floor(v.ub))

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

        return xl, xu, None, None

    def save_state(self, node):
        node.state = self.get_state()

    def load_state(self, node):
        self.current_node = node
        xl, xu, _, _ = node.state

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

            if lb == ub:
                v.fix(lb)
            else:
                v.unfix()

        for r in self.relaxation_objects:
            r.rebuild()

    def branch(self):
        xl, xu, _, _ = self.get_state()

        var_to_branch_on = None
        max_viol = 0
        for v in self.bin_and_int_vars:
            err = abs(v.value - round(v.value))
            if err > max_viol and err > self.config.integer_tol:
                var_to_branch_on = v
                max_viol = err

        if var_to_branch_on is None:
            for r in self.relaxation_objects:
                err = r.get_deviation()
                if err > max_viol and err > self.config.feasibility_tol:
                    var_to_branch_on = r.get_rhs_vars()[0]
                    max_viol = err

        if var_to_branch_on is None:
            # the relaxation was feasible
            # no nodes in this part of the tree need explored
            return pybnb.Node()

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
        
        child1.state = (xl1, xu1, None, None)
        child2.state = (xl2, xu2, None, None)

        yield child1
        yield child2


def solve_with_bnb(model: _BlockData, config: BnBConfig, comm=None):
    # we don't want to modify the original model
    model, orig_var_map = _get_clone_and_var_map(model)
    diving_obj, diving_sol = run_diving_heuristic(model, config.nlp_solver, config.feasibility_tol, config.integer_tol)
    prob = _BnB(model, config, feasible_objective=diving_obj)
    res: pybnb.SolverResults = pybnb.solve(
        prob,
        best_objective=diving_obj,
        queue_strategy=pybnb.QueueStrategy.bound,
        absolute_gap=config.abs_gap,
        relative_gap=config.mip_gap,
        comparison_tolerance=1e-4,
        comm=comm,
        time_limit=config.time_limit,
        node_limit=config.node_limit,
        # log=logger,
    )
    ret = Results()
    ret.best_feasible_objective = res.objective
    ret.best_objective_bound = res.bound
    ss = pybnb.SolutionStatus
    if res.solution_status == ss.optimal:
        ret.termination_condition = TerminationCondition.optimal
    elif res.solution_status == ss.infeasible:
        ret.termination_condition = TerminationCondition.infeasible
    elif res.solution_status == ss.unbounded:
        ret.termination_condition = TerminationCondition.unbounded
    else:
        ret.termination_condition = TerminationCondition.unknown
    best_node = res.best_node
    if best_node is None:
        if diving_obj is not None:
            ret.solution_loader = SolutionLoader(primals={id(orig_var_map[v]): (orig_var_map[v], val) for v, val in diving_sol.items()}, duals=None, slacks=None, reduced_costs=None)
    else:
        vals = best_node.state[2]
        primals = dict()
        orig_vars = ComponentSet(prob.nlp.vars)
        for v, val in zip(prob.all_vars, vals):
            if v in orig_vars:
                ov = orig_var_map[v]
                primals[id(ov)] = (ov, val)
        ret.solution_loader = SolutionLoader(primals=primals, duals=None, slacks=None, reduced_costs=None)
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
    

SolverFactory.register(name="coramin_bnb", doc="Coramin Branch and Bound Solver")(BnBSolver)
