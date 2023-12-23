import pyomo.environ as pe
import pybnb
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from typing import Tuple, List, Sequence, Optional, MutableMapping
from pyomo.contrib import appsi
from pyomo.contrib.coramin.utils.pyomo_utils import get_objective
import numpy as np
import math
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
from mpi4py import MPI
import time

np.set_printoptions(linewidth=1000)


def collect_vars(m: _BlockData) -> Tuple[List[_GeneralVarData], List[_GeneralVarData], List[_GeneralVarData]]:
    binary_vars = ComponentSet()
    integer_vars = ComponentSet()
    all_vars = ComponentSet()
    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=pe.Block):
        for v in identify_variables(c.body, include_fixed=False):
            all_vars.add(v)
            if v.is_binary():
                binary_vars.add(v)
            elif v.is_integer():
                integer_vars.add(v)
    obj = get_objective(m)
    if obj is not None:
        for v in identify_variables(obj.expr, include_fixed=False):
            all_vars.add(v)
            if v.is_binary():
                binary_vars.add(v)
            elif v.is_integer():
                integer_vars.add(v)
    return list(binary_vars), list(integer_vars), list(all_vars)


def relax_integers(binary_vars: Sequence[_GeneralVarData], integer_vars: Sequence[_GeneralVarData]):
    for v in list(binary_vars) + list(integer_vars):
        lb, ub = v.bounds
        v.domain = pe.Reals
        v.setlb(lb)
        v.setub(ub)


def restore_integers(binary_vars: Sequence[_GeneralVarData], integer_vars: Sequence[_GeneralVarData]):
    for v in binary_vars:
        v.domain = pe.Binary
    for v in integer_vars:
        v.domain = pe.Integers


class DivingHeuristic(pybnb.Problem):
    def __init__(self, m: _BlockData, nlp_solver: appsi.base.Solver) -> None:
        super().__init__()

        nlp_solver.config.load_solution = False

        binary_vars, integer_vars, all_vars = collect_vars(m)
        relax_integers(binary_vars, integer_vars)

        self.m = m
        self.opt = nlp_solver
        self.all_vars = all_vars
        self.binary_vars = binary_vars
        self.integer_vars = integer_vars
        self.bin_and_int_vars = list(binary_vars) + list(integer_vars)
        self.orig_lbs = [v.lb for v in self.bin_and_int_vars]
        self.orig_ubs = [v.ub for v in self.bin_and_int_vars]
        self.obj = get_objective(m)

        if self.obj.sense == pe.minimize:
            self._sense = pybnb.minimize
        else:
            self._sense = pybnb.maximize

        self.current_node: Optional[pybnb.Node] = None

    def sense(self):
        return self._sense
    
    def bound(self):
        res = self.opt.solve(self.m)
        if res.best_feasible_objective is None:
            return self.infeasible_objective()
        res.solution_loader.load_vars([v for v in self.bin_and_int_vars if not v.is_fixed()])
        ret = res.best_feasible_objective
        if self._sense == pybnb.minimize:
            ret = max(self.current_node.bound, ret)
        else:
            ret = min(self.current_node.bound, ret)
        return ret
    
    def objective(self):
        unfixed_vars = [v for v in self.bin_and_int_vars if not v.is_fixed()]
        vals = [v.value for v in unfixed_vars]
        for v in unfixed_vars:
            v.fix(round(v.value))
        res = self.opt.solve(self.m)
        if res.best_feasible_objective is None:
            ret = self.infeasible_objective()
        else:
            ret = res.best_feasible_objective
            res.solution_loader.load_vars()
            sol = np.array([v.value for v in self.all_vars], dtype=float)
            xl, xu, _ = self.current_node.state
            self.current_node.state = (xl, xu, sol)
        for v, val in zip(unfixed_vars, vals):
            v.unfix()
            # we have to restore the values so that branch() works properly
            v.set_value(val, skip_validation=True)
        return ret
    
    def get_state(self):
        xl = [math.ceil(v.lb) for v in self.bin_and_int_vars]
        xl = np.array(xl, dtype=int)

        xu = [math.floor(v.ub) for v in self.bin_and_int_vars]
        xu = np.array(xu, dtype=int)

        return xl, xu, None
    
    def save_state(self, node):
        node.state = self.get_state()

    def load_state(self, node):
        self.current_node = node
        xl, xu, _ = node.state
        xl = [int(i) for i in xl]
        xu = [int(i) for i in xu]

        for v, lb, ub in zip(self.bin_and_int_vars, xl, xu):
            v.setlb(lb)
            v.setub(ub)
            if lb == ub:
                v.fix(lb)
            else:
                v.unfix()

    def branch(self):
        xl, xu, _ = self.get_state()
        dist_list = [(abs(v.value - round(v.value)), ndx) for ndx, v in enumerate(self.bin_and_int_vars)]
        dist_list.sort(key=lambda i: i[0], reverse=True)
        ndx = dist_list[0][1]
        branching_var = self.bin_and_int_vars[ndx]

        xl1 = xl.copy()
        xu1 = xu.copy()
        xu1[ndx] = math.floor(branching_var.value)
        child1 = pybnb.Node()
        child1.state = (xl1, xu1, None)

        xl2 = xl.copy()
        xu2 = xu.copy()
        xl2[ndx] = math.ceil(branching_var.value)
        child2 = pybnb.Node()
        child2.state = (xl2, xu2, None)

        yield child1
        yield child2


def assert_feasible(m: _BlockData, var_list: Sequence[_GeneralVarData], feasibility_tol: float, integer_tol: float):
    for c in m.component_data_objects(pe.Constraint, active=True, descend_into=pe.Block):
        body_val = pe.value(c.body)
        if c.lb is not None:
            assert c.lb - feasibility_tol <= body_val or abs(c.lb - body_val)/abs(c.lb) <= feasibility_tol
        if c.ub is not None:
            assert body_val <= c.ub + feasibility_tol or abs(c.ub - body_val)/abs(c.ub) <= feasibility_tol

    for v in var_list:
        val = v.value
        lb, ub = v.bounds
        if lb is not None:
            assert lb - feasibility_tol <= val or abs(lb - val)/abs(lb) <= feasibility_tol
        if ub is not None:
            assert val <= ub + feasibility_tol or abs(ub - val)/abs(ub) <= feasibility_tol
        if v.is_integer():
            assert abs(val - round(val)) <= integer_tol


def run_diving_heuristic(m: _BlockData, nlp_solver: appsi.base.Solver, feasibility_tol: float = 1e-6, integer_tol: float = 1e-4, time_limit: float = 300, node_limit: int = 1000):
    prob = DivingHeuristic(m, nlp_solver)
    res: pybnb.SolverResults = pybnb.solve(prob, queue_strategy=pybnb.QueueStrategy.bound, objective_stop=prob.infeasible_objective(), node_limit=node_limit, time_limit=time_limit)
    ss = pybnb.SolutionStatus
    if res.solution_status in {ss.feasible, ss.optimal}:
        best_obj = res.objective
        best_sol: MutableMapping[_GeneralVarData, float] = ComponentMap(zip(prob.all_vars, res.best_node.state[2]))
    else:
        best_obj = None
        best_sol = None

    restore_integers(prob.binary_vars, prob.integer_vars)
    for v, lb, ub in zip(prob.bin_and_int_vars, prob.orig_lbs, prob.orig_ubs):
        v.unfix()
        v.setlb(lb)
        v.setub(ub)

    if best_sol is not None:
        # double check that the solution is feasible
        for v, val in best_sol.items():
            v.set_value(val, skip_validation=True)
        assert_feasible(m, prob.all_vars, feasibility_tol, integer_tol)

    return best_obj, best_sol
