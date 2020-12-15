#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
try:
    from mpi4py import MPI
    mpi4py_available = True
except:
    mpi4py_available = False
try:
    import numpy as np
    numpy_available = True
except:
    numpy_available = False
import logging


logger = logging.getLogger(__name__)


"""
It is easier to understand this code after reading "A note on feasibility in Benders Decomposition" by 
Grothey et al.

Original problem:

min f(x, y) + h0(y)
s.t.
    g(x, y) <= 0
    h(y) <= 0
    
where y are the complicating variables. Reformulate to 

min h0(y) + eta
s.t.
    g(x, y) <= 0
    f(x, y) <= eta
    h(y) <= 0
    
Master problem must be of the form

min h0(y) + eta
s.t.
    h(y) <= 0
    benders cuts
    
where the last constraint will be generated automatically with BendersCutGenerators. The BendersCutGenerators
must be handed a subproblem of the form

min f(x, y)
s.t.
    g(x, y) <= 0
    
except the constraints don't actually have to be in this form. The subproblem will automatically be transformed to

min _z
s.t.
    g(x, y) - z <= 0             (alpha)
    f(x, y) - eta - z <= 0       (beta)
    y - y_k = 0                  (gamma)
    eta - eta_k = 0              (delta)
"""


solver_dual_sign_convention = dict()
solver_dual_sign_convention['ipopt'] = -1
solver_dual_sign_convention['gurobi'] = -1
solver_dual_sign_convention['gurobi_direct'] = -1
solver_dual_sign_convention['gurobi_persistent'] = -1
solver_dual_sign_convention['cplex'] = -1
solver_dual_sign_convention['cplex_direct'] = -1
solver_dual_sign_convention['cplexdirect'] = -1
solver_dual_sign_convention['cplex_persistent'] = -1
solver_dual_sign_convention['glpk'] = -1
solver_dual_sign_convention['cbc'] = -1
solver_dual_sign_convention['xpress_direct'] = -1
solver_dual_sign_convention['xpress_persistent'] = -1


def _del_con(c):
    parent = c.parent_component()
    if parent.is_indexed():
        parent.__delitem__(c.index())
    else:
        assert parent is c
        c.parent_block().del_component(c)


def _any_common_elements(a, b):
    if len(a) < len(b):
        for i in a:
            if i in b:
                return True
    else:
        for i in b:
            if i in a:
                return True
    return False


def _setup_subproblem(b, master_vars, relax_subproblem_cons):
    # first get the objective and turn it into a constraint
    master_vars = ComponentSet(master_vars)

    objs = list(b.component_data_objects(pyo.Objective, descend_into=False, active=True))
    if len(objs) != 1:
        raise ValueError('Subproblem must have exactly one objective')
    orig_obj = objs[0]
    orig_obj_expr = orig_obj.expr
    b.del_component(orig_obj)

    b._z = pyo.Var(bounds=(0, None))
    b.objective = pyo.Objective(expr=b._z)
    b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    b._eta = pyo.Var()

    b.aux_cons = pyo.ConstraintList()
    for c in list(b.component_data_objects(pyo.Constraint, descend_into=True, active=True, sort=True)):
        if not relax_subproblem_cons:
            c_vars = ComponentSet(identify_variables(c.body, include_fixed=False))
            if not _any_common_elements(master_vars, c_vars):
                continue
        if c.equality:
            body = c.body
            rhs = pyo.value(c.lower)
            body -= rhs
            b.aux_cons.add(body - b._z <= 0)
            b.aux_cons.add(-body - b._z <= 0)
            _del_con(c)
        else:
            body = c.body
            lower = pyo.value(c.lower)
            upper = pyo.value(c.upper)
            if upper is not None:
                body_upper = body - upper - b._z
                b.aux_cons.add(body_upper <= 0)
            if lower is not None:
                body_lower = body - lower
                body_lower = -body_lower
                body_lower -= b._z
                b.aux_cons.add(body_lower <= 0)
            _del_con(c)

    b.obj_con = pyo.Constraint(expr=orig_obj_expr - b._eta - b._z <= 0)


@declare_custom_block(name='BendersCutGenerator')
class BendersCutGeneratorData(_BlockData):
    def __init__(self, component):
        if not mpi4py_available:
            raise ImportError('BendersCutGenerator requires mpi4py.')
        if not numpy_available:
            raise ImportError('BendersCutGenerator requires numpy.')
        _BlockData.__init__(self, component)
        
        self.num_subproblems_by_rank = 0 #np.zeros(self.comm.Get_size())
        self.subproblems = list()
        self.complicating_vars_maps = list()
        self.master_vars = list()
        self.master_vars_indices = pyo.ComponentMap()
        self.master_etas = list()
        self.cuts = None
        self.subproblem_solvers = list()
        self.tol = None
        self.all_master_etas = list()
        self._subproblem_ndx_map = dict()  # map from ndx in self.subproblems (local) to the global subproblem ndx


    def global_num_subproblems(self):
        return int(self.num_subproblems_by_rank.sum())

    def local_num_subproblems(self):
        return len(self.subproblems)

    def set_input(self, master_vars, tol=1e-6, comm = None):
        """
        It is very important for master_vars to be in the same order for every process.

        Parameters
        ----------
        master_vars
        tol
        """
        self.comm = None

        if comm is not None:
            self.comm = comm
        else:
            self.comm = MPI.COMM_WORLD
        self.num_subproblems_by_rank = np.zeros(self.comm.Get_size())
        del self.cuts
        self.cuts = pyo.ConstraintList()
        self.subproblems = list()
        self.master_etas = list()
        self.complicating_vars_maps = list()
        self.master_vars = list(master_vars)
        self.master_vars_indices = pyo.ComponentMap()
        for i, v in enumerate(self.master_vars):
            self.master_vars_indices[v] = i
        self.tol = tol
        self.subproblem_solvers = list()
        self.all_master_etas = list()
        self._subproblem_ndx_map = dict()

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, master_eta, subproblem_solver='gurobi_persistent', relax_subproblem_cons=False):
        _rank = np.argmin(self.num_subproblems_by_rank)
        self.num_subproblems_by_rank[_rank] += 1
        self.all_master_etas.append(master_eta)
        if _rank == self.comm.Get_rank():
            self.master_etas.append(master_eta)
            subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
            self.subproblems.append(subproblem)
            self.complicating_vars_maps.append(complicating_vars_map)
            _setup_subproblem(subproblem, master_vars=[complicating_vars_map[i] for i in self.master_vars if i in complicating_vars_map], relax_subproblem_cons=relax_subproblem_cons)
            self._subproblem_ndx_map[len(self.subproblems) - 1] = self.global_num_subproblems() - 1

            if isinstance(subproblem_solver, str):
                subproblem_solver = pyo.SolverFactory(subproblem_solver)
            self.subproblem_solvers.append(subproblem_solver)
            if isinstance(subproblem_solver, PersistentSolver):
                subproblem_solver.set_instance(subproblem)

    def generate_cut(self):
        coefficients = np.zeros(self.global_num_subproblems() * len(self.master_vars), dtype='d')
        constants = np.zeros(self.global_num_subproblems(), dtype='d')
        eta_coeffs = np.zeros(self.global_num_subproblems(), dtype='d')

        for local_subproblem_ndx in range(len(self.subproblems)):
            subproblem = self.subproblems[local_subproblem_ndx]
            global_subproblem_ndx = self._subproblem_ndx_map[local_subproblem_ndx]
            complicating_vars_map = self.complicating_vars_maps[local_subproblem_ndx]
            master_eta = self.master_etas[local_subproblem_ndx]
            coeff_ndx = global_subproblem_ndx * len(self.master_vars)

            subproblem.fix_complicating_vars = pyo.ConstraintList()
            var_to_con_map = pyo.ComponentMap()
            for master_var in self.master_vars:
                if master_var in complicating_vars_map:
                    sub_var = complicating_vars_map[master_var]
                    sub_var.value = master_var.value
                    new_con = subproblem.fix_complicating_vars.add(sub_var - master_var.value == 0)
                    var_to_con_map[master_var] = new_con
            subproblem.fix_eta = pyo.Constraint(expr=subproblem._eta - master_eta.value == 0)
            subproblem._eta.value = master_eta.value

            subproblem_solver = self.subproblem_solvers[local_subproblem_ndx]
            if subproblem_solver.name not in solver_dual_sign_convention:
                raise NotImplementedError('BendersCutGenerator is unaware of the dual sign convention of subproblem solver ' + subproblem_solver.name)
            sign_convention = solver_dual_sign_convention[subproblem_solver.name]

            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.add_constraint(c)
                subproblem_solver.add_constraint(subproblem.fix_eta)
                res = subproblem_solver.solve(tee=False, load_solutions=False, save_results=False)
                if res.solver.termination_condition != pyo.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem_solver.load_vars()
                subproblem_solver.load_duals()
            else:
                res = subproblem_solver.solve(subproblem, tee=False, load_solutions=False)
                if res.solver.termination_condition != pyo.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem.solutions.load_from(res)

            constants[global_subproblem_ndx] = pyo.value(subproblem._z)
            eta_coeffs[global_subproblem_ndx] = sign_convention * pyo.value(subproblem.dual[subproblem.obj_con])
            for master_var in self.master_vars:
                if master_var in complicating_vars_map:
                    c = var_to_con_map[master_var]
                    coefficients[coeff_ndx] = sign_convention * pyo.value(subproblem.dual[c])
                coeff_ndx += 1

            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.remove_constraint(c)
                subproblem_solver.remove_constraint(subproblem.fix_eta)
            del subproblem.fix_complicating_vars
            del subproblem.fix_complicating_vars_index
            del subproblem.fix_eta

        total_num_subproblems = self.global_num_subproblems()
        global_constants = np.zeros(total_num_subproblems, dtype='d')
        global_coeffs = np.zeros(total_num_subproblems*len(self.master_vars), dtype='d')
        global_eta_coeffs = np.zeros(total_num_subproblems, dtype='d')

        comm = self.comm
        comm.Allreduce([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        comm.Allreduce([eta_coeffs, MPI.DOUBLE], [global_eta_coeffs, MPI.DOUBLE])
        comm.Allreduce([coefficients, MPI.DOUBLE], [global_coeffs, MPI.DOUBLE])

        global_constants = [float(i) for i in global_constants]
        global_coeffs = [float(i) for i in global_coeffs]
        global_eta_coeffs = [float(i) for i in global_eta_coeffs]

        coeff_ndx = 0
        cuts_added = list()
        for global_subproblem_ndx in range(total_num_subproblems):
            cut_expr = global_constants[global_subproblem_ndx]
            if cut_expr > self.tol:
                master_eta = self.all_master_etas[global_subproblem_ndx]
                cut_expr -= global_eta_coeffs[global_subproblem_ndx] * (master_eta - master_eta.value)
                for master_var in self.master_vars:
                    coeff = global_coeffs[coeff_ndx]
                    cut_expr -= coeff * (master_var - master_var.value)
                    coeff_ndx += 1
                new_cut = self.cuts.add(cut_expr <= 0)
                cuts_added.append(new_cut)
            else:
                coeff_ndx += len(self.master_vars)

        return cuts_added
