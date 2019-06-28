from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pe
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.kernel.component_set import ComponentSet
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
solver_dual_sign_convention['cplex_persistent'] = -1
solver_dual_sign_convention['glpk'] = -1
solver_dual_sign_convention['cbc'] = -1


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

    objs = list(b.component_data_objects(pe.Objective, descend_into=False, active=True))
    if len(objs) != 1:
        raise ValueError('Subproblem must have exactly one objective')
    orig_obj = objs[0]
    orig_obj_expr = orig_obj.expr
    b.del_component(orig_obj)

    b._z = pe.Var(bounds=(0, None))
    b.objective = pe.Objective(expr=b._z)
    b.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    b._eta = pe.Var()

    b.aux_cons = pe.ConstraintList()
    for c in list(b.component_data_objects(pe.Constraint, descend_into=True, active=True, sort=True)):
        if not relax_subproblem_cons:
            c_vars = ComponentSet(identify_variables(c.body, include_fixed=False))
            if not _any_common_elements(master_vars, c_vars):
                continue
        if c.equality:
            body = c.body
            rhs = pe.value(c.lower)
            body -= rhs
            b.aux_cons.add(body - b._z <= 0)
            b.aux_cons.add(-body - b._z <= 0)
            _del_con(c)
        else:
            body = c.body
            lower = pe.value(c.lower)
            upper = pe.value(c.upper)
            if upper is not None:
                body_upper = body - upper - b._z
                b.aux_cons.add(body_upper <= 0)
            if lower is not None:
                body_lower = body - lower
                body_lower = -body_lower
                body_lower -= b._z
                b.aux_cons.add(body_lower <= 0)
            _del_con(c)

    b.obj_con = pe.Constraint(expr=orig_obj_expr - b._eta - b._z <= 0)


@declare_custom_block(name='BendersCutGenerator')
class BendersCutGeneratorData(_BlockData):
    def __init__(self, component):
        if not mpi4py_available:
            raise ImportError('BendersCutGenerator requires mpi4py.')
        if not numpy_available:
            raise ImportError('BendersCutGenerator requires numpy.')
        _BlockData.__init__(self, component)
        self.num_subproblems_by_rank = np.zeros(MPI.COMM_WORLD.Get_size())
        self.subproblems = list()
        self.complicating_vars_maps = list()
        self.master_vars = list()
        self.master_vars_indices = pe.ComponentMap()
        self.master_etas = list()
        self.cuts = None
        self.subproblem_solvers = list()
        self.tol = None
        self.all_master_etas = list()

    def set_input(self, master_vars, tol=1e-6):
        """
        It is very important for master_vars to be in the same order for every process.

        Parameters
        ----------
        master_vars
        master_eta
        tol

        Returns
        -------

        """
        self.num_subproblems_by_rank = np.zeros(MPI.COMM_WORLD.Get_size())
        del self.cuts
        self.cuts = pe.ConstraintList()
        self.subproblems = list()
        self.master_etas = list()
        self.complicating_vars_maps = list()
        self.master_vars = list(master_vars)
        self.master_vars_indices = pe.ComponentMap()
        for i, v in enumerate(self.master_vars):
            self.master_vars_indices[v] = i
        self.tol = tol
        self.subproblem_solvers = list()
        self.all_master_etas = list()

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, master_eta, subproblem_solver='gurobi_persistent', relax_subproblem_cons=False):
        _rank = np.argmin(self.num_subproblems_by_rank)
        self.num_subproblems_by_rank[_rank] += 1
        self.all_master_etas.append(master_eta)
        if _rank == MPI.COMM_WORLD.Get_rank():
            self.master_etas.append(master_eta)
            subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
            self.subproblems.append(subproblem)
            self.complicating_vars_maps.append(complicating_vars_map)
            _setup_subproblem(subproblem, master_vars=[complicating_vars_map[i] for i in self.master_vars if i in complicating_vars_map], relax_subproblem_cons=relax_subproblem_cons)

            if isinstance(subproblem_solver, str):
                subproblem_solver = pe.SolverFactory(subproblem_solver)
            self.subproblem_solvers.append(subproblem_solver)
            if isinstance(subproblem_solver, PersistentSolver):
                subproblem_solver.set_instance(subproblem)

    def generate_cut(self):
        coefficients = np.zeros(len(self.subproblems)*len(self.master_vars), dtype='d')
        constants = np.zeros(len(self.subproblems), dtype='d')
        eta_coeffs = np.zeros(len(self.subproblems), dtype='d')

        coeff_ndx = 0
        for subproblem_ndx in range(len(self.subproblems)):
            subproblem = self.subproblems[subproblem_ndx]
            complicating_vars_map = self.complicating_vars_maps[subproblem_ndx]
            master_eta = self.master_etas[subproblem_ndx]

            subproblem.fix_complicating_vars = pe.ConstraintList()
            var_to_con_map = pe.ComponentMap()
            for master_var in self.master_vars:
                if master_var in complicating_vars_map:
                    sub_var = complicating_vars_map[master_var]
                    sub_var.value = master_var.value
                    new_con = subproblem.fix_complicating_vars.add(sub_var - master_var.value == 0)
                    var_to_con_map[master_var] = new_con
            subproblem.fix_eta = pe.Constraint(expr=subproblem._eta - master_eta.value == 0)
            subproblem._eta.value = master_eta.value

            subproblem_solver = self.subproblem_solvers[subproblem_ndx]
            if subproblem_solver.name not in solver_dual_sign_convention:
                raise NotImplementedError('BendersCutGenerator is unaware of the dual sign convention of subproblem solver ' + self.subproblem_solver.name)
            sign_convention = solver_dual_sign_convention[subproblem_solver.name]

            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.add_constraint(c)
                subproblem_solver.add_constraint(subproblem.fix_eta)
                res = subproblem_solver.solve(tee=False, load_solutions=False, save_results=False)
                if res.solver.termination_condition != pe.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem_solver.load_vars()
                subproblem_solver.load_duals()
            else:
                res = subproblem_solver.solve(subproblem, tee=False, load_solutions=False)
                if res.solver.termination_condition != pe.TerminationCondition.optimal:
                    raise RuntimeError('Unable to generate cut because subproblem failed to converge.')
                subproblem.solutions.load_from(res)

            constants[subproblem_ndx] = pe.value(subproblem._z)
            eta_coeffs[subproblem_ndx] = sign_convention * pe.value(subproblem.dual[subproblem.obj_con])
            for master_var in self.master_vars:
                if master_var in complicating_vars_map:
                    c = var_to_con_map[master_var]
                    coefficients[coeff_ndx] = sign_convention * pe.value(subproblem.dual[c])
                coeff_ndx += 1

            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.remove_constraint(c)
                subproblem_solver.remove_constraint(subproblem.fix_eta)
            del subproblem.fix_complicating_vars
            del subproblem.fix_complicating_vars_index
            del subproblem.fix_eta

        total_num_subproblems = int(np.sum(self.num_subproblems_by_rank))
        global_constants = np.zeros(total_num_subproblems, dtype='d')
        global_coeffs = np.zeros(total_num_subproblems*len(self.master_vars), dtype='d')
        global_eta_coeffs = np.zeros(total_num_subproblems, dtype='d')

        comm = MPI.COMM_WORLD
        comm.Allgatherv([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        comm.Allgatherv([coefficients, MPI.DOUBLE], [global_coeffs, MPI.DOUBLE])
        comm.Allgatherv([eta_coeffs, MPI.DOUBLE], [global_eta_coeffs, MPI.DOUBLE])

        global_constants = [float(i) for i in global_constants]
        global_coeffs = [float(i) for i in global_coeffs]
        global_eta_coeffs = [float(i) for i in global_eta_coeffs]

        coeff_ndx = 0
        cuts_added = list()
        for subproblem_ndx in range(total_num_subproblems):
            cut_expr = global_constants[subproblem_ndx]
            if cut_expr > self.tol:
                master_eta = self.all_master_etas[subproblem_ndx]
                cut_expr -= global_eta_coeffs[subproblem_ndx] * (master_eta - master_eta.value)
                for master_var in self.master_vars:
                    coeff = global_coeffs[coeff_ndx]
                    cut_expr -= coeff * (master_var - master_var.value)
                    coeff_ndx += 1
                new_cut = self.cuts.add(cut_expr <= 0)
                cuts_added.append(new_cut)
            else:
                coeff_ndx += len(self.master_vars)
        return cuts_added
