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

import logging
import re
import sys

from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.core.base import Suffix, Var, Constraint, SOSConstraint, Objective
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
    DirectOrPersistentSolver,
)
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import numpy as np
import time

logger = logging.getLogger('pyomo.solvers')

cuopt, cuopt_available = attempt_import(
    'cuopt',
    )

@SolverFactory.register('cuopt_direct', doc='Direct python interface to CUOPT')
class CUOPTDirect(DirectSolver):
    def __init__(self, **kwds):
        kwds['type'] = 'cuoptdirect'
        super(CUOPTDirect, self).__init__(**kwds)
        self._python_api_exists = True

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()
        log_file = None
        if self._log_file:
            log_file = self._log_file
        t0 = time.time()
        self.solution = cuopt.linear_programming.solver.Solve(self._solver_model)
        t1 = time.time()
        self._wallclock_time = t1 - t0
        return Bunch(rc=None, log=None)

    def _add_constraint(self, constraints):
        c_lb, c_ub = [], []
        matrix_data, matrix_indptr, matrix_indices = [], [0], []
        for i, con in enumerate(constraints):
            repn = generate_standard_repn(con.body, quadratic=False)
            matrix_data.extend(repn.linear_coefs)
            matrix_indices.extend([self.var_name_dict[str(i)] for i in repn.linear_vars])
            """for v, c in zip(con.body.linear_vars, con.body.linear_coefs):
                matrix_data.append(value(c))
                matrix_indices.append(self.var_name_dict[str(v)])"""
            matrix_indptr.append(len(matrix_data))
            c_lb.append(value(con.lower) if con.lower is not None else -np.inf)
            c_ub.append(value(con.upper) if con.upper is not None else np.inf)
        self._solver_model.set_csr_constraint_matrix(np.array(matrix_data), np.array(matrix_indices), np.array(matrix_indptr))
        self._solver_model.set_constraint_lower_bounds(np.array(c_lb))
        self._solver_model.set_constraint_upper_bounds(np.array(c_ub))

    def _add_var(self, variables):
        # Map vriable to index and get var bounds
        var_type_dict = {"Integers": 'I', "Reals": 'C', "Binary": 'I'} # NonNegativeReals ?
        self.var_name_dict = {}
        v_lb, v_ub, v_type = [], [], []

        for i, v in enumerate(variables):
            v_type.append(var_type_dict[str(v.domain)])
            if v.domain == "Binary":
                v_lb.append(0)
                v_ub.append(1)
            else:
                v_lb.append(v.lb if v.lb is not None else -np.inf)
                v_ub.append(v.ub if v.ub is not None else np.inf)
            self.var_name_dict[str(v)] = i
            self._pyomo_var_to_ndx_map[v] = self._ndx_count
            self._ndx_count += 1

        self._solver_model.set_variable_lower_bounds(np.array(v_lb))
        self._solver_model.set_variable_upper_bounds(np.array(v_ub))
        self._solver_model.set_variable_types(np.array(v_type))
        self._solver_model.set_variable_names(np.array(list(self.var_name_dict.keys())))

    def _set_objective(self, objective):
        repn = generate_standard_repn(objective.expr, quadratic=False)
        obj_coeffs = [0] * len(self.var_name_dict)
        for i, coeff in enumerate(repn.linear_coefs):
            obj_coeffs[self.var_name_dict[str(repn.linear_vars[i])]] = coeff
        self._solver_model.set_objective_coefficients(np.array(obj_coeffs))
        if objective.sense == maximize:
            self._solver_model.set_maximize(True)

    def _set_instance(self, model, kwds={}):
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self.var_name_dict = None
        self._pyomo_var_to_ndx_map = ComponentMap()
        self._ndx_count = 0

        try:
            self._solver_model = cuopt.linear_programming.DataModel()
        except Exception:
            e = sys.exc_info()[1]
            msg = (
                "Unable to create CUOPT model. "
                "Have you installed the Python "
                "SDK for CUOPT?\n\n\t" + "Error message: {0}".format(e)
            )
        self._add_block(model)

    def _add_block(self, block):
        self._add_var(block.component_data_objects(
            ctype=Var, descend_into=True, active=True, sort=True)
        )

        for sub_block in block.block_data_objects(descend_into=True, active=True):
            self._add_constraint(sub_block.component_data_objects(
                ctype=Constraint, descend_into=False, active=True, sort=True)
            )
            obj_counter = 0
            for obj in sub_block.component_data_objects(
                ctype=Objective, descend_into=False, active=True
            ):
                obj_counter += 1
                if obj_counter > 1:
                    raise ValueError(
                        "Solver interface does not support multiple objectives."
                    )
                self._set_objective(obj)

    def _postsolve(self):
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The cuopt_direct solver plugin cannot extract solution suffix="
                    + suffix
                )

        solution = self.solution
        status = solution.get_termination_status()
        self.results = SolverResults()
        soln = Solution()
        self.results.solver.name = "CUOPT"
        self.results.solver.wallclock_time = self._wallclock_time

        prob_type = solution.problem_category

        if status in [1]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status in [3]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status in [8]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        elif status in [2]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status in [4]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = (
                TerminationCondition.maxIterations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status in [5]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = (
                TerminationCondition.maxTimeLimit
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status in [7]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = (
                TerminationCondition.other
            )
            soln.status = SolutionStatus.other
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        if self._solver_model.maximize:
            self.results.problem.sense = maximize
        else:
            self.results.problem.sense = minimize

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        try:
            self.results.problem.upper_bound = solution.get_primal_objective()
            self.results.problem.lower_bound = solution.get_primal_objective()
        except Exception as e:
            pass

        var_map = self._pyomo_var_to_ndx_map
        primal_solution = solution.get_primal_solution().tolist()
        for i, pyomo_var in enumerate(var_map.keys()):
            pyomo_var.set_value(primal_solution[i], skip_validation=True)

        if extract_reduced_costs:
            self._load_rc()

        self.results.solution.insert(soln)
        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return False

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        rc = self._pyomo_model.rc
        var_map = self._pyomo_var_to_ndx_map
        if vars_to_load is None:
            vars_to_load = var_map.keys()
        reduced_costs = self.solution.get_reduced_costs()
        for pyomo_var in vars_to_load:
            rc[pyomo_var] = reduced_costs[var_map[pyomo_var]]

    def load_rc(self, vars_to_load):
        """
        Load the reduced costs into the 'rc' suffix. The 'rc' suffix must live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        self._load_rc(vars_to_load)
