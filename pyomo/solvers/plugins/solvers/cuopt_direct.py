# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import re
import time

from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import numpy as np
from pyomo.core.base import Suffix, Var, Constraint, Objective
from pyomo.core.staleflag import StaleFlagManager
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
    DirectOrPersistentSolver,
)
from pyomo.common.enums import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory

logger = logging.getLogger(__name__)


def _get_cuopt_version(cuopt, avail):
    if not avail:
        return
    CUOPTDirect._version = tuple(cuopt.__version__.split('.'))
    CUOPTDirect._name = "cuOpt %s.%s%s" % CUOPTDirect._version


cuopt, cuopt_available = attempt_import("cuopt", callback=_get_cuopt_version)


@SolverFactory.register("cuopt", doc="Direct python interface to CUOPT")
class CUOPTDirect(DirectSolver):
    def __init__(self, **kwds):
        kwds["type"] = "cuoptdirect"
        super().__init__(**kwds)
        self._python_api_exists = cuopt_available
        # Note: Undefined capabilities default to None
        self._capabilities.linear = True
        self._capabilities.integer = True
        self.referenced_vars = ComponentSet()
        # remove the instance-level definition of the cuopt version:
        # because the version comes from an imported module, only one
        # version of cuopt is supported (and stored as a class attribute)
        del self._version

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()
        log_file = ""
        if self._log_file:
            log_file = self._log_file
        logger.debug("Applying cuOpt solver")
        t0 = time.time()
        settings = cuopt.linear_programming.solver_settings.SolverSettings()
        settings.set_parameter("log_file", log_file)
        for key, option in self.options.items():
            settings.set_parameter(key, option)
        self.solution = cuopt.linear_programming.solver.Solve(
            self._solver_model, settings
        )
        t1 = time.time()
        self._wallclock_time = t1 - t0
        logger.debug("cuOpt solver completed in %.4f seconds", self._wallclock_time)
        return Bunch(rc=None, log=None)

    def _add_constraints(self, constraints):
        # build constraint matrix for cuopt
        c_lb, c_ub = [], []
        matrix_data = []
        matrix_indptr = [0]
        matrix_indices = []

        # visitor walks expression trees and extracts linear coefficients
        visitor = LinearRepnVisitor({})
        con_idx = 0
        for con in constraints:
            if not con.active:
                continue

            lb, body, ub = con.to_bounded_expression(evaluate_bounds=True)
            if lb is None and ub is None:
                assert not con.equality
                continue  # non-binding, so skip
            repn = visitor.walk_expression(body)

            if repn.nonlinear is not None:
                raise ValueError(
                    f"Constraint '{con.name}' contains nonlinear terms which are "
                    "not supported by cuOpt solver."
                )

            # check for trivial constraints after getting repn (more efficient
            # than walking expression twice with is_fixed)
            if not repn.linear:
                if self._skip_trivial_constraints:
                    # verify feasibility before skipping
                    const = repn.constant if repn.constant else 0
                    lb_val = lb if lb is not None else -np.inf
                    ub_val = ub if ub is not None else np.inf
                    if not (lb_val <= const <= ub_val):
                        raise ValueError(
                            f"Trivial constraint {con.name} is infeasible "
                            f"(constant={const}, bounds=[{lb_val}, {ub_val}])"
                        )
                    continue
                # if not skipping, still need to add it (even if trivial)

            self._symbol_map.getSymbol(con, self._labeler)
            self._pyomo_con_to_solver_con_map[con] = con_idx
            con_idx += 1

            # repn.linear is keyed by id(var), use var_map to get actual vars
            for var_id, coef in repn.linear.items():
                var = visitor.var_map[var_id]
                matrix_data.append(coef)
                matrix_indices.append(self._pyomo_var_to_ndx_map[var])
                self.referenced_vars.add(var)

            matrix_indptr.append(len(matrix_data))
            const = repn.constant if repn.constant else 0
            c_lb.append(lb - const if lb is not None else -np.inf)
            c_ub.append(ub - const if ub is not None else np.inf)

        if len(matrix_data) == 0:
            matrix_data = [0]
            matrix_indices = [0]
            matrix_indptr = [0, 1]
            c_lb = [0]
            c_ub = [0]

        self._solver_model.set_csr_constraint_matrix(
            np.array(matrix_data), np.array(matrix_indices), np.array(matrix_indptr)
        )
        self._solver_model.set_constraint_lower_bounds(np.array(c_lb))
        self._solver_model.set_constraint_upper_bounds(np.array(c_ub))

    def _add_variables(self, variables):
        # Map variable to index and get var bounds
        v_lb, v_ub, v_type, v_names = [], [], [], []

        for v in variables:
            lb, ub = v.bounds
            if v.is_integer():
                v_type.append("I")
            elif v.is_continuous():
                v_type.append("C")
            else:
                logger.error("Unallowable domain for variable %s", v.name)
                raise ValueError(f"Unallowable domain for variable {v.name}")
            v_lb.append(lb if lb is not None else -np.inf)
            v_ub.append(ub if ub is not None else np.inf)
            v_names.append(self._symbol_map.getSymbol(v, self._labeler))
            self._pyomo_var_to_ndx_map[v] = self._ndx_count
            self._ndx_count += 1

        self._solver_model.set_variable_lower_bounds(np.array(v_lb))
        self._solver_model.set_variable_upper_bounds(np.array(v_ub))
        self._solver_model.set_variable_types(np.array(v_type))
        self._solver_model.set_variable_names(np.array(v_names))

    def _set_objective(self, objective):
        visitor = LinearRepnVisitor({})
        repn = visitor.walk_expression(objective.expr)
        if repn.nonlinear is not None:
            raise ValueError(
                f"Objective contains nonlinear terms which are "
                "not supported by cuOpt solver."
            )

        obj_coeffs = [0] * len(self._pyomo_var_to_ndx_map)
        # repn.linear is keyed by id(var), use var_map to get actual vars
        for var_id, coef in repn.linear.items():
            var = visitor.var_map[var_id]
            obj_coeffs[self._pyomo_var_to_ndx_map[var]] = coef
            self.referenced_vars.add(var)
        self._solver_model.set_objective_coefficients(np.array(obj_coeffs))
        self._solver_model.set_maximize(objective.sense == maximize)

    def _set_instance(self, model, kwds={}):
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_var_to_ndx_map = ComponentMap()
        self._ndx_count = 0

        try:
            self._solver_model = cuopt.linear_programming.DataModel()
        except Exception as e:
            msg = (
                "Unable to create CUOPT model. "
                "Have you installed the Python "
                "SDK for CUOPT?\n\n\t" + "Error message: {0}".format(e)
            )
            logger.error(msg)
            raise Exception(msg)
        self._add_block(model)

    def _add_block(self, block):
        self._add_variables(
            block.component_data_objects(ctype=Var, descend_into=True, sort=True)
        )
        self._add_constraints(
            block.component_data_objects(
                ctype=Constraint, descend_into=True, active=True, sort=True
            )
        )
        objectives = list(
            block.component_data_objects(Objective, descend_into=True, active=True)
        )
        if len(objectives) > 1:
            raise ValueError("Solver interface does not support multiple objectives.")
        elif objectives:
            self._set_objective(objectives[0])

    def _postsolve(self):
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
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

        is_mip = solution.get_problem_category()

        # Termination Status
        # 0 - CUOPT_TERIMINATION_STATUS_NO_TERMINATION
        # 1 - CUOPT_TERIMINATION_STATUS_OPTIMAL
        # 2 - CUOPT_TERIMINATION_STATUS_INFEASIBLE
        # 3 - CUOPT_TERIMINATION_STATUS_UNBOUNDED
        # 4 - CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT
        # 5 - CUOPT_TERIMINATION_STATUS_TIME_LIMIT
        # 6 - CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR
        # 7 - CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE
        # 8 - CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND
        # 9 - CUOPT_TERIMINATION_STATUS_CONCURRENT_LIMIT

        if status == 1:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == 3:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == 8:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.feasible
            soln.status = SolutionStatus.feasible
        elif status == 2:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == 4:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = (
                TerminationCondition.maxIterations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == 5:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = (
                TerminationCondition.maxTimeLimit
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == 7:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.other
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
        if is_mip:
            ObjBound = solution.get_milp_stats()["solution_bound"]
            ObjVal = solution.get_primal_objective()
            if self._solver_model.maximize:
                self.results.problem.upper_bound = ObjBound
                self.results.problem.lower_bound = ObjVal
            else:
                self.results.problem.upper_bound = ObjVal
                self.results.problem.lower_bound = ObjBound
        else:
            self.results.problem.upper_bound = solution.get_primal_objective()
            self.results.problem.lower_bound = solution.get_primal_objective()

        var_map = self._pyomo_var_to_ndx_map
        con_map = self._pyomo_con_to_solver_con_map

        primal_solution = solution.get_primal_solution().tolist()
        reduced_costs = None
        dual_solution = None
        if is_mip:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
        else:
            if extract_reduced_costs:
                reduced_costs = solution.get_reduced_cost()
            if extract_duals:
                dual_solution = solution.get_dual_solution()

        if self._save_results:
            soln_variables = soln.variable
            soln_constraints = soln.constraint
            for pyomo_var in var_map.keys():
                if len(primal_solution) > 0 and pyomo_var in self.referenced_vars:
                    name = self._symbol_map.getSymbol(pyomo_var, self._labeler)
                    soln_variables[name] = {
                        "Value": primal_solution[var_map[pyomo_var]]
                    }
                    if reduced_costs is not None and len(reduced_costs) > 0:
                        soln_variables[name]["Rc"] = reduced_costs[var_map[pyomo_var]]
            for pyomo_con in con_map.keys():
                if dual_solution is not None and len(dual_solution) > 0:
                    con_name = self._symbol_map.getSymbol(pyomo_con, self._labeler)
                    soln_constraints[con_name] = {
                        "Dual": dual_solution[con_map[pyomo_con]]
                    }

        elif self._load_solutions:
            if len(primal_solution) > 0:
                self.load_vars()

                if reduced_costs is not None:
                    self._load_rc()

                if dual_solution is not None:
                    self._load_duals()

        self.results.solution.insert(soln)
        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return False

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_ndx_map
        if vars_to_load is None:
            vars_to_load = var_map.keys()
        primal_solution = self.solution.get_primal_solution()
        for pyomo_var in vars_to_load:
            if pyomo_var in self.referenced_vars:
                pyomo_var.set_value(
                    primal_solution[var_map[pyomo_var]], skip_validation=True
                )

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, "rc"):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        rc = self._pyomo_model.rc
        var_map = self._pyomo_var_to_ndx_map
        if vars_to_load is None:
            vars_to_load = var_map.keys()
        reduced_costs = self.solution.get_reduced_cost()
        for pyomo_var in vars_to_load:
            rc[pyomo_var] = reduced_costs[var_map[pyomo_var]]

    def load_rc(self, vars_to_load=None):
        """
        Load the reduced costs into the 'rc' suffix. The 'rc' suffix must live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        is_mip = self.solution.get_problem_category()
        if is_mip:
            logger.warning("Cannot get reduced costs for MIP.")
        else:
            self._load_rc(vars_to_load)

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        dual = self._pyomo_model.dual
        con_map = self._pyomo_con_to_solver_con_map
        if cons_to_load is None:
            cons_to_load = con_map.keys()
        dual_solution = self.solution.get_dual_solution()
        for pyomo_con in cons_to_load:
            dual[pyomo_con] = dual_solution[con_map[pyomo_con]]

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the 'dual' suffix. The 'dual' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        is_mip = self.solution.get_problem_category()
        if is_mip:
            logger.warning("Cannot get duals for MIP.")
        else:
            self._load_duals(cons_to_load)
