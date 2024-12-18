#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import sys

from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import Var
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    ProductExpression,
    UnaryFunctionExpression,
    PowExpression,
    DivisionExpression,
)
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


logger = logging.getLogger("pyomo.solvers")


class DegreeError(ValueError):
    pass


def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


@SolverFactory.register("scip_direct", doc="Direct python interface to SCIP")
class SCIPDirect(DirectSolver):

    def __init__(self, **kwds):
        kwds["type"] = "scipdirect"
        DirectSolver.__init__(self, **kwds)
        self._init()
        self._solver_model = None

    def _init(self):
        try:
            import pyscipopt

            self._scip = pyscipopt
            self._python_api_exists = True
            self._version = tuple(
                int(k) for k in str(self._scip.Model().version()).split(".")
            )
            self._version_major = self._version[0]
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            print(f"Import of pyscipopt failed - SCIP message={str(e)}\n")
            self._python_api_exists = False

        # Note: Undefined capabilities default to None
        self._max_constraint_degree = None
        self._max_obj_degree = 1
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = False
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True
        self._skip_trivial_constraints = True

        # Dictionary used exclusively for SCIP, as we want the constraint expressions
        self._pyomo_var_to_solver_var_expr_map = ComponentMap()
        self._pyomo_con_to_solver_con_expr_map = dict()

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()

        # Suppress solver output if requested
        if self._tee:
            self._solver_model.hideOutput(quiet=False)
        else:
            self._solver_model.hideOutput(quiet=True)

        # Redirect solver output to a logfile if requested
        if self._keepfiles:
            # Only save log file when the user wants to keep it.
            self._solver_model.setLogfile(self._log_file)
            print(f"Solver log file: {self._log_file}")

        # Set user specified parameters
        for key, option in self.options.items():
            try:
                key_type = type(self._solver_model.getParam(key))
            except KeyError:
                raise ValueError(f"Key {key} is an invalid parameter for SCIP")

            if key_type == str:
                self._solver_model.setParam(key, option)
            else:
                if not _is_numeric(option):
                    raise ValueError(
                        f"Value {option} for parameter {key} is not a string and can't be converted to float"
                    )
                self._solver_model.setParam(key, float(option))

        self._solver_model.optimize()

        # TODO: Check if this is even needed, or if it is sufficient to close the open file
        # if self._keepfiles:
        #     self._solver_model.setLogfile(None)

        # FIXME: can we get a return code indicating if SCIP had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=None):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (max_degree is not None) and (degree > max_degree):
            raise DegreeError(
                "While SCIP supports general non-linear constraints, the objective must be linear. "
                "Please reformulate the objective by introducing a new variable. "
                "For min problems: min z s.t z >= f(x). For max problems: max z s.t z <= f(x). "
                "f(x) is the original non-linear objective."
            )

        new_expr = repn.constant

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr += sum(
                repn.linear_coefs[i] * self._pyomo_var_to_solver_var_expr_map[var]
                for i, var in enumerate(repn.linear_vars)
            )

        for i, v in enumerate(repn.quadratic_vars):
            x, y = v
            new_expr += (
                repn.quadratic_coefs[i]
                * self._pyomo_var_to_solver_var_expr_map[x]
                * self._pyomo_var_to_solver_var_expr_map[y]
            )
            referenced_vars.add(x)
            referenced_vars.add(y)

        if repn.nonlinear_expr is not None:

            def get_nl_expr_recursively(pyomo_expr):
                if not hasattr(pyomo_expr, "args"):
                    if not isinstance(pyomo_expr, Var):
                        return float(pyomo_expr)
                    else:
                        referenced_vars.add(pyomo_expr)
                        return self._pyomo_var_to_solver_var_expr_map[pyomo_expr]
                scip_expr_list = [0 for i in range(pyomo_expr.nargs())]
                for i in range(pyomo_expr.nargs()):
                    scip_expr_list[i] = get_nl_expr_recursively(pyomo_expr.args[i])
                if isinstance(pyomo_expr, PowExpression):
                    if len(scip_expr_list) != 2:
                        raise ValueError(
                            f"PowExpression has {len(scip_expr_list)} many terms instead of two!"
                        )
                    return scip_expr_list[0] ** (scip_expr_list[1])
                elif isinstance(pyomo_expr, ProductExpression):
                    return self._scip.quickprod(scip_expr_list)
                elif isinstance(pyomo_expr, SumExpression):
                    return self._scip.quicksum(scip_expr_list)
                elif isinstance(pyomo_expr, DivisionExpression):
                    if len(scip_expr_list) != 2:
                        raise ValueError(
                            f"DivisionExpression has {len(scip_expr_list)} many terms instead of two!"
                        )
                    return scip_expr_list[0] / scip_expr_list[1]
                elif isinstance(pyomo_expr, UnaryFunctionExpression):
                    if len(scip_expr_list) != 1:
                        raise ValueError(
                            f"UnaryExpression has {len(scip_expr_list)} many terms instead of one!"
                        )
                    if pyomo_expr.name == "sin":
                        return self._scip.sin(scip_expr_list[0])
                    elif pyomo_expr.name == "cos":
                        return self._scip.cos(scip_expr_list[0])
                    elif pyomo_expr.name == "exp":
                        return self._scip.exp(scip_expr_list[0])
                    elif pyomo_expr.name == "log":
                        return self._scip.log(scip_expr_list[0])
                    else:
                        raise NotImplementedError(
                            f"PySCIPOpt through Pyomo does not support the unary function {pyomo_expr.name}"
                        )
                else:
                    raise NotImplementedError(
                        f"PySCIPOpt through Pyomo does not yet support expression type {type(pyomo_expr)}"
                    )

            new_expr += get_nl_expr_recursively(repn.nonlinear_expr)

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=None):
        if max_degree is None or max_degree >= 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        scip_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree)

        return scip_expr, referenced_vars

    def _scip_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._solver_model.infinity()
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._solver_model.infinity()

        return lb, ub

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._scip_vtype_from_var(var)
        lb, ub = self._scip_lb_ub_from_var(var)

        scip_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_expr_map[var] = scip_var
        self._pyomo_var_to_solver_var_map[var] = scip_var.name
        self._solver_var_to_pyomo_var_map[varname] = var
        self._referenced_variables[var] = 0

    def close(self):
        """Frees SCIP resources used by this solver instance."""

        if self._solver_model is not None:
            self._solver_model.freeProb()
            self._solver_model = None

    def __exit__(self, t, v, traceback):
        super().__exit__(t, v, traceback)
        self.close()

    def _set_instance(self, model, kwds={}):
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self.available()
        try:
            self._solver_model = self._scip.Model()
        except Exception:
            e = sys.exc_info()[1]
            msg = (
                "Unable to create SCIP model. "
                f"Have you installed PySCIPOpt correctly?\n\n\t Error message: {e}"
            )
            raise Exception(msg)

        self._add_block(model)

        for var, n_ref in self._referenced_variables.items():
            if n_ref != 0:
                if var.fixed:
                    if not self._output_fixed_variable_bounds:
                        raise ValueError(
                            f"Encountered a fixed variable {var.name} inside "
                            "an active objective or constraint "
                            f"expression on model {self._pyomo_model.name}, which is usually "
                            "indicative of a preprocessing error. Use "
                            "the IO-option 'output_fixed_variable_bounds=True' "
                            "to suppress this error and fix the variable "
                            "by overwriting its bounds in the SCIP instance."
                        )

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body) and self._skip_trivial_constraints:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            scip_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(), self._max_constraint_degree
            )
        else:
            scip_expr, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body, self._max_constraint_degree
            )

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError(f"Lower bound of constraint {con} is not constant.")
            con_lower = value(con.lower)
            if type(con_lower) != float and type(con_lower) != int:
                logger.warning(
                    f"Constraint {conname} has LHS type {type(value(con.lower))}. "
                    f"Converting to float as type is not allowed for SCIP."
                )
                con_lower = float(con_lower)
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(f"Upper bound of constraint {con} is not constant.")
            con_upper = value(con.upper)
            if type(con_upper) != float and type(con_upper) != int:
                logger.warning(
                    f"Constraint {conname} has RHS type {type(value(con.upper))}. "
                    f"Converting to float as type is not allowed for SCIP."
                )
                con_upper = float(con_upper)

        if con.equality:
            scip_cons = self._solver_model.addCons(scip_expr == con_lower, name=conname)
        elif con.has_lb() and con.has_ub():
            scip_cons = self._solver_model.addCons(con_lower <= scip_expr, name=conname)
            rhs = con_upper
            if hasattr(con.body, "constant"):
                con_constant = value(con.body.constant)
                if not isinstance(con_constant, (float, int)):
                    con_constant = float(con_constant)
                rhs -= con_constant
            self._solver_model.chgRhs(scip_cons, rhs)
        elif con.has_lb():
            scip_cons = self._solver_model.addCons(con_lower <= scip_expr, name=conname)
        elif con.has_ub():
            scip_cons = self._solver_model.addCons(scip_expr <= con_upper, name=conname)
        else:
            raise ValueError(
                f"Constraint does not have a lower or an upper bound: {con} \n"
            )

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_expr_map[con] = scip_cons
        self._pyomo_con_to_solver_con_map[con] = scip_cons.name
        self._solver_con_to_pyomo_con_map[conname] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level not in [1, 2]:
            raise ValueError(f"Solver does not support SOS level {level} constraints")

        scip_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, "get_items"):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[con].add(v)
            scip_vars.append(self._pyomo_var_to_solver_var_expr_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        if level == 1:
            scip_cons = self._solver_model.addConsSOS1(
                scip_vars, weights=weights, name=conname
            )
        else:
            scip_cons = self._solver_model.addConsSOS2(
                scip_vars, weights=weights, name=conname
            )
        self._pyomo_con_to_solver_con_expr_map[con] = scip_cons
        self._pyomo_con_to_solver_con_map[con] = scip_cons.name
        self._solver_con_to_pyomo_con_map[conname] = con

    def _scip_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate SCIP variable type

        Parameters
        ----------
        var: pyomo.core.base.var.Var
            The pyomo variable that we want to retrieve the SCIP vtype of

        Returns
        -------
        vtype: str
            B for Binary, I for Integer, or C for Continuous
        """
        if var.is_binary():
            vtype = "B"
        elif var.is_integer():
            vtype = "I"
        elif var.is_continuous():
            vtype = "C"
        else:
            raise ValueError(f"Variable domain type is not recognized for {var.domain}")
        return vtype

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError("Cannot add inactive objective to solver.")

        if obj.sense == minimize:
            sense = "minimize"
        elif obj.sense == maximize:
            sense = "maximize"
        else:
            raise ValueError(f"Objective sense is not recognized: {obj.sense}")

        scip_expr, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree
        )

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(scip_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _get_solver_solution_status(self, scip, soln):
        """ """
        # Get the status of the SCIP Model currently
        status = scip.getStatus()

        # Go through each potential case and update appropriately
        if scip.getStage() == 1:  # SCIP Model is created but not yet optimized
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Model is loaded, but no solution information is available."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
        elif status == "optimal":  # optimal
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = (
                "Model was solved to optimality (subject to tolerances), "
                "and an optimal solution is available."
            )
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == "infeasible":
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be infeasible"
            )
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == "inforunbd":
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Problem proven to be infeasible or unbounded."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.infeasibleOrUnbounded
            )
            soln.status = SolutionStatus.unsure
        elif status == "unbounded":
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be unbounded."
            )
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == "gaplimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the gap dropped below "
                "the value specified in the "
                "limits/gap parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "stallnodelimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the stalling node limit "
                "exceeded the value specified in the "
                "limits/stallnodes parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "restartlimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the total number of restarts "
                "exceeded the value specified in the "
                "limits/restarts parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "nodelimit" or status == "totalnodelimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the number of "
                "branch-and-cut nodes explored exceeded the limits specified "
                "in the limits/nodes or limits/totalnodes parameter"
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxEvaluations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "timelimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the time expended exceeded "
                "the value specified in the limits/time parameter."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxTimeLimit
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "sollimit" or status == "bestsollimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the number of solutions found "
                "reached the value specified in the limits/solutions or"
                "limits/bestsol parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "memlimit":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the memory used exceeded "
                "the value specified in the limits/memory parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == "userinterrupt":
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization was terminated by the user."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                f"Unhandled SCIP status ({str(status)})"
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        return soln

    def _postsolve(self):
        # Constraint duals and variable
        # reduced-costs were removed as in SCIP they contain
        # too many caveats. Slacks were removed as later
        # planned interfaces do not intend to support.
        # Scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        for suffix in self._suffixes:
            raise RuntimeError(
                f"***The scip_direct solver plugin cannot extract solution suffix={suffix}"
            )

        scip = self._solver_model
        status = scip.getStatus()
        scip_vars = scip.getVars()
        n_bin_vars = sum([scip_var.vtype() == "BINARY" for scip_var in scip_vars])
        n_int_vars = sum([scip_var.vtype() == "INTEGER" for scip_var in scip_vars])
        n_con_vars = sum([scip_var.vtype() == "CONTINUOUS" for scip_var in scip_vars])

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = f"SCIP{self._version}"
        self.results.solver.wallclock_time = scip.getSolvingTime()

        soln = self._get_solver_solution_status(scip, soln)

        self.results.problem.name = scip.getProbName()

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if scip.getNSols() > 0:
            scip_has_sol = True
        else:
            scip_has_sol = False
        if not scip_has_sol and (status == "inforunbd" or status == "infeasible"):
            pass
        else:
            if n_bin_vars + n_int_vars == 0:
                self.results.problem.upper_bound = scip.getObjVal()
                self.results.problem.lower_bound = scip.getObjVal()
            elif scip.getObjectiveSense() == "minimize":  # minimizing
                if scip_has_sol:
                    self.results.problem.upper_bound = scip.getObjVal()
                else:
                    self.results.problem.upper_bound = scip.infinity()
                self.results.problem.lower_bound = scip.getDualbound()
            else:  # maximizing
                self.results.problem.upper_bound = scip.getDualbound()
                if scip_has_sol:
                    self.results.problem.lower_bound = scip.getObjVal()
                else:
                    self.results.problem.lower_bound = -scip.infinity()

            try:
                soln.gap = (
                    self.results.problem.upper_bound - self.results.problem.lower_bound
                )
            except TypeError:
                soln.gap = None

        self.results.problem.number_of_constraints = scip.getNConss(transformed=False)
        # self.results.problem.number_of_nonzeros = None
        self.results.problem.number_of_variables = scip.getNVars(transformed=False)
        self.results.problem.number_of_binary_variables = n_bin_vars
        self.results.problem.number_of_integer_variables = n_int_vars
        self.results.problem.number_of_continuous_variables = n_con_vars
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = scip.getNSols()

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatibility. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """

            if scip.getNSols() > 0:
                soln_variables = soln.variable

                scip_vars = scip.getVars()
                scip_var_names = [scip_var.name for scip_var in scip_vars]
                var_names = set(self._solver_var_to_pyomo_var_map.keys())
                assert set(scip_var_names) == var_names
                var_vals = [scip.getVal(scip_var) for scip_var in scip_vars]

                for scip_var, val, name in zip(scip_vars, var_vals, scip_var_names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[name]
                    if self._referenced_variables[pyomo_var] > 0:
                        soln_variables[name] = {"Value": val}

        elif self._load_solutions:
            if scip.getNSols() > 0:
                self.load_vars()

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        TempfileManager.pop(remove=not self._keepfiles)

        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        partial_sol = False
        for pyomo_var in self._pyomo_var_to_solver_var_expr_map:
            if pyomo_var.value is None:
                partial_sol = True
                break
        if partial_sol:
            scip_sol = self._solver_model.createPartialSol()
        else:
            scip_sol = self._solver_model.createSol()
        for pyomo_var, scip_var in self._pyomo_var_to_solver_var_expr_map.items():
            if pyomo_var.value is not None:
                scip_sol[scip_var] = value(pyomo_var)
        if partial_sol:
            self._solver_model.addSol(scip_sol)
        else:
            feasible = self._solver_model.checkSol(scip_sol, printreason=not self._tee)
            if feasible:
                self._solver_model.addSol(scip_sol)
            else:
                logger.warning("Warm start solution was not accepted by SCIP")
                self._solver_model.freeSol(scip_sol)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_expr_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        scip_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = [self._solver_model.getVal(scip_var) for scip_var in scip_vars_to_load]

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.set_value(val, skip_validation=True)

    def _load_rc(self, vars_to_load=None):
        raise NotImplementedError(
            "SCIP via Pyomo does not support reduced cost loading."
        )

    def _load_duals(self, cons_to_load=None):
        raise NotImplementedError(
            "SCIP via Pyomo does not support dual solution loading"
        )

    def _load_slacks(self, cons_to_load=None):
        raise NotImplementedError("SCIP via Pyomo does not support slack loading")

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the 'dual' suffix. The 'dual' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_duals(cons_to_load)

    def load_rc(self, vars_to_load):
        """
        Load the reduced costs into the 'rc' suffix. The 'rc' suffix must live on the parent model.

        Parameters
        ----------
        vars_to_load: list of Var
        """
        self._load_rc(vars_to_load)

    def load_slacks(self, cons_to_load=None):
        """
        Load the values of the slack variables into the 'slack' suffix. The 'slack' suffix must live on the parent
        model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_slacks(cons_to_load)
