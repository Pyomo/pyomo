#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import re
import sys
import pyomo.common
from pyutilib.misc import Bunch
from pyutilib.services import TempfileManager
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.kernel.component_map import ComponentMap
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
import time


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


class _CplexExpr(object):
    def __init__(self):
        self.variables = []
        self.coefficients = []
        self.offset = 0
        self.q_variables1 = []
        self.q_variables2 = []
        self.q_coefficients = []

def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


@SolverFactory.register('cplex_direct', doc='Direct python interface to CPLEX')
class CPLEXDirect(DirectSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'cplexdirect'
        DirectSolver.__init__(self, **kwds)
        self._init()
        self._wallclock_time = None
        self._pyomo_var_to_ndx_map = ComponentMap()
        self._ndx_count = 0

    def _init(self):
        try:
            import cplex
            self._cplex = cplex
            self._python_api_exists = True
            self._version = tuple(
                int(k) for k in self._cplex.Cplex().get_version().split('.'))
            while len(self._version) < 4:
                self._version += (0,)
            self._version = tuple(int(i) for i in self._version[:4])
            self._version_major = self._version[0]
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            # other forms of exceptions can be thrown by the gurobi python
            # import. for example, a gurobipy.GurobiError exception is thrown
            # if all tokens for Gurobi are already in use. assuming, of
            # course, the license is a token license. unfortunately, you can't
            # import without a license, which means we can't test for the
            # exception above!
            print("Import of cplex failed - cplex message=" + str(e) + "\n")
            self._python_api_exists = False

        self._range_constraints = set()

        self._max_constraint_degree = 2
        self._max_obj_degree = 2

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(descend_into=True, active=True):
                for var in block.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=False, active=True, sort=False):
                    var.stale = True
        if self._tee:
            def _process_stream(arg):
                sys.stdout.write(arg)
                return arg
            self._solver_model.set_results_stream(self._log_file, _process_stream)
        else:
            self._solver_model.set_results_stream(self._log_file)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        obj_degree = self._objective.expr.polynomial_degree()
        if obj_degree is None or obj_degree > 2:
            raise DegreeError('CPLEXDirect does not support expressions of degree {0}.'\
                              .format(obj_degree))
        elif obj_degree == 2:
            quadratic_objective = True
        else:
            quadratic_objective = False

        num_integer_vars = self._solver_model.variables.get_num_integer()
        num_binary_vars = self._solver_model.variables.get_num_binary()
        num_sos = self._solver_model.SOS.get_num()

        if self._solver_model.quadratic_constraints.get_num() != 0:
            quadratic_cons = True
        else:
            quadratic_cons = False

        if (num_integer_vars + num_binary_vars + num_sos) > 0:
            integer = True
        else:
            integer = False

        if integer:
            if quadratic_cons:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MIQCP)
            elif quadratic_objective:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MIQP)
            else:
                self._solver_model.set_problem_type(self._solver_model.problem_type.MILP)
        else:
            if quadratic_cons:
                self._solver_model.set_problem_type(self._solver_model.problem_type.QCP)
            elif quadratic_objective:
                self._solver_model.set_problem_type(self._solver_model.problem_type.QP)
            else:
                self._solver_model.set_problem_type(self._solver_model.problem_type.LP)

        for key, option in self.options.items():
            opt_cmd = self._solver_model.parameters
            key_pieces = key.split('_')
            for key_piece in key_pieces:
                opt_cmd = getattr(opt_cmd, key_piece)
            # When options come from the pyomo command, all
            # values are string types, so we try to cast
            # them to a numeric value in the event that
            # setting the parameter fails.
            try:
                opt_cmd.set(option)
            except self._cplex.exceptions.CplexError:
                # we place the exception handling for
                # checking the cast of option to a float in
                # another function so that we can simply
                # call raise here instead of except
                # TypeError as e / raise e, because the
                # latter does not preserve the Cplex stack
                # trace
                if not _is_numeric(option):
                    raise
                opt_cmd.set(float(option))

        t0 = time.time()
        self._solver_model.solve()
        t1 = time.time()
        self._wallclock_time = t1 - t0

        # FIXME: can we get a return code indicating if CPLEX had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError('CPLEXDirect does not support expressions of degree {0}.'.format(degree))

        new_expr = _CplexExpr()
        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr.variables.extend(self._pyomo_var_to_ndx_map[i] for i in repn.linear_vars)
            new_expr.coefficients.extend(repn.linear_coefs)

        for i, v in enumerate(repn.quadratic_vars):
            x, y = v
            new_expr.q_coefficients.append(repn.quadratic_coefs[i])
            new_expr.q_variables1.append(self._pyomo_var_to_ndx_map[x])
            new_expr.q_variables2.append(self._pyomo_var_to_ndx_map[y])
            referenced_vars.add(x)
            referenced_vars.add(y)

        new_expr.offset = repn.constant

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return cplex_expr, referenced_vars

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._cplex_vtype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._cplex.infinity
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._cplex.infinity

        self._solver_model.variables.add(lb=[lb], ub=[ub], types=[vtype], names=[varname])

        self._pyomo_var_to_solver_var_map[var] = varname
        self._solver_var_to_pyomo_var_map[varname] = var
        self._pyomo_var_to_ndx_map[var] = self._ndx_count
        self._ndx_count += 1
        self._referenced_variables[var] = 0

        if var.is_fixed():
            self._solver_model.variables.set_lower_bounds(varname, var.value)
            self._solver_model.variables.set_upper_bounds(varname, var.value)

    def _set_instance(self, model, kwds={}):
        self._pyomo_var_to_ndx_map = ComponentMap()
        self._ndx_count = 0
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        try:
            self._solver_model = self._cplex.Cplex()
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create CPLEX model. "
                   "Have you installed the Python "
                   "bindings for CPLEX?\n\n\t"+
                   "Error message: {0}".format(e))
            raise Exception(msg)

        self._add_block(model)

        for var, n_ref in self._referenced_variables.items():
            if n_ref != 0:
                if var.fixed:
                    if not self._output_fixed_variable_bounds:
                        raise ValueError(
                            "Encountered a fixed variable (%s) inside "
                            "an active objective or constraint "
                            "expression on model %s, which is usually "
                            "indicative of a preprocessing error. Use "
                            "the IO-option 'output_fixed_variable_bounds=True' "
                            "to suppress this error and fix the variable "
                            "by overwriting its bounds in the CPLEX instance."
                            % (var.name, self._pyomo_model.name,))

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),
                self._max_constraint_degree)
        else:
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body,
                self._max_constraint_degree)

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError("Lower bound of constraint {0} "
                                 "is not constant.".format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError("Upper bound of constraint {0} "
                                 "is not constant.".format(con))

        if con.equality:
            my_sense = 'E'
            my_rhs = [value(con.lower) - cplex_expr.offset]
            my_range = []
        elif con.has_lb() and con.has_ub():
            my_sense = 'R'
            lb = value(con.lower)
            ub = value(con.upper)
            my_rhs = [ub - cplex_expr.offset]
            my_range = [lb - ub]
            self._range_constraints.add(con)
        elif con.has_lb():
            my_sense = 'G'
            my_rhs = [value(con.lower) - cplex_expr.offset]
            my_range = []
        elif con.has_ub():
            my_sense = 'L'
            my_rhs = [value(con.upper) - cplex_expr.offset]
            my_range = []
        else:
            raise ValueError("Constraint does not have a lower "
                             "or an upper bound: {0} \n".format(con))

        if len(cplex_expr.q_coefficients) == 0:
            self._solver_model.linear_constraints.add(
                lin_expr=[[cplex_expr.variables,
                           cplex_expr.coefficients]],
                senses=my_sense,
                rhs=my_rhs,
                range_values=my_range,
                names=[conname])
        else:
            if my_sense == 'R':
                raise ValueError("The CPLEXDirect interface does not "
                                 "support quadratic range constraints: "
                                 "{0}".format(con))
            self._solver_model.quadratic_constraints.add(
                lin_expr=[cplex_expr.variables,
                          cplex_expr.coefficients],
                quad_expr=[cplex_expr.q_variables1,
                           cplex_expr.q_variables2,
                           cplex_expr.q_coefficients],
                sense=my_sense,
                rhs=my_rhs[0],
                name=conname)

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = conname
        self._solver_con_to_pyomo_con_map[conname] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level == 1:
            sos_type = self._solver_model.SOS.type.SOS1
        elif level == 2:
            sos_type = self._solver_model.SOS.type.SOS2
        else:
            raise ValueError("Solver does not support SOS "
                             "level {0} constraints".format(level))

        cplex_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, 'get_items'):
            # aml sos constraint
            sos_items = list(con.get_items())
        else:
            # kernel sos constraint
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[con].add(v)
            cplex_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        self._solver_model.SOS.add(type=sos_type, SOS=[cplex_vars, weights], name=conname)
        self._pyomo_con_to_solver_con_map[con] = conname
        self._solver_con_to_pyomo_con_map[conname] = con

    def _cplex_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate gurobi variable type
        :param var: pyomo.core.base.var.Var
        :return: gurobipy.GRB.CONTINUOUS or gurobipy.GRB.BINARY or gurobipy.GRB.INTEGER
        """
        if var.is_binary():
            vtype = self._solver_model.variables.type.binary
        elif var.is_integer():
            vtype = self._solver_model.variables.type.integer
        elif var.is_continuous():
            vtype = self._solver_model.variables.type.continuous
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vtype

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        self._solver_model.objective.set_linear([(i, 0.0) for i in range(len(self._pyomo_var_to_solver_var_map.values()))])
        self._solver_model.objective.set_quadratic([[[0], [0]] for i in self._pyomo_var_to_solver_var_map.keys()])

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._solver_model.objective.sense.minimize
        elif obj.sense == maximize:
            sense = self._solver_model.objective.sense.maximize
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        cplex_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)
        for i in range(len(cplex_expr.q_coefficients)):
            cplex_expr.q_coefficients[i] *= 2

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.objective.set_sense(sense)
        if hasattr(self._solver_model.objective, 'set_offset'):
            self._solver_model.objective.set_offset(cplex_expr.offset)
        if len(cplex_expr.coefficients) != 0:
            self._solver_model.objective.set_linear(list(zip(cplex_expr.variables, cplex_expr.coefficients)))
        if len(cplex_expr.q_coefficients) != 0:
            self._solver_model.objective.set_quadratic_coefficients(list(zip(cplex_expr.q_variables1,
                                                                             cplex_expr.q_variables2,
                                                                             cplex_expr.q_coefficients)))
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from CPLEX are
        # constraint duals, constraint slacks, and variable
        # reduced-costs. scan through the solver suffix list
        # and throw an exception if the user has specified
        # any others.
        extract_duals = False
        extract_slacks = False
        extract_reduced_costs = False
        for suffix in self._suffixes:
            flag = False
            if re.match(suffix, "dual"):
                extract_duals = True
                flag = True
            if re.match(suffix, "slack"):
                extract_slacks = True
                flag = True
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError("***The cplex_direct solver plugin cannot extract solution suffix="+suffix)

        cpxprob = self._solver_model
        status = cpxprob.solution.get_status()

        if cpxprob.get_problem_type() in [cpxprob.problem_type.MILP,
                                          cpxprob.problem_type.MIQP,
                                          cpxprob.problem_type.MIQCP]:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = ("CPLEX {0}".format(cpxprob.get_version()))
        self.results.solver.wallclock_time = self._wallclock_time

        if status in [1, 101, 102]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status in [2, 40, 118, 133, 134]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status in [4, 119, 134]:
            # Note: status of 4 means infeasible or unbounded
            #       and 119 means MIP infeasible or unbounded
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = \
                TerminationCondition.infeasibleOrUnbounded
            soln.status = SolutionStatus.unsure
        elif status in [3, 103]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status in [10]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit
        elif status in [11, 25, 107, 131]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        if cpxprob.objective.get_sense() == cpxprob.objective.sense.minimize:
            self.results.problem.sense = minimize
        elif cpxprob.objective.get_sense() == cpxprob.objective.sense.maximize:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError('Unrecognized cplex objective sense: {0}'.\
                               format(cpxprob.objective.get_sense()))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if (cpxprob.variables.get_num_binary() + cpxprob.variables.get_num_integer()) == 0:
            try:
                self.results.problem.upper_bound = cpxprob.solution.get_objective_value()
                self.results.problem.lower_bound = cpxprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
        elif cpxprob.objective.get_sense() == cpxprob.objective.sense.minimize:
            try:
                self.results.problem.upper_bound = cpxprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
            try:
                self.results.problem.lower_bound = cpxprob.solution.MIP.get_best_objective()
            except self._cplex.exceptions.CplexError:
                pass
        elif cpxprob.objective.get_sense() == cpxprob.objective.sense.maximize:
            try:
                self.results.problem.upper_bound = cpxprob.solution.MIP.get_best_objective()
            except self._cplex.exceptions.CplexError:
                pass
            try:
                self.results.problem.lower_bound = cpxprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
        else:
            raise RuntimeError('Unrecognized cplex objective sense: {0}'.\
                               format(cpxprob.objective.get_sense()))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            soln.gap = None

        self.results.problem.name = cpxprob.get_problem_name()
        assert cpxprob.indicator_constraints.get_num() == 0
        self.results.problem.number_of_constraints = \
            (cpxprob.linear_constraints.get_num() +
             cpxprob.quadratic_constraints.get_num() +
             cpxprob.SOS.get_num())
        self.results.problem.number_of_nonzeros = None
        self.results.problem.number_of_variables = cpxprob.variables.get_num()
        self.results.problem.number_of_binary_variables = cpxprob.variables.get_num_binary()
        self.results.problem.number_of_integer_variables = cpxprob.variables.get_num_integer()
        assert cpxprob.variables.get_num_semiinteger() == 0
        assert cpxprob.variables.get_num_semicontinuous() == 0
        self.results.problem.number_of_continuous_variables = \
            (cpxprob.variables.get_num() -
             cpxprob.variables.get_num_binary() -
             cpxprob.variables.get_num_integer())
        self.results.problem.number_of_objectives = 1

        # only try to get objective and variable values if a solution exists
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatability. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if cpxprob.solution.get_solution_type() > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                var_names = self._solver_model.variables.get_names()
                var_names = list(set(var_names).intersection(set(self._pyomo_var_to_solver_var_map.values())))
                var_vals = self._solver_model.solution.get_values(var_names)
                for i, name in enumerate(var_names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[name]
                    if self._referenced_variables[pyomo_var] > 0:
                        pyomo_var.stale = False
                        soln_variables[name] = {"Value":var_vals[i]}

                if extract_reduced_costs:
                    reduced_costs = self._solver_model.solution.get_reduced_costs(var_names)
                    for i, name in enumerate(var_names):
                        pyomo_var = self._solver_var_to_pyomo_var_map[name]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[name]["Rc"] = reduced_costs[i]

                if extract_slacks:
                    for con_name in self._solver_model.linear_constraints.get_names():
                        soln_constraints[con_name] = {}
                    for con_name in self._solver_model.quadratic_constraints.get_names():
                        soln_constraints[con_name] = {}
                elif extract_duals:
                    # CPLEX PYTHON API DOES NOT SUPPORT QUADRATIC DUAL COLLECTION
                    for con_name in self._solver_model.linear_constraints.get_names():
                        soln_constraints[con_name] = {}

                if extract_duals:
                    dual_values = self._solver_model.solution.get_dual_values()
                    for i, con_name in enumerate(self._solver_model.linear_constraints.get_names()):
                        soln_constraints[con_name]["Dual"] = dual_values[i]

                if extract_slacks:
                    linear_slacks = self._solver_model.solution.get_linear_slacks()
                    qudratic_slacks = self._solver_model.solution.get_quadratic_slacks()
                    for i, con_name in enumerate(self._solver_model.linear_constraints.get_names()):
                        pyomo_con = self._solver_con_to_pyomo_con_map[con_name]
                        if pyomo_con in self._range_constraints:
                            R_ = self._solver_model.linear_constraints.get_range_values(con_name)
                            if R_ == 0:
                                soln_constraints[con_name]["Slack"] = linear_slacks[i]
                            else:
                                Ls_ = linear_slacks[i]
                                Us_ = R_ - Ls_
                                if abs(Us_) > abs(Ls_):
                                    soln_constraints[con_name]["Slack"] = Us_
                                else:
                                    soln_constraints[con_name]["Slack"] = -Ls_
                        else:
                            soln_constraints[con_name]["Slack"] = linear_slacks[i]
                    for i, con_name in enumerate(self._solver_model.quadratic_constraints.get_names()):
                        soln_constraints[con_name]["Slack"] = qudratic_slacks[i]
        elif self._load_solutions:
            if cpxprob.solution.get_solution_type() > 0:
                self._load_vars()

                if extract_reduced_costs:
                    self._load_rc()

                if extract_duals:
                    self._load_duals()

                if extract_slacks:
                    self._load_slacks()

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        TempfileManager.pop(remove=not self._keepfiles)

        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        # here warm start means MIP start, which we can not add
        # if the problem type is not discrete
        cpxprob = self._solver_model
        if cpxprob.get_problem_type() in [cpxprob.problem_type.MILP,
                                          cpxprob.problem_type.MIQP,
                                          cpxprob.problem_type.MIQCP]:
            var_names = []
            var_values = []
            for pyomo_var, cplex_var in self._pyomo_var_to_solver_var_map.items():
                if pyomo_var.value is not None:
                    var_names.append(cplex_var)
                    var_values.append(value(pyomo_var))

            if len(var_names):
                self._solver_model.MIP_starts.add(
                    [var_names, var_values],
                    self._solver_model.MIP_starts.effort_level.auto)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        cplex_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.solution.get_values(cplex_vars_to_load)

        for i, pyomo_var in enumerate(vars_to_load):
            if ref_vars[pyomo_var] > 0:
                pyomo_var.stale = False
                pyomo_var.value = vals[i]

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        cplex_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.solution.get_reduced_costs(cplex_vars_to_load)

        for i, pyomo_var in enumerate(vars_to_load):
            if ref_vars[pyomo_var] > 0:
                rc[pyomo_var] = vals[i]

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.linear_constraints.get_names()
            vals = self._solver_model.solution.get_dual_values()
        else:
            cplex_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = cplex_cons_to_load.intersection(set(self._solver_model.linear_constraints.get_names()))
            vals = self._solver_model.solution.get_dual_values(linear_cons_to_load)

        for i, cplex_con in enumerate(linear_cons_to_load):
            pyomo_con = reverse_con_map[cplex_con]
            dual[pyomo_con] = vals[i]

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = self._pyomo_model.slack

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.linear_constraints.get_names()
            linear_vals = self._solver_model.solution.get_linear_slacks()
            quadratic_cons_to_load = self._solver_model.quadratic_constraints.get_names()
            quadratic_vals = self._solver_model.solution.get_quadratic_slacks()
        else:
            cplex_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = cplex_cons_to_load.intersection(set(self._solver_model.linear_constraints.get_names()))
            linear_vals = self._solver_model.solution.get_linear_slacks(linear_cons_to_load)
            quadratic_cons_to_load = cplex_cons_to_load.intersection(set(self._solver_model.quadratic_constraints.get_names()))
            quadratic_vals = self._solver_model.solution.get_quadratic_slacks(quadratic_cons_to_load)

        for i, cplex_con in enumerate(linear_cons_to_load):
            pyomo_con = reverse_con_map[cplex_con]
            if pyomo_con in self._range_constraints:
                R_ = self._solver_model.linear_constraints.get_range_values(cplex_con)
                if R_ == 0:
                    slack[pyomo_con] = linear_vals[i]
                else:
                    Ls_ = linear_vals[i]
                    Us_ = R_ - Ls_
                    if abs(Us_) > abs(Ls_):
                        slack[pyomo_con] = Us_
                    else:
                        slack[pyomo_con] = -Ls_
            else:
                slack[pyomo_con] = linear_vals[i]

        for i, cplex_con in enumerate(quadratic_cons_to_load):
            pyomo_con = reverse_con_map[cplex_con]
            slack[pyomo_con] = quadratic_vals[i]

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
