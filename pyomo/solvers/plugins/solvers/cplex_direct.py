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
import pyutilib.services
from pyutilib.misc import Bunch
from pyomo.util.plugin import alias
from pyomo.core.kernel.numvalue import is_fixed
from pyomo.repn import generate_canonical_repn, LinearCanonicalRepn, canonical_degree
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver
from pyomo.core.kernel.numvalue import value
import pyomo.core.kernel
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
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


class CPLEXDirect(DirectSolver):
    alias('cplex_direct', doc='Direct python interface to CPLEX')

    def __init__(self, **kwds):
        kwds['type'] = 'cplexdirect'
        DirectSolver.__init__(self, **kwds)
        self._init()
        self._wallclock_time = None

    def _init(self):
        try:
            import cplex
            self._cplex = cplex
            self._python_api_exists = True
            self._version = tuple(self._cplex.Cplex().get_version().split('.'))
            while len(self._version) < 4:
                self._version += (0,)
            self._version = self._version[:4]
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
        for var in self._pyomo_model.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=True,
                                                            active=None, sort=False):
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

        if len(self._solver_model.objective.get_quadratic()) != 0:
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
            opt_cmd.set(option)

        t0 = time.time()
        self._solver_model.solve()
        t1 = time.time()
        self._wallclock_time = t1 - t0

        # FIXME: can we get a return code indicating if CPLEX had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = canonical_degree(repn)
        if (degree is None) or (degree > max_degree):
            raise DegreeError('CPLEXDirect does not support expressions of degree {0}.'.format(degree))

        if isinstance(repn, LinearCanonicalRepn):
            new_expr = _CplexExpr()

            if repn.constant is not None:
                new_expr.offset = repn.constant

            if (repn.linear is not None) and (len(repn.linear) > 0):
                list(map(referenced_vars.add, repn.variables))
                new_expr.variables.extend(self._pyomo_var_to_solver_var_map[var] for var in repn.variables)
                new_expr.coefficients.extend(coeff for coeff in repn.linear)

        else:
            new_expr = _CplexExpr()
            if 0 in repn:
                new_expr.offset = repn[0][None]

            if 1 in repn:
                for ndx, coeff in repn[1].items():
                    new_expr.coefficients.append(coeff)
                    var = repn[-1][ndx]
                    new_expr.variables.append(self._pyomo_var_to_solver_var_map[var])
                    referenced_vars.add(var)

            if 2 in repn:
                for key, coeff in repn[2].items():
                    new_expr.q_coefficients.append(coeff)
                    indices = list(key.keys())
                    if len(indices) == 1:
                        ndx = indices[0]
                        var = repn[-1][ndx]
                        referenced_vars.add(var)
                        cplex_var = self._pyomo_var_to_solver_var_map[var]
                        new_expr.q_variables1.append(cplex_var)
                        new_expr.q_variables2.append(cplex_var)
                    else:
                        ndx = indices[0]
                        var = repn[-1][ndx]
                        referenced_vars.add(var)
                        cplex_var = self._pyomo_var_to_solver_var_map[var]
                        new_expr.q_variables1.append(cplex_var)
                        ndx = indices[1]
                        var = repn[-1][ndx]
                        referenced_vars.add(var)
                        cplex_var = self._pyomo_var_to_solver_var_map[var]
                        new_expr.q_variables2.append(cplex_var)

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        repn = generate_canonical_repn(expr)

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
        lb = value(var.lb)
        ub = value(var.ub)
        if lb is None:
            lb = -self._cplex.infinity
        if ub is None:
            ub = self._cplex.infinity

        self._solver_model.variables.add(lb=[lb], ub=[ub], types=[vtype], names=[varname])

        self._pyomo_var_to_solver_var_map[var] = varname
        self._referenced_variables[var] = 0

        if var.is_fixed():
            self._solver_model.variables.set_lower_bounds(varname, var.value)
            self._solver_model.variables.set_upper_bounds(varname, var.value)

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        try:
            self._solver_model = self._cplex.Cplex()
        except Exception:
            e = sys.exc_info()[1]
            msg = ('Unable to create CPLEX model. Have you installed the Python bindings for CPLEX?\n\n\t' +
                   'Error message: {0}'.format(e))
            raise Exception(msg)

        self._add_block(model)

        for var, n_ref in self._referenced_variables.items():
            if n_ref != 0:
                if var.fixed:
                    if not self._output_fixed_variable_bounds:
                        raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                         "or constraint expression on model %s, which is usually indicative of "
                                         "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                         "to suppress this error and fix the variable by overwriting its bounds in "
                                         "the Gurobi instance."
                                         % (var.name, self._pyomo_model.name,))

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_repn(con.canonical_form(),
                                                                         self._max_constraint_degree)
        elif isinstance(con, LinearCanonicalRepn):
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_repn(con, self._max_constraint_degree)
        else:
            cplex_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            my_sense = 'E'
            my_rhs = [value(con.lower) - cplex_expr.offset]
            my_range = []
        elif con.has_lb() and (value(con.lower) > -float('inf')) and con.has_ub() and (value(con.upper) < float('inf')):
            my_sense = 'R'
            lb = value(con.lower)
            ub = value(con.upper)
            my_rhs = [ub - cplex_expr.offset]
            my_range = [lb - ub]
            self._range_constraints.add(con)
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            my_sense = 'G'
            my_rhs = [value(con.lower) - cplex_expr.offset]
            my_range = []
        elif con.has_ub() and (value(con.upper) < float('inf')):
            my_sense = 'L'
            my_rhs = [value(con.upper) - cplex_expr.offset]
            my_range = []
        else:
            raise ValueError('Constraint does not have a lower or an upper bound: {0} \n'.format(con))

        if len(cplex_expr.q_coefficients) == 0:
            self._solver_model.linear_constraints.add(lin_expr=[[cplex_expr.variables, cplex_expr.coefficients]],
                                                      senses=my_sense, rhs=my_rhs, range_values=my_range,
                                                      names=[conname])
        else:
            if my_sense == 'R':
                raise ValueError('The CPLEXDirect interface does not support quadratic ' +
                                 'range constraints: {0}'.format(con))
            self._solver_model.quadratic_constraints.add(lin_expr=[cplex_expr.variables, cplex_expr.coefficients],
                                                         quad_expr=[cplex_expr.q_variables1,
                                                                    cplex_expr.q_variables2,
                                                                    cplex_expr.q_coefficients],
                                                         sense=my_sense, rhs=my_rhs, name=conname)

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = conname

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
            raise ValueError('Solver does not support SOS level {0} constraints'.format(level))

        cplex_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        for v, w in con.get_items():
            self._vars_referenced_by_con[con].add(v)
            cplex_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        self._solver_model.SOS.add(type=sos_type, SOS=[cplex_vars, weights], name=conname)
        self._pyomo_con_to_solver_con_map[con] = conname

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

    def _add_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        self._solver_model.objective.set_linear([(var, 0.0) for var in self._pyomo_var_to_solver_var_map.values()])
        self._solver_model.objective.set_quadratic([[[0], [0]] for i in self._pyomo_var_to_solver_var_map.keys()])

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == pyomo.core.kernel.minimize:
            sense = self._solver_model.objective.sense.minimize
        elif obj.sense == pyomo.core.kernel.maximize:
            sense = self._solver_model.objective.sense.maximize
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        cplex_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)
        for i in range(len(cplex_expr.q_coefficients)):
            cplex_expr.q_coefficients[i] *= 2

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.objective.set_sense(sense)
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
                if len(self._range_constraints) != 0:
                    err_msg = ('CPLEXDirect does not support range constraints and slack suffixes. \nIf you want ' +
                               'slack information, please split up the following constraints:\n')
                    for con in self._range_constraints:
                        err_msg += '{0}\n'.format(con)
                    raise ValueError(err_msg)
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError("***The cplex_direct solver plugin cannot extract solution suffix="+suffix)

        gprob = self._solver_model
        status = gprob.solution.get_status()

        if gprob.get_problem_type() in [gprob.problem_type.MILP, gprob.problem_type.MIQP, gprob.problem_type.MIQCP]:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = ("CPLEX {0}".format(gprob.get_version()))
        self.results.solver.wallclock_time = self._wallclock_time

        if status in [1, 101, 102]:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif soln_status in [2, 118]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif soln_status in [4, 119]:
            # Note: soln_status of 4 means infeasible or unbounded
            #       and 119 means MIP infeasible or unbounded
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
            soln.status = SolutionStatus.unsure
        elif soln_status in [3, 103]:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif soln_status in [10]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit
        elif soln_status in [11, 25]:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        if gprob.objective.get_sense() == gprob.objective.sense.minimize:
            self.results.problem.sense = pyomo.core.kernel.minimize
        elif gprob.objective.get_sense() == gprob.objective.sense.maximize:
            self.results.problem.sense = pyomo.core.kernel.maximize
        else:
            raise RuntimeError('Unrecognized cplex objective sense: {0}'.format(gprob.objective.get_sense()))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if (gprob.variables.get_num_binary() + gprob.variables.get_num_integer()) == 0:
            try:
                self.results.problem.upper_bound = gprob.solution.get_objective_value()
                self.results.problem.lower_bound = gprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
        elif gprob.objective.get_sense() == gprob.objective.sense.minimize:
            try:
                self.results.problem.upper_bound = gprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
            try:
                self.results.problem.lower_bound = gprob.solution.MIP.get_best_objective()
            except self._cplex.exceptions.CplexError:
                pass
        elif gprob.objective.get_sense() == gprob.objective.sense.maximize:
            try:
                self.results.problem.upper_bound = gprob.solution.MIP.get_best_objective()
            except self._cplex.exceptions.CplexError:
                pass
            try:
                self.results.problem.lower_bound = gprob.solution.get_objective_value()
            except self._cplex.exceptions.CplexError:
                pass
        else:
            raise RuntimeError('Unrecognized cplex objective sense: {0}'.format(gprob.objective.get_sense()))

        try:
            self.results.problem.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            self.results.problem.gap = None

        self.results.problem.name = gprob.get_problem_name()
        stats = gprob.get_stats()
        assert gprob.indicator_constraints.get_num() == 0
        self.results.problem.number_of_constraints = (gprob.linear_constraints.get_num() +
                                                      gprob.quadratic_constraints.get_num() +
                                                      gprob.SOS.get_num())
        self.results.problem.number_of_nonzeros = None
        self.results.problem.num_linear_nz = stats.num_linear_nz
        self.results.problem.num_quadratic_linear_nz = stats.num_quadratic_linear_nz
        self.results.problem.num_quadratic_nz = stats.num_quadratic_nz
        self.results.problem.num_indicator_nz = stats.num_indicator_nz
        self.results.problem.num_indicator_rhs_nz = stats.num_indicator_rhs_nz
        self.results.problem.num_lazy_nnz = stats.num_lazy_nnz
        self.results.problem.num_lazy_rhs_nnz = stats.num_lazy_rhs_nnz
        self.results.problem.num_linear_objective_nz = stats.num_linear_objective_nz
        self.results.problem.num_quadratic_objective_nz = stats.num_quadratic_objective_nz
        self.results.problem.num_linear_rhs_nz = stats.num_linear_rhs_nz
        self.results.problem.num_quadratic_rhs_nz = stats.num_quadratic_rhs_nz
        self.results.problem.number_of_variables = gprob.variables.get_num()
        self.results.problem.number_of_binary_variables = gprob.variables.get_num_binary()
        self.results.problem.number_of_integer_variables = gprob.variables.get_num_integer()
        assert gprob.variables.get_num_semiinteger() == 0
        assert gprob.variables.get_num_semicontinuous() == 0
        self.results.problem.number_of_continuous_variables = (gprob.variables.get_num() -
                                                               gprob.variables.get_num_binary() -
                                                               gprob.variables.get_num_integer())
        self.results.problem.number_of_objectives = 1

        # only try to get objective and variable values if a solution exists
        if self._load_solutions:
            if gprob.solution.get_solution_type() > 0:
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
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        var_names = []
        var_values = []
        for pyomo_var, cplex_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                var_names.append(cplex_var)
                var_values.append(value(pyomo_var))

        if len(var_names):
            self._solver_model.MIP_starts.add([var_names, var_values], self._solver_model.MIP_starts.effort_level.auto)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        for var in vars_to_load:
            if ref_vars[var] > 0:
                var.stale = False
                var.value = self._solver_model.solution.get_values(var_map[var])

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        for var in vars_to_load:
            if ref_vars[var] > 0:
                rc[var] = self._solver_model.solution.get_reduced_costs(var_map[var])

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        dual = self._pyomo_model.dual
        if cons_to_load is None:
            cons_to_load = ComponentSet(con_map.keys())

        reverse_con_map = {}
        for pyomo_con, con in con_map.items():
            reverse_con_map[con] = pyomo_con

        for cplex_con in self._solver_model.linear_constraints.get_names():
            pyomo_con = reverse_con_map[cplex_con]
            if pyomo_con in cons_to_load:
                dual[pyomo_con] = self._solver_model.solution.get_dual_values(cplex_con)

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        slack = self._pyomo_model.slack
        if cons_to_load is None:
            cons_to_load = ComponentSet(con_map.keys())

        reverse_con_map = {}
        for pyomo_con, con in con_map.items():
            reverse_con_map[con] = pyomo_con

        for cplex_con in self._solver_model.linear_constraints.get_names():
            pyomo_con = reverse_con_map[cplex_con]
            if pyomo_con in cons_to_load:
                slack[pyomo_con] = self._solver_model.solution.get_linear_slacks(cplex_con)

        for cplex_con in self._solver_model.quadratic_constraints.get_names():
            pyomo_con = reverse_con_map[cplex_con]
            if pyomo_con in cons_to_load:
                slack[pyomo_con] = self._solver_model.solution.get_quadratic_slacks(cplex_con)
