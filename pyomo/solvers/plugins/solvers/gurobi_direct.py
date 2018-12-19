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
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass

def _is_numeric(x):
    try:
        float(x)
    except ValueError:
        return False
    return True

@SolverFactory.register('gurobi_direct', doc='Direct python interface to Gurobi')
class GurobiDirect(DirectSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'gurobi_direct'
        DirectSolver.__init__(self, **kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._init()

    def _init(self):
        self._name = None
        try:
            import gurobipy
            self._gurobipy = gurobipy
            self._python_api_exists = True
            self._version = self._gurobipy.gurobi.version()
            self._name = "Gurobi %s.%s%s" % self._version
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
            print("Import of gurobipy failed - gurobi message=" + str(e) + "\n")
            self._python_api_exists = False

        self._range_constraints = set()

        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        # fix for compatibility with pre-5.0 Gurobi
        if self._python_api_exists and \
           (self._version_major < 5):
            self._max_constraint_degree = 1
            self._capabilities.quadratic_constraint = False

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(descend_into=True,
                                                              active=True):
                for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                        descend_into=False,
                                                        active=True,
                                                        sort=False):
                    var.stale = True
        if self._tee:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        self._solver_model.setParam('LogFile', self._log_file)

        if self._keepfiles:
            print("Solver log file: "+self._log_file)

        # Options accepted by gurobi (case insensitive):
        # ['Cutoff', 'IterationLimit', 'NodeLimit', 'SolutionLimit', 'TimeLimit',
        #  'FeasibilityTol', 'IntFeasTol', 'MarkowitzTol', 'MIPGap', 'MIPGapAbs',
        #  'OptimalityTol', 'PSDTol', 'Method', 'PerturbValue', 'ObjScale', 'ScaleFlag',
        #  'SimplexPricing', 'Quad', 'NormAdjust', 'BarIterLimit', 'BarConvTol',
        #  'BarCorrectors', 'BarOrder', 'Crossover', 'CrossoverBasis', 'BranchDir',
        #  'Heuristics', 'MinRelNodes', 'MIPFocus', 'NodefileStart', 'NodefileDir',
        #  'NodeMethod', 'PumpPasses', 'RINS', 'SolutionNumber', 'SubMIPNodes', 'Symmetry',
        #  'VarBranch', 'Cuts', 'CutPasses', 'CliqueCuts', 'CoverCuts', 'CutAggPasses',
        #  'FlowCoverCuts', 'FlowPathCuts', 'GomoryPasses', 'GUBCoverCuts', 'ImpliedCuts',
        #  'MIPSepCuts', 'MIRCuts', 'NetworkCuts', 'SubMIPCuts', 'ZeroHalfCuts', 'ModKCuts',
        #  'Aggregate', 'AggFill', 'PreDual', 'DisplayInterval', 'IISMethod', 'InfUnbdInfo',
        #  'LogFile', 'PreCrush', 'PreDepRow', 'PreMIQPMethod', 'PrePasses', 'Presolve',
        #  'ResultFile', 'ImproveStartTime', 'ImproveStartGap', 'Threads', 'Dummy', 'OutputFlag']
        for key, option in self.options.items():
            # When options come from the pyomo command, all
            # values are string types, so we try to cast
            # them to a numeric value in the event that
            # setting the parameter fails.
            try:
                self._solver_model.setParam(key, option)
            except TypeError:
                # we place the exception handling for
                # checking the cast of option to a float in
                # another function so that we can simply
                # call raise here instead of except
                # TypeError as e / raise e, because the
                # latter does not preserve the Gurobi stack
                # trace
                if not _is_numeric(option):
                    raise
                self._solver_model.setParam(key, float(option))

        if self._version_major >= 5:
            for suffix in self._suffixes:
                if re.match(suffix, "dual"):
                    self._solver_model.setParam(self._gurobipy.GRB.Param.QCPDual, 1)

        self._solver_model.optimize()

        self._solver_model.setParam('LogFile', 'default')

        # FIXME: can we get a return code indicating if Gurobi had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError('GurobiDirect does not support expressions of degree {0}.'.format(degree))

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr = self._gurobipy.LinExpr(repn.linear_coefs, [self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars])
        else:
            new_expr = 0.0

        for i,v in enumerate(repn.quadratic_vars):
            x,y = v
            new_expr += repn.quadratic_coefs[i] * self._pyomo_var_to_solver_var_map[x] * self._pyomo_var_to_solver_var_map[y]
            referenced_vars.add(x)
            referenced_vars.add(y)

        new_expr += repn.constant

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        if max_degree == 2:
            repn = generate_standard_repn(expr, quadratic=True)
        else:
            repn = generate_standard_repn(expr, quadratic=False)

        try:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return gurobi_expr, referenced_vars

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._gurobi_vtype_from_var(var)
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._gurobipy.GRB.INFINITY
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._gurobipy.GRB.INFINITY

        gurobipy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[var] = gurobipy_var
        self._solver_var_to_pyomo_var_map[gurobipy_var] = var
        self._referenced_variables[var] = 0

        if var.is_fixed():
            gurobipy_var.setAttr('lb', var.value)
            gurobipy_var.setAttr('ub', var.value)

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        try:
            if model.name is not None:
                self._solver_model = self._gurobipy.Model(model.name)
            else:
                self._solver_model = self._gurobipy.Model()
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create Gurobi model. "
                   "Have you installed the Python "
                   "bindings for Gurboi?\n\n\t"+
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
                            "by overwriting its bounds in the Gurobi instance."
                            % (var.name, self._pyomo_model.name,))

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)
        self._solver_model.update()

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),
                self._max_constraint_degree)
        #elif isinstance(con, LinearCanonicalRepn):
        #    gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(
        #        con,
        #        self._max_constraint_degree)
        else:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(
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
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.EQUAL,
                                                        rhs=value(con.lower),
                                                        name=conname)
        elif con.has_lb() and con.has_ub():
            gurobipy_con = self._solver_model.addRange(gurobi_expr,
                                                       value(con.lower),
                                                       value(con.upper),
                                                       name=conname)
            self._range_constraints.add(con)
        elif con.has_lb():
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.GREATER_EQUAL,
                                                        rhs=value(con.lower),
                                                        name=conname)
        elif con.has_ub():
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr,
                                                        sense=self._gurobipy.GRB.LESS_EQUAL,
                                                        rhs=value(con.upper),
                                                        name=conname)
        else:
            raise ValueError("Constraint does not have a lower "
                             "or an upper bound: {0} \n".format(con))

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[gurobipy_con] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level == 1:
            sos_type = self._gurobipy.GRB.SOS_TYPE1
        elif level == 2:
            sos_type = self._gurobipy.GRB.SOS_TYPE2
        else:
            raise ValueError("Solver does not support SOS "
                             "level {0} constraints".format(level))

        gurobi_vars = []
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
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[gurobipy_con] = con

    def _gurobi_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate gurobi variable type
        :param var: pyomo.core.base.var.Var
        :return: gurobipy.GRB.CONTINUOUS or gurobipy.GRB.BINARY or gurobipy.GRB.INTEGER
        """
        if var.is_binary():
            vtype = self._gurobipy.GRB.BINARY
        elif var.is_integer():
            vtype = self._gurobipy.GRB.INTEGER
        elif var.is_continuous():
            vtype = self._gurobipy.GRB.CONTINUOUS
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vtype

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = self._gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(gurobi_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from GUROBI are
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
                raise RuntimeError("***The gurobi_direct solver plugin cannot extract solution suffix="+suffix)

        gprob = self._solver_model
        grb = self._gurobipy.GRB
        status = gprob.Status

        if gprob.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = self._name
        self.results.solver.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Model is loaded, but no solution information is available."
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
        elif status == grb.OPTIMAL:  # optimal
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                      "and an optimal solution is available."
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == grb.INFEASIBLE:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = "Model was proven to be infeasible"
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == grb.INF_OR_UNBD:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = "Problem proven to be infeasible or unbounded."
            self.results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
            soln.status = SolutionStatus.unsure
        elif status == grb.UNBOUNDED:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = "Model was proven to be unbounded."
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == grb.CUTOFF:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimal objective for model was proven to be worse than the " \
                                                      "value specified in the Cutoff parameter. No solution " \
                                                      "information is available."
            self.results.solver.termination_condition = TerminationCondition.minFunctionValue
            soln.status = SolutionStatus.unknown
        elif status == grb.ITERATION_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimization terminated because the total number of simplex " \
                                                      "iterations performed exceeded the value specified in the " \
                                                      "IterationLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.NODE_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimization terminated because the total number of " \
                                                      "branch-and-cut nodes explored exceeded the value specified " \
                                                      "in the NodeLimit parameter"
            self.results.solver.termination_condition = TerminationCondition.maxEvaluations
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.TIME_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimization terminated because the time expended exceeded " \
                                                      "the value specified in the TimeLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.SOLUTION_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimization terminated because the number of solutions found " \
                                                      "reached the value specified in the SolutionLimit parameter."
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.INTERRUPTED:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "Optimization was terminated by the user."
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == grb.NUMERIC:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical " \
                                                      "difficulties."
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == grb.SUBOPTIMAL:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = "Unable to satisfy optimality tolerances; a sub-optimal " \
                                                      "solution is available."
            self.results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.feasible
        # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
        elif (status is not None) and \
             (status == getattr(grb,'USER_OBJ_LIMIT',None)):
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = "User specified an objective limit " \
                                                      "(a bound on either the best objective " \
                                                      "or the best bound), and that limit has " \
                                                      "been reached. Solution is available."
            self.results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = \
                ("Unhandled Gurobi solve status "
                 "("+str(status)+")")
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        self.results.problem.name = gprob.ModelName

        if gprob.ModelSense == 1:
            self.results.problem.sense = minimize
        elif gprob.ModelSense == -1:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if (gprob.NumBinVars + gprob.NumIntVars) == 0:
            try:
                self.results.problem.upper_bound = gprob.ObjVal
                self.results.problem.lower_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == 1:  # minimizing
            try:
                self.results.problem.upper_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == -1:  # maximizing
            try:
                self.results.problem.upper_bound = gprob.ObjBound
            except (self._gurobipy.GurobiError, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjVal
            except (self._gurobipy.GurobiError, AttributeError):
                pass
        else:
            raise RuntimeError('Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = gprob.NumConstrs + gprob.NumQConstrs + gprob.NumSOS
        self.results.problem.number_of_nonzeros = gprob.NumNZs
        self.results.problem.number_of_variables = gprob.NumVars
        self.results.problem.number_of_binary_variables = gprob.NumBinVars
        self.results.problem.number_of_integer_variables = gprob.NumIntVars
        self.results.problem.number_of_continuous_variables = gprob.NumVars - gprob.NumIntVars - gprob.NumBinVars
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = gprob.SolCount

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatability. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if gprob.SolCount > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                gurobi_vars = self._solver_model.getVars()
                gurobi_vars = list(set(gurobi_vars).intersection(set(self._pyomo_var_to_solver_var_map.values())))
                var_vals = self._solver_model.getAttr("X", gurobi_vars)
                names = self._solver_model.getAttr("VarName", gurobi_vars)
                for gurobi_var, val, name in zip(gurobi_vars, var_vals, names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[gurobi_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        pyomo_var.stale = False
                        soln_variables[name] = {"Value": val}

                if extract_reduced_costs:
                    vals = self._solver_model.getAttr("Rc", gurobi_vars)
                    for gurobi_var, val, name in zip(gurobi_vars, vals, names):
                        pyomo_var = self._solver_var_to_pyomo_var_map[gurobi_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[name]["Rc"] = val

                if extract_duals or extract_slacks:
                    gurobi_cons = self._solver_model.getConstrs()
                    con_names = self._solver_model.getAttr("ConstrName", gurobi_cons)
                    for name in con_names:
                        soln_constraints[name] = {}
                    if self._version_major >= 5:
                        gurobi_q_cons = self._solver_model.getQConstrs()
                        q_con_names = self._solver_model.getAttr("QCName", gurobi_q_cons)
                        for name in q_con_names:
                            soln_constraints[name] = {}

                if extract_duals:
                    vals = self._solver_model.getAttr("Pi", gurobi_cons)
                    for val, name in zip(vals, con_names):
                        soln_constraints[name]["Dual"] = val
                    if self._version_major >= 5:
                        q_vals = self._solver_model.getAttr("QCPi", gurobi_q_cons)
                        for val, name in zip(q_vals, q_con_names):
                            soln_constraints[name]["Dual"] = val

                if extract_slacks:
                    gurobi_range_con_vars = set(self._solver_model.getVars()) - set(self._pyomo_var_to_solver_var_map.values())
                    vals = self._solver_model.getAttr("Slack", gurobi_cons)
                    for gurobi_con, val, name in zip(gurobi_cons, vals, con_names):
                        pyomo_con = self._solver_con_to_pyomo_con_map[gurobi_con]
                        if pyomo_con in self._range_constraints:
                            lin_expr = self._solver_model.getRow(gurobi_con)
                            for i in reversed(range(lin_expr.size())):
                                v = lin_expr.getVar(i)
                                if v in gurobi_range_con_vars:
                                    Us_ = v.X
                                    Ls_ = v.UB - v.X
                                    if Us_ > Ls_:
                                        soln_constraints[name]["Slack"] = Us_
                                    else:
                                        soln_constraints[name]["Slack"] = -Ls_
                                    break
                        else:
                            soln_constraints[name]["Slack"] = val
                    if self._version_major >= 5:
                        q_vals = self._solver_model.getAttr("QCSlack", gurobi_q_cons)
                        for val, name in zip(q_vals, q_con_names):
                            soln_constraints[name]["Slack"] = val
        elif self._load_solutions:
            if gprob.SolCount > 0:

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
        for pyomo_var, gurobipy_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                gurobipy_var.setAttr(self._gurobipy.GRB.Attr.Start, value(pyomo_var))

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.stale = False
                var.value = val

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        rc = self._pyomo_model.rc
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("Rc", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            if self._version_major >= 5:
                quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs()))
            if self._version_major >= 5:
                quadratic_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs()))
        linear_vals = self._solver_model.getAttr("Pi", linear_cons_to_load)
        if self._version_major >= 5:
            quadratic_vals = self._solver_model.getAttr("QCPi", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[gurobi_con]
            dual[pyomo_con] = val
        if self._version_major >= 5:
            for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
                pyomo_con = reverse_con_map[gurobi_con]
                dual[pyomo_con] = val

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        reverse_con_map = self._solver_con_to_pyomo_con_map
        slack = self._pyomo_model.slack

        gurobi_range_con_vars = set(self._solver_model.getVars()) - set(self._pyomo_var_to_solver_var_map.values())

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            if self._version_major >= 5:
                quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set([con_map[pyomo_con] for pyomo_con in cons_to_load])
            linear_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getConstrs()))
            if self._version_major >= 5:
                quadratic_cons_to_load = gurobi_cons_to_load.intersection(set(self._solver_model.getQConstrs()))
        linear_vals = self._solver_model.getAttr("Slack", linear_cons_to_load)
        if self._version_major >= 5:
            quadratic_vals = self._solver_model.getAttr("QCSlack", quadratic_cons_to_load)

        for gurobi_con, val in zip(linear_cons_to_load, linear_vals):
            pyomo_con = reverse_con_map[gurobi_con]
            if pyomo_con in self._range_constraints:
                lin_expr = self._solver_model.getRow(gurobi_con)
                for i in reversed(range(lin_expr.size())):
                    v = lin_expr.getVar(i)
                    if v in gurobi_range_con_vars:
                        Us_ = v.X
                        Ls_ = v.UB - v.X
                        if Us_ > Ls_:
                            slack[pyomo_con] = Us_
                        else:
                            slack[pyomo_con] = -Ls_
                        break
            else:
                slack[pyomo_con] = val
        if self._version_major >= 5:
            for gurobi_con, val in zip(quadratic_cons_to_load, quadratic_vals):
                pyomo_con = reverse_con_map[gurobi_con]
                slack[pyomo_con] = val

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
