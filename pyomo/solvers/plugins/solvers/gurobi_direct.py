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
from pyomo.core.kernel.numvalue import value
import pyomo.core.kernel
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition


logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


class GurobiDirect(DirectSolver):
    alias('gurobi_direct', doc='Direct python interface to Gurobi')

    def __init__(self, **kwds):
        kwds['type'] = 'gurobi_direct'
        super(GurobiDirect, self).__init__(**kwds)
        self._init()

    def _init(self):
        try:
            import gurobipy
            self._gurobipy = gurobipy
            self._python_api_exists = True
            self._version = self._gurobipy.gurobi.version()
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

        if self._version_major < 5:
            self._max_constraint_degree = 1
        else:
            self._max_constraint_degree = 2
        self._max_obj_degree = 2

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        if self._version_major < 5:
            self._capabilities.quadratic_constraint = False
        else:
            self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _presolve(self, *args, **kwds):
        super(GurobiDirect, self)._presolve(*args, **kwds)

    def _apply_solver(self):
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
            self._solver_model.setParam(key, option)

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

        degree = canonical_degree(repn)
        if (degree is None) or (degree > max_degree):
            raise DegreeError('GurobiDirect does not support expressions of degree {0}.'.format(degree))

        if isinstance(repn, LinearCanonicalRepn):
            new_expr = 0.0

            if repn.constant is not None:
                new_expr += repn.constant

            if (repn.linear is not None) and (len(repn.linear) > 0):
                list(map(referenced_vars.add, repn.variables))
                new_expr = repn.constant + sum(coeff*self._pyomo_var_to_solver_var_map[repn.variables[i]]
                                               for i, coeff in enumerate(repn.linear))

        else:
            new_expr = 0
            if 0 in repn:
                new_expr += repn[0][None]

            if 1 in repn:
                for ndx, coeff in repn[1].items():
                    new_expr += coeff * self._pyomo_var_to_solver_var_map[repn[-1][ndx]]
                    referenced_vars.add(repn[-1][ndx])

            if 2 in repn:
                for key, coeff in repn[2].items():
                    tmp_expr = coeff
                    for ndx, power in key.items():
                        referenced_vars.add(repn[-1][ndx])
                        for i in range(power):
                            tmp_expr *= self._pyomo_var_to_solver_var_map[repn[-1][ndx]]
                    new_expr += tmp_expr

        return new_expr, referenced_vars

    def _get_expr_from_pyomo_expr(self, expr, max_degree=2):
        repn = generate_canonical_repn(expr)

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
        lb = value(var.lb)
        ub = value(var.ub)
        if lb is None:
            lb = -self._gurobipy.GRB.INFINITY
        if ub is None:
            ub = self._gurobipy.GRB.INFINITY

        gurobipy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[var] = gurobipy_var
        self._referenced_variables[var] = 0

        if var.is_fixed():
            gurobipy_var.setAttr('lb', var.value)
            gurobipy_var.setAttr('ub', var.value)

    def _compile_instance(self, model, **kwds):
        self._range_constraints = set()
        super(GurobiDirect, self)._compile_instance(model, **kwds)
        try:
            self._solver_model = self._gurobipy.Model()
        except Exception:
            e = sys.exc_info()[1]
            msg = ('Unable to create Gurobi model. Have you ihnstalled the Python bindings for Gurboi?\n\n\t' +
                   'Error message: {0}'.format(e))
            raise Exception(msg)

        self._add_block(model)

        for var in self._referenced_variables:
            if var.fixed:
                if not self._output_fixed_variable_bounds:
                    raise ValueError("Encountered a fixed variable (%s) inside an active objective "
                                     "or constraint expression on model %s, which is usually indicative of "
                                     "a preprocessing error. Use the IO-option 'output_fixed_variable_bounds=True' "
                                     "to suppress this error and fix the variable by overwriting its bounds in "
                                     "the Gurobi instance."
                                     % (var.name, self._pyomo_model.name,))

    def _add_block(self, block):
        for var in block.component_data_objects(ctype=pyomo.core.base.var.Var, descend_into=True, active=True):
            self._add_var(var)
        self._solver_model.update()

        for sub_block in block.block_data_objects(descend_into=True, active=True):
            for con in sub_block.component_data_objects(ctype=pyomo.core.base.constraint.Constraint,
                                                        descend_into=False, active=True):
                self._add_constraint(con)

            for con in sub_block.component_data_objects(ctype=pyomo.core.base.sos.SOSConstraint,
                                                        descend_into=False, active=True):
                self._add_sos_constraint(con)

            if len([obj for obj in sub_block.component_data_objects(ctype=pyomo.core.base.objective.Objective,
                                                                    descend_into=False, active=True)]) != 0:
                self._compile_objective()

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(con.canonical_form(),
                                                                          self._max_constraint_degree)
        elif isinstance(con, LinearCanonicalRepn):
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(con, self._max_constraint_degree)
        else:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))

        if con.equality:
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr, sense=self._gurobipy.GRB.EQUAL,
                                                        rhs=value(con.lower), name=conname)
        elif con.has_lb() and (value(con.lower) > -float('inf')) and con.has_ub() and (value(con.upper) < float('inf')):
            gurobipy_con = self._solver_model.addRange(gurobi_expr, value(con.lower), value(con.upper), name=conname)
            self._range_constraints.add(con)
        elif con.has_lb() and (value(con.lower) > -float('inf')):
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr, sense=self._gurobipy.GRB.GREATER_EQUAL,
                                                        rhs=value(con.lower), name=conname)
        elif con.has_ub() and (value(con.upper) < float('inf')):
            gurobipy_con = self._solver_model.addConstr(lhs=gurobi_expr, sense=self._gurobipy.GRB.LESS_EQUAL,
                                                        rhs=value(con.upper), name=conname)
        else:
            raise ValueError('Constraint does not have a lower or an upper bound: {0} \n'.format(con))

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con

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
            raise ValueError('Solver does not support SOS level {0} constraints'.format(level))

        gurobi_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        for v, w in con.get_items():
            self._vars_referenced_by_con[con].add(v)
            gurobi_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        gurobipy_con = self._solver_model.addSOS(sos_type, gurobi_vars, weights)
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con

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

    def _compile_objective(self):
        obj_counter = 0

        for obj in self._pyomo_model.component_data_objects(ctype=pyomo.core.base.objective.Objective,
                                                            descend_into=True, active=True):
            obj_counter += 1
            if obj_counter > 1:
                raise ValueError('Multiple active objectives found. Solver only handles one active objective')

            if obj.sense == pyomo.core.kernel.minimize:
                sense = self._gurobipy.GRB.MINIMIZE
            elif obj.sense == pyomo.core.kernel.maximize:
                sense = self._gurobipy.GRB.MAXIMIZE
            else:
                raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)

            for var in referenced_vars:
                self._referenced_variables[var] += 1

            self._solver_model.setObjective(gurobi_expr, sense=sense)
            self._objective_label = self._symbol_map.getSymbol(obj, self._labeler)

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
                if len(self._range_constraints) != 0:
                    err_msg = ('GurobiDirect does not support range constraints and slack suffixes. \nIf you want ' +
                               'slack information, please split up the following constraints:\n')
                    for con in self._range_constraints:
                        err_msg += '{0}\n'.format(con)
                    raise ValueError(err_msg)
            if re.match(suffix, "rc"):
                extract_reduced_costs = True
                flag = True
            if not flag:
                raise RuntimeError(
                    "***The gurobi_direct solver plugin "
                    "cannot extract solution suffix="+suffix)

        gprob = self._solver_model

        if gprob.getAttr(self._gurobipy.GRB.Attr.IsMIP):
            extract_reduced_costs = False
            extract_duals = False

        pvars = gprob.getVars()
        cons = gprob.getConstrs()
        qcons = []
        if self._version_major >= 5:
            qcons = gprob.getQConstrs()

        self.results = SolverResults()
        soln = Solution()

        # cache the variable and constraint dictionaries -
        # otherwise, each invocation will include a lookup in a
        # MapContainer, which is extremely expensive.
        soln_variables = soln.variable
        soln_constraints = soln.constraint

        self.results.solver.name = ("Gurobi %s.%s%s"
                                    % self._gurobipy.gurobi.version())
        self.results.solver.wallclock_time = gprob.Runtime

        if gprob.Status == 1:  # problem is loaded, but no solution
            self.results.solver.termination_message = "Model is loaded, but no solution information is available."
            self.results.solver.termination_condition = TerminationCondition.other
        if gprob.Status == 2:  # optimal
            self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances)."
            self.results.solver.termination_condition = TerminationCondition.optimal
        elif gprob.Status == 3:  # infeasible
            self.results.solver.termination_message = "Model was proven to be infeasible."
            self.results.solver.termination_condition = TerminationCondition.infeasible
        elif gprob.Status == 4:  # infeasible or unbounded
            self.results.solver.termination_message = "Model was proven to be either infeasible or unbounded."
            self.results.solver.termination_condition = TerminationCondition.infeasibleOrUnbounded
        elif gprob.Status == 5:  # unbounded
            self.results.solver.termination_message = "Model was proven to be unbounded."
            self.results.solver.termination_condition = TerminationCondition.unbounded
        elif gprob.Status == 6:  # cutoff
            self.results.solver.termination_message = "Optimal objective for model was proven to be worse than the " \
                                                      "value specified in the Cutoff parameter."
            self.results.solver.termination_condition = TerminationCondition.minFunctionValue
        elif gprob.Status == 7:  # iteration limit
            self.results.solver.termination_message = "Optimization terminated because the total number of simplex " \
                                                      "or barrier iterations exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxIterations
        elif gprob.Status == 8:  # node limit
            self.results.solver.termination_message = "Optimization terminated because the total number of " \
                                                      "branch-and-cut nodes exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxEvaluations
        elif gprob.Status == 9:  # time limit
            self.results.solver.termination_message = "Optimization terminated because the total time expended " \
                                                      "exceeded specified limits."
            self.results.solver.termination_condition = TerminationCondition.maxTimeLimit
        elif gprob.Status == 10:  # solution limit
            self.results.solver.termination_message = "Optimization terminated because the number of solutions " \
                                                      "found reached specified limits."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 11:  # interrupted
            self.results.solver.termination_message = "Optimization was terminated by the user."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 12:  # numeric issues
            self.results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical " \
                                                      "difficulties."
            self.results.solver.termination_condition = TerminationCondition.other
        elif gprob.Status == 13:  # suboptimal
            self.results.solver.termination_message = "Optimization was unable to satisfy optimality tolerances; " \
                                                      "returned solution is sub-optimal."
            self.results.solver.termination_condition = TerminationCondition.other
        else:  # unknown
            self.results.solver.termination_message = "Unknown termination condition received following optimization."
            self.results.solver.termination_condition = TerminationCondition.other

        self.results.problem.name = gprob.ModelName
        if gprob.ModelSense == 1:  # minimizing
            self.results.problem.sense = pyomo.core.kernel.minimize
            try:
                self.results.problem.upper_bound = gprob.ObjVal
            except self._gurobipy.GurobiError:
                self.results.problem.upper_bound = None
            try:
                self.results.problem.lower_bound = gprob.ObjBound
            except self._gurobipy.GurobiError:
                self.results.problem.lower_bound = None
        elif gprob.ModelSense == -1:  # maximizing
            self.results.problem.sense = pyomo.core.kernel.maximize
            try:
                self.results.problem.upper_bound = gprob.ObjBound
            except self._gurobipy.GurobiError:
                self.results.problem.upper_bound = None
            try:
                self.results.problem.lower_bound = gprob.ObjVal
            except self._gurobipy.GurobiError:
                self.results.problem.lower_bound = None
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

        status = self._solver_model.Status
        if self._gurobipy.GRB.OPTIMAL == status:
            soln.status = SolutionStatus.optimal
        elif self._gurobipy.GRB.INFEASIBLE == status:
            soln.status = SolutionStatus.infeasible
        elif self._gurobipy.GRB.CUTOFF == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.INF_OR_UNBD == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.INTERRUPTED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.LOADED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.SUBOPTIMAL == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.UNBOUNDED == status:
            soln.status = SolutionStatus.other
        elif self._gurobipy.GRB.ITERATION_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.NODE_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.SOLUTION_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.TIME_LIMIT == status:
            soln.status = SolutionStatus.stoppedByLimit
        elif self._gurobipy.GRB.NUMERIC == status:
            soln.status = SolutionStatus.error
        else:
            raise RuntimeError("Unknown solution status returned by "
                               "Gurobi solver")

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if (soln.status == SolutionStatus.optimal) or \
           ((soln.status == SolutionStatus.stoppedByLimit) and (gprob.SolCount > 0)):

            soln.objective[self._objective_label] = {'Value': gprob.ObjVal}

            for pyomo_var, gurobipy_var in self._pyomo_var_to_solver_var_map.items():
                if self._referenced_variables[pyomo_var] > 0:
                    soln_variables[gurobipy_var.VarName] = {"Value": gurobipy_var.x}

            if extract_reduced_costs:
                for var in pvars:
                    soln_variables[var.VarName]["Rc"] = var.Rc

            if extract_duals or extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName] = {}
                for con in qcons:
                    soln_constraints[con.QCName] = {}

            if extract_duals:
                for con in cons:
                    # Pi attributes in Gurobi are the constraint duals
                    soln_constraints[con.ConstrName]["Dual"] = con.Pi
                for con in qcons:
                    # QCPI attributes in Gurobi are the constraint duals
                    soln_constraints[con.QCName]["Dual"] = con.QCPi

            if extract_slacks:
                for con in cons:
                    soln_constraints[con.ConstrName]["Slack"] = con.Slack
                for con in qcons:
                    soln_constraints[con.QCName]["Slack"] = con.QCSlack

        self.results.solution.insert(soln)

        # finally, clean any temporary files registered with the temp file
        # manager, created populated *directly* by this plugin.
        pyutilib.services.TempfileManager.pop(remove=not self._keepfiles)

        return super(GurobiDirect, self)._postsolve()

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        for pyomo_var, gurobipy_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                gurobipy_var.setAttr(self._gurobipy.GRB.Attr.Start, value(pyomo_var))

    def _load_vars(self, vars_to_load=None):
        if vars_to_load is None:
            vars_to_load = self._pyomo_var_to_solver_var_map.keys()

        for var in vars_to_load:
            if self._referenced_variables[var] > 0:
                var.value = self._pyomo_var_to_solver_var_map[var].x
