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
import time

from pyomo.common.tempfiles import TempfileManager
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr.numvalue import is_fixed
from pyomo.core.expr.numvalue import value
from pyomo.repn import generate_standard_repn
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import DirectOrPersistentSolver
from pyomo.core.kernel.objective import minimize, maximize
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solution import Solution, SolutionStatus
from pyomo.opt.results.solver import TerminationCondition, SolverStatus
from pyomo.opt.base import SolverFactory
from pyomo.core.base.suffix import Suffix
import pyomo.core.base.var


logger = logging.getLogger('pyomo.solvers')

class DegreeError(ValueError):
    pass

def _is_convertable(conv_type,x):
    try:
        conv_type(x)
    except ValueError:
        return False
    return True

def _print_message(xp_prob, _, msg, *args):
    if msg is not None:
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()

@SolverFactory.register('xpress_direct', doc='Direct python interface to XPRESS')
class XpressDirect(DirectSolver):

    def __init__(self, **kwds):
        if 'type' not in kwds:
            kwds['type'] = 'xpress_direct'
        super(XpressDirect, self).__init__(**kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()

        self._name = None
        try:
            import xpress 
            self._xpress = xpress
            self._python_api_exists = True
            self._version = tuple(
                    int(k) for k in self._xpress.getversion().split('.'))
            self._name = "Xpress %s.%s.%s" % self._version
            self._version_major = self._version[0]
            # in versions prior to 34, xpress raised a RuntimeError, but in more
            # recent versions it raises a xpress.ModelError. We'll cache the appropriate
            # one here
            if self._version_major < 34:
                self._XpressException = RuntimeError
            else:
                self._XpressException = xpress.ModelError
        except ImportError:
            self._python_api_exists = False
        except Exception as e:
            # other forms of exceptions can be thrown by the xpress python
            # import. for example, a xpress.InterfaceError exception is thrown
            # if the Xpress license is not valid. Unfortunately, you can't
            # import without a license, which means we can't test for the
            # exception above!
            print("Import of xpress failed - xpress message=" + str(e) + "\n")
            self._python_api_exists = False
            
        self._range_constraints = set()

        # TODO: this isn't a limit of XPRESS, which implements an SLP
        #       method for NLPs. But it is a limit of *this* interface
        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        # There does not seem to be an easy way to get the
        # wallclock time out of xpress, so we will measure it
        # ourselves
        self._opt_time = None

        # Note: Undefined capabilites default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

    def _apply_solver(self):
        if not self._save_results:
            for block in self._pyomo_model.block_data_objects(descend_into=True,
                                                              active=True):
                for var in block.component_data_objects(ctype=pyomo.core.base.var.Var,
                                                        descend_into=False,
                                                        active=True,
                                                        sort=False):
                    var.stale = True

        self._solver_model.setlogfile(self._log_file)
        if self._keepfiles:
            print("Solver log file: "+self.log_file)

        # setting a log file in xpress disables all output
        # this callback prints all messages to stdout
        if self._tee:
            self._solver_model.addcbmessage(_print_message, None, 0)

        # set xpress options
        # if the user specifies a 'mipgap', set it, and
        # set xpress's related options to 0.
        if self.options.mipgap is not None:
            self._solver_model.setControl('miprelstop', float(self.options.mipgap))
            self._solver_model.setControl('miprelcutoff', 0.0)
            self._solver_model.setControl('mipaddcutoff', 0.0)
        # xpress is picky about the type which is passed
        # into a control. So we will infer and cast
        # get the xpress valid controls
        xp_controls = self._xpress.controls
        for key, option in self.options.items():
            if key == 'mipgap': # handled above
                continue
            try: 
                self._solver_model.setControl(key, option)
            except self._XpressException:
                # take another try, converting to its type
                # we'll wrap this in a function to raise the
                # xpress error
                contr_type = type(getattr(xp_controls, key))
                if not _is_convertable(contr_type, option):
                    raise
                self._solver_model.setControl(key, contr_type(option))

        start_time = time.time()
        self._solver_model.solve()
        self._opt_time = time.time() - start_time

        self._solver_model.setlogfile('')
        if self._tee:
            self._solver_model.removecbmessage(_print_message, None)

        # FIXME: can we get a return code indicating if XPRESS had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError('XpressDirect does not support expressions of degree {0}.'.format(degree))

        # NOTE: xpress's python interface only allows for expresions
        #       with native numeric types. Others, like numpy.float64,
        #       will cause an exception when constructing expressions
        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr = self._xpress.Sum(float(coef)*self._pyomo_var_to_solver_var_map[var] for coef,var in zip(repn.linear_coefs, repn.linear_vars))
        else:
            new_expr = 0.0

        for coef,(x,y) in zip(repn.quadratic_coefs,repn.quadratic_vars):
            new_expr += float(coef) * self._pyomo_var_to_solver_var_map[x] * self._pyomo_var_to_solver_var_map[y]
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
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(repn, max_degree)
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return xpress_expr, referenced_vars

    def _xpress_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -self._xpress.infinity
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = self._xpress.infinity
        return lb, ub

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vartype = self._xpress_vartype_from_var(var)
        lb, ub = self._xpress_lb_ub_from_var(var)

        xpress_var = self._xpress.var(name=varname, lb=lb, ub=ub, vartype=vartype)
        self._solver_model.addVariable(xpress_var)

        ## bounds on binary variables don't seem to be set correctly
        ## by the method above
        if vartype == self._xpress.binary:
            if lb == ub:
                self._solver_model.chgbounds([xpress_var], ['B'], [lb])
            else:
                self._solver_model.chgbounds([xpress_var, xpress_var], ['L', 'U'], [lb,ub])

        self._pyomo_var_to_solver_var_map[var] = xpress_var
        self._solver_var_to_pyomo_var_map[xpress_var] = var
        self._referenced_variables[var] = 0

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        try:
            if model.name is not None:
                self._solver_model = self._xpress.problem(name=model.name)
            else:
                self._solver_model = self._xpress.problem()
        except Exception:
            e = sys.exc_info()[1]
            msg = ("Unable to create Xpress model. "
                   "Have you installed the Python "
                   "bindings for Xpress?\n\n\t" +
                   "Error message: {0}".format(e))
            raise Exception(msg)
        self._add_block(model)

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)

    def _add_constraint(self, con):
        if not con.active:
            return None

        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(),
                self._max_constraint_degree)
        else:
            xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(
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
            xpress_con = self._xpress.constraint(body=xpress_expr,
                                                 sense=self._xpress.eq,
                                                 rhs=value(con.lower),
                                                 name=conname)
        elif con.has_lb() and con.has_ub():
            xpress_con = self._xpress.constraint(body=xpress_expr,
                                                 sense=self._xpress.range,
                                                 lb=value(con.lower),
                                                 ub=value(con.upper),
                                                 name=conname)
            self._range_constraints.add(xpress_con)
        elif con.has_lb():
            xpress_con = self._xpress.constraint(body=xpress_expr,
                                                 sense=self._xpress.geq,
                                                 rhs=value(con.lower),
                                                 name=conname)
        elif con.has_ub():
            xpress_con = self._xpress.constraint(body=xpress_expr,
                                                 sense=self._xpress.leq,
                                                 rhs=value(con.upper),
                                                 name=conname)
        else:
            raise ValueError("Constraint does not have a lower "
                             "or an upper bound: {0} \n".format(con))

        self._solver_model.addConstraint(xpress_con)

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = xpress_con
        self._solver_con_to_pyomo_con_map[xpress_con] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level not in [1,2]:
            raise ValueError("Solver does not support SOS "
                             "level {0} constraints".format(level))

        xpress_vars = []
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
            xpress_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        xpress_con = self._xpress.sos(xpress_vars, weights, level, conname)
        self._solver_model.addSOS(xpress_con)
        self._pyomo_con_to_solver_con_map[con] = xpress_con
        self._solver_con_to_pyomo_con_map[xpress_con] = con

    def _xpress_vartype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate xpress variable type
        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer
        """
        if var.is_binary():
            vartype = self._xpress.binary
        elif var.is_integer():
            vartype = self._xpress.integer
        elif var.is_continuous():
            vartype = self._xpress.continuous
        else:
            raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
        return vartype
    
    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = self._xpress.minimize
        elif obj.sense == maximize:
            sense = self._xpress.maximize
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        xpress_expr, referenced_vars = self._get_expr_from_pyomo_expr(obj.expr, self._max_obj_degree)

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(xpress_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _postsolve(self):
        # the only suffixes that we extract from XPRESS are
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
                raise RuntimeError("***The xpress_direct solver plugin cannot extract solution suffix="+suffix)

        xprob = self._solver_model
        xp = self._xpress
        xprob_attrs = xprob.attributes

        ## XPRESS's status codes depend on this
        ## (number of integer vars > 0) or (number of special order sets > 0)
        is_mip = (xprob_attrs.mipents > 0) or (xprob_attrs.sets > 0)

        if is_mip:
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = self._name
        self.results.solver.wallclock_time = self._opt_time

        if is_mip:
            status = xprob_attrs.mipstatus
            mip_sols = xprob_attrs.mipsols
            if status == xp.mip_not_loaded:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is not loaded; no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            #no MIP solution, first LP did not solve, second LP did, third search started but incomplete
            elif status == xp.mip_lp_not_optimal \
                    or status == xp.mip_lp_optimal \
                    or status == xp.mip_no_sol_found:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is loaded, but no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            elif status == xp.mip_solution: # some solution available
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Unable to satisfy optimality tolerances; a sub-optimal " \
                                                          "solution is available."
                self.results.solver.termination_condition = TerminationCondition.other
                soln.status = SolutionStatus.feasible
            elif status == xp.mip_infeas: # MIP proven infeasible
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be infeasible"
                self.results.solver.termination_condition = TerminationCondition.infeasible
                soln.status = SolutionStatus.infeasible
            elif status == xp.mip_optimal: # optimal
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                          "and an optimal solution is available."
                self.results.solver.termination_condition = TerminationCondition.optimal
                soln.status = SolutionStatus.optimal
            elif status == xp.mip_unbounded and mip_sols > 0:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "LP relaxation was proven to be unbounded, " \
                                                          "but a solution is available."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            elif status == xp.mip_unbounded and mip_sols <= 0:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "LP relaxation was proven to be unbounded."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            else:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = \
                    ("Unhandled Xpress solve status "
                     "("+str(status)+")")
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
        else: ## an LP, we'll check the lpstatus
            status = xprob_attrs.lpstatus
            if status == xp.lp_unstarted:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Model is not loaded; no solution information is available."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.unknown
            elif status == xp.lp_optimal:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Model was solved to optimality (subject to tolerances), " \
                                                          "and an optimal solution is available."
                self.results.solver.termination_condition = TerminationCondition.optimal
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_infeas:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be infeasible"
                self.results.solver.termination_condition = TerminationCondition.infeasible
                soln.status = SolutionStatus.infeasible
            elif status == xp.lp_cutoff:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Optimal objective for model was proven to be worse than the " \
                                                          "cutoff value specified; a solution is available."
                self.results.solver.termination_condition = TerminationCondition.minFunctionValue
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_unfinished:
                self.results.solver.status = SolverStatus.aborted
                self.results.solver.termination_message = "Optimization was terminated by the user."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            elif status == xp.lp_unbounded:
                self.results.solver.status = SolverStatus.warning
                self.results.solver.termination_message = "Model was proven to be unbounded."
                self.results.solver.termination_condition = TerminationCondition.unbounded
                soln.status = SolutionStatus.unbounded
            elif status == xp.lp_cutoff_in_dual:
                self.results.solver.status = SolverStatus.ok
                self.results.solver.termination_message = "Xpress reported the LP was cutoff in the dual."
                self.results.solver.termination_condition = TerminationCondition.minFunctionValue
                soln.status = SolutionStatus.optimal
            elif status == xp.lp_unsolved:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = "Optimization was terminated due to unrecoverable numerical " \
                                                          "difficulties."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            elif status == xp.lp_nonconvex:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = "Optimization was terminated because nonconvex quadratic data " \
                                                          "were found."
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error
            else:
                self.results.solver.status = SolverStatus.error
                self.results.solver.termination_message = \
                    ("Unhandled Xpress solve status "
                     "("+str(status)+")")
                self.results.solver.termination_condition = TerminationCondition.error
                soln.status = SolutionStatus.error


        self.results.problem.name = xprob_attrs.matrixname

        if xprob_attrs.objsense == 1.0:
            self.results.problem.sense = minimize
        elif xprob_attrs.objsense == -1.0:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError('Unrecognized Xpress objective sense: {0}'.format(xprob_attrs.objsense))

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if not is_mip: #LP or continuous problem
            try:
                self.results.problem.upper_bound = xprob_attrs.lpobjval
                self.results.problem.lower_bound = xprob_attrs.lpobjval
            except (self._XpressException, AttributeError):
                pass
        elif xprob_attrs.objsense == 1.0:  # minimizing MIP
            try:
                self.results.problem.upper_bound = xprob_attrs.mipbestobjval
            except (self._XpressException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = xprob_attrs.bestbound
            except (self._XpressException, AttributeError):
                pass
        elif xprob_attrs.objsense == -1.0:  # maximizing MIP
            try:
                self.results.problem.upper_bound = xprob_attrs.bestbound
            except (self._XpressException, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = xprob_attrs.mipbestobjval
            except (self._XpressException, AttributeError):
                pass
        else:
            raise RuntimeError('Unrecognized xpress objective sense: {0}'.format(xprob_attrs.objsense))

        try:
            soln.gap = self.results.problem.upper_bound - self.results.problem.lower_bound
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = xprob_attrs.rows + xprob_attrs.sets + xprob_attrs.qconstraints
        self.results.problem.number_of_nonzeros = xprob_attrs.elems
        self.results.problem.number_of_variables = xprob_attrs.cols
        self.results.problem.number_of_integer_variables = xprob_attrs.mipents
        self.results.problem.number_of_continuous_variables = xprob_attrs.cols - xprob_attrs.mipents
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = xprob_attrs.mipsols if is_mip else 1 

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatability. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if xprob_attrs.lpstatus in \
                    [xp.lp_optimal, xp.lp_cutoff, xp.lp_cutoff_in_dual] or \
                    xprob_attrs.mipsols > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                xpress_vars = list(self._solver_var_to_pyomo_var_map.keys())
                var_vals = xprob.getSolution(xpress_vars)
                for xpress_var, val in zip(xpress_vars, var_vals):
                    pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        pyomo_var.stale = False
                        soln_variables[xpress_var.name] = {"Value": val}

                if extract_reduced_costs:
                    vals = xprob.getRCost(xpress_vars)
                    for xpress_var, val in zip(xpress_vars, vals):
                        pyomo_var = self._solver_var_to_pyomo_var_map[xpress_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[xpress_var.name]["Rc"] = val

                if extract_duals or extract_slacks:
                    xpress_cons = list(self._solver_con_to_pyomo_con_map.keys())
                    for con in xpress_cons:
                        soln_constraints[con.name] = {}

                if extract_duals:
                    vals = xprob.getDual(xpress_cons)
                    for val, con in zip(vals, xpress_cons):
                        soln_constraints[con.name]["Dual"] = val

                if extract_slacks:
                    vals = xprob.getSlack(xpress_cons)
                    for con, val in zip(xpress_cons, vals):
                        if con in self._range_constraints:
                            ## for xpress, the slack on a range constraint
                            ## is based on the upper bound
                            lb = con.lb
                            ub = con.ub
                            ub_s = val
                            expr_val = ub-ub_s
                            lb_s = lb-expr_val
                            if abs(ub_s) > abs(lb_s):
                                soln_constraints[con.name]["Slack"] = ub_s
                            else:
                                soln_constraints[con.name]["Slack"] = lb_s
                        else:
                            soln_constraints[con.name]["Slack"] = val

        elif self._load_solutions:
            if xprob_attrs.lpstatus == xp.lp_optimal and \
                    ((not is_mip) or (xprob_attrs.mipsols > 0)):

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
        mipsolval = list()
        mipsolcol = list()
        for pyomo_var, xpress_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                mipsolval.append(value(pyomo_var))
                mipsolcol.append(xpress_var)
        self._solver_model.addmipsol(mipsolval, mipsolcol)

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getSolution(xpress_vars_to_load)

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

        xpress_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getRCost(xpress_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        dual = self._pyomo_model.dual

        if cons_to_load is None:
            cons_to_load = con_map.keys()

        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._solver_model.getDual(xpress_cons_to_load)

        for pyomo_con, val in zip(cons_to_load, vals):
            dual[pyomo_con] = val

    def _load_slacks(self, cons_to_load=None):
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)
        con_map = self._pyomo_con_to_solver_con_map
        slack = self._pyomo_model.slack

        if cons_to_load is None:
            cons_to_load = con_map.keys()

        xpress_cons_to_load = [con_map[pyomo_con] for pyomo_con in cons_to_load]
        vals = self._solver_model.getSlack(xpress_cons_to_load)

        for pyomo_con, xpress_con, val in zip(cons_to_load, xpress_cons_to_load, vals):
            if xpress_con in self._range_constraints:
                ## for xpress, the slack on a range constraint
                ## is based on the upper bound
                lb = con.lb
                ub = con.ub
                ub_s = val
                expr_val = ub-ub_s
                lb_s = lb-expr_val
                if abs(ub_s) > abs(lb_s):
                    slack[pyomo_con] = ub_s
                else:
                    slack[pyomo_con] = lb_s
            else:
                slack[pyomo_con] = val

    def load_duals(self, cons_to_load=None):
        """
        Load the duals into the 'dual' suffix. The 'dual' suffix must live on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_duals(cons_to_load)

    def load_rc(self, vars_to_load=None):
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
