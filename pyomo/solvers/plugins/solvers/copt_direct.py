#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
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
from pyomo.common.errors import ApplicationError
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.expr.numvalue import value, is_fixed
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

logger = logging.getLogger('pyomo.solvers')


class DegreeError(ValueError):
    pass


def _parse_coptpy_version(coptpy, avail):
    if not avail:
        return
    coptpy_major = coptpy.COPT.VERSION_MAJOR
    coptpy_minor = coptpy.COPT.VERSION_MINOR
    coptpy_tech = coptpy.COPT.VERSION_TECHNICAL
    CoptDirect._version = (coptpy_major, coptpy_minor, coptpy_tech)
    CoptDirect._name = "COPT %s.%s.%s" % CoptDirect._version
    while len(CoptDirect._version) < 4:
        CoptDirect._version += (0,)
    CoptDirect._version = CoptDirect._version[:4]


coptpy, coptpy_available = attempt_import(
    'coptpy', catch_exceptions=(Exception,), callback=_parse_coptpy_version
)


@SolverFactory.register('copt_direct', doc='Direct python interface to COPT')
class CoptDirect(DirectSolver):
    _name = None
    _version = 0
    _coptenv = None

    def __init__(self, **kwds):
        if 'type' not in kwds:
            kwds['type'] = 'copt_direct'

        super(CoptDirect, self).__init__(**kwds)

        self._python_api_exists = True

        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()

        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        if coptpy_available and self._coptenv is None:
            self._coptenv = coptpy.Envr()
        self._coptmodel_name = "coptprob"
        self._solver_model = None

    def available(self, exception_flag=True):
        if not coptpy_available:
            if exception_flag:
                raise ApplicationError(
                    "No Python bindings available for %d solver plugin" % (type(self),)
                )
            return False
        else:
            return True

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()

        if not self._tee:
            self._solver_model.setParam('Logging', 0)

        if self._keepfiles:
            self._solver_model.setLogFile(self._log_file)
            print("Solver log file: " + self._log_file)

        for key, option in self.options.items():
            if key.lower() == "writeprob":
                self._solver_model.write(option)
            else:
                self._solver_model.setParam(key, option)

        self._solver_model.solve()

        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError(
                'CoptDirect does not support expressions of degree {0}.'.format(degree)
            )

        if len(repn.quadratic_vars) > 0:
            new_expr = coptpy.QuadExpr(0.0)
        else:
            new_expr = coptpy.LinExpr(0.0)

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr += coptpy.LinExpr(
                [self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars],
                repn.linear_coefs,
            )

        for i, v in enumerate(repn.quadratic_vars):
            x, y = v
            new_expr.addTerm(
                repn.quadratic_coefs[i],
                self._pyomo_var_to_solver_var_map[x],
                self._pyomo_var_to_solver_var_map[y],
            )
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
            copt_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                repn, max_degree
            )
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return copt_expr, referenced_vars

    def _copt_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -coptpy.COPT.INFINITY
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = +coptpy.COPT.INFINITY
        return lb, ub

    def _copt_vtype_from_var(self, var):
        if var.is_binary():
            vtype = coptpy.COPT.BINARY
        elif var.is_integer():
            vtype = coptpy.COPT.INTEGER
        elif var.is_continuous():
            vtype = coptpy.COPT.CONTINUOUS
        else:
            raise ValueError(
                'Variable domain type is not recognized for {0}'.format(var.domain)
            )
        return vtype

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._copt_vtype_from_var(var)
        lb, ub = self._copt_lb_ub_from_var(var)

        coptpy_var = self._solver_model.addVar(lb=lb, ub=ub, vtype=vtype, name=varname)

        self._pyomo_var_to_solver_var_map[var] = coptpy_var
        self._solver_var_to_pyomo_var_map[coptpy_var] = var
        self._referenced_variables[var] = 0

    def _add_constraint(self, con):
        if not con.active:
            return None
        if is_fixed(con.body):
            if self._skip_trivial_constraints:
                return None

        conname = self._symbol_map.getSymbol(con, self._labeler)

        if con._linear_canonical_form:
            copt_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(), self._max_constraint_degree
            )
        else:
            copt_expr, referenced_vars = self._get_expr_from_pyomo_expr(
                con.body, self._max_constraint_degree
            )

        if con.has_lb():
            if not is_fixed(con.lower):
                raise ValueError(
                    "Lower bound of constraint {0} is not constant.".format(con)
                )

        if con.has_ub():
            if not is_fixed(con.upper):
                raise ValueError(
                    "Upper bound of constraint {0} is not constant.".format(con)
                )

        if con.equality:
            coptpy_con = self._solver_model.addQConstr(
                copt_expr == value(con.lower), name=conname
            )
        elif con.has_lb() and con.has_ub():
            coptpy_con = self._solver_model.addBoundConstr(
                copt_expr, value(con.lower), value(con.upper), name=conname
            )
        elif con.has_lb():
            coptpy_con = self._solver_model.addQConstr(
                copt_expr >= value(con.lower), name=conname
            )
        elif con.has_ub():
            coptpy_con = self._solver_model.addQConstr(
                copt_expr <= value(con.upper), name=conname
            )
        else:
            raise ValueError(
                "Constraint does not has lower/upper bound: {0} \n".format(con)
            )

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars

        self._pyomo_con_to_solver_con_map[con] = coptpy_con
        self._solver_con_to_pyomo_con_map[coptpy_con] = con

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        self._symbol_map.getSymbol(con, self._labeler)

        level = con.level
        if level == 1:
            sos_type = coptpy.COPT.SOS_TYPE1
        elif level == 2:
            sos_type = coptpy.COPT.SOS_TYPE2
        else:
            raise ValueError(
                "Solver does not support SOS level {0} constraints".format(level)
            )

        copt_vars = []
        weights = []

        self._vars_referenced_by_con[con] = ComponentSet()

        if hasattr(con, 'get_items'):
            sos_items = list(con.get_items())
        else:
            sos_items = list(con.items())

        for v, w in sos_items:
            self._vars_referenced_by_con[con].add(v)
            copt_vars.append(self._pyomo_var_to_solver_var_map[v])
            self._referenced_variables[v] += 1
            weights.append(w)

        coptpy_con = self._solver_model.addSOS(sos_type, copt_vars, weights)
        self._pyomo_con_to_solver_con_map[con] = coptpy_con
        self._solver_con_to_pyomo_con_map[coptpy_con] = con

    def _set_objective(self, obj):
        if self._objective is not None:
            for var in self._vars_referenced_by_obj:
                self._referenced_variables[var] -= 1
            self._vars_referenced_by_obj = ComponentSet()
            self._objective = None

        if obj.active is False:
            raise ValueError('Cannot add inactive objective to solver.')

        if obj.sense == minimize:
            sense = coptpy.COPT.MINIMIZE
        elif obj.sense == maximize:
            sense = coptpy.COPT.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        copt_expr, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree
        )

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(copt_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

    def _add_block(self, block):
        DirectOrPersistentSolver._add_block(self, block)

    def _set_instance(self, model, kwds={}):
        DirectOrPersistentSolver._set_instance(self, model, kwds)

        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()

        if self._solver_model is not None:
            self._solver_model.clear()
            self._solver_model = None
        try:
            if model.name is not None:
                self._solver_model = self._coptenv.createModel(model.name)
                self._coptmodel_name = model.name
            else:
                self._solver_model = self._coptenv.createModel()
        except Exception:
            e = sys.exc_info()[1]
            msg = (
                "Unable to create COPT model. Have you installed the Python bindings for COPT?\n\n\t"
                + "Error message: {0}".format(e)
            )
            raise Exception(msg)

        self._add_block(model)

        for var, n_ref in self._referenced_variables.items():
            if n_ref != 0:
                if var.fixed:
                    if not self._output_fixed_variable_bounds:
                        raise ValueError(
                            "Encountered a fixed variable (%s) inside an active objective or constraint "
                            "expression on model %s, which is usually indicative of a preprocessing error."
                            "Use the IO-option 'output_fixed_variable_bounds=True' to suppress this error"
                            "and fix the variable by overwriting its bounds in the COPT instance."
                            % (var.name, self._pyomo_model.name)
                        )

    def _postsolve(self):
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
                raise RuntimeError(
                    "***The copt_direct solver plugin cannot extract solution suffix="
                    + suffix
                )

        if self._solver_model.ismip:
            # TODO: Fix getting slacks for MIP
            if extract_slacks:
                logger.warning("Cannot get slacks for MIP.")
                extract_slacks = False
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
                extract_reduced_costs = False
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
                extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = self._name
        self.results.solver.wallclock_time = self._solver_model.SolvingTime

        status = self._solver_model.status
        if status == coptpy.COPT.UNSTARTED:
            self.results.solver.status = SolverStatus.unknown
            self.results.solver.termination_message = "Model was not solved yet."
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.unknown
        elif status == coptpy.COPT.OPTIMAL:
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = (
                "Model was solved to optimality within tolerances."
            )
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == coptpy.COPT.INFEASIBLE:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be infeasible."
            )
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == coptpy.COPT.UNBOUNDED:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be unbounded."
            )
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == coptpy.COPT.INF_OR_UNB:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be infeasible or unbounded."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.infeasibleOrUnbounded
            )
            soln.status = SolutionStatus.unsure
        elif status == coptpy.COPT.NUMERICAL:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                "Optimization was terminated due to numerical difficulties."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == coptpy.COPT.NODELIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the node limit was reached."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxEvaluations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == coptpy.COPT.IMPRECISE:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = "Unable to satisfy optimality tolerances and returns a sub-optimal solution."
            self.results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.feasible
        elif status == coptpy.COPT.TIMEOUT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the time limit was reached."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxTimeLimit
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == coptpy.COPT.UNFINISHED:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                "Optimization was terminated unexpectedly."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == coptpy.COPT.INTERRUPTED:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization was terminated by the user."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.userInterrupt
            )
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                "Unknown COPT status " + "(" + str(status) + ")"
            )
            self.results.solver.termination_condition = TerminationCondition.error
            self.status = SolutionStatus.error

        self.results.problem.name = self._coptmodel_name

        if self._solver_model.objsense == coptpy.COPT.MINIMIZE:
            self.results.problem.sense = minimize
        elif self._solver_model.objsense == coptpy.COPT.MAXIMIZE:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError(
                'Unrecognized COPT objective sense: {0}'.format(
                    self._solver_model.objsense
                )
            )

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if self._solver_model.ismip == 0:
            self.results.problem.upper_bound = self._solver_model.lpobjval
            self.results.problem.lower_bound = self._solver_model.lpobjval
        elif self._solver_model.objsense == coptpy.COPT.MINIMIZE:
            self.results.problem.upper_bound = self._solver_model.objval
            self.results.problem.lower_bound = self._solver_model.bestbnd
        elif self._solver_model.objsense == coptpy.COPT.MAXIMIZE:
            self.results.problem.upper_bound = self._solver_model.bestbnd
            self.results.problem.lower_bound = self._solver_model.objval
        else:
            raise RuntimeError(
                'Unrecognized COPT objective sense: {0}'.format(
                    self._solver_model.objsense
                )
            )

        try:
            soln.gap = (
                self.results.problem.upper_bound - self.results.problem.lower_bound
            )
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = (
            self._solver_model.rows
            + self._solver_model.qconstrs
            + self._solver_model.soss
        )
        self.results.problem.number_of_nonzeros = self._solver_model.elems
        self.results.problem.number_of_variables = self._solver_model.cols
        self.results.problem.number_of_binary_variables = self._solver_model.bins
        self.results.problem.number_of_integer_variables = self._solver_model.ints
        self.results.problem.number_of_continuous_variables = (
            self._solver_model.cols - self._solver_model.ints - self._solver_model.bins
        )
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = (
            self._solver_model.haslpsol or self._solver_model.hasmipsol
        )

        if self._save_results:
            if self._solver_model.haslpsol or self._solver_model.hasmipsol:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                var_map = self._pyomo_var_to_solver_var_map
                vars_to_load = var_map.keys()
                copt_vars = [var_map[pyomo_var] for pyomo_var in vars_to_load]

                var_vals = self._solver_model.getInfo('Value', copt_vars)
                names = []
                for copt_var in copt_vars:
                    names.append(copt_var.name)
                for copt_var, val, name in zip(copt_vars, var_vals, names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[copt_var]
                    if self._referenced_variables[pyomo_var] > 0:
                        soln_variables[name] = {'Value': val}

                if extract_reduced_costs:
                    vals = self._solver_model.getInfo('RedCost', copt_vars)
                    for copt_var, val, name in zip(copt_vars, vals, names):
                        pyomo_var = self._solver_var_to_pyomo_var_map[copt_var]
                        if self._referenced_variables[pyomo_var] > 0:
                            soln_variables[name]['Rc'] = val

                if extract_duals or extract_slacks:
                    copt_cons = self._solver_model.getConstrs()
                    con_names = []
                    for copt_con in copt_cons:
                        con_names.append(copt_con.name)
                    for name in con_names:
                        soln_constraints[name] = {}

                    if self._solver_model.qconstrs > 0:
                        copt_q_cons = self._solver_model.getQConstrs()
                        q_con_names = []
                        for copt_q_con in copt_q_cons:
                            q_con_names.append(copt_q_con.name)
                        for name in q_con_names:
                            soln_constraints[name] = {}

                if extract_duals:
                    vals = self._solver_model.getInfo('Dual', copt_cons)
                    for val, name in zip(vals, con_names):
                        soln_constraints[name]['Dual'] = val
                    # TODO: Get duals for quadratic constraints

                if extract_slacks:
                    # NOTE: Slacks in COPT are activities of constraints
                    vals = self._solver_model.getInfo('Slack', copt_cons)
                    for val, name in zip(vals, con_names):
                        soln_constraints[name]['Slack'] = val

                    if self._solver_model.qconstrs > 0:
                        q_vals = self._solver_model.getInfo('Slack', copt_q_cons)
                        for val, name in zip(q_vals, q_con_names):
                            soln_constraints[name]['Slack'] = val
        elif self._load_solutions:
            if self._solver_model.haslpsol or self._solver_model.hasmipsol:
                self.load_vars()
                if extract_reduced_costs:
                    self._load_rc()
                if extract_duals:
                    self._load_duals()
                # TODO: Fix getting slacks for MIP
                if extract_slacks:
                    self._load_slacks()
        self.results.solution.insert(soln)

        TempfileManager.pop(remove=not self._keepfiles)

        return DirectOrPersistentSolver._postsolve(self)

    def warm_start_capable(self):
        return True

    def _warm_start(self):
        for pyomo_var, coptpy_var in self._pyomo_var_to_solver_var_map.items():
            if pyomo_var.value is not None:
                self._solver_model.setMipStart(coptpy_var, value(pyomo_var))
        self._solver_model.loadMipStart()

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables

        if vars_to_load is None:
            vars_to_load = var_map.keys()

        copt_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getInfo('Value', copt_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.set_value(val, skip_validation=True)

    def _load_rc(self, vars_to_load=None):
        if not hasattr(self._pyomo_model, 'rc'):
            self._pyomo_model.rc = Suffix(direction=Suffix.IMPORT)

        rc = self._pyomo_model.rc
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables

        if vars_to_load is None:
            vars_to_load = var_map.keys()

        copt_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getInfo('RedCost', copt_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                rc[var] = val

    def _load_duals(self, cons_to_load=None):
        # TODO: Dual solution for quadratic constraints are not available
        if not hasattr(self._pyomo_model, 'dual'):
            self._pyomo_model.dual = Suffix(direction=Suffix.IMPORT)

        dual = self._pyomo_model.dual
        con_map = self._pyomo_con_to_solver_con_map

        if cons_to_load is None:
            pyomo_cons_to_load = con_map.keys()
        else:
            pyomo_cons_to_load = cons_to_load

        for pyomo_con in pyomo_cons_to_load:
            dual[pyomo_con] = con_map[pyomo_con].dual

    def _load_slacks(self, cons_to_load=None):
        # NOTE: Slacks in COPT are activities of constraints
        if not hasattr(self._pyomo_model, 'slack'):
            self._pyomo_model.slack = Suffix(direction=Suffix.IMPORT)

        slack = self._pyomo_model.slack
        con_map = self._pyomo_con_to_solver_con_map

        if cons_to_load is None:
            pyomo_cons_to_load = con_map.keys()
        else:
            pyomo_cons_to_load = cons_to_load

        for pyomo_con in pyomo_cons_to_load:
            slack[pyomo_con] = con_map[pyomo_con].slack

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
        Load the values of the slack variables into the 'slack' suffix. The 'slack' suffix must live
        on the parent model.

        Parameters
        ----------
        cons_to_load: list of Constraint
        """
        self._load_slacks(cons_to_load)
