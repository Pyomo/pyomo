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


def _parse_gurobi_version(gurobipy, avail):
    if not avail:
        return
    GurobiDirect._version = gurobipy.gurobi.version()
    GurobiDirect._name = "Gurobi %s.%s%s" % GurobiDirect._version
    while len(GurobiDirect._version) < 4:
        GurobiDirect._version += (0,)
    GurobiDirect._version = GurobiDirect._version[:4]
    GurobiDirect._version_major = GurobiDirect._version[0]


gurobipy, gurobipy_available = attempt_import(
    'gurobipy',
    # Other forms of exceptions can be thrown by the gurobi python
    # import.  For example, a gurobipy.GurobiError exception is thrown
    # if all tokens for Gurobi are already in use; assuming, of course,
    # the license is a token license.  Unfortunately, you can't import
    # without a license, which means we can't explicitly test for that
    # exception!
    catch_exceptions=(Exception,),
    callback=_parse_gurobi_version,
)


def _set_options(model_or_env, options):
    # Set a parameters from the dictionary 'options' on the given gurobipy
    # model or environment.
    for key, option in options.items():
        # When options come from the pyomo command, all
        # values are string types, so we try to cast
        # them to a numeric value in the event that
        # setting the parameter fails.
        try:
            model_or_env.setParam(key, option)
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
            model_or_env.setParam(key, float(option))


@SolverFactory.register('gurobi_direct', doc='Direct python interface to Gurobi')
class GurobiDirect(DirectSolver):
    """A direct interface to Gurobi using gurobipy.

    :param manage_env: Set to True if this solver instance should create and
        manage its own Gurobi environment (defaults to False)
    :type manage_env: bool
    :param options: Dictionary of Gurobi parameters to set
    :type options: dict

    If ``manage_env`` is set to True, the ``GurobiDirect`` object creates a local
    Gurobi environment and manage all associated Gurobi resources. Importantly,
    this enables Gurobi licenses to be freed and connections terminated when the
    solver context is exited::

        with SolverFactory('gurobi', solver_io='python', manage_env=True) as opt:
            opt.solve(model)

        # All Gurobi models and environments are freed

    If ``manage_env`` is set to False (the default), the ``GurobiDirect`` object
    uses the global default Gurobi environment::

        with SolverFactory('gurobi', solver_io='python') as opt:
            opt.solve(model)

        # Only models created by `opt` are freed, the global default
        # environment remains active

    ``manage_env=True`` is required when setting license or connection parameters
    programmatically. The ``options`` argument is used to pass parameters to the
    Gurobi environment. For example, to connect to a Gurobi Cluster Manager::

        options = {
            "CSManager": "<url>",
            "CSAPIAccessID": "<access-id>",
            "CSAPISecret": "<api-key>",
        }
        with SolverFactory(
            'gurobi', solver_io='python', manage_env=True, options=options
        ) as opt:
            opt.solve(model)  # Model solved on compute server
        # Compute server connection terminated
    """

    _name = None
    _version = 0
    _version_major = 0
    _default_env_started = False

    def __init__(self, manage_env=False, **kwds):
        if 'type' not in kwds:
            kwds['type'] = 'gurobi_direct'
        super(GurobiDirect, self).__init__(**kwds)
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._needs_updated = True  # flag that indicates if solver_model.update() needs called before getting variable and constraint attributes
        self._callback = None
        self._callback_func = None

        self._python_api_exists = gurobipy_available
        self._range_constraints = set()

        self._max_obj_degree = 2
        self._max_constraint_degree = 2

        # Note: Undefined capabilities default to None
        self._capabilities.linear = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.integer = True
        self._capabilities.sos1 = True
        self._capabilities.sos2 = True

        # fix for compatibility with pre-5.0 Gurobi
        #
        # Note: Unfortunately, this will trigger the immediate import
        #    of the gurobipy module
        if gurobipy_available and GurobiDirect._version_major < 5:
            self._max_constraint_degree = 1
            self._capabilities.quadratic_constraint = False

        # remove the instance-level definition of the gurobi version:
        # because the version comes from an imported module, only one
        # version of gurobi is supported (and stored as a class attribute)
        del self._version

        self._manage_env = manage_env
        self._env = None
        self._env_options = None
        self._solver_model = None

    def available(self, exception_flag=True):
        """Returns True if the solver is available.

        :param exception_flag: If True, raise an exception instead of returning
            False if the solver is unavailable (defaults to False)
        :type exception_flag: bool

        In general, ``available()`` does not need to be called by the user, as
        the check is run automatically when solving a model. However it is useful
        for a simple retry loop when using a shared Gurobi license::

            with SolverFactory('gurobi', solver_io='python') as opt:
                while not available(exception_flag=False):
                    time.sleep(1)
                opt.solve(model)

        """
        # First check gurobipy is imported
        if not gurobipy_available:
            if exception_flag:
                gurobipy.log_import_warning(logger=__name__)
                raise ApplicationError(
                    "No Python bindings available for %s solver plugin" % (type(self),)
                )
            return False

        # Ensure environment is started to check for a valid license
        with capture_output(capture_fd=True) as OUT:
            try:
                self._init_env()
                return True
            except gurobipy.GurobiError as e:
                msg = "Could not create Model - gurobi message=%s\n" % (e,)
        if OUT.getvalue():
            msg += "\n" + OUT.getvalue()
        # Didn't return, so environment start failed
        if exception_flag:
            logger.warning(msg)
            raise ApplicationError(
                "Could not create Model for %s solver plugin - gurobi message=%s"
                % (type(self), msg)
            )
        else:
            return False

    def _apply_solver(self):
        StaleFlagManager.mark_all_as_stale()

        if self._tee:
            self._solver_model.setParam('OutputFlag', 1)
        else:
            self._solver_model.setParam('OutputFlag', 0)

        if self._keepfiles:
            # Only save log file when the user wants to keep it.
            self._solver_model.setParam('LogFile', self._log_file)
            print("Solver log file: " + self._log_file)

        # Only pass along changed parameters to the model
        if self._env_options:
            new_options = {
                key: option
                for key, option in self.options.items()
                if key not in self._env_options or self._env_options[key] != option
            }
        else:
            new_options = self.options
        _set_options(self._solver_model, new_options)

        if self._version_major >= 5:
            for suffix in self._suffixes:
                if re.match(suffix, "dual"):
                    self._solver_model.setParam(gurobipy.GRB.Param.QCPDual, 1)

        self._solver_model.optimize(self._callback)
        self._needs_updated = False

        if self._keepfiles:
            # Change LogFile to make Gurobi close the original log file.
            # May not work for all Gurobi versions, like ver. 9.5.0.
            self._solver_model.setParam('LogFile', 'default')

        # FIXME: can we get a return code indicating if Gurobi had a significant failure?
        return Bunch(rc=None, log=None)

    def _get_expr_from_pyomo_repn(self, repn, max_degree=2):
        referenced_vars = ComponentSet()

        degree = repn.polynomial_degree()
        if (degree is None) or (degree > max_degree):
            raise DegreeError(
                'GurobiDirect does not support expressions of degree {0}.'.format(
                    degree
                )
            )

        if len(repn.linear_vars) > 0:
            referenced_vars.update(repn.linear_vars)
            new_expr = gurobipy.LinExpr(
                repn.linear_coefs,
                [self._pyomo_var_to_solver_var_map[i] for i in repn.linear_vars],
            )
        else:
            new_expr = 0.0

        for i, v in enumerate(repn.quadratic_vars):
            x, y = v
            new_expr += (
                repn.quadratic_coefs[i]
                * self._pyomo_var_to_solver_var_map[x]
                * self._pyomo_var_to_solver_var_map[y]
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
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                repn, max_degree
            )
        except DegreeError as e:
            msg = e.args[0]
            msg += '\nexpr: {0}'.format(expr)
            raise DegreeError(msg)

        return gurobi_expr, referenced_vars

    def _gurobi_lb_ub_from_var(self, var):
        if var.is_fixed():
            val = var.value
            return val, val
        if var.has_lb():
            lb = value(var.lb)
        else:
            lb = -gurobipy.GRB.INFINITY
        if var.has_ub():
            ub = value(var.ub)
        else:
            ub = gurobipy.GRB.INFINITY
        return lb, ub

    def _add_var(self, var):
        varname = self._symbol_map.getSymbol(var, self._labeler)
        vtype = self._gurobi_vtype_from_var(var)
        lb, ub = self._gurobi_lb_ub_from_var(var)

        gurobipy_var = self._solver_model.addVar(
            lb=lb, ub=ub, vtype=vtype, name=varname
        )

        self._pyomo_var_to_solver_var_map[var] = gurobipy_var
        self._solver_var_to_pyomo_var_map[gurobipy_var] = var
        self._referenced_variables[var] = 0

        self._needs_updated = True

    def close_global(self):
        """Frees all Gurobi models used by this solver, and frees the global
        default Gurobi environment.

        The default environment is used by all ``GurobiDirect`` solvers started
        with ``manage_env=False`` (the default). To guarantee that all Gurobi
        resources are freed, all instantiated ``GurobiDirect`` solvers must also
        be correctly closed.

        The following example will free all Gurobi resources assuming the user did
        not create any other models (e.g. via another ``GurobiDirect`` object with
        ``manage_env=False``)::

            opt = SolverFactory('gurobi', solver_io='python')
            try:
                opt.solve(model)
            finally:
                opt.close_global()
            # All Gurobi models created by `opt` are freed and the default
            # Gurobi environment is closed
        """
        self.close()
        with capture_output(capture_fd=True):
            gurobipy.disposeDefaultEnv()
        GurobiDirect._default_env_started = False

    def _init_env(self):
        if self._manage_env:
            # Ensure an environment is active for this instance
            if self._env is None:
                assert self._solver_model is None
                env = gurobipy.Env(empty=True)
                _set_options(env, self.options)
                env.start()
                # Successful start (no errors): store the environment
                self._env = env
                self._env_options = dict(self.options)
        else:
            # Ensure the (global) default env is started
            if not GurobiDirect._default_env_started:
                m = gurobipy.Model()
                m.close()
                GurobiDirect._default_env_started = True

    def _create_model(self, model):
        self._init_env()
        if self._solver_model is not None:
            self._solver_model.close()
        if model.name is not None:
            self._solver_model = gurobipy.Model(model.name, env=self._env)
        else:
            self._solver_model = gurobipy.Model(env=self._env)

    def close(self):
        """Frees local Gurobi resources used by this solver instance.

        All Gurobi models created by the solver are freed. If the solver was
        created with ``manage_env=True``, this method also closes the Gurobi
        environment used by this solver instance. Calling ``.close()`` achieves
        the same result as exiting the solver context (although using context
        managers is preferred where possible)::

            opt = SolverFactory('gurobi', solver_io='python', manage_env=True)
            try:
                opt.solve(model)
            finally:
                opt.close()
            # Gurobi models and environments created by `opt` are freed

        As with the context manager, if ``manage_env=False`` (the default) was
        used, only the Gurobi models created by this solver are freed. The
        default global Gurobi environment will still be active::

            opt = SolverFactory('gurobi', solver_io='python')
            try:
                opt.solve(model)
            finally:
                opt.close()
            # Gurobi models created by `opt` are freed; however the
            # default/global Gurobi environment is still active
        """

        if self._solver_model is not None:
            self._solver_model.close()
            self._solver_model = None
        if self._manage_env:
            if self._env is not None:
                self._env.close()
                self._env = None
                self._env_options = None

    def __exit__(self, t, v, traceback):
        super().__exit__(t, v, traceback)
        self.close()

    def _set_instance(self, model, kwds={}):
        self._range_constraints = set()
        DirectOrPersistentSolver._set_instance(self, model, kwds)
        self._pyomo_con_to_solver_con_map = dict()
        self._solver_con_to_pyomo_con_map = ComponentMap()
        self._pyomo_var_to_solver_var_map = ComponentMap()
        self._solver_var_to_pyomo_var_map = ComponentMap()
        try:
            self._create_model(model)
        except Exception:
            e = sys.exc_info()[1]
            msg = (
                "Unable to create Gurobi model. "
                "Have you installed the Python "
                "bindings for Gurobi?\n\n\t" + "Error message: {0}".format(e)
            )
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
                            % (var.name, self._pyomo_model.name)
                        )

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
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(
                con.canonical_form(), self._max_constraint_degree
            )
        # elif isinstance(con, LinearCanonicalRepn):
        #    gurobi_expr, referenced_vars = self._get_expr_from_pyomo_repn(
        #        con,
        #        self._max_constraint_degree)
        else:
            gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(
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
            gurobipy_con = self._solver_model.addConstr(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.EQUAL,
                rhs=value(con.lower),
                name=conname,
            )
        elif con.has_lb() and con.has_ub():
            gurobipy_con = self._solver_model.addRange(
                gurobi_expr, value(con.lower), value(con.upper), name=conname
            )
            self._range_constraints.add(con)
        elif con.has_lb():
            gurobipy_con = self._solver_model.addConstr(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.GREATER_EQUAL,
                rhs=value(con.lower),
                name=conname,
            )
        elif con.has_ub():
            gurobipy_con = self._solver_model.addConstr(
                lhs=gurobi_expr,
                sense=gurobipy.GRB.LESS_EQUAL,
                rhs=value(con.upper),
                name=conname,
            )
        else:
            raise ValueError(
                "Constraint does not have a lower "
                "or an upper bound: {0} \n".format(con)
            )

        for var in referenced_vars:
            self._referenced_variables[var] += 1
        self._vars_referenced_by_con[con] = referenced_vars
        self._pyomo_con_to_solver_con_map[con] = gurobipy_con
        self._solver_con_to_pyomo_con_map[gurobipy_con] = con

        self._needs_updated = True

    def _add_sos_constraint(self, con):
        if not con.active:
            return None

        conname = self._symbol_map.getSymbol(con, self._labeler)
        level = con.level
        if level == 1:
            sos_type = gurobipy.GRB.SOS_TYPE1
        elif level == 2:
            sos_type = gurobipy.GRB.SOS_TYPE2
        else:
            raise ValueError(
                "Solver does not support SOS level {0} constraints".format(level)
            )

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

        self._needs_updated = True

    def _gurobi_vtype_from_var(self, var):
        """
        This function takes a pyomo variable and returns the appropriate gurobi variable type
        :param var: pyomo.core.base.var.Var
        :return: gurobipy.GRB.CONTINUOUS or gurobipy.GRB.BINARY or gurobipy.GRB.INTEGER
        """
        if var.is_binary():
            vtype = gurobipy.GRB.BINARY
        elif var.is_integer():
            vtype = gurobipy.GRB.INTEGER
        elif var.is_continuous():
            vtype = gurobipy.GRB.CONTINUOUS
        else:
            raise ValueError(
                'Variable domain type is not recognized for {0}'.format(var.domain)
            )
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
            sense = gurobipy.GRB.MINIMIZE
        elif obj.sense == maximize:
            sense = gurobipy.GRB.MAXIMIZE
        else:
            raise ValueError('Objective sense is not recognized: {0}'.format(obj.sense))

        gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(
            obj.expr, self._max_obj_degree
        )

        for var in referenced_vars:
            self._referenced_variables[var] += 1

        self._solver_model.setObjective(gurobi_expr, sense=sense)
        self._objective = obj
        self._vars_referenced_by_obj = referenced_vars

        self._needs_updated = True

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
                raise RuntimeError(
                    "***The gurobi_direct solver plugin cannot extract solution suffix="
                    + suffix
                )

        gprob = self._solver_model
        grb = gurobipy.GRB
        status = gprob.Status

        if gprob.getAttr(gurobipy.GRB.Attr.IsMIP):
            if extract_reduced_costs:
                logger.warning("Cannot get reduced costs for MIP.")
            if extract_duals:
                logger.warning("Cannot get duals for MIP.")
            extract_reduced_costs = False
            extract_duals = False

        self.results = SolverResults()
        soln = Solution()

        self.results.solver.name = GurobiDirect._name
        self.results.solver.wallclock_time = gprob.Runtime

        if status == grb.LOADED:  # problem is loaded, but no solution
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Model is loaded, but no solution information is available."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.unknown
        elif status == grb.OPTIMAL:  # optimal
            self.results.solver.status = SolverStatus.ok
            self.results.solver.termination_message = (
                "Model was solved to optimality (subject to tolerances), "
                "and an optimal solution is available."
            )
            self.results.solver.termination_condition = TerminationCondition.optimal
            soln.status = SolutionStatus.optimal
        elif status == grb.INFEASIBLE:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be infeasible"
            )
            self.results.solver.termination_condition = TerminationCondition.infeasible
            soln.status = SolutionStatus.infeasible
        elif status == grb.INF_OR_UNBD:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Problem proven to be infeasible or unbounded."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.infeasibleOrUnbounded
            )
            soln.status = SolutionStatus.unsure
        elif status == grb.UNBOUNDED:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Model was proven to be unbounded."
            )
            self.results.solver.termination_condition = TerminationCondition.unbounded
            soln.status = SolutionStatus.unbounded
        elif status == grb.CUTOFF:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimal objective for model was proven to be worse than the "
                "value specified in the Cutoff parameter. No solution "
                "information is available."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.minFunctionValue
            )
            soln.status = SolutionStatus.unknown
        elif status == grb.ITERATION_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the total number of simplex "
                "iterations performed exceeded the value specified in the "
                "IterationLimit parameter."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxIterations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.NODE_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the total number of "
                "branch-and-cut nodes explored exceeded the value specified "
                "in the NodeLimit parameter"
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxEvaluations
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.TIME_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the time expended exceeded "
                "the value specified in the TimeLimit parameter."
            )
            self.results.solver.termination_condition = (
                TerminationCondition.maxTimeLimit
            )
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.SOLUTION_LIMIT:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization terminated because the number of solutions found "
                "reached the value specified in the SolutionLimit parameter."
            )
            self.results.solver.termination_condition = TerminationCondition.unknown
            soln.status = SolutionStatus.stoppedByLimit
        elif status == grb.INTERRUPTED:
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "Optimization was terminated by the user."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == grb.NUMERIC:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                "Optimization was terminated due to unrecoverable numerical "
                "difficulties."
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error
        elif status == grb.SUBOPTIMAL:
            self.results.solver.status = SolverStatus.warning
            self.results.solver.termination_message = (
                "Unable to satisfy optimality tolerances; a sub-optimal "
                "solution is available."
            )
            self.results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.feasible
        # note that USER_OBJ_LIMIT was added in Gurobi 7.0, so it may not be present
        elif (status is not None) and (status == getattr(grb, 'USER_OBJ_LIMIT', None)):
            self.results.solver.status = SolverStatus.aborted
            self.results.solver.termination_message = (
                "User specified an objective limit "
                "(a bound on either the best objective "
                "or the best bound), and that limit has "
                "been reached. Solution is available."
            )
            self.results.solver.termination_condition = TerminationCondition.other
            soln.status = SolutionStatus.stoppedByLimit
        else:
            self.results.solver.status = SolverStatus.error
            self.results.solver.termination_message = (
                "Unhandled Gurobi solve status (" + str(status) + ")"
            )
            self.results.solver.termination_condition = TerminationCondition.error
            soln.status = SolutionStatus.error

        self.results.problem.name = gprob.ModelName

        if gprob.ModelSense == 1:
            self.results.problem.sense = minimize
        elif gprob.ModelSense == -1:
            self.results.problem.sense = maximize
        else:
            raise RuntimeError(
                'Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense)
            )

        self.results.problem.upper_bound = None
        self.results.problem.lower_bound = None
        if (gprob.NumBinVars + gprob.NumIntVars) == 0:
            try:
                self.results.problem.upper_bound = gprob.ObjVal
                self.results.problem.lower_bound = gprob.ObjVal
            except (gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == 1:  # minimizing
            try:
                self.results.problem.upper_bound = gprob.ObjVal
            except (gurobipy.GurobiError, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                pass
        elif gprob.ModelSense == -1:  # maximizing
            try:
                self.results.problem.upper_bound = gprob.ObjBound
            except (gurobipy.GurobiError, AttributeError):
                pass
            try:
                self.results.problem.lower_bound = gprob.ObjVal
            except (gurobipy.GurobiError, AttributeError):
                pass
        else:
            raise RuntimeError(
                'Unrecognized gurobi objective sense: {0}'.format(gprob.ModelSense)
            )

        try:
            soln.gap = (
                self.results.problem.upper_bound - self.results.problem.lower_bound
            )
        except TypeError:
            soln.gap = None

        self.results.problem.number_of_constraints = (
            gprob.NumConstrs + gprob.NumQConstrs + gprob.NumSOS
        )
        self.results.problem.number_of_nonzeros = gprob.NumNZs
        self.results.problem.number_of_variables = gprob.NumVars
        self.results.problem.number_of_binary_variables = gprob.NumBinVars
        self.results.problem.number_of_integer_variables = gprob.NumIntVars
        self.results.problem.number_of_continuous_variables = (
            gprob.NumVars - gprob.NumIntVars - gprob.NumBinVars
        )
        self.results.problem.number_of_objectives = 1
        self.results.problem.number_of_solutions = gprob.SolCount

        # if a solve was stopped by a limit, we still need to check to
        # see if there is a solution available - this may not always
        # be the case, both in LP and MIP contexts.
        if self._save_results:
            """
            This code in this if statement is only needed for backwards compatibility. It is more efficient to set
            _save_results to False and use load_vars, load_duals, etc.
            """
            if gprob.SolCount > 0:
                soln_variables = soln.variable
                soln_constraints = soln.constraint

                gurobi_vars = self._solver_model.getVars()
                gurobi_vars = list(
                    set(gurobi_vars).intersection(
                        set(self._pyomo_var_to_solver_var_map.values())
                    )
                )
                var_vals = self._solver_model.getAttr("X", gurobi_vars)
                names = self._solver_model.getAttr("VarName", gurobi_vars)
                for gurobi_var, val, name in zip(gurobi_vars, var_vals, names):
                    pyomo_var = self._solver_var_to_pyomo_var_map[gurobi_var]
                    if self._referenced_variables[pyomo_var] > 0:
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
                        q_con_names = self._solver_model.getAttr(
                            "QCName", gurobi_q_cons
                        )
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
                    gurobi_range_con_vars = set(self._solver_model.getVars()) - set(
                        self._pyomo_var_to_solver_var_map.values()
                    )
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
                self.load_vars()

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
                gurobipy_var.setAttr(gurobipy.GRB.Attr.Start, value(pyomo_var))
        self._needs_updated = True

    def _load_vars(self, vars_to_load=None):
        var_map = self._pyomo_var_to_solver_var_map
        ref_vars = self._referenced_variables
        if vars_to_load is None:
            vars_to_load = var_map.keys()

        gurobi_vars_to_load = [var_map[pyomo_var] for pyomo_var in vars_to_load]
        vals = self._solver_model.getAttr("X", gurobi_vars_to_load)

        for var, val in zip(vars_to_load, vals):
            if ref_vars[var] > 0:
                var.set_value(val, skip_validation=True)

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
            gurobi_cons_to_load = set(
                [con_map[pyomo_con] for pyomo_con in cons_to_load]
            )
            linear_cons_to_load = gurobi_cons_to_load.intersection(
                set(self._solver_model.getConstrs())
            )
            if self._version_major >= 5:
                quadratic_cons_to_load = gurobi_cons_to_load.intersection(
                    set(self._solver_model.getQConstrs())
                )
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

        gurobi_range_con_vars = set(self._solver_model.getVars()) - set(
            self._pyomo_var_to_solver_var_map.values()
        )

        if cons_to_load is None:
            linear_cons_to_load = self._solver_model.getConstrs()
            if self._version_major >= 5:
                quadratic_cons_to_load = self._solver_model.getQConstrs()
        else:
            gurobi_cons_to_load = set(
                [con_map[pyomo_con] for pyomo_con in cons_to_load]
            )
            linear_cons_to_load = gurobi_cons_to_load.intersection(
                set(self._solver_model.getConstrs())
            )
            if self._version_major >= 5:
                quadratic_cons_to_load = gurobi_cons_to_load.intersection(
                    set(self._solver_model.getQConstrs())
                )
        linear_vals = self._solver_model.getAttr("Slack", linear_cons_to_load)
        if self._version_major >= 5:
            quadratic_vals = self._solver_model.getAttr(
                "QCSlack", quadratic_cons_to_load
            )

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

    def _update(self):
        self._solver_model.update()
        self._needs_updated = False
