# -*- coding: utf-8 -*-
"""Decomposition solver for Generalized Disjunctive Programming (GDP) problems.

The GDPopt (Generalized Disjunctive Programming optimizer) solver applies a
variety of decomposition-based approaches to solve Generalized Disjunctive
Programming (GDP) problems. GDP models can include nonlinear, continuous
variables and constraints, as well as logical conditions.

These approaches include:

- Outer approximation
- Partial surrogate cuts [pending]
- Generalized Bender decomposition [pending]

This solver implementation was developed by Carnegie Mellon University in the
research group of Ignacio Grossmann.

For nonconvex problems, the bounds self.LB and self.UB may not be rigorous.

Questions: Please make a post at StackOverflow and/or contact Qi Chen
<https://github.com/qtothec>.

"""
import logging
from copy import deepcopy
from math import copysign, fabs

from six import iteritems

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base import (Block, Constraint, ConstraintList, Expression,
                             Objective, Set, Suffix, TransformationFactory,
                             Var, maximize, minimize, value)
from pyomo.core.base.block import generate_cuid_names
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import (ComponentMap, ComponentSet, NonNegativeReals,
                               Reals)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import ProblemSense, SolverResults

logger = logging.getLogger('pyomo.contrib.gdpopt')

__version__ = (0, 1, 0)


class GDPoptSolver(pyomo.util.plugin.Plugin):
    """A decomposition-based GDP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('gdpopt',
                            doc='The GDPopt decomposition-based GDP solver')

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    def solve(self, model, **kwds):
        """Solve the model.

        Warning: this solver is still in beta. Keyword arguments subject to
        change. Undocumented keyword arguments definitely subject to change.

        Args:
            model (Block): a Pyomo model or block to be solved

        Kwargs:
            tol (float): bound tolerance
            iterlim (int): maximum number of master iterations
            strategy (str): decomposition strategy to use. Possible values:
                LOA, LGBD
            init_strategy (str): initialization strategy to use when generating
                the initial cuts to construct the master problem.
            max_slack (float): upper bound on slack variable values
            OA_penalty (float): multiplier on objective penalization for slack
                variables.
            nlp (str): Solver to use for nonlinear subproblems
            nlp_kwargs (dict): Keyword arguments to pass to NLP solver
            mip (str): Solver to use for linear discrete problems
            mip_kwargs (dict): Keyword arguments to pass to MIP solver
            solve_in_place (bool): If true, GDPopt manipulations are performed
                directly upon the model. Otherwise, the model is first copied
                and solution values are copied over afterwards.
            master_postsolve (func): callback hook after a solution of the
                master problem
            subprob_presolve (func): callback hook before calling the
                subproblem solver
            subprob_postsolve (func): callback hook after a solution of the
                nonlinear subproblem
            subprob_postfeas (func): callback hook after feasible solution of
                the nonlinear subproblem

        """
        solve_data = GDPoptSolveData()
        solve_data.bound_tolerance = kwds.pop('tol', 1E-6)
        solve_data.iteration_limit = kwds.pop('iterlim', 30)
        solve_data.decomposition_strategy = kwds.pop('strategy', 'LOA')
        solve_data.initialization_strategy = kwds.pop('init_strategy', None)
        solve_data.custom_init_disjuncts = kwds.pop(
            'custom_init_disjuncts', None)
        solve_data.max_slack = kwds.pop('max_slack', 1000)
        solve_data.OA_penalty_factor = kwds.pop('OA_penalty', 1000)
        solve_data.nlp_solver_name = kwds.pop('nlp', 'ipopt')
        solve_data.nlp_solver_kwargs = kwds.pop('nlp_kwargs', {})
        solve_data.mip_solver_name = kwds.pop('mip', 'gurobi')
        solve_data.mip_solver_kwargs = kwds.pop('mip_kwargs', {})
        solve_data.modify_in_place = kwds.pop('solve_in_place', True)
        solve_data.master_postsolve = kwds.pop(
            'master_postsolve', _DoNothing())
        solve_data.subproblem_presolve = kwds.pop(
            'subprob_presolve', _DoNothing())
        solve_data.subproblem_postsolve = kwds.pop(
            'subprob_postsolve', _DoNothing())
        solve_data.subproblem_postfeasible = kwds.pop(
            'subprob_postfeas', _DoNothing())
        solve_data.algorithm_stall_after = kwds.pop('algorithm_stall_after', 2)
        solve_data.tee = kwds.pop('tee', False)

        if solve_data.tee:
            old_logger_level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)

        if kwds:
            logger.warn("Unrecognized arguments passed to GDPopt solver: %s"
                        % (kwds,))

        # Verify that decomposition strategy chosen is one of the supported
        # strategies
        valid_strategies = ['LOA']
        if solve_data.decomposition_strategy not in valid_strategies:
            raise ValueError('Unrecognized decomposition strategy %s. '
                             'Valid strategies include: %s'
                             % (solve_data.decomposition_strategy,
                                valid_strategies))

        # To accommodate multi-phase hybrid strategies, define an attribute to
        # track the current phase.
        if solve_data.decomposition_strategy == 'hPSC':
            solve_data.current_strategy = 'PSC'
        else:
            solve_data.current_strategy = solve_data.decomposition_strategy

        # When generating cuts, small duals multiplied by expressions can cause
        # problems. Exclude all duals smaller in absolue value than the
        # following.
        solve_data.small_dual_tolerance = 1E-8
        solve_data.integer_tolerance = 1E-5

        solve_data.round_NLP_binaries = True
        """bool: flag to round binary values to exactly 0 or 1.
        Rounding is done before solving NLP subproblem"""

        # Modify in place decides whether to run the algorithm on a copy of the
        # originally model passed to the solver, or whether to manipulate the
        # original model directly.
        solve_data.original_model = model
        if solve_data.modify_in_place:
            solve_data.working_model = m = model
        else:
            # print('Clone for working model')
            solve_data.working_model = m = model.clone()

        # Store the initial model state as the best solution found. If we find
        # no better solution, then we will restore from this copy.
        # print('Initial clone for best_solution_found')
        solve_data.best_solution_found = model.clone()

        # Save model initial values. These are used later to initialize NLP
        # subproblems.
        model_obj_to_cuid = generate_cuid_names(
            model, ctype=(Var, Disjunct), descend_into=(Block, Disjunct))
        model_cuid_to_obj = dict((cuid, obj)
                                 for obj, cuid in iteritems(model_obj_to_cuid))
        solve_data.initial_variable_values = dict(
            (model_obj_to_cuid[v], value(v, exception=False))
            for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct)))

        # Create a model block on which to store GDPopt-specific utility
        # modeling objects.
        GDPopt = m.GDPopt_utils = Block()

        # Create the solver results object
        res = solve_data.results = SolverResults()
        res.problem.name = m.name
        res.problem.number_of_nonzeros = None  # TODO
        res.solver.name = 'GDPopt ' + str(self.version())
        # TODO work on termination condition and message
        res.solver.termination_condition = None
        res.solver.message = None
        # TODO add some kind of timing
        res.solver.user_time = None
        res.solver.system_time = None
        res.solver.wallclock_time = None
        res.solver.termination_message = None

        # Validate the model to ensure that GDPopt is able to solve it.
        #
        # This needs to take place before the detection of nonlinear
        # constraints, because if the objective is nonlinear, it will be moved
        # to the constraints.
        assert(not hasattr(solve_data, 'nonlinear_constraints'))
        self._validate_model(solve_data)

        # Maps in order to keep track of certain generated constraints
        GDPopt.oa_cut_map = Suffix(direction=Suffix.LOCAL, datatype=None)

        # Integer cuts exclude particular discrete decisions
        GDPopt.integer_cuts = ConstraintList(doc='integer cuts')
        # Feasible integer cuts exclude discrete realizations that have been
        # explored via an NLP subproblem. Depending on model characteristics,
        # the user may wish to revisit NLP subproblems (with a different
        # initialization, for example). Therefore, these cuts are not enabled
        # by default, unless the initial model has no discrete decisions.
        #
        # Note: these cuts will only exclude integer realizations that are not
        # already in the primary GDPopt_integer_cuts ConstraintList.
        GDPopt.feasible_integer_cuts = ConstraintList(
            doc='explored integer cuts')
        if not solve_data._no_discrete_decisions:
            GDPopt.feasible_integer_cuts.deactivate()

        # # Build a list of binary variables
        # self.binary_vars = [v for v in m.component_data_objects(
        #     ctype=Var, descend_into=True)
        #     if v.is_binary() and not v.fixed]
        #
        # Build list of nonlinear constraints
        solve_data.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if v.body.polynomial_degree() not in (0, 1)]

        # Set up iteration counters
        solve_data.nlp_iter = 0
        solve_data.mip_iter = 0
        solve_data.mip_subiter = 0

        # set up bounds
        solve_data.LB = float('-inf')
        solve_data.UB = float('inf')
        solve_data.LB_progress = [solve_data.LB]
        solve_data.UB_progress = [solve_data.UB]

        # Flag indicating whether the solution improved in the past iteration
        # or not
        solve_data.solution_improved = False

        # Set up solvers
        solve_data.nlp_solver = SolverFactory(solve_data.nlp_solver_name)
        solve_data.mip_solver = SolverFactory(solve_data.mip_solver_name)

        # Initialize the master problem
        self._GDPopt_initialize_master(solve_data)

        # Algorithm main loop
        self._GDPopt_iteration_loop(solve_data)

        # Update values in original model
        self._copy_values(solve_data.best_solution_found, model,
                          to_map=model_cuid_to_obj)

        solve_data.results.problem.lower_bound = solve_data.LB
        solve_data.results.problem.upper_bound = solve_data.UB

        if solve_data.tee:
            logger.setLevel(old_logger_level)

    def _copy_values(self, from_model, to_model, from_map=None, to_map=None):
        """Copy variable values from one model to another.

        from_map: a mapping of source model objects to uid names
        to_map: a mapping of uid names to destination model objects
        """
        if from_map is None:
            from_map = generate_cuid_names(from_model,
                                           ctype=(Var, Disjunct),
                                           descend_into=(Block, Disjunct))
        if to_map is None:
            tm_obj_to_uid = generate_cuid_names(to_model,
                                                ctype=(Var, Disjunct),
                                                descend_into=(Block, Disjunct))
            to_map = dict((cuid, obj)
                          for obj, cuid in iteritems(tm_obj_to_uid))
        for v in from_model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct)):
            uid = from_map[v]
            dest_model_var = to_map.get(uid, None)
            if dest_model_var is not None:
                try:
                    dest_model_var.set_value(value(v))
                except ValueError as err:
                    if 'is not in domain Binary' in err.message:
                        # check to see whether this is just a tolerance
                        # issue
                        if (fabs(value(v) - 1) <= self.integer_tolerance or
                                fabs(value(v)) <= self.integer_tolerance):
                            dest_model_var.set_value(round(value(v)))
                        else:
                            raise
                    elif 'No value for uninitialized' in err.message:
                        # Variable value was None
                        dest_model_var.set_value(None)

    def _copy_dual_suffixes(self, from_model, to_model,
                            from_map=None, to_map=None):
        """Copy suffix values from one model to another."""
        self._copy_suffix(from_model.dual, to_model.dual,
                          from_map=from_map, to_map=to_map)
        if hasattr(from_model, 'ipopt_zL_out'):
            self._copy_suffix(from_model.ipopt_zL_out, to_model.ipopt_zL_out,
                              from_map=from_map, to_map=to_map)
        if hasattr(from_model, 'ipopt_zU_out'):
            self._copy_suffix(from_model.ipopt_zU_out, to_model.ipopt_zU_out,
                              from_map=from_map, to_map=to_map)

    def _copy_suffix(self, from_suffix, to_suffix, from_map=None, to_map=None):
        """Copy suffix values from one model to another."""
        if from_map is None:
            from_map = generate_cuid_names(from_suffix.model(),
                                           ctype=(Var, Constraint, Disjunct),
                                           descend_into=(Block, Disjunct))
        if to_map is None:
            tm_obj_to_uid = generate_cuid_names(
                to_suffix.model(), ctype=(Var, Constraint, Disjunct),
                descend_into=(Block, Disjunct))
            to_map = dict((cuid, obj)
                          for obj, cuid in iteritems(tm_obj_to_uid))

        to_suffix.clear()
        for model_obj in from_suffix:
            to_model_obj = to_map[from_map[model_obj]]
            to_suffix[to_model_obj] = from_suffix[model_obj]

    def _validate_model(self, solve_data):
        """Validate that the model is solveable by GDPopt.

        Also populates results object with problem information.

        """
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils

        # Get count of constraints and variables
        solve_data.results.problem.number_of_constraints = 0
        var_set = ComponentSet()
        binary_var_set = ComponentSet()
        integer_var_set = ComponentSet()
        continuous_var_set = ComponentSet()
        for constr in m.component_data_objects(ctype=Constraint, active=True,
                                               descend_into=(Disjunct, Block)):
            solve_data.results.problem.number_of_constraints += 1
            for v in EXPR.identify_variables(constr.body, include_fixed=False):
                var_set.add(v)
                if v.is_binary():
                    binary_var_set.add(v)
                elif v.is_integer():
                    integer_var_set.add(v)
                elif v.is_continuous():
                    continuous_var_set.add(v)
                else:
                    raise TypeError('Variable {0} has unknown domain of {1}'.
                                    format(v.name, v.domain))

        active_disjunctions = ComponentSet(
            disj for disj in m.component_data_objects(
                ctype=Disjunction, active=True,
                descend_into=(Disjunct, Block)))
        solve_data.results.problem.number_of_disjunctions = \
            len(active_disjunctions)

        solve_data.results.problem.number_of_variables = len(var_set)
        solve_data.results.problem.number_of_binary_variables = \
            len(binary_var_set)
        solve_data.results.problem.number_of_integer_variables = \
            len(integer_var_set)
        solve_data.results.problem.number_of_continuous_variables = \
            len(continuous_var_set)

        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, active=True, descend_into=(Block, Disjunct))
                if v.is_integer() and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'GDPopt does not currently support solution of '
                             'such problems.')
            # TODO add in the reformulation using base 2

        # Handle LP/NLP being passed to the solver
        if len(binary_var_set) == 0 and len(active_disjunctions) == 0:
            logger.info('Problem has no discrete decisions.')
            solve_data._no_discrete_decisions = True
        else:
            solve_data._no_discrete_decisions = False

        # Handle missing or multiple objectives
        objs = list(m.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        num_objs = len(objs)
        solve_data.results.problem.number_of_objectives = num_objs
        if num_objs == 0:
            logger.warning(
                'Model has no active objectives. Adding dummy objective.')
            GDPopt.dummy_objective = Objective(expr=1)
            main_obj = GDPopt.dummy_objective
        elif num_objs > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = objs[0]

        # Move the objective to the constraints
        GDPopt.objective_value = Var(domain=Reals, initialize=0)
        if main_obj.sense == minimize:
            GDPopt.objective_expr = Constraint(
                expr=GDPopt.objective_value >= main_obj.expr)
            solve_data.results.problem.sense = ProblemSense.minimize
        else:
            GDPopt.objective_expr = Constraint(
                expr=GDPopt.objective_value <= main_obj.expr)
            solve_data.results.problem.sense = ProblemSense.maximize
        main_obj.deactivate()
        GDPopt.objective = Objective(
            expr=GDPopt.objective_value, sense=main_obj.sense)

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation (Glover?) or throw an error message

    def _GDPopt_initialize_master(self, solve_data):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        if (solve_data.current_strategy == 'LOA' or
                solve_data.decomposition_strategy == 'hPSC' or
                solve_data.current_strategy == 'LGBD'):
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            m.dual.activate()
        if solve_data.current_strategy == 'PSC':
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            m.dual.activate()
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            GDPopt.psc_cuts = ConstraintList()
        if solve_data.current_strategy == 'LGBD':
            GDPopt.gbd_cuts = ConstraintList(doc='Generalized Benders cuts')

        if solve_data.initialization_strategy is None:
            self._init_set_covering(solve_data)
        elif solve_data.initialization_strategy == 'max_binary':
            self._init_max_binaries(solve_data)
            self._solve_NLP_subproblem(solve_data)
        elif solve_data.initialization_strategy == 'fixed_binary':
            self._validate_disjunctions(solve_data)
            self._solve_NLP_subproblem(solve_data)
        elif solve_data.initialization_strategy == 'custom_disjuncts':
            self._init_custom_disjuncts(solve_data)
        else:
            raise ValueError('Unknown initialization strategy: %s'
                             % (self.initialization_strategy,))

    def _validate_disjunctions(self, solve_data):
        """Validate if the disjunctions are satisfied by the current values."""
        # TODO implement this? If not, the user will simply get an infeasible
        # return value
        pass

    def _init_custom_disjuncts(self, solve_data):
        """Initialize by using user-specified custom disjuncts."""
        m = solve_data.working_model
        # error checking to make sure that the user gave proper disjuncts
        orig_model_cuids = generate_cuid_names(
            solve_data.original_model, ctype=Disjunct,
            descend_into=(Block, Disjunct))
        working_model_cuids = generate_cuid_names(
            m, ctype=Disjunct, descend_into=(Block, Disjunct))
        model_cuid_to_obj = dict(
            (cuid, obj)
            for obj, cuid in iteritems(working_model_cuids))
        for disj_set in solve_data.custom_init_disjuncts:
            fixed_disjs = ComponentSet()
            for disj in disj_set:
                disj_to_fix = model_cuid_to_obj[orig_model_cuids[disj]]
                if not disj_to_fix.indicator_var.fixed:
                    disj_to_fix.indicator_var.fix(1)
                    fixed_disjs.add(disj_to_fix)
            self._solve_init_MIP(solve_data)
            for disj in fixed_disjs:
                disj.indicator_var.unfix()
            self._solve_NLP_subproblem(solve_data)

    def _solve_init_MIP(self, solve_data):
        """Solves the initialization MIP corresponding to the passed model.

        Intended to consolidate some MIP solution code.

        """
        # print('Clone working model for init MIP')
        m = solve_data.working_model.clone()
        GDPopt = m.GDPopt_utils

        # deactivate nonlinear constraints
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)

        # Propagate variable bounds
        TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Propagate fixed variables
        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        # Remove zero terms in linear expressions
        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Transform bound constraints
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Remove trivial constraints
        TransformationFactory(
            'contrib.deactivate_trivial_constraints').apply_to(m)

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = solve_data.mip_solver.solve(
            m, load_solutions=False,
            **solve_data.mip_solver_kwargs)
        terminate_cond = results.solver.termination_condition
        if terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(
                m, load_solutions=False,
                **solve_data.mip_solver_kwargs)
            terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)

        if terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            logger.info('Solved set covering MIP')
            return True
        elif terminate_cond is tc.infeasible:
            logger.info('Set covering problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if solve_data.mip_iter <= 1:
                logger.warn('Problem was infeasible. '
                            'Check your linear and logical constraints '
                            'for contradictions.')
            if GDPopt.objective.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
            return False
        else:
            raise ValueError(
                'GDPopt unable to handle set covering MILP '
                'termination condition '
                'of %s. Solver message: %s' %
                (terminate_cond, results.solver.message))

    def _init_max_binaries(self, solve_data):
        """Initialize by maximizing binary variables and disjuncts.

        This function activates as many binary variables and disjucts as
        feasible. The user would usually want to call _solve_NLP_subproblem()
        after an invocation of this function.

        """
        solve_data.mip_subiter += 1
        # print('Clone working model for init max binaries')
        m = solve_data.working_model.clone()
        logger.info("MIP %s.%s: maximize value of binaries" %
                    (solve_data.mip_iter, solve_data.mip_subiter))
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()
        m.GDPopt_utils.objective.deactivate()
        binary_vars = (
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed)
        m.GDPopt_utils.max_binary_obj = Objective(
            expr=sum(v for v in binary_vars), sense=maximize)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK
        # TODO: chull too?
        results = solve_data.mip_solver.solve(
            m, **solve_data.mip_solver_kwargs)
        # m.display()
        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            # Transfer variable values back to main working model
            self._copy_values(m, solve_data.working_model)
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError('Linear relaxation is infeasible. '
                             'Problem is infeasible.')
        else:
            raise ValueError('Cannot handle termination condition %s'
                             % (solve_terminate_cond,))

    def _init_set_covering(self, solve_data, iterlim=8):
        """Initialize by solving problems to cover the set of all disjuncts.

        The purpose of this initialization is to generate linearizations
        corresponding to each of the disjuncts.

        This work is based upon prototyping work done by Eloy Fernandez at
        Carnegie Mellon University.

        """
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        working_map = generate_cuid_names(m, ctype=(Var, Disjunct),
                                          descend_into=(Block, Disjunct))
        nonlinear_disjuncts = frozenset(
            working_map[disj] for disj in m.component_data_objects(
                ctype=Disjunct, active=True, descend_into=(Block, Disjunct))
            if any(constr.body.polynomial_degree() not in (0, 1)
                   for constr in disj.component_data_objects(
                       ctype=Constraint, active=True,
                       descend_into=(Block, Disjunct))))
        covered_disjuncts = set()
        not_covered_disjuncts = set(nonlinear_disjuncts)
        iter_count = 1
        GDPopt.feasible_integer_cuts.activate()
        while not_covered_disjuncts and iter_count <= iterlim:
            # Solve set covering MIP
            if not self._solve_set_cover_MIP(
                    solve_data,
                    covered_disjuncts=covered_disjuncts,
                    not_covered_disjuncts=not_covered_disjuncts):
                # problem is infeasible. break
                return False
            # solve local NLP
            if self._solve_NLP_subproblem(solve_data):
                # if successful, updated sets
                active_disjuncts = frozenset(
                    working_map[disj] for disj in m.component_data_objects(
                        ctype=Disjunct, active=True,
                        descend_into=(Block, Disjunct))
                    if fabs(value(disj.indicator_var) - 1) <= 1E-5)
                covered_disjuncts.update(active_disjuncts)
                not_covered_disjuncts.difference_update(active_disjuncts)
            iter_count += 1
            # m.GDPopt_utils.integer_cuts.pprint()
        GDPopt.feasible_integer_cuts.deactivate()
        if not_covered_disjuncts:
            # Iteration limit was hit without a full covering of all nonlinear
            # disjuncts
            logger.warn('Iteration limit reached for set covering '
                        'initialization.')
            return False
        return True

    def _solve_set_cover_MIP(self, solve_data, covered_disjuncts,
                             not_covered_disjuncts):
        # print('Clone working model for set cover MIP')
        m = solve_data.working_model.clone()
        GDPopt = m.GDPopt_utils

        cuid_map = generate_cuid_names(m, ctype=Disjunct,
                                       descend_into=(Block, Disjunct))
        reverse_map = dict((cuid, obj) for obj, cuid in iteritems(cuid_map))

        # Set up set covering objective
        weights = dict((disjID, 1) for disjID in covered_disjuncts)
        weights.update(dict((disjID, len(covered_disjuncts) + 1)
                            for disjID in not_covered_disjuncts))
        GDPopt.objective.deactivate()
        GDPopt.set_cover_obj = Objective(
            expr=sum(weights[disjID] * reverse_map[disjID].indicator_var
                     for disjID in weights),
            sense=maximize)

        # deactivate nonlinear constraints
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()

        # Deactivate potentially non-rigorous generated cuts
        for constr in m.component_objects(ctype=Constraint, active=True,
                                          descend_into=(Block, Disjunct)):
            if (constr.local_name == 'GDPopt_OA_cuts' or
                    constr.local_name == 'psc_cuts'):
                constr.deactivate()

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)

        # Propagate variable bounds
        TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Propagate fixed variables
        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        # Remove zero terms in linear expressions
        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Transform bound constraints
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Remove trivial constraints
        TransformationFactory(
            'contrib.deactivate_trivial_constraints').apply_to(m)

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = solve_data.mip_solver.solve(
            m, load_solutions=False,
            **solve_data.mip_solver_kwargs)
        terminate_cond = results.solver.termination_condition
        if terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(
                m, load_solutions=False,
                **solve_data.mip_solver_kwargs)
            terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)

        if terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            # m.print_all_units()
            # int_tol = 1E-4
            # binary_vars = [
            #     v for v in m.component_data_objects(
            #         ctype=Var, descend_into=(Block, Disjunct))
            #     if v.is_binary() and not v.fixed]
            #
            # from pprint import pprint
            # pprint(list(v.name for v in binary_vars
            #             if fabs(v.value - 1) <= int_tol))
            # pprint(list(v.name for v in binary_vars
            #             if fabs(v.value) <= int_tol))
            logger.info('Solved set covering MIP')
            return True
        elif terminate_cond is tc.infeasible:
            logger.info('Set covering problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if solve_data.mip_iter <= 1:
                logger.warn('Problem was infeasible. '
                            'Check your linear and logical constraints '
                            'for contradictions.')
            if GDPopt.objective.sense == minimize:
                solve_data.LB = float('inf')
            else:
                solve_data.UB = float('-inf')
            return False
        else:
            raise ValueError(
                'GDPopt unable to handle set covering MILP '
                'termination condition '
                'of %s. Solver message: %s' %
                (terminate_cond, results.solver.message))

    def _GDPopt_iteration_loop(self, solve_data):
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, solve_data.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            logger.info('')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if solve_data.LB + solve_data.bound_tolerance >= solve_data.UB:
                logger.info('GDPopt exiting on bound convergence. '
                            'LB: %s + (tol %s) >= UB: %s' %
                            (solve_data.LB, solve_data.bound_tolerance,
                             solve_data.UB))
                break
            # Check iteration limit
            if solve_data.mip_iter >= solve_data.iteration_limit:
                logger.info('GDPopt unable to converge bounds '
                            'after %s master iterations.'
                            % (solve_data.mip_iter,))
                logger.info('Final bound values: LB: %s  UB: %s'
                            % (solve_data.LB, solve_data.UB))
                break
            solve_data.mip_subiter = 0
            # solve MILP master problem
            if solve_data.current_strategy == 'LOA':
                self._solve_OA_master(solve_data)
            elif solve_data.current_strategy == 'PSC':
                self._solve_PSC_master(solve_data)
            elif solve_data.current_strategy == 'LGBD':
                self._solve_GBD_master(solve_data)
            # Check bound convergence
            if solve_data.LB + solve_data.bound_tolerance >= solve_data.UB:
                logger.info('GDPopt exiting on bound convergence. '
                            'LB: %s + (tol %s) >= UB: %s'
                            % (solve_data.LB, solve_data.bound_tolerance,
                               solve_data.UB))
                break
            # Solve NLP subproblem
            self._solve_NLP_subproblem(solve_data)

            # If the hybrid algorithm is not making progress, switch to OA.
            required_relax_prog = 1E-6
            required_feas_prog = 1E-6
            if GDPopt.objective.sense == minimize:
                relax_prog_log = solve_data.LB_progress
                feas_prog_log = solve_data.UB_progress
                sign_adjust = 1
            else:
                relax_prog_log = solve_data.UB_progress
                feas_prog_log = solve_data.LB_progress
                sign_adjust = -1
            # Maximum number of iterations in which the lower (optimistic)
            # bound does not improve sufficiently before switching to OA
            LB_stall_after = 5
            if (len(relax_prog_log) > LB_stall_after and
                (sign_adjust * relax_prog_log[-1] <= sign_adjust * (
                    relax_prog_log[-1 - LB_stall_after]
                    + required_relax_prog))):
                if (solve_data.decomposition_strategy == 'hPSC' and
                        solve_data.current_strategy == 'PSC'):
                    logger.info('Relaxation not making enough progress '
                                'for %s iterations. '
                                'Switching to OA.' % (LB_stall_after,))
                    solve_data.current_strategy = 'LOA'

            # Max number of iterations in which upper (feasible) bound does not
            # improve before turning on no-backtracking
            no_backtrack_after = 1
            if (len(feas_prog_log) > no_backtrack_after and
                (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
                 >= sign_adjust * feas_prog_log[-1 - no_backtrack_after])):
                if not GDPopt.feasible_integer_cuts.active:
                    logger.info('Feasible solutions not making enough '
                                'progress for %s iterations. '
                                'Turning on no-backtracking '
                                'integer cuts.' % (no_backtrack_after,))
                    GDPopt.feasible_integer_cuts.activate()

            # Maximum number of iterations in which feasible bound does not
            # improve before terminating algorithm
            if (len(feas_prog_log) > solve_data.algorithm_stall_after and
                (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
                 >= sign_adjust *
                 feas_prog_log[-1 - solve_data.algorithm_stall_after])):
                logger.info('Feasible solutions not making enough progress '
                            'for %s iterations. Algorithm stalled. Exiting.\n'
                            'To continue, increase value of parameter '
                            'algorithm_stall_after.'
                            % (solve_data.algorithm_stall_after,))
                break

    def _solve_OA_master(self, solve_data):
        solve_data.mip_iter += 1
        m = solve_data.working_model.clone()
        GDPopt = m.GDPopt_utils
        logger.info('MIP %s: Solve master problem.' %
                    (solve_data.mip_iter,))

        # Deactivate nonlinear constraints
        for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct)):
            if c.body.polynomial_degree() not in (0, 1):
                c.deactivate()

        # Set up augmented Lagrangean penalty objective
        GDPopt.objective.deactivate()
        sign_adjust = 1 if GDPopt.objective.sense == minimize else -1
        GDPopt.OA_penalty_expr = Expression(
            expr=sign_adjust * solve_data.OA_penalty_factor *
            sum(v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
                if v.parent_component().local_name == 'GDPopt_OA_slack'))
        GDPopt.oa_obj = Objective(
            expr=GDPopt.objective.expr + GDPopt.OA_penalty_expr,
            sense=GDPopt.objective.sense)

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)

        # Propagate variable bounds
        TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Propagate fixed variables
        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        # Remove zero terms in linear expressions
        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Transform bound constraints
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Remove trivial constraints
        TransformationFactory(
            'contrib.deactivate_trivial_constraints').apply_to(m)

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        # Solve
        results = solve_data.mip_solver.solve(
            m, load_solutions=False,
            **solve_data.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that the problem is
            # infeasible or unbounded during presolve, but fails to
            # distinguish. We need to resolve with a solver option flag on.
            old_options = deepcopy(solve_data.mip_solver.options)
            # This solver option is specific to Gurobi.
            solve_data.mip_solver.options['DualReductions'] = 0
            results = solve_data.mip_solver.solve(
                m, load_solutions=False,
                **solve_data.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            solve_data.mip_solver.options.update(old_options)

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if GDPopt.objective.sense == minimize:
                solve_data.LB = max(value(GDPopt.oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(GDPopt.oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            # m.print_selected_units()
            logger.info('MIP %s: OBJ: %s  LB: %s  UB: %s'
                        % (solve_data.mip_iter, value(GDPopt.oa_obj.expr),
                           solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.maxTimeLimit:
            # TODO check that status is actually ok and everything is feasible
            logger.info('Unable to optimize MILP master problem '
                        'within time limit. '
                        'Using current solver feasible solution.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if GDPopt.objective.sense == minimize:
                solve_data.LB = max(
                    value(GDPopt.objective.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(
                    value(GDPopt.objective.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            logger.info('MIP %s: OBJ: %s  LB: %s  UB: %s'
                        % (self.mip_iter, value(GDPopt.objective.expr),
                           self.LB, self.UB))
        elif (master_terminate_cond is tc.other and
                results.solution.status is SolutionStatus.feasible):
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            logger.info('MILP solver reported feasible solution, '
                        'but not guaranteed to be optimal.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            if GDPopt.objective.sense == minimize:
                solve_data.LB = max(value(GDPopt.oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(GDPopt.oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            logger.info('MIP %s: OBJ: %s  LB: %s  UB: %s'
                        % (solve_data.mip_iter, value(GDPopt.oa_obj.expr),
                           solve_data.LB, solve_data.UB))
        elif master_terminate_cond is tc.infeasible:
            logger.info('MILP master problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if solve_data.mip_iter == 1:
                logger.warn('GDPopt initialization may have generated poor '
                            'quality cuts.')
            # set optimistic bound to infinity
            if GDPopt.objective.sense == minimize:
                solve_data.LB = float('inf')
                solve_data.LB_progress.append(solve_data.UB)
            else:
                solve_data.UB = float('-inf')
                solve_data.UB_progress.append(solve_data.UB)
        else:
            raise ValueError(
                'GDPopt unable to handle MILP master termination condition '
                'of %s. Solver message: %s' %
                (master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        solve_data.master_postsolve(m, solve_data)

    def _solve_NLP_subproblem(self, solve_data):
        # print('Clone working model for NLP')
        m = solve_data.working_model.clone()
        GDPopt = m.GDPopt_utils
        solve_data.nlp_iter += 1
        logger.info('NLP %s: Solve subproblem for fixed binaries and '
                    'logical realizations.'
                    % (solve_data.nlp_iter,))
        # Fix binary variables
        binary_vars = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed]
        for v in binary_vars:
            if v.value is None:
                logger.warning('No value is defined for binary variable %s'
                               ' for the NLP subproblem.' % (v.name,))
            else:
                # round the integer variable values so that they are exactly 0
                # or 1
                if solve_data.round_NLP_binaries:
                    v.set_value(round(v.value))
            v.fix()

        # Deactivate the OA and PSC cuts
        for constr in m.component_objects(ctype=Constraint, active=True,
                                          descend_into=(Block, Disjunct)):
            if (constr.local_name == 'GDPopt_OA_cuts' or
                    constr.local_name == 'psc_cuts'):
                constr.deactivate()

        # Activate or deactivate disjuncts according to the value of their
        # indicator variable
        for disj in m.component_data_objects(
                ctype=Disjunct, descend_into=(Block, Disjunct)):
            if (fabs(value(disj.indicator_var) - 1)
                    <= solve_data.integer_tolerance):
                # Disjunct is active. Convert to Block.
                disj.parent_block().reclassify_component_type(disj, Block)
            elif (fabs(value(disj.indicator_var))
                    <= solve_data.integer_tolerance):
                disj.deactivate()
            else:
                raise ValueError(
                    'Non-binary value of disjunct indicator variable '
                    'for %s: %s' % (disj.name, value(disj.indicator_var)))

        for d in m.component_data_objects(Disjunction, active=True):
            d.deactivate()

        # Propagate variable bounds
        TransformationFactory('contrib.propagate_eq_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Propagate fixed variables
        TransformationFactory('contrib.propagate_fixed_vars').apply_to(m)
        # Remove zero terms in linear expressions
        TransformationFactory('contrib.remove_zero_terms').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Transform bound constraints
        TransformationFactory('contrib.constraints_to_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('contrib.detect_fixed_vars').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('contrib.propagate_zero_sum').apply_to(m)
        # Remove trivial constraints
        TransformationFactory(
            'contrib.deactivate_trivial_constraints').apply_to(m)

        # restore original variable values
        obj_to_cuid = generate_cuid_names(m, ctype=(Var, Constraint, Disjunct),
                                          descend_into=(Block, Disjunct))
        for v in m.component_data_objects(ctype=Var,
                                          descend_into=(Block, Disjunct)):
            if not v.fixed and not v.is_binary():
                try:
                    old_value = solve_data.initial_variable_values[
                        obj_to_cuid[v]]
                    # Ensure that the value is within the bounds
                    if old_value is not None:
                        if v.has_lb() and old_value < v.lb:
                            old_value = v.lb
                        if v.has_ub() and old_value > v.ub:
                            old_value = v.ub
                        # Set the value
                        v.set_value(old_value)
                except KeyError:
                    continue

        solve_data.subproblem_presolve(m, solve_data)

        # Solve the NLP
        results = solve_data.nlp_solver.solve(
            m, load_solutions=False,
            **solve_data.nlp_solver_kwargs)
        solve_data.solve_results = results

        solnFeasible = False

        def process_feasible_solution():
            self._copy_values(m, solve_data.working_model,
                              from_map=obj_to_cuid)
            self._copy_dual_suffixes(m, solve_data.working_model,
                                     from_map=obj_to_cuid)
            if GDPopt.objective.sense == minimize:
                solve_data.UB = min(
                    value(GDPopt.objective.expr), solve_data.UB)
                solve_data.solution_improved = (
                    solve_data.UB < solve_data.UB_progress[-1])
            else:
                solve_data.LB = max(
                    value(GDPopt.objective.expr), solve_data.LB)
                solve_data.solution_improved = (
                    solve_data.LB > solve_data.LB_progress[-1])
            logger.info('NLP %s: OBJ: %s  LB: %s  UB: %s'
                        % (solve_data.nlp_iter,
                           value(GDPopt.objective.expr),
                           solve_data.LB, solve_data.UB))
            if solve_data.solution_improved:
                # print('Clone model for best_solution_found')
                solve_data.best_solution_found = m.clone()

            # Add the linear cut
            if solve_data.current_strategy == 'LOA':
                self._add_oa_cut(m, solve_data)
            elif solve_data.current_strategy == 'PSC':
                self._add_psc_cut(solve_data)
            elif solve_data.current_strategy == 'LGBD':
                self._add_gbd_cut(solve_data)

            # This adds an integer cut to the GDPopt_feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(solve_data, feasible=True)

            solve_data.subproblem_postfeasible(m, solve_data)

        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            process_feasible_solution()
            solnFeasible = True
        elif subprob_terminate_cond is tc.infeasible:
            # TODO try something else? Reinitialize with different initial
            # value?
            logger.info('NLP subproblem was locally infeasible.')
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, solve_data.working_model)
            self._copy_dual_suffixes(m, solve_data.working_model,
                                     from_map=obj_to_cuid)
            if solve_data.current_strategy == 'PSC':
                logger.info('Adding PSC feasibility cut.')
                self._add_psc_cut(solve_data, nlp_feasible=False)
            elif solve_data.current_strategy == 'LGBD':
                logger.info('Adding GBD feasibility cut.')
                self._add_gbd_cut(solve_data, nlp_feasible=False)
            # Add an integer cut to exclude this discrete option
            self._add_int_cut(solve_data)
        elif subprob_terminate_cond is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial
            # value?
            logger.info('NLP subproblem failed to converge '
                        'within iteration limit.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            if self._is_feasible(m):
                logger.info('NLP solution is still feasible. '
                            'Using potentially suboptimal feasible solution.')
                process_feasible_solution()
                solnFeasible = True
            else:
                # Add an integer cut to exclude this discrete option
                self._add_int_cut(solve_data)
        else:
            raise ValueError(
                'GDPopt unable to handle NLP subproblem termination '
                'condition of %s. Results: %s'
                % (subprob_terminate_cond, results))

        if GDPopt.objective.sense == minimize:
            solve_data.UB_progress.append(solve_data.UB)
        else:
            solve_data.LB_progress.append(solve_data.LB)

        # Call the NLP post-solve callback
        solve_data.subproblem_postsolve(m, solve_data)
        return solnFeasible

    def _is_feasible(self, m, constr_tol=1E-6, var_tol=1E-8):
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct)):
            # constraint is an equality
            if constr.equality:
                if fabs(value(constr.lower) -
                        value(constr.body)) >= constr_tol:
                    logger.info('%s: %s != %s' % (
                        constr.name, value(constr.body), value(constr.lower)))
                    return False
            if constr.lower is not None:
                if value(constr.lower) - value(constr.body) >= constr_tol:
                    logger.info('%s: %s < %s' % (
                        constr.name, value(constr.body), value(constr.lower)))
                    return False
            if constr.upper is not None:
                if value(constr.body) - value(constr.upper) >= constr_tol:
                    logger.info('%s: %s > %s' % (
                        constr.name, value(constr.body), value(constr.upper)))
                    return False
        for var in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct)):
            if var.lb is not None:
                if value(var.lb) - value(var) >= var_tol:
                    logger.info('%s: %s < %s' % (
                        var.name, value(var), value(var.lb)))
                    return False
            if var.ub is not None:
                if value(var) - value(var.ub) >= var_tol:
                    logger.info('%s: %s > %s' % (
                        var.name, value(var), value(var.ub)))
                    return False
        return True

    def _add_oa_cut(self, nlp_solution, solve_data, for_GBD=False):
        """Add outer approximation cuts to working model.

        If for_GBD flag is True, then place the cuts in a component called
        GDPopt_OA_cuts_for_GBD and deactivate them by default.
        """
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        sign_adjust = -1 if GDPopt.objective.sense == minimize else 1

        # From the NLP solution, we need to figure out for which nonlinear
        # constraints to generate cuts
        nlp_nonlinear_constr = (c for c in nlp_solution.component_data_objects(
            ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        nlp_constr_to_cuid = generate_cuid_names(
            nlp_solution, ctype=(Constraint, Disjunct),
            descend_into=(Block, Disjunct))
        model_cuid_to_obj = dict((cuid, obj) for obj, cuid in iteritems(
            generate_cuid_names(
                solve_data.working_model, ctype=(Constraint, Disjunct),
                descend_into=(Block, Disjunct))))
        nonlinear_constraints = (model_cuid_to_obj[nlp_constr_to_cuid[c]]
                                 for c in nlp_nonlinear_constr)

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        logger.info('Adding OA cuts.')
        counter = 0
        for constr in nonlinear_constraints:
            if not m.dual.get(constr, None):
                continue
            parent_block = constr.parent_block()
            ignore_set = getattr(parent_block, 'GDPopt_ignore_OA', None)
            logger.debug('Ignore_set %s' % ignore_set)
            if (ignore_set and (constr in ignore_set or
                                constr.parent_component() in ignore_set)):
                logger.debug('OA cut addition for %s skipped because it is in '
                             'the ignore set.' % constr.name)
                continue

            logger.debug("Adding OA cut for %s with dual value %s"
                         % (constr.name, m.dual.get(constr)))

            constr_vars = list(EXPR.identify_variables(constr.body))
            jac_list = differentiate(constr.body, wrt_list=constr_vars)
            jacobians = ComponentMap(zip(constr_vars, jac_list))

            if not for_GBD:
                oa_utils = parent_block.component('GDPopt_OA')
                if oa_utils is None:
                    oa_utils = parent_block.GDPopt_OA = Block()
                    oa_utils.GDPopt_OA_cuts = ConstraintList()
                    oa_utils.next_idx = 1
                    oa_utils.GDPopt_OA_slacks = Set(dimen=1)
                    oa_utils.GDPopt_OA_slack = Var(
                        oa_utils.GDPopt_OA_slacks,
                        bounds=(0, solve_data.max_slack),
                        domain=NonNegativeReals, initialize=0)

                oa_cuts = oa_utils.GDPopt_OA_cuts
                oa_utils.GDPopt_OA_slacks.add(oa_utils.next_idx)
                slack_var = oa_utils.GDPopt_OA_slack[oa_utils.next_idx]
                oa_utils.next_idx += 1
            else:
                oa_cuts = parent_block.component('GDPopt_OA_cuts_for_GBD')
                if oa_cuts is None:
                    oa_cuts = parent_block.GDPopt_OA_cuts_for_GBD = \
                        ConstraintList()
                oa_cuts.deactivate()
            oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * (
                    value(constr.body) + sum(
                        value(jacobians[var]) * (var - value(var))
                        for var in constr_vars)) + slack_var <= 0)
            counter += 1

        logger.info('Added %s OA cuts' % counter)

    def _add_int_cut(self, solve_data, feasible=False):
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        int_tol = solve_data.integer_tolerance
        binary_vars = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed]
        # check to make sure that binary variables are all 0 or 1
        for v in binary_vars:
            if fabs(v.value - 1) > int_tol and fabs(value(v)) > int_tol:
                raise ValueError('Binary %s = %s is not 0 or 1' % (
                    v.name, value(v)))

        if not binary_vars:
            # if no binary variables, add infeasible constraints.
            if not feasible:
                logger.info(
                    'Adding integer cut to a model without binary variables. '
                    'Model is now infeasible.')
                GDPopt.integer_cuts.add(expr=GDPopt.objective_value >= 1)
                GDPopt.integer_cuts.add(expr=GDPopt.objective_value <= 0)
            else:
                GDPopt.feasible_integer_cuts.add(
                    expr=GDPopt.objective_value >= 1)
                GDPopt.feasible_integer_cuts.add(
                    expr=GDPopt.objective_value <= 0)
            return

        int_cut = (sum(1 - v for v in binary_vars
                       if fabs(v.value - 1) <= int_tol) +
                   sum(v for v in binary_vars
                       if fabs(value(v)) <= int_tol) >= 1)

        # print('Active: {}'.format(
        #     list(v.name for v in binary_vars
        #          if fabs(v.value - 1) <= int_tol)))
        # print('Inactive: {}'.format(
        #     list(v.name for v in binary_vars
        #          if fabs(v.value) <= int_tol)))

        if not feasible:
            # Add the integer cut
            logger.info('Adding integer cut')
            GDPopt.integer_cuts.add(expr=int_cut)
        else:
            logger.info('Adding feasible integer cut')
            GDPopt.feasible_integer_cuts.add(expr=int_cut)


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class GDPoptSolveData(object):
    """Data container to hold solve-instance data."""
    pass
