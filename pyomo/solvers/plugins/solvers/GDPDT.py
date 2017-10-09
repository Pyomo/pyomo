# -*- coding: UTF-8 -*-
"""Decomposition solver for Generalized Disjunctive Programming (GDP) problems.

The DAGPy (Decomposition Algorithms for GDP in Python) solver applies a variety
of decomposition-based approaches to solve Generalized Disjunctive Programming
(GDP) problems. GDP models can include nonlinear, continuous variables and
constraints, as well as logical conditions.

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
import sys
from copy import deepcopy
from math import copysign, fabs

import pyomo.util.plugin
from pyomo.core.base import expr as EXPR
from pyomo.core.base import (Block, Constraint, ConstraintList, Expression,
                             Objective, RangeSet, Set, Suffix,
                             TransformationFactory, Var, maximize, minimize,
                             value)
from pyomo.core.base.block import generate_cuid_names
from pyomo.core.base.expr import identify_variables
from pyomo.core.base.expr_common import clone_expression
from pyomo.core.base.numvalue import NumericConstant
from pyomo.core.base.symbolic import differentiate
from pyomo.core.kernel import ComponentSet, NonNegativeReals, Reals
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolutionStatus, SolverFactory, SolverStatus
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import ProblemSense, SolverResults
from pyomo.repn.canonical_repn import generate_canonical_repn
from six import iteritems

logger = logging.getLogger('pyomo.solvers')

__version__ = (0, 0, 1)


class DAGPySolver(pyomo.util.plugin.Plugin):
    """A decomposition-based GDP solver."""

    pyomo.util.plugin.implements(IOptSolver)
    pyomo.util.plugin.alias('dagpy',
                            doc='The DAGPy decomposition-based GDP solver')

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
            solve_in_place (bool): If true, DAGPy manipulations are performed
                directly upon the model. Otherwise, the model is first copied
                and solution values are copied over afterwards.
            master_postsolve (func): callback hook after a solution of the
                master problem
            subprob_postsolve (func): callback hook after a solution of the
                nonlinear subproblem
            subprob_postfeas (func): callback hook after feasible solution of
                the nonlinear subproblem

        """
        self.bound_tolerance = kwds.pop('tol', 1E-6)
        self.iteration_limit = kwds.pop('iterlim', 30)
        self.decomposition_strategy = kwds.pop('strategy', 'LOA')
        self.initialization_strategy = kwds.pop('init_strategy', None)
        self.max_slack = kwds.pop('max_slack', 1000)
        self.OA_penalty_factor = kwds.pop('OA_penalty', 1000)
        self.nlp_solver_name = kwds.pop('nlp', 'ipopt')
        self.nlp_solver_kwargs = kwds.pop('nlp_kwargs', {})
        self.mip_solver_name = kwds.pop('mip', 'gurobi')
        self.mip_solver_kwargs = kwds.pop('mip_kwargs', {})
        self.modify_in_place = kwds.pop('solve_in_place', True)
        self.master_postsolve = kwds.pop('master_postsolve', _DoNothing())
        self.subproblem_postsolve = kwds.pop('subprob_postsolve', _DoNothing())
        self.subproblem_postfeasible = kwds.pop('subprob_postfeas',
                                                _DoNothing())
        self.tee = kwds.pop('tee', False)

        if self.tee:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setFormatter(logging.Formatter('%(message)s'))
            existing_handlers = list(logger.handlers)
            old_logger_level = logger.getEffectiveLevel()
            logger.setLevel(logging.INFO)
            logger.addHandler(ch)
            for handle in existing_handlers:
                logger.removeHandler(handle)

        if kwds:
            logger.warn("Unrecognized arguments passed to DAGPy solver: {}"
                        .format(kwds))

        # Verify that decomposition strategy chosen is one of the supported
        # strategies
        valid_strategies = [
            'LOA', 'LGBD']
        if self.decomposition_strategy not in valid_strategies:
            raise ValueError('Unrecognized decomposition strategy {}. '
                             'Valid strategies include: {}'.format(
                                 self.decomposition_strategy,
                                 valid_strategies))

        # If decomposition strategy is a hybrid, set the initial strategy
        if self.decomposition_strategy == 'hPSC':
            self._decomposition_strategy = 'PSC'
        else:
            self._decomposition_strategy = self.decomposition_strategy

        # When generating cuts, small duals multiplied by expressions can cause
        # problems. Exclude all duals smaller in absolue value than the
        # following.
        self.small_dual_tolerance = 1E-8
        self.integer_tolerance = 1E-5

        self.round_NLP_binaries = True
        """bool: flag to round binary values to exactly 0 or 1.
        Rounding is done before solving NLP subproblem"""

        # Modify in place decides whether to run the algorithm on a copy of the
        # originally model passed to the solver, or whether to manipulate the
        # original model directly.
        if self.modify_in_place:
            self.working_model = m = model
        else:
            self.working_model = m = model.clone()

        # Store the initial model state as the best solution found. If we find
        # no better solution, then we will restore from this copy.
        self.best_solution_found = model.clone()

        # Save model initial values. These are used later to initialize NLP
        # subproblems.
        model_obj_to_cuid = generate_cuid_names(
            model, ctype=(Var, Disjunct), descend_into=(Block, Disjunct))
        model_cuid_to_obj = {cuid: obj
                             for obj, cuid in iteritems(model_obj_to_cuid)}
        self.initial_variable_values = {
            model_obj_to_cuid[v]: value(v, exception=False)
            for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))}

        # Create a model block on which to store DAGPy-specific utility
        # modeling objects.
        DAGPy = m.DAGPy_utils = Block()

        # Create the solver results object
        res = self.results = SolverResults()
        res.problem.name = m.name
        res.problem.number_of_nonzeros = None  # TODO
        res.solver.name = 'DAGPy ' + str(self.version())
        # TODO work on termination condition and message
        res.solver.termination_condition = None
        res.solver.message = None
        # TODO add some kind of timing
        res.solver.user_time = None
        res.solver.system_time = None
        res.solver.wallclock_time = None
        res.solver.termination_message = None

        # Validate the model to ensure that DAGPy is able to solve it.
        #
        # This needs to take place before the detection of nonlinear
        # constraints, because if the objective is nonlinear, it will be moved
        # to the constraints.
        assert(not hasattr(self, 'nonlinear_constraints'))
        self._validate_model()

        # Maps in order to keep track of certain generated constraints
        DAGPy.oa_cut_map = Suffix(direction=Suffix.LOCAL, datatype=None)

        # Integer cuts exclude particular discrete decisions
        DAGPy.integer_cuts = ConstraintList(doc='integer cuts')
        # Feasible integer cuts exclude discrete realizations that have been
        # explored via an NLP subproblem. Depending on model characteristics,
        # the user may wish to revisit NLP subproblems (with a different
        # initialization, for example). Therefore, these cuts are not enabled
        # by default.
        #
        # Note: these cuts will only exclude integer realizations that are not
        # already in the primary DAGPy_integer_cuts ConstraintList.
        DAGPy.feasible_integer_cuts = ConstraintList(
            doc='explored integer cuts')
        DAGPy.feasible_integer_cuts.deactivate()

        # # Build a list of binary variables
        # self.binary_vars = [v for v in m.component_data_objects(
        #     ctype=Var, descend_into=True)
        #     if v.is_binary() and not v.fixed]
        #
        # Build list of nonlinear constraints
        self.nonlinear_constraints = [
            v for v in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if v.body.polynomial_degree() not in (0, 1)]

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0
        self.mip_subiter = 0

        # Set of NLP iterations for which cuts were generated
        DAGPy.nlp_iters = Set(dimen=1)

        # Create an integer index set over the nonlinear constraints
        DAGPy.nl_constraint_set = RangeSet(len(self.nonlinear_constraints))
        # Mapping Constraint -> integer index
        self.nl_map = {}
        # Mapping integer index -> Constraint
        self.nl_inverse_map = {}
        # Generate the two maps. These maps may be helpful for later
        # interpreting indices on the slack variables or generated cuts.
        for c, n in zip(self.nonlinear_constraints, DAGPy.nl_constraint_set):
            self.nl_map[c] = n
            self.nl_inverse_map[n] = c

        # Create slack variables
        DAGPy.slack_vars = Var(DAGPy.nlp_iters, DAGPy.nl_constraint_set,
                               domain=NonNegativeReals,
                               bounds=(0, self.max_slack), initialize=0)

        # set up bounds
        self.LB = float('-inf')
        self.UB = float('inf')
        self.LB_progress = [self.LB]
        self.UB_progress = [self.UB]

        # Flag indicating whether the solution improved in the past iteration
        # or not
        self.solution_improved = False

        # Set up solvers
        self.nlp_solver = SolverFactory(self.nlp_solver_name)
        self.mip_solver = SolverFactory(self.mip_solver_name)

        # Initialize the master problem
        self._DAGPy_initialize_master()

        # Algorithm main loop
        self._DAGPy_iteration_loop()

        # Update values in original model
        self._copy_values(self.best_solution_found, model,
                          to_map=model_cuid_to_obj)

        self.results.problem.lower_bound = self.LB
        self.results.problem.upper_bound = self.UB

        if self.tee:
            logger.removeHandler(ch)
            logger.setLevel(old_logger_level)
            for handle in existing_handlers:
                logger.addHandler(handle)

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
            to_map = {cuid: obj
                      for obj, cuid in iteritems(tm_obj_to_uid)}
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
            to_map = {cuid: obj
                      for obj, cuid in iteritems(tm_obj_to_uid)}

        to_suffix.clear()
        for model_obj in from_suffix:
            to_model_obj = to_map[from_map[model_obj]]
            to_suffix[to_model_obj] = from_suffix[model_obj]

    def _validate_model(self):
        """Validate that the model is solveable by DAGPy.

        Also populates results object with problem information.

        """
        m, DAGPy = self.working_model, self.working_model.DAGPy_utils

        # Get count of constraints and variables
        self.results.problem.number_of_constraints = 0
        var_set = ComponentSet()
        binary_var_set = ComponentSet()
        integer_var_set = ComponentSet()
        continuous_var_set = ComponentSet()
        for constr in m.component_data_objects(ctype=Constraint, active=True,
                                               descend_into=(Disjunct, Block)):
            self.results.problem.number_of_constraints += 1
            for v in identify_variables(constr.body):
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

        self.results.problem.number_of_variables = len(var_set)
        self.results.problem.number_of_binary_variables = len(binary_var_set)
        self.results.problem.number_of_integer_variables = len(integer_var_set)
        self.results.problem.number_of_continuous_variables = \
            len(continuous_var_set)

        # Check for any integer variables
        if any(True for v in m.component_data_objects(
                ctype=Var, descend_into=True)
                if v.is_integer() and not v.fixed):
            raise ValueError('Model contains unfixed integer variables. '
                             'DAGPy does not currently support solution of '
                             'such problems.')
            # TODO add in the reformulation using base 2
            # TODO need to look inside of disjuncts for integer variables too

        # Handle missing or multiple objectives
        objs = list(m.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        num_objs = len(objs)
        self.results.problem.number_of_objectives = num_objs
        if num_objs == 0:
            raise ValueError('Model has no active objectives.')
        elif num_objs > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = objs[0]

        # Move the objective to the constraints
        DAGPy.objective_value = Var(domain=Reals, initialize=0)
        if main_obj.sense == minimize:
            DAGPy.objective_expr = Constraint(
                expr=DAGPy.objective_value >= main_obj.expr)
            self.results.problem.sense = ProblemSense.minimize
        else:
            DAGPy.objective_expr = Constraint(
                expr=DAGPy.objective_value <= main_obj.expr)
            self.results.problem.sense = ProblemSense.maximize
        main_obj.deactivate()
        DAGPy.objective = Objective(
            expr=DAGPy.objective_value, sense=main_obj.sense)

        # TODO if any continuous variables are multipled with binary ones, need
        # to do some kind of transformation (Glover?) or throw an error message

    def _DAGPy_initialize_master(self):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        m, DAGPy = self.working_model, self.working_model.DAGPy_utils
        if (self._decomposition_strategy == 'LOA' or
                self.decomposition_strategy == 'hPSC' or
                self._decomposition_strategy == 'LGBD'):
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            m.dual.activate()
            # Map Constraint, nlp_iter -> generated OA Constraint
            DAGPy.OA_constr_map = {}
            self._calc_jacobians()  # preload jacobians
        if self._decomposition_strategy == 'PSC':
            if not hasattr(m, 'dual'):  # Set up dual value reporting
                m.dual = Suffix(direction=Suffix.IMPORT)
            m.dual.activate()
            if not hasattr(m, 'ipopt_zL_out'):
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(m, 'ipopt_zU_out'):
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            DAGPy.psc_cuts = ConstraintList()
        if self._decomposition_strategy == 'LGBD':
            DAGPy.gbd_cuts = ConstraintList(doc='Generalized Benders cuts')

        if self.initialization_strategy is None:
            self._init_set_covering()
        elif self.initialization_strategy == 'max_binary':
            self._init_max_binaries()
            self._solve_NLP_subproblem()
        elif self.initialization_strategy == 'fixed_binary':
            self._validate_disjunctions()
            self._solve_NLP_subproblem()
        else:
            raise ValueError('Unknown initialization strategy: {}'
                             .format(self.initialization_strategy))

    def _validate_disjunctions(self):
        """Validate if the disjunctions are satisfied by the current values."""
        # TODO implement this? If not, the user will simply get an infeasible
        # return value
        pass

    def _init_max_binaries(self):
        """Initialize by maximizing binary variables and disjuncts.

        This function activates as many binary variables and disjucts as
        feasible. The user would usually want to call _solve_NLP_subproblem()
        after an invocation of this function.

        """
        self.mip_subiter += 1
        m = self.working_model.clone()
        logger.info("MIP {}.{}: maximize value of binaries".format(
            self.mip_iter, self.mip_subiter))
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()
        m.DAGPy_utils.objective.deactivate()
        binary_vars = (
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed)
        m.DAGPy_utils.max_binary_obj = Objective(
            expr=sum(v for v in binary_vars), sense=maximize)
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK
        # TODO: chull too?
        results = self.mip_solver.solve(m, **self.mip_solver_kwargs)
        # m.display()
        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            # Transfer variable values back to main working model
            self._copy_values(m, self.working_model)
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError('Linear relaxation is infeasible. '
                             'Problem is infeasible.')
        else:
            raise ValueError('Cannot handle termination condition {}'.format(
                solve_terminate_cond))

    def _init_set_covering(self, iterlim=8):
        """Initialize by solving problems to cover the set of all disjuncts.

        The purpose of this initialization is to generate linearizations
        corresponding to each of the disjuncts.

        This work is based upon prototyping work done by Eloy Fernandez at
        Carnegie Mellon University.

        """
        m = self.working_model
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
        while not_covered_disjuncts and iter_count <= iterlim:
            # Solve set covering MIP
            if not self._solve_set_cover_MIP(
                    covered_disjuncts=covered_disjuncts,
                    not_covered_disjuncts=not_covered_disjuncts):
                # problem is infeasible. break
                return False
            # solve local NLP
            if self._solve_NLP_subproblem():
                # if successful, updated sets
                active_disjuncts = frozenset(
                    working_map[disj] for disj in m.component_data_objects(
                        ctype=Disjunct, active=True,
                        descend_into=(Block, Disjunct))
                    if fabs(value(disj.indicator_var) - 1) <= 1E-5)
                covered_disjuncts.update(active_disjuncts)
                not_covered_disjuncts.difference_update(active_disjuncts)
            iter_count += 1
        if not_covered_disjuncts:
            # Iteration limit was hit without a full covering of all nonlinear
            # disjuncts
            logger.warn('Iteration limit reached for set covering '
                        'initialization.')
            return False
        return True

    def _solve_set_cover_MIP(self, covered_disjuncts, not_covered_disjuncts):
        m = self.working_model.clone()
        DAGPy = m.DAGPy_utils

        cuid_map = generate_cuid_names(m, ctype=Disjunct,
                                       descend_into=(Block, Disjunct))
        reverse_map = {cuid: obj for obj, cuid in iteritems(cuid_map)}

        # Set up set covering objective
        weights = {disjID: 1 for disjID in covered_disjuncts}
        weights.update({disjID: len(covered_disjuncts) + 1
                        for disjID in not_covered_disjuncts})
        DAGPy.objective.deactivate()
        DAGPy.set_cover_obj = Objective(
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
            if (constr.local_name == 'DAGPy_OA_cuts' or
                    constr.local_name == 'psc_cuts'):
                constr.deactivate()

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        terminate_cond = results.solver.termination_condition
        if terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            **self.mip_solver_kwargs)
            terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)

        if terminate_cond is tc.optimal:
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            logger.info('Solved set covering MIP')
            return True
        elif terminate_cond is tc.infeasible:
            logger.info('Set covering problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if self.mip_iter <= 1:
                logger.warn('Problem was infeasible. '
                            'Check your linear and logical constraints '
                            'for contradictions.')
            if DAGPy.objective.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
            return False
        else:
            raise ValueError(
                'DAGPy unable to handle set covering MILP '
                'termination condition '
                'of {}. Solver message: {}'.format(
                    terminate_cond, results.solver.message))

    def _DAGPy_iteration_loop(self):
        m = self.working_model
        DAGPy = m.DAGPy_utils
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, self.iteration_limit)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            logger.info('')  # print blank lines for visual display
            backup_iter += 1
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                logger.info('DAGPy exiting on bound convergence. '
                            'LB: {} + (tol {}) >= UB: {}'.format(
                                self.LB, self.bound_tolerance, self.UB))
                break
            # Check iteration limit
            if self.mip_iter >= self.iteration_limit:
                logger.info('DAGPy unable to converge bounds '
                            'after {} master iterations.'
                            .format(self.mip_iter))
                logger.info('Final bound values: LB: {}  UB: {}'.
                            format(self.LB, self.UB))
                break
            self.mip_subiter = 0
            # solve MILP master problem
            if self._decomposition_strategy == 'LOA':
                self._solve_OA_master()
            elif self._decomposition_strategy == 'PSC':
                self._solve_PSC_master()
            elif self._decomposition_strategy == 'LGBD':
                self._solve_GBD_master()
            # Check bound convergence
            if self.LB + self.bound_tolerance >= self.UB:
                logger.info('DAGPy exiting on bound convergence. '
                            'LB: {} + (tol {}) >= UB: {}'.format(
                                self.LB, self.bound_tolerance, self.UB))
                break
            # Solve NLP subproblem
            self._solve_NLP_subproblem()

            # If the hybrid algorithm is not making progress, switch to OA.
            required_relax_prog = 1E-6
            required_feas_prog = 1E-6
            if DAGPy.objective.sense == minimize:
                relax_prog_log = self.LB_progress
                feas_prog_log = self.UB_progress
                sign_adjust = 1
            else:
                relax_prog_log = self.UB_progress
                feas_prog_log = self.LB_progress
                sign_adjust = -1
            # Maximum number of iterations in which the lower (optimistic)
            # bound does not improve sufficiently before switching to OA
            LB_stall_after = 5
            if (len(relax_prog_log) > LB_stall_after and
                (sign_adjust * relax_prog_log[-1] <= sign_adjust * (
                    relax_prog_log[-1 - LB_stall_after]
                    + required_relax_prog))):
                if (self.decomposition_strategy == 'hPSC' and
                        self._decomposition_strategy == 'PSC'):
                    logger.info('Relaxation not making enough progress '
                                'for {} iterations. '
                                'Switching to OA.'.format(LB_stall_after))
                    self._decomposition_strategy = 'LOA'

            # Max number of iterations in which upper (feasible) bound does not
            # improve before turning on no-backtracking
            no_backtrack_after = 5
            if (len(feas_prog_log) > no_backtrack_after and
                (sign_adjust * feas_prog_log[-1] <= sign_adjust * (
                    feas_prog_log[-1 - no_backtrack_after]
                    + required_feas_prog))):
                if not DAGPy.feasible_integer_cuts.active:
                    logger.info('Feasible solutions not making enough '
                                'progress for {} iterations. '
                                'Turning on no-backtracking '
                                'integer cuts.'.format(no_backtrack_after))
                    DAGPy.feasible_integer_cuts.activate()

            # Maximum number of iterations in which feasible bound does not
            # improve before terminating algorithm
            algorithm_stall_after = 8
            if (len(feas_prog_log) > algorithm_stall_after and
                (sign_adjust * feas_prog_log[-1] <= sign_adjust * (
                    feas_prog_log[-1 - algorithm_stall_after]
                    + required_feas_prog))):
                logger.info('Feasible solutions not making enough progress '
                            'for {} iterations. Algorithm stalled. Exiting.\n'
                            'To continue, increase value of parameter '
                            'algorithm_stall_after.'
                            .format(algorithm_stall_after))
                break

    def _solve_OA_master(self):
        self.mip_iter += 1
        m = self.working_model.clone()
        DAGPy = m.DAGPy_utils
        logger.info('MIP {}: Solve master problem.'.format(self.mip_iter))

        # Deactivate nonlinear constraints
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()

        # Set up augmented Lagrangean penalty objective
        DAGPy.objective.deactivate()
        sign_adjust = 1 if DAGPy.objective.sense == minimize else -1
        DAGPy.OA_penalty_expr = Expression(
            expr=sign_adjust * self.OA_penalty_factor * sum(
                v for v in DAGPy.slack_vars[...]))
        DAGPy.oa_obj = Objective(
            expr=DAGPy.objective.expr + DAGPy.OA_penalty_expr,
            sense=DAGPy.objective.sense)

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK

        # Deactivate extraneous IMPORT/EXPORT suffixes
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        # Solve
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            **self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.oa_obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.oa_obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.oa_obj.expr),
                                self.LB, self.UB))
        elif master_terminate_cond is tc.maxTimeLimit:
            # TODO check that status is actually ok and everything is feasible
            logger.info('Unable to optimize MILP master problem '
                        'within time limit. '
                        'Using current solver feasible solution.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.objective.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.objective.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.objective.expr),
                                self.LB, self.UB))
        elif (master_terminate_cond is tc.other and
                results.solution.status is SolutionStatus.feasible):
            # load the solution and suppress the warning message by setting
            # solver status to ok.
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.oa_obj.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.oa_obj.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.oa_obj.expr),
                                self.LB, self.UB))
        elif master_terminate_cond is tc.infeasible:
            logger.info('MILP master problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if self.mip_iter == 1:
                logger.warn('DAGPy initialization may have generated poor '
                            'quality cuts.')
            # set optimistic bound to infinity
            if DAGPy.objective.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        else:
            raise ValueError(
                'DAGPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_PSC_master(self):
        self.mip_iter += 1
        m = self.working_model.clone()
        DAGPy = m.DAGPy_utils
        logger.info('MIP {}: Solve master problem.'.format(self.mip_iter))

        # Deactivate nonlinear constraints
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))
        for c in nonlinear_constraints:
            c.deactivate()

        # Transform disjunctions
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK

        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        # Solve
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)

        # Process master problem result
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.objective.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.objective.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.objective.expr),
                                self.LB, self.UB))
        elif master_terminate_cond is tc.infeasible:
            logger.info('MILP master problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if self.mip_iter == 1:
                logger.warn('DAGPy initialization may have generated poor '
                            'quality cuts.')
            # set optimistic bound to infinity
            if DAGPy.objective.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            logger.info('MILP master problem is unbounded. ')
            m.solutions.load_from(results)
        elif master_terminate_cond is tc.maxTimeLimit:
            # TODO check that status is actually ok and everything is feasible
            logger.info('Unable to optimize MILP master problem '
                        'within time limit. '
                        'Using current solver feasible solution.')
            results.solver.status = SolverStatus.ok
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.objective.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.objective.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.objective.expr),
                                self.LB, self.UB))
        else:
            raise ValueError(
                'DAGPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_GBD_master(self, leave_linear_active=False):
        m = self.working_model
        DAGPy = m.DAGPy_utils
        self.mip_iter += 1
        logger.info('MIP {}: Solve master problem.'.format(self.mip_iter))
        if not leave_linear_active:
            # Deactivate all constraints except those in DAGPy_linear_cuts
            _DAGPy_linear_cuts = set(
                c for c in m.DAGPy_linear_cuts.component_data_objects(
                    ctype=Constraint, descend_into=True))
            to_deactivate = set(c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True)
                if c not in _DAGPy_linear_cuts)
            for c in to_deactivate:
                c.deactivate()
        else:
            for c in self.nonlinear_constraints:
                c.deactivate()
        m.DAGPy_linear_cuts.activate()
        m.DAGPy_objective_expr.activate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.mip_solver_kwargs)
        master_terminate_cond = results.solver.termination_condition
        if master_terminate_cond is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it is infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            old_options = deepcopy(self.mip_solver.options)
            # This solver option is specific to Gurobi.
            self.mip_solver.options['DualReductions'] = 0
            results = self.mip_solver.solve(m, load_solutions=False,
                                            **self.mip_solver_kwargs)
            master_terminate_cond = results.solver.termination_condition
            self.mip_solver.options.update(old_options)
        if not leave_linear_active:
            for c in to_deactivate:
                c.activate()
        else:
            for c in self.nonlinear_constraints:
                c.activate()
        m.DAGPy_linear_cuts.deactivate()
        getattr(m, 'ipopt_zL_out', _DoNothing()).activate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).activate()

        # Process master problem result
        if master_terminate_cond is tc.optimal:
            # proceed. Just need integer values
            m.solutions.load_from(results)
            if DAGPy.objective.sense == minimize:
                self.LB = max(value(DAGPy.objective.expr), self.LB)
                self.LB_progress.append(self.LB)
            else:
                self.UB = min(value(DAGPy.objective.expr), self.UB)
                self.UB_progress.append(self.UB)
            logger.info('MIP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.mip_iter, value(DAGPy.objective.expr),
                                self.LB, self.UB))
        elif master_terminate_cond is tc.infeasible:
            logger.info('MILP master problem is infeasible. '
                        'Problem may have no more feasible '
                        'binary configurations.')
            if self.mip_iter == 1:
                logger.warn('DAGPy initialization may have generated poor '
                            'quality cuts.')
            # set optimistic bound to infinity
            if DAGPy.objective.sense == minimize:
                self.LB = float('inf')
            else:
                self.UB = float('-inf')
        elif master_terminate_cond is tc.unbounded:
            logger.info('MILP master problem is unbounded. ')
            # Change the integer values to something new, re-solve.
            m.DAGPy_linear_cuts.activate()
            m.DAGPy_feasible_integer_cuts.activate()
            self._init_max_binaries()
            m.DAGPy_linear_cuts.deactivate()
            m.DAGPy_feasible_integer_cuts.deactivate()
        else:
            raise ValueError(
                'DAGPy unable to handle MILP master termination condition '
                'of {}. Solver message: {}'.format(
                    master_terminate_cond, results.solver.message))

        # Call the MILP post-solve callback
        self.master_postsolve(m, self)

    def _solve_NLP_subproblem(self):
        m = self.working_model.clone()
        DAGPy = m.DAGPy_utils
        self.nlp_iter += 1
        logger.info('NLP {}: Solve subproblem for fixed binaries and '
                    'logical realizations.'
                    .format(self.nlp_iter))
        # Fix binary variables
        binary_vars = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed]
        for v in binary_vars:
            if v.value is None:
                logger.warning('No value is defined for binary variable {}'
                               ' for the NLP subproblem.'.format(v.name))
            else:
                # round the integer variable values so that they are exactly 0
                # or 1
                if self.round_NLP_binaries:
                    v.set_value(round(v.value))
            v.fix()

        # Deactivate the OA and PSC cuts
        for constr in m.component_objects(ctype=Constraint, active=True,
                                          descend_into=(Block, Disjunct)):
            if (constr.local_name == 'DAGPy_OA_cuts' or
                    constr.local_name == 'psc_cuts'):
                constr.deactivate()

        # Activate or deactivate disjuncts according to the value of their
        # indicator variable
        for disj in m.component_data_objects(
                ctype=Disjunct, descend_into=(Block, Disjunct)):
            if fabs(value(disj.indicator_var) - 1) <= self.integer_tolerance:
                # Disjunct is active. Convert to Block.
                disj.parent_block().reclassify_component_type(disj, Block)
            elif fabs(value(disj.indicator_var)) <= self.integer_tolerance:
                disj.deactivate()
            else:
                raise ValueError(
                    'Non-binary value of disjunct indicator variable '
                    'for {}: {}'.format(disj.name, value(disj.indicator_var)))

        # Propagate variable bounds
        TransformationFactory('core.propagate_eq_var_bounds').apply_to(m)
        # Detect fixed variables
        TransformationFactory('core.detect_fixed_vars').apply_to(m)
        # Propagate fixed variables
        TransformationFactory('core.propagate_fixed_vars').apply_to(m)
        # Remove zero terms in linear expressions
        TransformationFactory('core.remove_zero_terms').apply_to(m)
        # Remove terms in equal to zero summations
        TransformationFactory('core.propagate_zero_sum').apply_to(m)
        # Transform bound constraints
        TransformationFactory('core.constraints_to_var_bounds').apply_to(m)
        # Remove trivial constraints
        TransformationFactory(
            'core.deactivate_trivial_constraints').apply_to(m)

        # restore original variable values
        obj_to_cuid = generate_cuid_names(m, ctype=(Var, Constraint, Disjunct),
                                          descend_into=(Block, Disjunct))
        for v in m.component_data_objects(ctype=Var,
                                          descend_into=(Block, Disjunct)):
            if not v.fixed and not v.is_binary():
                try:
                    old_value = self.initial_variable_values[obj_to_cuid[v]]
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

        # Solve the NLP
        results = self.nlp_solver.solve(m, load_solutions=False,
                                        **self.nlp_solver_kwargs)

        solnFeasible = False

        def process_feasible_solution():
            self._copy_values(m, self.working_model, from_map=obj_to_cuid)
            self._copy_dual_suffixes(m, self.working_model,
                                     from_map=obj_to_cuid)
            if DAGPy.objective.sense == minimize:
                self.UB = min(value(DAGPy.objective.expr), self.UB)
                self.solution_improved = self.UB < self.UB_progress[-1]
                self.UB_progress.append(self.UB)
            else:
                self.LB = max(value(DAGPy.objective.expr), self.LB)
                self.solution_improved = self.LB > self.LB_progress[-1]
                self.LB_progress.append(self.LB)
            logger.info('NLP {}: OBJ: {}  LB: {}  UB: {}'
                        .format(self.nlp_iter, value(DAGPy.objective.expr),
                                self.LB, self.UB))
            if self.solution_improved:
                self.best_solution_found = m.clone()

            # Add the linear cut
            if self._decomposition_strategy == 'LOA':
                self._add_oa_cut()
            elif self._decomposition_strategy == 'PSC':
                self._add_psc_cut()
            elif self._decomposition_strategy == 'LGBD':
                self._add_gbd_cut()

            # This adds an integer cut to the DAGPy_feasible_integer_cuts
            # ConstraintList, which is not activated by default. However, it
            # may be activated as needed in certain situations or for certain
            # values of option flags.
            self._add_int_cut(feasible=True)

            self.subproblem_postfeasible(m, self)

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
            self._copy_values(m, self.working_model)
            self._copy_dual_suffixes(m, self.working_model,
                                     from_map=obj_to_cuid)
            if self._decomposition_strategy == 'PSC':
                logger.info('Adding PSC feasibility cut.')
                self._add_psc_cut(nlp_feasible=False)
            elif self._decomposition_strategy == 'LGBD':
                logger.info('Adding GBD feasibility cut.')
                self._add_gbd_cut(nlp_feasible=False)
            # Add an integer cut to exclude this discrete option
            self._add_int_cut()
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
                self._add_int_cut()
        else:
            raise ValueError(
                'DAGPy unable to handle NLP subproblem termination '
                'condition of {}. Results: {}'.format(
                    subprob_terminate_cond, results))

        # Call the NLP post-solve callback
        self.subproblem_postsolve(m, self)
        return solnFeasible

    def _is_feasible(self, m, constr_tol=1E-6, var_tol=1E-8):
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct)):
            # constraint is an equality
            if constr.equality:
                if fabs(value(constr.lower) -
                        value(constr.body)) >= constr_tol:
                    logger.info('{}: {} ≠ {}'.format(
                        constr.name, value(constr.body), value(constr.lower)))
                    return False
            if constr.lower is not None:
                if value(constr.lower) - value(constr.body) >= constr_tol:
                    logger.info('{}: {} < {}'.format(
                        constr.name, value(constr.body), value(constr.lower)))
                    return False
            if constr.upper is not None:
                if value(constr.body) - value(constr.upper) >= constr_tol:
                    logger.info('{}: {} > {}'.format(
                        constr.name, value(constr.body), value(constr.upper)))
                    return False
        for var in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct)):
            if var.lb is not None:
                if value(var.lb) - value(var) >= var_tol:
                    logger.info('{}: {} < {}'.format(
                        var.name, value(var), value(var.lb)))
                    return False
            if var.ub is not None:
                if value(var) - value(var.ub) >= var_tol:
                    logger.info('{}: {} > {}'.format(
                        var.name, value(var), value(var.ub)))
                    return False
        return True

    def _calc_jacobians(self):
        self.jacs = {}
        for c in self.nonlinear_constraints:
            constraint_vars = list(EXPR.identify_variables(c.body))
            jac_list = differentiate(c.body, wrt_list=constraint_vars)
            self.jacs[c] = {id(var): jac
                            for var, jac in zip(constraint_vars, jac_list)}

    def _add_oa_cut(self, for_GBD=False):
        """Add outer approximation cuts to working model.

        If for_GBD flag is True, then place the cuts in a component called
        DAGPy_OA_cuts_for_GBD and deactivate them by default.
        """
        m, DAGPy = self.working_model, self.working_model.DAGPy_utils
        DAGPy.nlp_iters.add(self.nlp_iter)
        sign_adjust = -1 if DAGPy.objective.sense == minimize else 1

        # deactivate inactive disjuncts
        deactivated_disj = set()
        for disj in m.component_data_objects(ctype=Disjunct, active=True,
                                             descend_into=(Block, Disjunct)):
            if fabs(value(disj.indicator_var)) <= self.integer_tolerance:
                disj.deactivate()
                deactivated_disj.add(disj)
            elif fabs(value(disj.indicator_var) - 1) <= self.integer_tolerance:
                pass  # This is fine, keep going.
            else:
                raise ValueError("Disjunct {} has an indicator variable "
                                 "value that falls outside of integer "
                                 "tolerance from 0 or 1: {}"
                                 .format(disj.name, disj.indicator_var.value))

        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))

        # generate new constraints
        # TODO some kind of special handling if the dual is phenomenally small?
        logger.info('Adding OA cuts.')
        for constr in nonlinear_constraints:
            if not m.dual.get(constr, None):
                continue
            parent_block = constr.parent_block()

            if not for_GBD:
                oa_cuts = parent_block.component('DAGPy_OA_cuts')
                if oa_cuts is None:
                    oa_cuts = parent_block.DAGPy_OA_cuts = ConstraintList()
            else:
                oa_cuts = parent_block.component('DAGPy_OA_cuts_for_GBD')
                if oa_cuts is None:
                    oa_cuts = parent_block.DAGPy_OA_cuts_for_GBD = \
                        ConstraintList()
                oa_cuts.deactivate()
            # m.dual.display()
            c = oa_cuts.add(
                expr=copysign(1, sign_adjust * m.dual[constr]) * sum(
                    value(self.jacs[constr][id(var)]) * (var - value(var))
                    for var in list(EXPR.identify_variables(constr.body))) +
                DAGPy.slack_vars[self.nlp_iter, self.nl_map[constr]] <= 0)
            DAGPy.OA_constr_map[constr, self.nlp_iter] = c

        # Restore deactivated constraints
        for disj in deactivated_disj:
            disj.activate()

    def _add_psc_cut(self, nlp_feasible=True):
        m, DAGPy = self.working_model, self.working_model.DAGPy_utils

        sign_adjust = 1 if DAGPy.objective.sense == minimize else -1

        nonlinear_variables, nonlinear_variable_IDs = \
            self._detect_nonlinear_vars(m)
        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))

        # generate the sum of all multipliers with the nonlinear constraints
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in nonlinear_variables}
        sum_nonlinear = (
            # Address constraints of form f(x) <= upper
            sum(value(m.dual[c]) * -1 *
                (clone_expression(c.body, substitute=var_to_val) - c.upper)
                for c in nonlinear_constraints
                if fabs(value(m.dual.get(c, 0))) > self.small_dual_tolerance
                and c.upper is not None) +
            # Address constraints of form f(x) >= lower
            sum(value(m.dual[c]) *
                (c.lower - clone_expression(c.body, substitute=var_to_val))
                for c in nonlinear_constraints
                if fabs(value(m.dual.get(c, 0))) > self.small_dual_tolerance
                and c.lower is not None and not c.upper == c.lower))

        # Generate the sum of all multipliers with linear constraints
        # containing nonlinear variables
        #
        # For now, need to generate canonical representation in order to get
        # the coefficients on the linear terms.
        lin_cons = [c for c in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True)
            if c.body.polynomial_degree() in (0, 1)]
        # Create a coefficient dictionary mapping variables to their
        # coefficient in the expression. Constraint -> (id(Var) -> coefficient)
        coef_dict = {}
        constr_vars = {}
        for constr in lin_cons:
            repn = generate_canonical_repn(constr.body)
            if repn.variables is None or repn.linear is None:
                repn.variables = []
                repn.linear = []
            coef_dict[constr] = {id(var): coef for var, coef in
                                 zip(repn.variables, repn.linear)}
            constr_vars[constr] = repn.variables
        sum_linear = sum(
            m.dual[c] *
            sum(coef_dict[c][id(var)] * (var - value(var))
                for var in constr_vars[c]
                if id(var) in nonlinear_variable_IDs)
            for c in lin_cons
            if fabs(value(m.dual.get(c, 0))) > self.small_dual_tolerance)

        # Generate the sum of all bound multipliers with nonlinear variables
        sum_var_bounds = (
            sum(m.ipopt_zL_out.get(var, 0) * (var - value(var))
                for var in nonlinear_variables
                if fabs(value(m.ipopt_zL_out.get(var, 0))) >
                self.small_dual_tolerance) +
            sum(m.ipopt_zU_out.get(var, 0) * (var - value(var))
                for var in nonlinear_variables
                if fabs(value(m.ipopt_zU_out.get(var, 0))) >
                self.small_dual_tolerance))

        if nlp_feasible:
            # Optimality cut (for feasible NLP)
            DAGPy.psc_cuts.add(
                expr=DAGPy.objective.expr * sign_adjust >= sign_adjust * (
                    DAGPy.objective.expr + sum_nonlinear + sum_linear +
                    sum_var_bounds))
        else:
            # Feasibility cut (for infeasible NLP)
            DAGPy.psc_cuts.add(
                expr=sign_adjust * (
                    sum_nonlinear + sum_linear + sum_var_bounds) <= 0)

    def _add_gbd_cut(self, nlp_feasible=True):
        m = self.working_model
        DAGPy = m.DAGPy_utils

        sign_adjust = 1 if DAGPy.objective.sense == minimize else -1

        # linearize the GDP, store linearized cuts somewhere.
        if nlp_feasible:
            self._add_oa_cut(for_GBD=True)

        # Fix the binary variables. Solve resulting LP.
        self._solve_LP_subproblem_for_GBD()

        # Generate Benders cuts based on dual values.
        # Objective constraints
        obj_constr_list = [self.OA_constr_map[DAGPy.objective_expr, i]
                           for i in self.nlp_iters]
        var_to_val = {id(var): NumericConstant(value(var))
                      for var in m.component_data_objects(
                          ctype=Var, descend_into=(Block, Constraint))
                      if not var.is_binary() and not var.is_integer()}

        DAGPy.gbd_cuts.add(
            expr=DAGPy.objective.expr * sign_adjust >= sign_adjust * (
                DAGPy.objective.expr +
                # Address constraints of form f(x) <= upper
                sum((m.dual[c] * - 1 *
                     (clone_expression(c.body, substitute=var_to_val) -
                      c.upper))
                    for c in obj_constr_list
                    if fabs(m.dual[c]) > self.small_dual_tolerance
                    and c.upper is not None) +
                # Address constraints of form f(x) >= lower
                sum((m.dual[c] *
                     (c.lower -
                      clone_expression(c.body, substitute=var_to_val))
                     for c in obj_constr_list
                     if fabs(m.dual[c]) > self.small_dual_tolerance
                     and c.lower is not None and not c.upper == c.lower))
            ))

    def _solve_LP_subproblem_for_GBD(self):
        m = self.working_model.clone()
        DAGPy = m.DAGPy_utils

        # Activate the OA cuts for GBD, deactivate nonlinear constraints
        for constr in m.component_objects(ctype=Constraint, active=None,
                                          descend_into=(Block, Disjunct)):
            if (constr.local_name == 'DAGPy_OA_cuts' or
                    constr.local_name == 'DAGPy_OA_cuts_for_GBD'):
                constr.activate()
        for constr in m.component_data_objects(ctype=Constraint, active=True,
                                               descend_into=(Block, Disjunct)):
            if constr.body.polynomial_degree() not in (0, 1):
                constr.deactivate()
        # Fix the slacks to zero
        DAGPy.slack_vars.fix(0)
        # Transform the model
        TransformationFactory('gdp.bigm').apply_to(m)
        TransformationFactory('gdp.reclassify').apply_to(m)  # HACK

        # Identify transformed OA constraints
        m.display()
        exit()

        # Fix the binary variables
        # TODO should this be done before transformation?
        binary_vars = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed]
        for v in binary_vars:
            if v.value is None:
                logger.warning('No value is defined for binary variable {}'
                               ' for the NLP subproblem.'.format(v.name))
            v.fix()

        results = self.mip_solver.solve(m, load_solutions=False,
                                        **self.nlp_solver_kwargs)
        terminate_cond = results.solver.termination_condition
        if terminate_cond is tc.optimal:
            # copy the values back
            m.solutions.load_from(results)
            self._copy_values(m, self.working_model)
            self._copy_dual_suffixes(m, self.working_model)
        elif terminate_cond is tc.infeasible:
            # Something went wrong
            raise ValueError('DAGPy LP subproblem for logic-based GBD '
                             'is infeasible. This should not be possible. '
                             'Was the NLP subproblem feasible?')

    def _add_int_cut(self, feasible=False):
        m, DAGPy = self.working_model, self.working_model.DAGPy_utils
        int_tol = self.integer_tolerance
        binary_vars = [
            v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v.is_binary() and not v.fixed]
        # check to make sure that binary variables are all 0 or 1
        for v in binary_vars:
            if fabs(v.value - 1) > int_tol and fabs(value(v)) > int_tol:
                raise ValueError('Binary {} = {} is not 0 or 1'.format(
                    v.name, value(v)))

        if not binary_vars:  # if no binary variables, skip.
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
            DAGPy.integer_cuts.add(expr=int_cut)
            # DAGPy.integer_cuts.pprint()
        else:
            DAGPy.feasible_integer_cuts.add(expr=int_cut)

    def _detect_nonlinear_vars(self, m):
        """Identify the variables that participate in nonlinear terms."""
        nonlinear_variables = []
        # This is a workaround because Var is not hashable, and I do not want
        # duplicates in nonlinear_variables.
        seen = set()

        nonlinear_constraints = (
            c for c in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block, Disjunct))
            if c.body.polynomial_degree() not in (0, 1))

        for constr in nonlinear_constraints:
            if isinstance(constr.body, EXPR._SumExpression):
                # go through each term and check to see if the term is
                # nonlinear
                for expr in constr.body._args:
                    # Check to see if the expression is nonlinear
                    if expr.polynomial_degree() not in (0, 1):
                        # collect variables
                        for var in EXPR.identify_variables(
                                expr, include_fixed=False):
                            if id(var) not in seen:
                                seen.add(id(var))
                                nonlinear_variables.append(var)
            # if the root expression object is not a summation, then something
            # else is the cause of the nonlinearity. Collect all participating
            # variables.
            else:
                # collect variables
                for var in EXPR.identify_variables(constr.body,
                                                   include_fixed=False):
                    if id(var) not in seen:
                        seen.add(id(var))
                        nonlinear_variables.append(var)
        nonlinear_variable_IDs = set(
            id(v) for v in nonlinear_variables)

        return nonlinear_variables, nonlinear_variable_IDs


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
