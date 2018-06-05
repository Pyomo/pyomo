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
from __future__ import division

import logging

import pyomo.common.plugin
from pyomo.common.config import (ConfigBlock, ConfigList, ConfigValue, In,
                                 NonNegativeFloat, NonNegativeInt)
from pyomo.contrib.gdpopt.master_initialize import (init_custom_disjuncts,
                                                    init_set_covering,
                                                    init_max_binaries)
from pyomo.contrib.gdpopt.nlp_solve import (solve_NLP,
                                            update_nlp_progress_indicators)
from pyomo.contrib.gdpopt.cut_generation import (add_integer_cut,
                                                 add_outer_approximation_cuts)
from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.util import GDPoptSolveData, _DoNothing, a_logger
from pyomo.core.base import (Block, Constraint, ConstraintList, Expression,
                             Objective, Suffix, TransformationFactory,
                             Var, minimize, value, Reals)
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt.base import IOptSolver
from pyomo.opt.results import ProblemSense, SolverResults


__version__ = (0, 2, 0)


class GDPoptSolver(pyomo.common.plugin.Plugin):
    """A decomposition-based GDP solver."""

    pyomo.common.plugin.implements(IOptSolver)
    pyomo.common.plugin.alias(
        'gdpopt', doc='The GDPopt decomposition-based GDP solver')

    CONFIG = ConfigBlock("GDPopt")
    CONFIG.declare("bound_tolerance", ConfigValue(
        default=1E-6, domain=NonNegativeFloat,
        description="Tolerance for bound convergence."
    ))
    CONFIG.declare("iterlim", ConfigValue(
        default=30, domain=NonNegativeInt,
        description="Iteration limit."
    ))
    CONFIG.declare("strategy", ConfigValue(
        default="LOA", domain=In(["LOA"]),
        description="Decomposition strategy to use."
    ))
    CONFIG.declare("init_strategy", ConfigValue(
        default="set_covering", domain=In([
            "set_covering", "max_binary", "fixed_binary", "custom_disjuncts"]),
        description="Initialization strategy to use.",
        doc="""Selects the initialization strategy to use when generating
        the initial cuts to construct the master problem."""
    ))
    CONFIG.declare("custom_init_disjuncts", ConfigList(
        # domain=ComponentSets of Disjuncts,
        default=None,
        description="List of disjunct sets to use for initialization."
    ))
    CONFIG.declare("max_slack", ConfigValue(
        default=1000, domain=NonNegativeFloat,
        description="Upper bound on slack variables for OA"
    ))
    CONFIG.declare("OA_penalty_factor", ConfigValue(
        default=1000, domain=NonNegativeFloat,
        description="Penalty multiplication term for slack variables on the "
        "objective value."
    ))
    CONFIG.declare("set_cover_iterlim", ConfigValue(
        default=8, domain=NonNegativeInt,
        description="Limit on the number of set covering iterations."
    ))
    CONFIG.declare("mip", ConfigValue(
        default="gurobi",
        description="Mixed integer linear solver to use."
    ))
    mip_options = CONFIG.declare("mip_options", ConfigBlock(implicit=True))
    CONFIG.declare("nlp", ConfigValue(
        default="ipopt",
        description="Nonlinear solver to use"))
    nlp_options = CONFIG.declare("nlp_options", ConfigBlock(implicit=True))
    CONFIG.declare("master_postsolve", ConfigValue(
        default=_DoNothing,
        description="callback hook after a solution of the master problem"
    ))
    CONFIG.declare("subprob_presolve", ConfigValue(
        default=_DoNothing,
        description="callback hook before calling the subproblem solver"
    ))
    CONFIG.declare("subprob_postsolve", ConfigValue(
        default=_DoNothing,
        description="callback hook after a solution of the "
        "nonlinear subproblem"
    ))
    CONFIG.declare("subprob_postfeas", ConfigValue(
        default=_DoNothing,
        description="callback hook after feasible solution of "
        "the nonlinear subproblem"
    ))
    CONFIG.declare("algorithm_stall_after", ConfigValue(
        default=2,
        description="number of non-improving master iterations after which "
        "the algorithm will stall and exit."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        description="Stream output to terminal.",
        domain=bool
    ))
    CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.gdpopt',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    CONFIG.declare("small_dual_tolerance", ConfigValue(
        default=1E-8,
        description="When generating cuts, small duals multiplied "
        "by expressions can cause problems. Exclude all duals "
        "smaller in absolue value than the following."
    ))
    CONFIG.declare("integer_tolerance", ConfigValue(
        default=1E-5,
        description="Tolerance on integral values."
    ))
    CONFIG.declare("constraint_tolerance", ConfigValue(
        default=1E-6,
        description="Tolerance on constraint satisfaction."
    ))
    CONFIG.declare("variable_tolerance", ConfigValue(
        default=1E-8,
        description="Tolerance on variable bounds."
    ))
    CONFIG.declare("round_NLP_binaries", ConfigValue(
        default=True,
        description="flag to round binary values to exactly 0 or 1. "
        "Rounding is done before solving NLP subproblem"
    ))

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

        """
        config = self.CONFIG(kwds.pop('options', {}))

        config.set_value(kwds)

        solve_data = GDPoptSolveData()

        old_logger_level = config.logger.getEffectiveLevel()
        try:
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)

            # Modify in place decides whether to run the algorithm on a copy of
            # the originally model passed to the solver, or whether to
            # manipulate the original model directly.
            solve_data.working_model = m = model

            solve_data.current_strategy = config.strategy

            # Create a model block on which to store GDPopt-specific utility
            # modeling objects.
            # TODO check if name is taken already
            GDPopt = m.GDPopt_utils = Block()

            GDPopt.initial_var_list = list(v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct)
            ))
            GDPopt.initial_var_values = list(
                v.value for v in GDPopt.initial_var_list)
            GDPopt.initial_disjuncts_list = list(
                m.component_data_objects(
                    ctype=Disjunct, descend_into=(Block, Disjunct)))
            GDPopt.initial_constraints_list = list(
                m.component_data_objects(
                    ctype=Constraint, descend_into=(Block, Disjunct)))
            GDPopt.initial_nonlinear_constraints = [
                v for v in GDPopt.initial_constraints_list
                if v.body.polynomial_degree() not in (0, 1)]

            # Store the initial model state as the best solution found. If we
            # find no better solution, then we will restore from this copy.
            # print('Initial clone for best_solution_found')
            solve_data.best_solution_found = model.clone()

            # Save model initial values. These are used later to initialize NLP
            # subproblems.
            GDPopt.initial_var_values = list(
                v.value for v in GDPopt.initial_var_list)

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
            self._validate_model(config, solve_data)

            # Maps in order to keep track of certain generated constraints
            GDPopt.oa_cut_map = Suffix(direction=Suffix.LOCAL, datatype=None)

            # Integer cuts exclude particular discrete decisions
            GDPopt.integer_cuts = ConstraintList(doc='integer cuts')

            # Feasible integer cuts exclude discrete realizations that have
            # been explored via an NLP subproblem. Depending on model
            # characteristics, the user may wish to revisit NLP subproblems
            # (with a different initialization, for example). Therefore, these
            # cuts are not enabled by default, unless the initial model has no
            # discrete decisions.

            # Note: these cuts will only exclude integer realizations that are
            # not already in the primary GDPopt_integer_cuts ConstraintList.
            GDPopt.exclude_explored_configurations = ConstraintList(
                doc='explored integer cuts')
            if not GDPopt._no_discrete_decisions:
                GDPopt.exclude_explored_configurations.deactivate()

            # Set up iteration counters
            solve_data.nlp_iter = 0
            solve_data.mip_iter = 0
            solve_data.mip_subiter = 0

            # set up bounds
            solve_data.LB = float('-inf')
            solve_data.UB = float('inf')
            solve_data.LB_progress = [solve_data.LB]
            solve_data.UB_progress = [solve_data.UB]

            # Flag indicating whether the solution improved in the past
            # iteration or not
            solve_data.solution_improved = False

            # Initialize the master problem
            self._GDPopt_initialize_master(solve_data, config)

            # Algorithm main loop
            self._GDPopt_iteration_loop(solve_data, config)

            # Update values in original model
            for var, val in zip(GDPopt.initial_var_list,
                                solve_data.best_solution_found):
                var.value = val

            solve_data.results.problem.lower_bound = solve_data.LB
            solve_data.results.problem.upper_bound = solve_data.UB

        finally:
            config.logger.setLevel(old_logger_level)

    def _validate_model(self, config, solve_data):
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
            config.logger.info('Problem has no discrete decisions.')
            GDPopt._no_discrete_decisions = True
        else:
            GDPopt._no_discrete_decisions = False

        # Handle missing or multiple objectives
        objs = list(m.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        num_objs = len(objs)
        solve_data.results.problem.number_of_objectives = num_objs
        if num_objs == 0:
            config.logger.warning(
                'Model has no active objectives. Adding dummy objective.')
            GDPopt.dummy_objective = Objective(expr=1)
            main_obj = GDPopt.dummy_objective
        elif num_objs > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = objs[0]

        # Move the objective to the constraints

        # TODO only move the objective if nonlinear?
        GDPopt.objective_value = Var(domain=Reals, initialize=0)
        solve_data.objective_sense = main_obj.sense
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

    def _GDPopt_initialize_master(self, solve_data, config):
        """Initialize the decomposition algorithm.

        This includes generating the initial cuts require to build the master
        problem.

        """
        m = solve_data.working_model
        if not hasattr(m, 'dual'):  # Set up dual value reporting
            m.dual = Suffix(direction=Suffix.IMPORT)
        m.dual.activate()

        solve_data.linear_GDP = m.clone()
        # deactivate nonlinear constraints
        for c in solve_data.linear_GDP.GDPopt_utils.\
                initial_nonlinear_constraints:
            c.deactivate()

        if config.init_strategy == 'set_covering':
            init_set_covering(solve_data, config)
        elif config.init_strategy == 'max_binary':
            init_max_binaries(solve_data, config)
        elif config.init_strategy == 'fixed_binary':
            self._validate_disjunctions(solve_data, config)
            self._solve_NLP_subproblem(solve_data, config)
        elif config.init_strategy == 'custom_disjuncts':
            init_custom_disjuncts(solve_data, config)
        else:
            raise ValueError('Unknown initialization strategy: %s'
                             % (config.init_strategy,))

    def _validate_disjunctions(self, solve_data, config):
        """Validate if the disjunctions are satisfied by the current values."""
        # TODO implement this? If not, the user will simply get an infeasible
        # return value
        pass

    def _GDPopt_iteration_loop(self, solve_data, config):
        m = solve_data.working_model
        GDPopt = m.GDPopt_utils
        # Backup counter to prevent infinite loop
        backup_max_iter = max(1000, config.iterlim)
        backup_iter = 0
        while backup_iter < backup_max_iter:
            # print line for visual display
            config.logger.info('---Master Iteration---')
            backup_iter += 1
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                config.logger.info(
                    'GDPopt exiting on bound convergence. '
                    'LB: %s + (tol %s) >= UB: %s' %
                    (solve_data.LB, config.bound_tolerance,
                     solve_data.UB))
                break
            # Check iteration limit
            if solve_data.mip_iter >= config.iterlim:
                config.logger.info(
                    'GDPopt unable to converge bounds '
                    'after %s master iterations.'
                    % (solve_data.mip_iter,))
                config.logger.info(
                    'Final bound values: LB: %s  UB: %s'
                    % (solve_data.LB, solve_data.UB))
                break
            solve_data.mip_subiter = 0
            # solve MILP master problem
            if solve_data.current_strategy == 'LOA':
                mip_results = self._solve_OA_master(solve_data, config)
                _, mip_var_values = mip_results
            # Check bound convergence
            if solve_data.LB + config.bound_tolerance >= solve_data.UB:
                config.logger.info(
                    'GDPopt exiting on bound convergence. '
                    'LB: %s + (tol %s) >= UB: %s'
                    % (solve_data.LB, config.bound_tolerance,
                       solve_data.UB))
                break
            # Solve NLP subproblem
            nlp_model = solve_data.working_model.clone()
            # copy in the discrete variable values
            for var, val in zip(nlp_model.GDPopt_utils.initial_var_list,
                                mip_var_values):
                if val is None:
                    continue
                else:
                    var.value = val
            TransformationFactory('gdp.fix_disjuncts').apply_to(nlp_model)
            solve_data.nlp_iter += 1
            nlp_result = solve_NLP(nlp_model, solve_data, config)
            nlp_feasible, nlp_var_values, nlp_duals = nlp_result
            if nlp_feasible:
                update_nlp_progress_indicators(nlp_model, solve_data, config)
                add_outer_approximation_cuts(
                    nlp_var_values, nlp_duals, solve_data, config)
            add_integer_cut(
                mip_var_values, solve_data, config, feasible=nlp_feasible)

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

            # Max number of iterations in which upper (feasible) bound does not
            # improve before turning on no-backtracking
            no_backtrack_after = 1
            if (len(feas_prog_log) > no_backtrack_after and
                (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
                 >= sign_adjust * feas_prog_log[-1 - no_backtrack_after])):
                if not GDPopt.exclude_explored_configurations.active:
                    config.logger.info(
                        'Feasible solutions not making enough '
                        'progress for %s iterations. '
                        'Turning on no-backtracking '
                        'integer cuts.' % (no_backtrack_after,))
                    GDPopt.exclude_explored_configurations.activate()

            # Maximum number of iterations in which feasible bound does not
            # improve before terminating algorithm
            if (len(feas_prog_log) > config.algorithm_stall_after and
                (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
                 >= sign_adjust *
                 feas_prog_log[-1 - config.algorithm_stall_after])):
                config.logger.info(
                    'Feasible solutions not making enough progress '
                    'for %s iterations. Algorithm stalled. Exiting.\n'
                    'To continue, increase value of parameter '
                    'algorithm_stall_after.'
                    % (config.algorithm_stall_after,))
                break

    def _solve_OA_master(self, solve_data, config):
        solve_data.mip_iter += 1
        m = solve_data.linear_GDP.clone()
        GDPopt = m.GDPopt_utils
        config.logger.info(
            'MIP %s: Solve master problem.' %
            (solve_data.mip_iter,))

        # Set up augmented Lagrangean penalty objective
        GDPopt.objective.deactivate()
        sign_adjust = 1 if GDPopt.objective.sense == minimize else -1
        GDPopt.OA_penalty_expr = Expression(
            expr=sign_adjust * config.OA_penalty_factor *
            sum(v for v in m.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
                if v.parent_component().local_name == 'GDPopt_OA_slacks'))
        GDPopt.oa_obj = Objective(
            expr=GDPopt.objective.expr + GDPopt.OA_penalty_expr,
            sense=GDPopt.objective.sense)

        mip_results = solve_linear_GDP(m, solve_data, config)
        if mip_results:
            if GDPopt.objective.sense == minimize:
                solve_data.LB = max(value(GDPopt.oa_obj.expr), solve_data.LB)
                solve_data.LB_progress.append(solve_data.LB)
            else:
                solve_data.UB = min(value(GDPopt.oa_obj.expr), solve_data.UB)
                solve_data.UB_progress.append(solve_data.UB)
            config.logger.info(
                'MIP %s: OBJ: %s  LB: %s  UB: %s'
                % (solve_data.mip_iter, value(GDPopt.oa_obj.expr),
                   solve_data.LB, solve_data.UB))
        else:
            if solve_data.mip_iter == 1:
                config.logger.warn(
                    'GDPopt initialization may have generated poor '
                    'quality cuts.')
            # set optimistic bound to infinity
            if GDPopt.objective.sense == minimize:
                solve_data.LB = float('inf')
                solve_data.LB_progress.append(solve_data.UB)
            else:
                solve_data.UB = float('-inf')
                solve_data.UB_progress.append(solve_data.UB)
        # Call the MILP post-solve callback
        config.master_postsolve(m, solve_data)

        return mip_results

    def _is_feasible(self, m, config, constr_tol=1E-6, var_tol=1E-8):
        for constr in m.component_data_objects(
                ctype=Constraint, active=True, descend_into=True):
            # Check constraint lower bound
            if (constr.lower is not None and (
                value(constr.lower) - value(constr.body)
                >= config.constraint_tolerance
            )):
                config.logger.info('%s: body %s < LB %s' % (
                    constr.name, value(constr.body), value(constr.lower)))
                return False
            # check constraint upper bound
            if (constr.upper is not None and (
                value(constr.body) - value(constr.upper)
                >= config.constraint_tolerance
            )):
                config.logger.info('%s: body %s > UB %s' % (
                    constr.name, value(constr.body), value(constr.upper)))
                return False
        for var in m.component_data_objects(ctype=Var, descend_into=True):
            # Check variable lower bound
            if (var.has_lb() and
                    value(var.lb) - value(var) >= config.variable_tolerance):
                config.logger.info('%s: %s < LB %s' % (
                    var.name, value(var), value(var.lb)))
                return False
            # Check variable upper bound
            if (var.has_ub() and
                    value(var) - value(var.ub) >= config.variable_tolerance):
                config.logger.info('%s: %s > UB %s' % (
                    var.name, value(var), value(var.ub)))
                return False
        return True
