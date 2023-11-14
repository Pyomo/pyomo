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

"""Iteration loop for MindtPy."""
import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
    SolverFactory,
    SolverResults,
    ProblemSense,
    SolutionStatus,
    SolverStatus,
)
from pyomo.core import (
    minimize,
    maximize,
    Objective,
    VarList,
    Reals,
    ConstraintList,
    Constraint,
    Block,
    TransformationFactory,
    NonNegativeReals,
    Suffix,
    Var,
    RangeSet,
    value,
    Expression,
)
from pyomo.contrib.gdpopt.util import (
    SuppressInfeasibleWarning,
    _DoNothing,
    lower_logger_level_to,
    copy_var_list_values,
    get_main_elapsed_time,
    time_code,
)
from pyomo.contrib.gdpopt.solve_discrete_problem import (
    distinguish_mip_infeasible_or_unbounded,
)
from pyomo.contrib.mindtpy.util import (
    generate_norm1_objective_function,
    generate_norm2sq_objective_function,
    generate_norm_inf_objective_function,
    generate_lag_objective_function,
    GurobiPersistent4MindtPy,
    setup_results_object,
    get_integer_solution,
    initialize_feas_subproblem,
    epigraph_reformulation,
    add_var_bound,
    copy_var_list_values_from_solution_pool,
    generate_norm_constraint,
    fp_converged,
    add_orthogonality_cuts,
    set_solver_mipgap,
    set_solver_constraint_violation_tolerance,
    update_solver_timelimit,
)

single_tree, single_tree_available = attempt_import('pyomo.contrib.mindtpy.single_tree')
tabu_list, tabu_list_available = attempt_import('pyomo.contrib.mindtpy.tabu_list')
egb, egb_available = attempt_import(
    'pyomo.contrib.pynumero.interfaces.external_grey_box'
)


class _MindtPyAlgorithm(object):
    def __init__(self, **kwds):
        """
        This is a common init method for all the MindtPy algorithms, so that we
        correctly set up the config arguments and initialize the generic parts
        of the algorithm state.

        """
        self.working_model = None
        self.mip = None
        self.fixed_nlp = None

        # We store bounds, timing info, iteration count, incumbent, and the
        # expression of the original (possibly nonlinear) objective function.
        self.results = SolverResults()
        self.timing = Bunch()
        self.curr_int_sol = []
        self.should_terminate = False
        self.integer_list = []

        # Set up iteration counters
        self.nlp_iter = 0
        self.mip_iter = 0
        self.mip_subiter = 0
        self.nlp_infeasible_counter = 0
        self.fp_iter = 1

        self.primal_bound_progress_time = [0]
        self.dual_bound_progress_time = [0]
        self.abs_gap = float('inf')
        self.rel_gap = float('inf')
        self.log_formatter = (
            ' {:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
        )
        self.fixed_nlp_log_formatter = (
            '{:1}{:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
        )
        self.log_note_formatter = ' {:>9}   {:>15}   {:>15}'

        # Flag indicating whether the solution improved in the past
        # iteration or not
        self.primal_bound_improved = False
        self.dual_bound_improved = False

        # Store the initial model state as the best solution found. If we
        # find no better solution, then we will restore from this copy.
        self.best_solution_found = None
        self.best_solution_found_time = None

        self.stored_bound = {}
        self.num_no_good_cuts_added = {}
        self.last_iter_cuts = False
        # Store the OA cuts generated in the mip_start_process.
        self.mip_start_lazy_oa_cuts = []
        # Whether to load solutions in solve() function
        self.load_solutions = True

    # Support use as a context manager under current solver API
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Solver is always available. Though subsolvers may not be, they will
        raise an error when the time comes.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

    _metasolver = False

    def _log_solver_intro_message(self):
        self.config.logger.info(
            "Starting MindtPy version %s using %s algorithm"
            % (".".join(map(str, self.version())), self.config.strategy)
        )
        os = StringIO()
        self.config.display(ostream=os)
        self.config.logger.info(os.getvalue())
        self.config.logger.info(
            '-----------------------------------------------------------------------------------------------\n'
            '               Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)                \n'
            '-----------------------------------------------------------------------------------------------\n'
            'For more information, please visit \n'
            'https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html'
        )
        self.config.logger.info(
            'If you use this software, please cite the following:\n'
            'Bernal, David E., et al. Mixed-integer nonlinear decomposition toolbox for Pyomo (MindtPy).\n'
            'Computer Aided Chemical Engineering. Vol. 44. Elsevier, 2018. 895-900.\n'
        )

    def set_up_logger(self):
        """Set up the formatter and handler for logger."""
        self.config.logger.handlers.clear()
        self.config.logger.propagate = False
        ch = logging.StreamHandler()
        ch.setLevel(self.config.logging_level)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        # add the handlers to logger
        self.config.logger.addHandler(ch)

    def _log_header(self, logger):
        # TODO: rewrite
        logger.info(
            '================================================================='
            '============================'
        )
        logger.info(
            '{:^9} | {:^15} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration',
                'Subproblem Type',
                'Lower Bound',
                'Upper Bound',
                ' Gap ',
                'Time(s)',
            )
        )

    def create_utility_block(self, model, name):
        created_util_block = False
        # Create a model block on which to store MindtPy-specific utility
        # modeling objects.
        if hasattr(model, name):
            raise RuntimeError(
                "MindtPy needs to create a Block named %s "
                "on the model object, but an attribute with that name "
                "already exists." % name
            )
        else:
            created_util_block = True
            setattr(
                model,
                name,
                Block(doc="Container for MindtPy solver utility modeling objects"),
            )
            self.util_block_name = name

            # Save ordered lists of main modeling components, so that data can
            # be easily transferred between future model clones.
            self.build_ordered_component_lists(model)
            self.add_cuts_components(model)

    def model_is_valid(self):
        """Determines whether the model is solvable by MindtPy.

        Returns
        -------
        bool
            True if model is solvable in MindtPy, False otherwise.
        """
        m = self.working_model
        MindtPy = m.MindtPy_utils
        config = self.config

        # Handle LP/NLP being passed to the solver
        prob = self.results.problem
        if len(MindtPy.discrete_variable_list) == 0:
            config.logger.info('Problem has no discrete decisions.')
            obj = next(m.component_data_objects(ctype=Objective, active=True))
            if (
                any(
                    c.body.polynomial_degree()
                    not in self.mip_constraint_polynomial_degree
                    for c in MindtPy.constraint_list
                )
                or obj.expr.polynomial_degree()
                not in self.mip_objective_polynomial_degree
            ):
                config.logger.info(
                    'Your model is a NLP (nonlinear program). '
                    'Using NLP solver %s to solve.' % config.nlp_solver
                )
                update_solver_timelimit(
                    self.nlp_opt, config.nlp_solver, self.timing, config
                )
                self.nlp_opt.solve(
                    self.original_model,
                    tee=config.nlp_solver_tee,
                    **config.nlp_solver_args,
                )
                return False
            else:
                config.logger.info(
                    'Your model is an LP (linear program). '
                    'Using LP solver %s to solve.' % config.mip_solver
                )
                if isinstance(self.mip_opt, PersistentSolver):
                    self.mip_opt.set_instance(self.original_model)
                update_solver_timelimit(
                    self.mip_opt, config.mip_solver, self.timing, config
                )
                results = self.mip_opt.solve(
                    self.original_model,
                    tee=config.mip_solver_tee,
                    load_solutions=self.load_solutions,
                    **config.mip_solver_args,
                )
                if len(results.solution) > 0:
                    self.original_model.solutions.load_from(results)
                return False

        # Set up dual value reporting
        if config.calculate_dual_at_solution:
            if not hasattr(m, 'dual'):
                m.dual = Suffix(direction=Suffix.IMPORT)
            elif not isinstance(m.dual, Suffix):
                raise ValueError(
                    "dual is not defined as a Suffix in the original model."
                )

        # TODO if any continuous variables are multiplied with binary ones,
        #  need to do some kind of transformation (Glover?) or throw an error message
        return True

    def build_ordered_component_lists(self, model):
        """Define lists used for future data transfer.

        Also attaches ordered lists of the variables, constraints to the model so that they can be used for mapping back and
        forth.

        """
        util_block = getattr(model, self.util_block_name)
        var_set = ComponentSet()
        util_block.constraint_list = list(
            model.component_data_objects(
                ctype=Constraint, active=True, descend_into=(Block)
            )
        )
        if egb_available:
            util_block.grey_box_list = list(
                model.component_data_objects(
                    ctype=egb.ExternalGreyBoxBlock, active=True, descend_into=(Block)
                )
            )
        else:
            util_block.grey_box_list = []
        util_block.linear_constraint_list = list(
            c
            for c in util_block.constraint_list
            if c.body.polynomial_degree() in self.mip_constraint_polynomial_degree
        )
        util_block.nonlinear_constraint_list = list(
            c
            for c in util_block.constraint_list
            if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree
        )
        util_block.objective_list = list(
            model.component_data_objects(
                ctype=Objective, active=True, descend_into=(Block)
            )
        )

        # Identify the non-fixed variables in (potentially) active constraints and
        # objective functions
        for constr in getattr(util_block, 'constraint_list'):
            for v in EXPR.identify_variables(constr.body, include_fixed=False):
                var_set.add(v)
        for obj in model.component_data_objects(ctype=Objective, active=True):
            for v in EXPR.identify_variables(obj.expr, include_fixed=False):
                var_set.add(v)

        # We use component_data_objects rather than list(var_set) in order to
        # preserve a deterministic ordering.
        if egb_available:
            util_block.variable_list = list(
                v
                for v in model.component_data_objects(
                    ctype=Var, descend_into=(Block, egb.ExternalGreyBoxBlock)
                )
                if v in var_set
            )
        else:
            util_block.variable_list = list(
                v
                for v in model.component_data_objects(ctype=Var, descend_into=(Block))
                if v in var_set
            )
        util_block.discrete_variable_list = list(
            v for v in util_block.variable_list if v in var_set and v.is_integer()
        )
        util_block.continuous_variable_list = list(
            v for v in util_block.variable_list if v in var_set and v.is_continuous()
        )

    def add_cuts_components(self, model):
        config = self.config
        MindtPy = model.MindtPy_utils

        # Create a model block in which to store the generated feasibility
        # slack constraints. Do not leave the constraints on by default.
        feas = MindtPy.feas_opt = Block()
        feas.deactivate()
        feas.feas_constraints = ConstraintList(doc='Feasibility Problem Constraints')

        # Create a model block in which to store the generated linear
        # constraints. Do not leave the constraints on by default.
        lin = MindtPy.cuts = Block()
        lin.deactivate()

        # no-good cuts exclude particular discrete decisions
        lin.no_good_cuts = ConstraintList(doc='no-good cuts')
        # Feasible no-good cuts exclude discrete realizations that have
        # been explored via an NLP subproblem. Depending on model
        # characteristics, the user may wish to revisit NLP subproblems
        # (with a different initialization, for example). Therefore, these
        # cuts are not enabled by default.

        if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
            feas.nl_constraint_set = RangeSet(
                len(MindtPy.nonlinear_constraint_list),
                doc='Integer index set over the nonlinear constraints.',
            )
            # Create slack variables for feasibility problem
            feas.slack_var = Var(
                feas.nl_constraint_set, domain=NonNegativeReals, initialize=1
            )
        else:
            feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

        # Create slack variables for OA cuts
        if config.add_slack:
            lin.slack_vars = VarList(
                bounds=(0, config.max_slack), initialize=0, domain=NonNegativeReals
            )

    def get_dual_integral(self):
        """Calculate the dual integral.
        Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

        Returns
        -------
        float
            The dual integral.
        """
        dual_integral = 0
        dual_bound_progress = self.dual_bound_progress.copy()
        # Initial dual bound is set to inf or -inf. To calculate dual integral, we set
        # initial_dual_bound to 10% greater or smaller than the first_found_dual_bound.
        # TODO: check if the calculation of initial_dual_bound needs to be modified.
        for dual_bound in dual_bound_progress:
            if dual_bound != dual_bound_progress[0]:
                break
        for i in range(len(dual_bound_progress)):
            if dual_bound_progress[i] == self.dual_bound_progress[0]:
                dual_bound_progress[i] = dual_bound * (
                    1
                    - self.config.initial_bound_coef
                    * self.objective_sense
                    * math.copysign(1, dual_bound)
                )
            else:
                break
        for i in range(len(dual_bound_progress)):
            if i == 0:
                dual_integral += abs(dual_bound_progress[i] - self.dual_bound) * (
                    self.dual_bound_progress_time[i]
                )
            else:
                dual_integral += abs(dual_bound_progress[i] - self.dual_bound) * (
                    self.dual_bound_progress_time[i]
                    - self.dual_bound_progress_time[i - 1]
                )
        self.config.logger.info(
            ' {:<25}:   {:>7.4f} '.format('Dual integral', dual_integral)
        )
        return dual_integral

    def get_primal_integral(self):
        """Calculate the primal integral.
        Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

        Returns
        -------
        float
            The primal integral.
        """
        primal_integral = 0
        primal_bound_progress = self.primal_bound_progress.copy()
        # Initial primal bound is set to inf or -inf. To calculate primal integral, we set
        # initial_primal_bound to 10% greater or smaller than the first_found_primal_bound.
        # TODO: check if the calculation of initial_primal_bound needs to be modified.
        for primal_bound in primal_bound_progress:
            if primal_bound != primal_bound_progress[0]:
                break
        for i in range(len(primal_bound_progress)):
            if primal_bound_progress[i] == self.primal_bound_progress[0]:
                primal_bound_progress[i] = primal_bound * (
                    1
                    + self.config.initial_bound_coef
                    * self.objective_sense
                    * math.copysign(1, primal_bound)
                )
            else:
                break
        for i in range(len(primal_bound_progress)):
            if i == 0:
                primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * (
                    self.primal_bound_progress_time[i]
                )
            else:
                primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * (
                    self.primal_bound_progress_time[i]
                    - self.primal_bound_progress_time[i - 1]
                )

        self.config.logger.info(
            ' {:<25}:   {:>7.4f} '.format('Primal integral', primal_integral)
        )
        return primal_integral

    def get_integral_info(self):
        '''
        Obtain primal integral, dual integral and primal dual gap integral.
        '''
        self.primal_integral = self.get_primal_integral()
        self.dual_integral = self.get_dual_integral()
        self.primal_dual_gap_integral = self.primal_integral + self.dual_integral

    def update_gap(self):
        """Update the relative gap and the absolute gap."""
        if self.objective_sense == minimize:
            self.abs_gap = self.primal_bound - self.dual_bound
        else:
            self.abs_gap = self.dual_bound - self.primal_bound
        self.rel_gap = self.abs_gap / (abs(self.primal_bound) + 1e-10)

    def update_dual_bound(self, bound_value):
        """Update the dual bound.

        Call after solving relaxed problem, including relaxed NLP and MIP main problem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the dual bound.
        """
        if math.isnan(bound_value):
            return
        if self.objective_sense == minimize:
            self.dual_bound = max(bound_value, self.dual_bound)
            self.dual_bound_improved = self.dual_bound > self.dual_bound_progress[-1]
        else:
            self.dual_bound = min(bound_value, self.dual_bound)
            self.dual_bound_improved = self.dual_bound < self.dual_bound_progress[-1]
        self.dual_bound_progress.append(self.dual_bound)
        self.dual_bound_progress_time.append(get_main_elapsed_time(self.timing))
        if self.dual_bound_improved:
            self.update_gap()

    def update_suboptimal_dual_bound(self, results):
        """If the relaxed problem is not solved to optimality, the dual bound is updated
        according to the dual bound of relaxed problem.

        Parameters
        ----------
        results : SolverResults
            Results from solving the relaxed problem.
            The dual bound of the relaxed problem can only be obtained from the result object.
        """
        if self.objective_sense == minimize:
            bound_value = results.problem.lower_bound
        else:
            bound_value = results.problem.upper_bound
        self.update_dual_bound(bound_value)

    def update_primal_bound(self, bound_value):
        """Update the primal bound.

        Call after solve fixed NLP subproblem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the primal bound.
        """
        if math.isnan(bound_value):
            return
        if self.objective_sense == minimize:
            self.primal_bound = min(bound_value, self.primal_bound)
            self.primal_bound_improved = (
                self.primal_bound < self.primal_bound_progress[-1]
            )
        else:
            self.primal_bound = max(bound_value, self.primal_bound)
            self.primal_bound_improved = (
                self.primal_bound > self.primal_bound_progress[-1]
            )
        self.primal_bound_progress.append(self.primal_bound)
        self.primal_bound_progress_time.append(get_main_elapsed_time(self.timing))
        if self.primal_bound_improved:
            self.update_gap()

    def process_objective(self, update_var_con_list=True):
        """Process model objective function.

        Check that the model has only 1 valid objective.
        If the objective is nonlinear, move it into the constraints.
        If no objective function exists, emit a warning and create a dummy objective.

        Parameters
        ----------
        update_var_con_list : bool, optional
            Whether to update the variable/constraint/objective lists, by default True.
            Currently, update_var_con_list will be set to False only when add_regularization is not None in MindtPy.
        """
        config = self.config
        m = self.working_model
        util_block = getattr(m, self.util_block_name)
        # Handle missing or multiple objectives
        active_objectives = list(
            m.component_data_objects(ctype=Objective, active=True, descend_into=True)
        )
        self.results.problem.number_of_objectives = len(active_objectives)
        if len(active_objectives) == 0:
            config.logger.warning(
                'Model has no active objectives. Adding dummy objective.'
            )
            util_block.dummy_objective = Objective(expr=1)
            main_obj = util_block.dummy_objective
        elif len(active_objectives) > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = active_objectives[0]
        self.results.problem.sense = (
            ProblemSense.minimize if main_obj.sense == 1 else ProblemSense.maximize
        )
        self.objective_sense = main_obj.sense

        # Move the objective to the constraints if it is nonlinear or move_objective is True.
        if (
            main_obj.expr.polynomial_degree()
            not in self.mip_objective_polynomial_degree
            or config.move_objective
        ):
            if config.move_objective:
                config.logger.info("Moving objective to constraint set.")
            else:
                config.logger.info(
                    "Objective is nonlinear. Moving it to constraint set."
                )
            util_block.objective_value = VarList(domain=Reals, initialize=0)
            util_block.objective_constr = ConstraintList()
            if (
                main_obj.expr.polynomial_degree()
                not in self.mip_objective_polynomial_degree
                and config.partition_obj_nonlinear_terms
                and main_obj.expr.__class__ is EXPR.SumExpression
            ):
                repn = generate_standard_repn(
                    main_obj.expr, quadratic=2 in self.mip_objective_polynomial_degree
                )
                # the following code will also work if linear_subexpr is a constant.
                linear_subexpr = (
                    repn.constant
                    + sum(
                        coef * var
                        for coef, var in zip(repn.linear_coefs, repn.linear_vars)
                    )
                    + sum(
                        coef * var1 * var2
                        for coef, (var1, var2) in zip(
                            repn.quadratic_coefs, repn.quadratic_vars
                        )
                    )
                )
                # only need to generate one epigraph constraint for the sum of all linear terms and constant
                epigraph_reformulation(
                    linear_subexpr,
                    util_block.objective_value,
                    util_block.objective_constr,
                    config.use_mcpp,
                    main_obj.sense,
                )
                nonlinear_subexpr = repn.nonlinear_expr
                if nonlinear_subexpr.__class__ is EXPR.SumExpression:
                    for subsubexpr in nonlinear_subexpr.args:
                        epigraph_reformulation(
                            subsubexpr,
                            util_block.objective_value,
                            util_block.objective_constr,
                            config.use_mcpp,
                            main_obj.sense,
                        )
                else:
                    epigraph_reformulation(
                        nonlinear_subexpr,
                        util_block.objective_value,
                        util_block.objective_constr,
                        config.use_mcpp,
                        main_obj.sense,
                    )
            else:
                epigraph_reformulation(
                    main_obj.expr,
                    util_block.objective_value,
                    util_block.objective_constr,
                    config.use_mcpp,
                    main_obj.sense,
                )

            main_obj.deactivate()
            util_block.objective = Objective(
                expr=sum(util_block.objective_value[:]), sense=main_obj.sense
            )

            if (
                main_obj.expr.polynomial_degree()
                not in self.mip_objective_polynomial_degree
                or (config.move_objective and update_var_con_list)
            ):
                util_block.variable_list.extend(util_block.objective_value[:])
                util_block.continuous_variable_list.extend(
                    util_block.objective_value[:]
                )
                util_block.constraint_list.extend(util_block.objective_constr[:])
                util_block.objective_list.append(util_block.objective)
                for constr in util_block.objective_constr[:]:
                    if (
                        constr.body.polynomial_degree()
                        in self.mip_constraint_polynomial_degree
                    ):
                        util_block.linear_constraint_list.append(constr)
                    else:
                        util_block.nonlinear_constraint_list.append(constr)

    def set_up_solve_data(self, model):
        """Set up the solve data.

        Parameters
        ----------
        model : Pyomo model
            The original model to be solved in MindtPy.
        """
        config = self.config
        # if the objective function is a constant, dual bound constraint is not added.
        obj = next(model.component_data_objects(ctype=Objective, active=True))
        if obj.expr.polynomial_degree() == 0:
            config.logger.info(
                'The model has a constant objecitive function. use_dual_bound is set to False.'
            )
            config.use_dual_bound = False

        if config.use_fbbt:
            fbbt(model)
            # TODO: logging_level is not logging.INFO here
            config.logger.info('Use the fbbt to tighten the bounds of variables')

        self.original_model = model
        self.working_model = model.clone()

        # set up bounds
        if obj.sense == minimize:
            self.primal_bound = float('inf')
            self.dual_bound = float('-inf')
        else:
            self.primal_bound = float('-inf')
            self.dual_bound = float('inf')
        self.primal_bound_progress = [self.primal_bound]
        self.dual_bound_progress = [self.dual_bound]

        if config.nlp_solver in {'ipopt', 'cyipopt'}:
            if not hasattr(self.working_model, 'ipopt_zL_out'):
                self.working_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            if not hasattr(self.working_model, 'ipopt_zU_out'):
                self.working_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)

        if config.quadratic_strategy == 0:
            self.mip_objective_polynomial_degree = {0, 1}
            self.mip_constraint_polynomial_degree = {0, 1}
        elif config.quadratic_strategy == 1:
            self.mip_objective_polynomial_degree = {0, 1, 2}
            self.mip_constraint_polynomial_degree = {0, 1}
        elif config.quadratic_strategy == 2:
            self.mip_objective_polynomial_degree = {0, 1, 2}
            self.mip_constraint_polynomial_degree = {0, 1, 2}

    # -----------------------------------------------------------------------------------------
    # initialization

    def MindtPy_initialization(self):
        """Initializes the decomposition algorithm.

        This function initializes the decomposition algorithm, which includes generating the
        initial cuts required to build the main MIP.
        """
        # Do the initialization
        config = self.config
        if config.init_strategy == 'rNLP':
            self.init_rNLP()
        elif config.init_strategy == 'max_binary':
            self.init_max_binaries()
        elif config.init_strategy == 'initial_binary':
            try:
                self.curr_int_sol = get_integer_solution(self.working_model)
            except TypeError as e:
                config.logger.error(e)
                raise ValueError(
                    'The initial integer combination is not provided or not complete. '
                    'Please provide the complete integer combination or use other initialization strategy.'
                )
            self.integer_list.append(self.curr_int_sol)
            fixed_nlp, fixed_nlp_result = self.solve_subproblem()
            self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)
        elif config.init_strategy == 'FP':
            self.init_rNLP()
            self.fp_loop()

    def init_rNLP(self, add_oa_cuts=True):
        """Initialize the problem by solving the relaxed NLP and then store the optimal variable
        values obtained from solving the rNLP.

        Parameters
        ----------
        add_oa_cuts : Bool
            Whether add OA cuts after solving the relaxed NLP problem.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the relaxed NLP.
        """
        config = self.config
        self.rnlp = self.working_model.clone()
        config.logger.debug('Relaxed NLP: Solve relaxed integrality')
        MindtPy = self.rnlp.MindtPy_utils
        TransformationFactory('core.relax_integer_vars').apply_to(self.rnlp)
        nlp_args = dict(config.nlp_solver_args)
        update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
        with SuppressInfeasibleWarning():
            results = self.nlp_opt.solve(
                self.rnlp,
                tee=config.nlp_solver_tee,
                load_solutions=self.load_solutions,
                **nlp_args,
            )
            if len(results.solution) > 0:
                self.rnlp.solutions.load_from(results)
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond in {tc.optimal, tc.feasible, tc.locallyOptimal}:
            main_objective = MindtPy.objective_list[-1]
            if subprob_terminate_cond == tc.optimal:
                self.update_dual_bound(value(main_objective.expr))
            else:
                config.logger.info('relaxed NLP is not solved to optimality.')
                self.update_suboptimal_dual_bound(results)
            config.logger.info(
                self.log_formatter.format(
                    '-',
                    'Relaxed NLP',
                    value(main_objective.expr),
                    self.primal_bound,
                    self.dual_bound,
                    self.rel_gap,
                    get_main_elapsed_time(self.timing),
                )
            )
            # Add OA cut
            if add_oa_cuts:
                if (
                    self.config.nlp_solver == 'cyipopt'
                    and self.objective_sense == minimize
                ):
                    # TODO: recover the opposite dual when cyipopt issue #2831 is solved.
                    dual_values = (
                        list(-1 * self.rnlp.dual[c] for c in MindtPy.constraint_list)
                        if config.calculate_dual_at_solution
                        else None
                    )
                else:
                    dual_values = (
                        list(self.rnlp.dual[c] for c in MindtPy.constraint_list)
                        if config.calculate_dual_at_solution
                        else None
                    )
                copy_var_list_values(
                    self.rnlp.MindtPy_utils.variable_list,
                    self.mip.MindtPy_utils.variable_list,
                    config,
                )
                if config.init_strategy == 'FP':
                    copy_var_list_values(
                        self.rnlp.MindtPy_utils.variable_list,
                        self.working_model.MindtPy_utils.variable_list,
                        config,
                    )
                self.add_cuts(
                    dual_values=dual_values,
                    linearize_active=True,
                    linearize_violated=True,
                    cb_opt=None,
                    nlp=self.rnlp,
                )
                for var in self.mip.MindtPy_utils.discrete_variable_list:
                    # We don't want to trigger the reset of the global stale
                    # indicator, so we will set this variable to be "stale",
                    # knowing that set_value will switch it back to "not
                    # stale"
                    var.stale = True
                    var.set_value(int(round(var.value)), skip_validation=True)
        elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
            # TODO fail? try something else?
            config.logger.info(
                'Initial relaxed NLP problem is infeasible. '
                'Problem may be infeasible.'
            )
        elif subprob_terminate_cond is tc.maxTimeLimit:
            config.logger.info('NLP subproblem failed to converge within time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif subprob_terminate_cond is tc.maxIterations:
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.'
            )
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of %s. Solver message: %s'
                % (subprob_terminate_cond, results.solver.message)
            )

    def init_max_binaries(self):
        """Modifies model by maximizing the number of activated binary variables.

        Note - The user would usually want to call solve_subproblem after an invocation
        of this function.

        Raises
        ------
        ValueError
            MILP main problem is infeasible.
        ValueError
            MindtPy unable to handle the termination condition of the MILP main problem.
        """
        config = self.config
        m = self.working_model.clone()
        if hasattr(m, 'dual') and isinstance(m.dual, Suffix):
            m.del_component('dual')
        MindtPy = m.MindtPy_utils
        self.mip_subiter += 1
        config.logger.debug('Initialization: maximize value of binaries')
        for c in MindtPy.nonlinear_constraint_list:
            c.deactivate()
        objective = next(m.component_data_objects(Objective, active=True))
        objective.deactivate()
        binary_vars = (
            v
            for v in m.MindtPy_utils.discrete_variable_list
            if v.is_binary() and not v.fixed
        )
        MindtPy.max_binary_obj = Objective(
            expr=sum(v for v in binary_vars), sense=maximize
        )

        getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
        getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()

        if isinstance(self.mip_opt, PersistentSolver):
            self.mip_opt.set_instance(m)
        mip_args = dict(config.mip_solver_args)
        update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
        results = self.mip_opt.solve(
            m, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **mip_args
        )
        if len(results.solution) > 0:
            m.solutions.load_from(results)

        solve_terminate_cond = results.solver.termination_condition
        if solve_terminate_cond is tc.optimal:
            copy_var_list_values(
                MindtPy.variable_list,
                self.working_model.MindtPy_utils.variable_list,
                config,
            )
            config.logger.info(
                self.log_formatter.format(
                    '-',
                    'Max binary MILP',
                    value(MindtPy.max_binary_obj.expr),
                    self.primal_bound,
                    self.dual_bound,
                    self.rel_gap,
                    get_main_elapsed_time(self.timing),
                )
            )
        elif solve_terminate_cond is tc.infeasible:
            raise ValueError(
                'MIP main problem is infeasible. '
                'Problem may have no more feasible '
                'binary configurations.'
            )
        elif solve_terminate_cond is tc.maxTimeLimit:
            config.logger.info('NLP subproblem failed to converge within time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif solve_terminate_cond is tc.maxIterations:
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.'
            )
        else:
            raise ValueError(
                'MindtPy unable to handle MILP main termination condition '
                'of %s. Solver message: %s'
                % (solve_terminate_cond, results.solver.message)
            )

    ##################################################################################################################################################################################################################
    # nlp_solve.py

    def solve_subproblem(self):
        """Solves the Fixed-NLP (with fixed integers).

        This function sets up the 'fixed_nlp' by fixing binaries, sets continuous variables to their initial var values,
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Returns
        -------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        results : SolverResults
            Results from solving the Fixed-NLP.
        """
        config = self.config
        MindtPy = self.fixed_nlp.MindtPy_utils
        self.nlp_iter += 1

        MindtPy.cuts.deactivate()
        if config.calculate_dual_at_solution:
            self.fixed_nlp.tmp_duals = ComponentMap()
            # tmp_duals are the value of the dual variables stored before using deactivate trivial constraints
            # The values of the duals are computed as follows: (Complementary Slackness)
            #
            # | constraint | c_geq | status at x1 | tmp_dual (violation) |
            # |------------|-------|--------------|----------------------|
            # | g(x) <= b  | -1    | g(x1) <= b   | 0                    |
            # | g(x) <= b  | -1    | g(x1) > b    | g(x1) - b            |
            # | g(x) >= b  | +1    | g(x1) >= b   | 0                    |
            # | g(x) >= b  | +1    | g(x1) < b    | b - g(x1)            |
            evaluation_error = False
            for c in self.fixed_nlp.MindtPy_utils.constraint_list:
                # We prefer to include the upper bound as the right hand side since we are
                # considering c by default a (hopefully) convex function, which would make
                # c >= lb a nonconvex inequality which we wouldn't like to add linearizations
                # if we don't have to
                rhs = value(c.upper) if c.has_ub() else value(c.lower)
                c_geq = -1 if c.has_ub() else 1
                try:
                    self.fixed_nlp.tmp_duals[c] = c_geq * max(
                        0, c_geq * (rhs - value(c.body))
                    )
                except (ValueError, OverflowError) as e:
                    config.logger.error(e)
                    self.fixed_nlp.tmp_duals[c] = None
                    evaluation_error = True
            if evaluation_error:
                for nlp_var, orig_val in zip(
                    MindtPy.variable_list, self.initial_var_values
                ):
                    if not nlp_var.fixed and not nlp_var.is_binary():
                        nlp_var.set_value(orig_val, skip_validation=True)
        try:
            TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
                self.fixed_nlp,
                tmp=True,
                ignore_infeasible=False,
                tolerance=config.constraint_tolerance,
            )
        except InfeasibleConstraintException as e:
            config.logger.error(
                str(e) + '\nInfeasibility detected in deactivate_trivial_constraints.'
            )
            results = SolverResults()
            results.solver.termination_condition = tc.infeasible
            return self.fixed_nlp, results
        # Solve the NLP
        nlp_args = dict(config.nlp_solver_args)
        update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
        with SuppressInfeasibleWarning():
            with time_code(self.timing, 'fixed subproblem'):
                results = self.nlp_opt.solve(
                    self.fixed_nlp,
                    tee=config.nlp_solver_tee,
                    load_solutions=self.load_solutions,
                    **nlp_args,
                )
                if len(results.solution) > 0:
                    self.fixed_nlp.solutions.load_from(results)
        TransformationFactory('contrib.deactivate_trivial_constraints').revert(
            self.fixed_nlp
        )
        return self.fixed_nlp, results

    def handle_nlp_subproblem_tc(self, fixed_nlp, result, cb_opt=None):
        """This function handles different terminaton conditions of the fixed-NLP subproblem.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        result : SolverResults
            Results from solving the NLP subproblem.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        """
        if result.solver.termination_condition in {
            tc.optimal,
            tc.locallyOptimal,
            tc.feasible,
        }:
            self.handle_subproblem_optimal(fixed_nlp, cb_opt)
        elif result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
            self.handle_subproblem_infeasible(fixed_nlp, cb_opt)
        elif result.solver.termination_condition is tc.maxTimeLimit:
            self.config.logger.info(
                'NLP subproblem failed to converge within the time limit.'
            )
            self.results.solver.termination_condition = tc.maxTimeLimit
            self.should_terminate = True
        elif result.solver.termination_condition is tc.maxEvaluations:
            self.config.logger.info('NLP subproblem failed due to maxEvaluations.')
            self.results.solver.termination_condition = tc.maxEvaluations
            self.should_terminate = True
        else:
            self.handle_subproblem_other_termination(
                fixed_nlp, result.solver.termination_condition, cb_opt
            )

    def handle_subproblem_optimal(self, fixed_nlp, cb_opt=None, fp=False):
        """This function copies the result of the NLP solver function ('solve_subproblem') to the working model, updates
        the bounds, adds OA and no-good cuts, and then stores the new solution if it is the new best solution. This
        function handles the result of the latest iteration of solving the NLP subproblem given an optimal solution.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        fp : bool, optional
            Whether it is in the loop of feasibility pump, by default False.
        """
        # TODO: check what is this copy_value function used for?
        # Warmstart?
        config = self.config
        copy_var_list_values(
            fixed_nlp.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            config,
        )
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.tmp_duals:
                if fixed_nlp.dual.get(c, None) is None:
                    fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
                elif (
                    self.config.nlp_solver == 'cyipopt'
                    and self.objective_sense == minimize
                ):
                    # TODO: recover the opposite dual when cyipopt issue #2831 is solved.
                    fixed_nlp.dual[c] = -fixed_nlp.dual[c]
            dual_values = list(
                fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list
            )
        else:
            dual_values = None
        main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
        self.update_primal_bound(value(main_objective.expr))
        if self.primal_bound_improved:
            self.best_solution_found = fixed_nlp.clone()
            self.best_solution_found_time = get_main_elapsed_time(self.timing)
        # Add the linear cut
        copy_var_list_values(
            fixed_nlp.MindtPy_utils.variable_list,
            self.mip.MindtPy_utils.variable_list,
            config,
        )
        self.add_cuts(
            dual_values=dual_values,
            linearize_active=True,
            linearize_violated=True,
            cb_opt=cb_opt,
            nlp=self.fixed_nlp,
        )

        var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_no_good_cuts:
            add_no_good_cuts(
                self.mip, var_values, config, self.timing, self.mip_iter, cb_opt
            )

        config.call_after_subproblem_feasible(fixed_nlp)

        config.logger.info(
            self.fixed_nlp_log_formatter.format(
                '*' if self.primal_bound_improved else ' ',
                self.nlp_iter if not fp else self.fp_iter,
                'Fixed NLP',
                value(main_objective.expr),
                self.primal_bound,
                self.dual_bound,
                self.rel_gap,
                get_main_elapsed_time(self.timing),
            )
        )

    def handle_subproblem_infeasible(self, fixed_nlp, cb_opt=None):
        """Solves feasibility problem and adds cut according to the specified strategy.

        This function handles the result of the latest iteration of solving the NLP subproblem given an infeasible
        solution and copies the solution of the feasibility problem to the working model.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        """
        # TODO try something else? Reinitialize with different initial
        # value?
        config = self.config
        config.logger.info('NLP subproblem was locally infeasible.')
        self.nlp_infeasible_counter += 1
        if config.calculate_dual_at_solution:
            for c in fixed_nlp.MindtPy_utils.constraint_list:
                rhs = value(c.upper) if c.has_ub() else value(c.lower)
                c_geq = -1 if c.has_ub() else 1
                fixed_nlp.dual[c] = c_geq * max(0, c_geq * (rhs - value(c.body)))
            dual_values = list(
                fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list
            )
        else:
            dual_values = None

        # if config.strategy == 'PSC' or config.strategy == 'GBD':
        #     for var in fixed_nlp.component_data_objects(ctype=Var, descend_into=True):
        #         fixed_nlp.ipopt_zL_out[var] = 0
        #         fixed_nlp.ipopt_zU_out[var] = 0
        #         if var.has_ub() and abs(var.ub - value(var)) < config.absolute_bound_tolerance:
        #             fixed_nlp.ipopt_zL_out[var] = 1
        #         elif var.has_lb() and abs(value(var) - var.lb) < config.absolute_bound_tolerance:
        #             fixed_nlp.ipopt_zU_out[var] = -1

        config.logger.info('Solving feasibility problem')
        feas_subproblem, feas_subproblem_results = self.solve_feasibility_subproblem()
        # TODO: do we really need this?
        if self.should_terminate:
            return
        copy_var_list_values(
            feas_subproblem.MindtPy_utils.variable_list,
            self.mip.MindtPy_utils.variable_list,
            config,
        )
        self.add_cuts(
            dual_values=dual_values,
            linearize_active=True,
            linearize_violated=True,
            cb_opt=cb_opt,
            nlp=feas_subproblem,
        )
        # Add a no-good cut to exclude this discrete option
        var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
        if config.add_no_good_cuts:
            # excludes current discrete option
            add_no_good_cuts(
                self.mip, var_values, config, self.timing, self.mip_iter, cb_opt
            )

    def handle_subproblem_other_termination(
        self, fixed_nlp, termination_condition, cb_opt=None
    ):
        """Handles the result of the latest iteration of solving the fixed NLP subproblem given
        a solution that is neither optimal nor infeasible.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        termination_condition : Pyomo TerminationCondition
            The termination condition of the fixed NLP subproblem.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.

        Raises
        ------
        ValueError
            MindtPy unable to handle the NLP subproblem termination condition.
        """
        if termination_condition is tc.maxIterations:
            # TODO try something else? Reinitialize with different initial value?
            self.config.logger.info(
                'NLP subproblem failed to converge within iteration limit.'
            )
            var_values = list(v.value for v in fixed_nlp.MindtPy_utils.variable_list)
            if self.config.add_no_good_cuts:
                # excludes current discrete option
                add_no_good_cuts(
                    self.mip,
                    var_values,
                    self.config,
                    self.timing,
                    self.mip_iter,
                    cb_opt,
                )

        else:
            raise ValueError(
                'MindtPy unable to handle NLP subproblem termination '
                'condition of {}'.format(termination_condition)
            )

    def solve_feasibility_subproblem(self):
        """Solves a feasibility NLP if the fixed_nlp problem is infeasible.

        Returns
        -------
        feas_subproblem : Pyomo model
            Feasibility NLP from the model.
        feas_soln : SolverResults
            Results from solving the feasibility NLP.
        """
        config = self.config
        feas_subproblem = self.fixed_nlp
        MindtPy = feas_subproblem.MindtPy_utils
        MindtPy.feas_opt.activate()
        if MindtPy.component('objective_value') is not None:
            MindtPy.objective_value[:].set_value(0, skip_validation=True)

        active_obj = next(
            feas_subproblem.component_data_objects(Objective, active=True)
        )
        active_obj.deactivate()
        for constr in MindtPy.nonlinear_constraint_list:
            constr.deactivate()

        MindtPy.feas_opt.activate()
        MindtPy.feas_obj.activate()
        nlp_args = dict(config.nlp_solver_args)
        update_solver_timelimit(
            self.feasibility_nlp_opt, config.nlp_solver, self.timing, config
        )
        TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
            feas_subproblem,
            tmp=True,
            ignore_infeasible=False,
            tolerance=config.constraint_tolerance,
        )
        with SuppressInfeasibleWarning():
            try:
                with time_code(self.timing, 'feasibility subproblem'):
                    feas_soln = self.feasibility_nlp_opt.solve(
                        feas_subproblem,
                        tee=config.nlp_solver_tee,
                        load_solutions=config.nlp_solver != 'appsi_ipopt',
                        **nlp_args,
                    )
                    if len(feas_soln.solution) > 0:
                        feas_subproblem.solutions.load_from(feas_soln)
            except (ValueError, OverflowError) as e:
                config.logger.error(e)
                for nlp_var, orig_val in zip(
                    MindtPy.variable_list, self.initial_var_values
                ):
                    if not nlp_var.fixed and not nlp_var.is_binary():
                        nlp_var.set_value(orig_val, skip_validation=True)
                with time_code(self.timing, 'feasibility subproblem'):
                    feas_soln = self.feasibility_nlp_opt.solve(
                        feas_subproblem,
                        tee=config.nlp_solver_tee,
                        load_solutions=config.nlp_solver != 'appsi_ipopt',
                        **nlp_args,
                    )
                    if len(feas_soln.solution) > 0:
                        feas_soln.solutions.load_from(feas_soln)
        self.handle_feasibility_subproblem_tc(
            feas_soln.solver.termination_condition, MindtPy
        )
        MindtPy.feas_opt.deactivate()
        for constr in MindtPy.nonlinear_constraint_list:
            constr.activate()
        active_obj.activate()
        MindtPy.feas_obj.deactivate()
        TransformationFactory('contrib.deactivate_trivial_constraints').revert(
            feas_subproblem
        )
        return feas_subproblem, feas_soln

    def handle_feasibility_subproblem_tc(self, subprob_terminate_cond, MindtPy):
        """Handles the result of the latest iteration of solving the feasibility NLP subproblem.

        Parameters
        ----------
        subprob_terminate_cond : Pyomo TerminationCondition
            The termination condition of the feasibility NLP subproblem.
        MindtPy : Pyomo Block
            The MindtPy_utils block.
        """
        config = self.config
        if subprob_terminate_cond in {tc.optimal, tc.locallyOptimal, tc.feasible}:
            # TODO: check what is this copy_value used for?
            copy_var_list_values(
                MindtPy.variable_list,
                self.working_model.MindtPy_utils.variable_list,
                config,
            )
            if value(MindtPy.feas_obj.expr) <= config.zero_tolerance:
                config.logger.warning(
                    'The objective value %.4E of feasibility problem is less than zero_tolerance. '
                    'This indicates that the nlp subproblem is feasible, although it is found infeasible in the previous step. '
                    'Check the nlp solver output' % value(MindtPy.feas_obj.expr)
                )
        elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
            config.logger.error(
                'Feasibility subproblem infeasible. This should never happen.'
            )
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
        elif subprob_terminate_cond is tc.maxIterations:
            config.logger.error(
                'Subsolver reached its maximum number of iterations without converging, '
                'consider increasing the iterations limit of the subsolver or reviewing your formulation.'
            )
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error
        else:
            config.logger.error(
                'MindtPy unable to handle feasibility subproblem termination condition '
                'of {}'.format(subprob_terminate_cond)
            )
            self.should_terminate = True
            self.results.solver.status = SolverStatus.error

    ######################################################################################################################################################
    # iterate.py

    def algorithm_should_terminate(self, check_cycling):
        """Checks if the algorithm should terminate at the given point.

        This function determines whether the algorithm should terminate based on the solver options and progress.
        (Sets the self.results.solver.termination_condition to the appropriate condition, i.e. optimal,
        maxIterations, maxTimeLimit).

        Parameters
        ----------
        check_cycling : bool
            Whether to check for a special case that causes the discrete variables to loop through the same values.

        Returns
        -------
        bool
            True if the algorithm should terminate, False otherwise.
        """
        if self.should_terminate:
            # self.primal_bound_progress[0] can only be inf or -inf.
            # If the current primal bound equals inf or -inf, we can infer there is no solution.
            if self.primal_bound == self.primal_bound_progress[0]:
                self.results.solver.termination_condition = tc.noSolution
            else:
                self.results.solver.termination_condition = tc.feasible
            return True
        return (
            self.bounds_converged()
            or self.reached_iteration_limit()
            or self.reached_time_limit()
            or self.reached_stalling_limit()
            or (check_cycling and self.iteration_cycling())
        )

    def fix_dual_bound(self, last_iter_cuts):
        """Fix the dual bound when no-good cuts or tabu list is activated.

        Parameters
        ----------
        last_iter_cuts : bool
            Whether the cuts in the last iteration have been added.
        """
        # If no-good cuts or tabu list is activated, the dual bound is not valid for the final optimal solution.
        # Therefore, we need to correct it at the end.
        # In singletree implementation, the dual bound at one iteration before the optimal solution, is valid for the optimal solution.
        # So we will set the dual bound to it.
        config = self.config
        if config.single_tree:
            config.logger.info(
                'Fix the bound to the value of one iteration before optimal solution is found.'
            )
            try:
                self.dual_bound = self.stored_bound[self.primal_bound]
            except KeyError as e:
                config.logger.error(
                    str(e) + '\nNo stored bound found. Bound fix failed.'
                )
        else:
            config.logger.info(
                'Solve the main problem without the last no_good cut to fix the bound.'
                'zero_tolerance is set to 1E-4'
            )
            config.zero_tolerance = 1e-4
            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            if not last_iter_cuts:
                fixed_nlp, fixed_nlp_result = self.solve_subproblem()
                self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)

            MindtPy = self.mip.MindtPy_utils
            # deactivate the integer cuts generated after the best solution was found.
            self.deactivate_no_good_cuts_when_fixing_bound(MindtPy.cuts.no_good_cuts)
            if (
                config.add_regularization is not None
                and MindtPy.component('mip_obj') is None
            ):
                MindtPy.objective_list[-1].activate()
            # determine if persistent solver is called.
            if isinstance(self.mip_opt, PersistentSolver):
                self.mip_opt.set_instance(self.mip, symbolic_solver_labels=True)
            mip_args = dict(config.mip_solver_args)
            update_solver_timelimit(
                self.mip_opt, config.mip_solver, self.timing, config
            )
            main_mip_results = self.mip_opt.solve(
                self.mip,
                tee=config.mip_solver_tee,
                load_solutions=self.load_solutions,
                **mip_args,
            )
            if len(main_mip_results.solution) > 0:
                self.mip.solutions.load_from(main_mip_results)

            if main_mip_results.solver.termination_condition is tc.infeasible:
                config.logger.info(
                    'Bound fix failed. The bound fix problem is infeasible'
                )
            else:
                self.update_suboptimal_dual_bound(main_mip_results)
                config.logger.info(
                    'Fixed bound values: Primal Bound: {}  Dual Bound: {}'.format(
                        self.primal_bound, self.dual_bound
                    )
                )
            # Check bound convergence
            if (
                abs(self.primal_bound - self.dual_bound)
                <= config.absolute_bound_tolerance
            ):
                self.results.solver.termination_condition = tc.optimal

    def set_up_tabulist_callback(self):
        """Sets up the tabulist using IncumbentCallback.
        Currently only support CPLEX.
        """
        tabulist = self.mip_opt._solver_model.register_callback(
            tabu_list.IncumbentCallback_cplex
        )
        tabulist.opt = self.mip_opt
        tabulist.config = self.config
        tabulist.mindtpy_solver = self
        self.mip_opt.options['preprocessing_reduce'] = 1
        # If the callback is used to reject incumbents, the user must set the
        # parameter c.parameters.preprocessing.reduce either to the value 1 (one)
        # to restrict presolve to primal reductions only or to 0 (zero) to disable all presolve reductions
        self.mip_opt._solver_model.set_warning_stream(None)
        self.mip_opt._solver_model.set_log_stream(None)
        self.mip_opt._solver_model.set_error_stream(None)

    def set_up_lazy_OA_callback(self):
        """Sets up the lazy OA using LazyConstraintCallback.
        Currently only support CPLEX and Gurobi.
        """
        if self.config.mip_solver == 'cplex_persistent':
            lazyoa = self.mip_opt._solver_model.register_callback(
                single_tree.LazyOACallback_cplex
            )
            # pass necessary data and parameters to lazyoa
            lazyoa.main_mip = self.mip
            lazyoa.config = self.config
            lazyoa.opt = self.mip_opt
            lazyoa.mindtpy_solver = self
            self.mip_opt._solver_model.set_warning_stream(None)
            self.mip_opt._solver_model.set_log_stream(None)
            self.mip_opt._solver_model.set_error_stream(None)
        if self.config.mip_solver == 'gurobi_persistent':
            self.mip_opt.set_callback(single_tree.LazyOACallback_gurobi)
            self.mip_opt.mindtpy_solver = self
            self.mip_opt.config = self.config

    ##########################################################################################################################################
    # mip_solve.py

    def solve_main(self):
        """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
        config = self.config
        self.mip_iter += 1

        # setup main problem
        self.setup_main()
        mip_args = self.set_up_mip_solver()

        try:
            main_mip_results = self.mip_opt.solve(
                self.mip,
                tee=config.mip_solver_tee,
                load_solutions=self.load_solutions,
                **mip_args,
            )
            # update_attributes should be before load_from(main_mip_results), since load_from(main_mip_results) may fail.
            if len(main_mip_results.solution) > 0:
                self.mip.solutions.load_from(main_mip_results)
        except (ValueError, AttributeError, RuntimeError) as e:
            config.logger.error(e)
            if config.single_tree:
                config.logger.warning('Single tree terminate.')
                if get_main_elapsed_time(self.timing) >= config.time_limit:
                    config.logger.warning('due to the timelimit.')
                    self.results.solver.termination_condition = tc.maxTimeLimit
                if config.strategy == 'GOA' or config.add_no_good_cuts:
                    config.logger.warning(
                        'Error: Cannot load a SolverResults object with bad status: error. '
                        'MIP solver failed. This usually happens in the single-tree GOA algorithm. '
                        "No-good cuts are added and GOA algorithm doesn't converge within the time limit. "
                        'No integer solution is found, so the CPLEX solver will report an error status. '
                    )
            return None, None
        if config.solution_pool:
            main_mip_results._solver_model = self.mip_opt._solver_model
            main_mip_results._pyomo_var_to_solver_var_map = (
                self.mip_opt._pyomo_var_to_solver_var_map
            )
        if main_mip_results.solver.termination_condition is tc.optimal:
            if config.single_tree and not config.add_no_good_cuts:
                self.update_suboptimal_dual_bound(main_mip_results)
        elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
                self.mip, config
            )
        return self.mip, main_mip_results

    def solve_fp_main(self):
        """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
        # setup main problem
        config = self.config
        self.setup_fp_main()
        mip_args = self.set_up_mip_solver()

        main_mip_results = self.mip_opt.solve(
            self.mip,
            tee=config.mip_solver_tee,
            load_solutions=self.load_solutions,
            **mip_args,
        )
        # update_attributes should be before load_from(main_mip_results), since load_from(main_mip_results) may fail.
        # if config.single_tree or config.use_tabu_list:
        #     self.update_attributes()
        if len(main_mip_results.solution) > 0:
            self.mip.solutions.load_from(main_mip_results)
        if main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
                self.mip, config
            )

        return self.mip, main_mip_results

    def solve_regularization_main(self):
        """This function solves the MIP main problem.

        Returns
        -------
        self.mip : Pyomo model
            The MIP stored in self.
        main_mip_results : SolverResults
            Results from solving the main MIP.
        """
        config = self.config
        # setup main problem
        self.setup_regularization_main()

        if isinstance(self.regularization_mip_opt, PersistentSolver):
            self.regularization_mip_opt.set_instance(self.mip)
        update_solver_timelimit(
            self.regularization_mip_opt,
            config.mip_regularization_solver,
            self.timing,
            config,
        )
        main_mip_results = self.regularization_mip_opt.solve(
            self.mip,
            tee=config.mip_solver_tee,
            load_solutions=self.load_solutions,
            **dict(config.mip_solver_args),
        )
        if len(main_mip_results.solution) > 0:
            self.mip.solutions.load_from(main_mip_results)
        if main_mip_results.solver.termination_condition is tc.optimal:
            config.logger.info(
                self.log_formatter.format(
                    self.mip_iter,
                    'Reg ' + self.regularization_mip_type,
                    value(self.mip.MindtPy_utils.roa_proj_mip_obj),
                    self.primal_bound,
                    self.dual_bound,
                    self.rel_gap,
                    get_main_elapsed_time(self.timing),
                )
            )

        elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            # Linear solvers will sometimes tell me that it's infeasible or
            # unbounded during presolve, but fails to distinguish. We need to
            # resolve with a solver option flag on.
            main_mip_results, _ = distinguish_mip_infeasible_or_unbounded(
                self.mip, config
            )

        self.mip.MindtPy_utils.objective_constr.deactivate()
        self.mip.MindtPy_utils.del_component('roa_proj_mip_obj')
        self.mip.MindtPy_utils.cuts.del_component('obj_reg_estimate')
        if config.add_regularization == 'level_L1':
            self.mip.MindtPy_utils.del_component('L1_obj')
        elif config.add_regularization == 'level_L_infinity':
            self.mip.MindtPy_utils.del_component('L_infinity_obj')

        return self.mip, main_mip_results

    def set_up_mip_solver(self):
        """Set up the MIP solver.

        Returns
        -------
        mainopt : SolverFactory
            The customized MIP solver.
        """
        # determine if persistent solver is called.
        config = self.config
        if isinstance(self.mip_opt, PersistentSolver):
            self.mip_opt.set_instance(self.mip, symbolic_solver_labels=True)
        if config.single_tree:
            self.set_up_lazy_OA_callback()
        if config.use_tabu_list:
            self.set_up_tabulist_callback()
        mip_args = dict(config.mip_solver_args)
        if config.mip_solver in {
            'cplex',
            'cplex_persistent',
            'gurobi',
            'gurobi_persistent',
        }:
            mip_args['warmstart'] = True
        return mip_args

    # The following functions deal with handling the solution we get from the above MIP solver function

    def handle_main_optimal(self, main_mip, update_bound=True):
        """This function copies the results from 'solve_main' to the working model and updates
        the upper/lower bound. This function is called after an optimal solution is found for
        the main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        update_bound : bool, optional
            Whether to update the bound, by default True.
            Bound will not be updated when handling regularization problem.
        """
        # proceed. Just need integer values
        MindtPy = main_mip.MindtPy_utils
        # check if the value of binary variable is valid
        for var in MindtPy.discrete_variable_list:
            if var.value is None:
                self.config.logger.warning(
                    f"Integer variable {var.name} not initialized.  "
                    "Setting it to its lower bound"
                )
                # nlp_var.bounds[0]
                var.set_value(var.lb, skip_validation=True)
        # warm start for the nlp subproblem
        copy_var_list_values(
            main_mip.MindtPy_utils.variable_list,
            self.fixed_nlp.MindtPy_utils.variable_list,
            self.config,
            skip_fixed=False,
        )

        if update_bound:
            self.update_dual_bound(value(MindtPy.mip_obj.expr))
            self.config.logger.info(
                self.log_formatter.format(
                    self.mip_iter,
                    'MILP',
                    value(MindtPy.mip_obj.expr),
                    self.primal_bound,
                    self.dual_bound,
                    self.rel_gap,
                    get_main_elapsed_time(self.timing),
                )
            )

    def handle_main_infeasible(self):
        """This function handles the result of the latest iteration of solving
        the MIP problem given an infeasible solution.
        """
        self.config.logger.info(
            'MIP main problem is infeasible. '
            'Problem may have no more feasible '
            'binary configurations.'
        )
        if self.mip_iter == 1:
            self.config.logger.warning(
                'MindtPy initialization may have generated poor quality cuts.'
            )
        # TODO no-good cuts for single tree case
        # set optimistic bound to infinity
        self.config.logger.info(
            'MindtPy exiting due to MILP main problem infeasibility.'
        )
        if self.results.solver.termination_condition is None:
            if (
                self.primal_bound == float('inf') and self.objective_sense == minimize
            ) or (
                self.primal_bound == float('-inf') and self.objective_sense == maximize
            ):
                # if self.mip_iter == 0:
                self.results.solver.termination_condition = tc.infeasible
            else:
                self.results.solver.termination_condition = tc.feasible

    def handle_main_max_timelimit(self, main_mip, main_mip_results):
        """This function handles the result of the latest iteration of solving the MIP problem
        given that solving the MIP takes too long.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : [type]
            Results from solving the MIP main subproblem.
        """
        # If we have found a valid feasible solution, we take that. If not, we can at least use the dual bound.
        MindtPy = main_mip.MindtPy_utils
        self.config.logger.info(
            'Unable to optimize MILP main problem '
            'within time limit. '
            'Using current solver feasible solution.'
        )
        copy_var_list_values(
            main_mip.MindtPy_utils.variable_list,
            self.fixed_nlp.MindtPy_utils.variable_list,
            self.config,
            skip_fixed=False,
        )
        self.update_suboptimal_dual_bound(main_mip_results)
        self.config.logger.info(
            self.log_formatter.format(
                self.mip_iter,
                'MILP',
                value(MindtPy.mip_obj.expr),
                self.primal_bound,
                self.dual_bound,
                self.rel_gap,
                get_main_elapsed_time(self.timing),
            )
        )

    def handle_main_unbounded(self, main_mip):
        """This function handles the result of the latest iteration of solving the MIP
        problem given an unbounded solution due to the relaxation.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.

        Returns
        -------
        main_mip_results : SolverResults
            The results of the bounded main problem.
        """
        # Solution is unbounded. Add an arbitrary bound to the objective and resolve.
        # This occurs when the objective is nonlinear. The nonlinear objective is moved
        # to the constraints, and deactivated for the linear main problem.
        config = self.config
        MindtPy = main_mip.MindtPy_utils
        config.logger.warning(
            'main MILP was unbounded. '
            'Resolving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective. '
            'You can change this bound with the option obj_bound.'.format(
                config.obj_bound
            )
        )
        MindtPy.objective_bound = Constraint(
            expr=(-config.obj_bound, MindtPy.mip_obj.expr, config.obj_bound)
        )
        if isinstance(self.mip_opt, PersistentSolver):
            self.mip_opt.set_instance(main_mip)
        update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
        with SuppressInfeasibleWarning():
            main_mip_results = self.mip_opt.solve(
                main_mip,
                tee=config.mip_solver_tee,
                load_solutions=self.load_solutions,
                **config.mip_solver_args,
            )
            if len(main_mip_results.solution) > 0:
                self.mip.solutions.load_from(main_mip_results)
        return main_mip_results

    def handle_regularization_main_tc(self, main_mip, main_mip_results):
        """Handles the result of the regularization main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : SolverResults
            Results from solving the regularization main subproblem.

        Raises
        ------
        ValueError
            MindtPy unable to handle the regularization problem termination condition.
        """
        if main_mip_results is None:
            self.config.logger.info(
                'Failed to solve the regularization problem.'
                'The solution of the OA main problem will be adopted.'
            )
        elif main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
            self.handle_main_optimal(main_mip, update_bound=False)
        elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
            self.config.logger.info(
                'Regularization problem failed to converge within the time limit.'
            )
            self.results.solver.termination_condition = tc.maxTimeLimit
            # break
        elif main_mip_results.solver.termination_condition is tc.infeasible:
            self.config.logger.info('Regularization problem infeasible.')
        elif main_mip_results.solver.termination_condition is tc.unbounded:
            self.config.logger.info(
                'Regularization problem unbounded.'
                'Sometimes solving MIQCP in CPLEX, unbounded means infeasible.'
            )
        elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
            self.config.logger.info(
                'Regularization problem is infeasible or unbounded.'
                'It might happen when using CPLEX to solve MIQP.'
            )
        elif main_mip_results.solver.termination_condition is tc.unknown:
            self.config.logger.info(
                'Termination condition of the regularization problem is unknown.'
            )
            if main_mip_results.problem.lower_bound != float('-inf'):
                self.config.logger.info('Solution limit has been reached.')
                self.handle_main_optimal(main_mip, update_bound=False)
            else:
                self.config.logger.info(
                    'No solution obtained from the regularization subproblem.'
                    'Please set mip_solver_tee to True for more information.'
                    'The solution of the OA main problem will be adopted.'
                )
        else:
            raise ValueError(
                'MindtPy unable to handle regularization problem termination condition '
                'of %s. Solver message: %s'
                % (
                    main_mip_results.solver.termination_condition,
                    main_mip_results.solver.message,
                )
            )

    def setup_main(self):
        """Set up main problem/main regularization problem for OA, ECP, Feasibility Pump and ROA methods."""
        config = self.config
        MindtPy = self.mip.MindtPy_utils

        for c in MindtPy.constraint_list:
            if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
                c.deactivate()

        MindtPy.cuts.activate()

        sign_adjust = 1 if self.objective_sense == minimize else -1
        MindtPy.del_component('mip_obj')
        if config.add_regularization is not None and config.add_no_good_cuts:
            MindtPy.cuts.no_good_cuts.deactivate()

        if config.add_slack:
            MindtPy.del_component('aug_penalty_expr')

            MindtPy.aug_penalty_expr = Expression(
                expr=sign_adjust
                * config.OA_penalty_factor
                * sum(v for v in MindtPy.cuts.slack_vars.values())
            )
        main_objective = MindtPy.objective_list[-1]
        MindtPy.mip_obj = Objective(
            expr=main_objective.expr
            + (MindtPy.aug_penalty_expr if config.add_slack else 0),
            sense=self.objective_sense,
        )

        if config.use_dual_bound:
            # Delete previously added dual bound constraint
            MindtPy.cuts.del_component('dual_bound')
            if self.dual_bound not in {float('inf'), float('-inf')}:
                if self.objective_sense == minimize:
                    MindtPy.cuts.dual_bound = Constraint(
                        expr=main_objective.expr
                        + (MindtPy.aug_penalty_expr if config.add_slack else 0)
                        >= self.dual_bound,
                        doc='Objective function expression should improve on the best found dual bound',
                    )
                else:
                    MindtPy.cuts.dual_bound = Constraint(
                        expr=main_objective.expr
                        + (MindtPy.aug_penalty_expr if config.add_slack else 0)
                        <= self.dual_bound,
                        doc='Objective function expression should improve on the best found dual bound',
                    )

    def setup_fp_main(self):
        """Set up main problem for Feasibility Pump method."""
        MindtPy = self.mip.MindtPy_utils

        for c in MindtPy.constraint_list:
            if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
                c.deactivate()

        MindtPy.cuts.activate()
        MindtPy.del_component('mip_obj')
        MindtPy.del_component('fp_mip_obj')
        if self.config.fp_main_norm == 'L1':
            MindtPy.fp_mip_obj = generate_norm1_objective_function(
                self.mip, self.working_model, discrete_only=self.config.fp_discrete_only
            )
        elif self.config.fp_main_norm == 'L2':
            MindtPy.fp_mip_obj = generate_norm2sq_objective_function(
                self.mip, self.working_model, discrete_only=self.config.fp_discrete_only
            )
        elif self.config.fp_main_norm == 'L_infinity':
            MindtPy.fp_mip_obj = generate_norm_inf_objective_function(
                self.mip, self.working_model, discrete_only=self.config.fp_discrete_only
            )

    def setup_regularization_main(self):
        """Set up main regularization problem for ROA method."""
        config = self.config
        MindtPy = self.mip.MindtPy_utils

        for c in MindtPy.constraint_list:
            if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
                c.deactivate()

        MindtPy.cuts.activate()

        sign_adjust = 1 if self.objective_sense == minimize else -1
        MindtPy.del_component('mip_obj')
        if config.single_tree:
            MindtPy.del_component('roa_proj_mip_obj')
            MindtPy.cuts.del_component('obj_reg_estimate')
        if config.add_regularization is not None and config.add_no_good_cuts:
            MindtPy.cuts.no_good_cuts.activate()

        # The epigraph constraint is very "flat" for branching rules.
        # In ROA, if the objective function is linear(or quadratic when quadratic_strategy = 1 or 2), the original objective function is used in the MIP problem.
        # In the MIP projection problem, we need to reactivate the epigraph constraint(objective_constr).
        if (
            MindtPy.objective_list[0].expr.polynomial_degree()
            in self.mip_objective_polynomial_degree
        ):
            MindtPy.objective_constr.activate()
        if config.add_regularization == 'level_L1':
            MindtPy.roa_proj_mip_obj = generate_norm1_objective_function(
                self.mip, self.best_solution_found, discrete_only=False
            )
        elif config.add_regularization == 'level_L2':
            MindtPy.roa_proj_mip_obj = generate_norm2sq_objective_function(
                self.mip, self.best_solution_found, discrete_only=False
            )
        elif config.add_regularization == 'level_L_infinity':
            MindtPy.roa_proj_mip_obj = generate_norm_inf_objective_function(
                self.mip, self.best_solution_found, discrete_only=False
            )
        elif config.add_regularization in {
            'grad_lag',
            'hess_lag',
            'hess_only_lag',
            'sqp_lag',
        }:
            MindtPy.roa_proj_mip_obj = generate_lag_objective_function(
                self.mip,
                self.best_solution_found,
                config,
                self.timing,
                discrete_only=False,
            )
        if self.objective_sense == minimize:
            MindtPy.cuts.obj_reg_estimate = Constraint(
                expr=sum(MindtPy.objective_value[:])
                <= (1 - config.level_coef) * self.primal_bound
                + config.level_coef * self.dual_bound
            )
        else:
            MindtPy.cuts.obj_reg_estimate = Constraint(
                expr=sum(MindtPy.objective_value[:])
                >= (1 - config.level_coef) * self.primal_bound
                + config.level_coef * self.dual_bound
            )

    def update_result(self):
        if self.objective_sense == minimize:
            self.results.problem.lower_bound = self.dual_bound
            self.results.problem.upper_bound = self.primal_bound
        else:
            self.results.problem.lower_bound = self.primal_bound
            self.results.problem.upper_bound = self.dual_bound

        self.results.solver.timing = self.timing
        self.results.solver.user_time = self.timing.total
        self.results.solver.wallclock_time = self.timing.total
        self.results.solver.iterations = self.mip_iter
        self.results.solver.num_infeasible_nlp_subproblem = self.nlp_infeasible_counter
        self.results.solver.best_solution_found_time = self.best_solution_found_time
        self.results.solver.primal_integral = self.primal_integral
        self.results.solver.dual_integral = self.dual_integral
        self.results.solver.primal_dual_gap_integral = self.primal_dual_gap_integral

    def load_solution(self):
        # Update values in original model
        config = self.config
        MindtPy = self.working_model.MindtPy_utils
        copy_var_list_values(
            from_list=self.best_solution_found.MindtPy_utils.variable_list,
            to_list=MindtPy.variable_list,
            config=config,
        )
        # The original does not have variable list.
        # Use get_vars_from_components() should be used for both working_model and original_model to exclude the unused variables.
        self.working_model.MindtPy_utils.deactivate()
        # The original objective should be activated to make sure the variable list is in the same order (get_vars_from_components).
        self.working_model.MindtPy_utils.objective_list[0].activate()
        if self.working_model.component("_int_to_binary_reform") is not None:
            self.working_model._int_to_binary_reform.deactivate()
        # exclude fixed variables here. This is consistent with the definition of variable_list.
        working_model_variable_list = list(
            get_vars_from_components(
                block=self.working_model,
                ctype=(Constraint, Objective),
                include_fixed=False,
                active=True,
                sort=True,
                descend_into=True,
                descent_order=None,
            )
        )
        original_model_variable_list = list(
            get_vars_from_components(
                block=self.original_model,
                ctype=(Constraint, Objective),
                include_fixed=False,
                active=True,
                sort=True,
                descend_into=True,
                descent_order=None,
            )
        )
        for v_from, v_to in zip(
            working_model_variable_list, original_model_variable_list
        ):
            if v_from.name != v_to.name:
                raise DeveloperError(
                    'The name of the two variables is not the same. Loading final solution'
                )
        copy_var_list_values(
            working_model_variable_list, original_model_variable_list, config=config
        )

    def check_subsolver_validity(self):
        """Check if the subsolvers are available and licensed."""
        if not self.mip_opt.available():
            raise ValueError(self.config.mip_solver + ' is not available.')
        if not self.mip_opt.license_is_valid():
            raise ValueError(self.config.mip_solver + ' is not licensed.')
        if not self.nlp_opt.available():
            raise ValueError(self.config.nlp_solver + ' is not available.')
        if not self.nlp_opt.license_is_valid():
            raise ValueError(self.config.nlp_solver + ' is not licensed.')
        if self.config.add_regularization is not None:
            if not self.regularization_mip_opt.available():
                raise ValueError(
                    self.config.mip_regularization_solver + ' is not available.'
                )
            if not self.regularization_mip_opt.license_is_valid():
                raise ValueError(
                    self.config.mip_regularization_solver + ' is not licensed.'
                )

    def check_config(self):
        """Checks if the configuration options make sense."""
        config = self.config
        # configuration confirmation
        if config.init_strategy == 'FP':
            config.add_no_good_cuts = True
            config.use_tabu_list = False

        if config.nlp_solver == 'baron':
            config.equality_relaxation = False
        if config.nlp_solver == 'gams' and config.nlp_solver.__contains__('solver'):
            if config.nlp_solver_args['solver'] == 'baron':
                config.equality_relaxation = False

        if config.solver_tee:
            config.mip_solver_tee = True
            config.nlp_solver_tee = True
        if config.add_no_good_cuts:
            config.integer_to_binary = True
        if config.use_tabu_list:
            config.mip_solver = 'cplex_persistent'
            if config.threads > 1:
                config.threads = 1
                config.logger.info(
                    'The threads parameter is corrected to 1 since incumbent callback conflicts with multi-threads mode.'
                )
        if config.solution_pool:
            if config.mip_solver not in {'cplex_persistent', 'gurobi_persistent'}:
                if config.mip_solver in {'appsi_cplex', 'appsi_gurobi'}:
                    config.logger.info("Solution pool does not support APPSI solver.")
                config.mip_solver = 'cplex_persistent'

        # related to https://github.com/Pyomo/pyomo/issues/2363
        if (
            'appsi' in config.mip_solver
            or 'appsi' in config.nlp_solver
            or (
                config.mip_regularization_solver is not None
                and 'appsi' in config.mip_regularization_solver
            )
        ):
            self.load_solutions = False

    ################################################################################################################################
    # Feasibility Pump

    def solve_fp_subproblem(self):
        """Solves the feasibility pump NLP subproblem.

        This function sets up the 'fp_nlp' by relax integer variables.
        precomputes dual values, deactivates trivial constraints, and then solves NLP model.

        Returns
        -------
        fp_nlp : Pyomo model
            Fixed-NLP from the model.
        results : SolverResults
            Results from solving the fixed-NLP subproblem.
        """
        fp_nlp = self.working_model.clone()
        MindtPy = fp_nlp.MindtPy_utils
        config = self.config

        # Set up NLP
        fp_nlp.MindtPy_utils.objective_list[-1].deactivate()
        if self.objective_sense == minimize:
            fp_nlp.improving_objective_cut = Constraint(
                expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) <= self.primal_bound
            )
        else:
            fp_nlp.improving_objective_cut = Constraint(
                expr=sum(fp_nlp.MindtPy_utils.objective_value[:]) >= self.primal_bound
            )

        # Add norm_constraint, which guarantees the monotonicity of the norm objective value sequence of all iterations
        # Ref: Paper 'A storm of feasibility pumps for nonconvex MINLP'   https://doi.org/10.1007/s10107-012-0608-x
        # the norm type is consistent with the norm obj of the FP-main problem.
        if config.fp_norm_constraint:
            generate_norm_constraint(fp_nlp, self.mip, config)

        MindtPy.fp_nlp_obj = generate_norm2sq_objective_function(
            fp_nlp, self.mip, discrete_only=config.fp_discrete_only
        )

        MindtPy.cuts.deactivate()
        TransformationFactory('core.relax_integer_vars').apply_to(fp_nlp)
        try:
            TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(
                fp_nlp,
                tmp=True,
                ignore_infeasible=False,
                tolerance=config.constraint_tolerance,
            )
        except InfeasibleConstraintException as e:
            config.logger.error(
                str(e) + '\nInfeasibility detected in deactivate_trivial_constraints.'
            )
            results = SolverResults()
            results.solver.termination_condition = tc.infeasible
            return fp_nlp, results
        # Solve the NLP
        nlp_args = dict(config.nlp_solver_args)
        update_solver_timelimit(self.nlp_opt, config.nlp_solver, self.timing, config)
        with SuppressInfeasibleWarning():
            with time_code(self.timing, 'fp subproblem'):
                results = self.nlp_opt.solve(
                    fp_nlp,
                    tee=config.nlp_solver_tee,
                    load_solutions=self.load_solutions,
                    **nlp_args,
                )
                if len(results.solution) > 0:
                    fp_nlp.solutions.load_from(results)
        return fp_nlp, results

    def handle_fp_subproblem_optimal(self, fp_nlp):
        """Copies the solution to the working model, updates bound, adds OA cuts / no-good cuts /
        increasing objective cut, calculates the duals and stores incumbent solution if it has been improved.

        Parameters
        ----------
        fp_nlp : Pyomo model
            The feasibility pump NLP subproblem.
        """
        copy_var_list_values(
            fp_nlp.MindtPy_utils.variable_list,
            self.working_model.MindtPy_utils.variable_list,
            self.config,
        )
        add_orthogonality_cuts(self.working_model, self.mip, self.config)

        # if OA-like or fp converged, update Upper bound,
        # add no_good cuts and increasing objective cuts (fp)
        if fp_converged(
            self.working_model,
            self.mip,
            proj_zero_tolerance=self.config.fp_projzerotol,
            discrete_only=self.config.fp_discrete_only,
        ):
            copy_var_list_values(
                self.mip.MindtPy_utils.variable_list,
                self.fixed_nlp.MindtPy_utils.variable_list,
                self.config,
                skip_fixed=False,
            )
            fixed_nlp, fixed_nlp_results = self.solve_subproblem()
            if fixed_nlp_results.solver.termination_condition in {
                tc.optimal,
                tc.locallyOptimal,
                tc.feasible,
            }:
                self.handle_subproblem_optimal(fixed_nlp)
                if self.primal_bound_improved:
                    self.mip.MindtPy_utils.cuts.del_component('improving_objective_cut')
                    if self.objective_sense == minimize:
                        self.mip.MindtPy_utils.cuts.improving_objective_cut = (
                            Constraint(
                                expr=sum(self.mip.MindtPy_utils.objective_value[:])
                                <= self.primal_bound
                                - self.config.fp_cutoffdecr
                                * max(1, abs(self.primal_bound))
                            )
                        )
                    else:
                        self.mip.MindtPy_utils.cuts.improving_objective_cut = (
                            Constraint(
                                expr=sum(self.mip.MindtPy_utils.objective_value[:])
                                >= self.primal_bound
                                + self.config.fp_cutoffdecr
                                * max(1, abs(self.primal_bound))
                            )
                        )
            else:
                self.config.logger.error(
                    'Feasibility pump Fixed-NLP is infeasible, something might be wrong. '
                    'There might be a problem with the precisions - the feasibility pump seems to have converged'
                )

    def handle_fp_main_tc(self, fp_main_results):
        """Handle the termination condition of the feasibility pump main problem.

        Parameters
        ----------
        fp_main_results : SolverResults
            The results from solving the FP main problem.

        Returns
        -------
        bool
            True if FP loop should terminate, False otherwise.
        """
        if fp_main_results.solver.termination_condition is tc.optimal:
            self.config.logger.info(
                self.log_formatter.format(
                    self.fp_iter,
                    'FP-MIP',
                    value(self.mip.MindtPy_utils.fp_mip_obj),
                    self.primal_bound,
                    self.dual_bound,
                    self.rel_gap,
                    get_main_elapsed_time(self.timing),
                )
            )
            return False
        elif fp_main_results.solver.termination_condition is tc.maxTimeLimit:
            self.config.logger.warning('FP-MIP reaches max TimeLimit')
            self.results.solver.termination_condition = tc.maxTimeLimit
            return True
        elif fp_main_results.solver.termination_condition is tc.infeasible:
            self.config.logger.warning('FP-MIP infeasible')
            no_good_cuts = self.mip.MindtPy_utils.cuts.no_good_cuts
            if no_good_cuts.__len__() > 0:
                no_good_cuts[no_good_cuts.__len__()].deactivate()
            return True
        elif fp_main_results.solver.termination_condition is tc.unbounded:
            self.config.logger.warning('FP-MIP unbounded')
            return True
        elif (
            fp_main_results.solver.termination_condition is tc.other
            and fp_main_results.solution.status is SolutionStatus.feasible
        ):
            self.config.logger.warning(
                'MILP solver reported feasible solution of FP-MIP, '
                'but not guaranteed to be optimal.'
            )
            return False
        else:
            self.config.logger.warning('Unexpected result of FP-MIP')
            return True

    def fp_loop(self):
        """Feasibility pump loop.

        This is the outermost function for the Feasibility Pump algorithm in this package; this function
        controls the progress of solving the model.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the FP-NLP subproblem.
        """
        config = self.config
        while self.fp_iter < config.fp_iteration_limit:
            # solve MIP main problem
            with time_code(self.timing, 'fp main'):
                fp_main, fp_main_results = self.solve_fp_main()
            fp_should_terminate = self.handle_fp_main_tc(fp_main_results)
            if fp_should_terminate:
                break

            # Solve NLP subproblem
            # The constraint linearization happens in the handlers
            fp_nlp, fp_nlp_result = self.solve_fp_subproblem()

            if fp_nlp_result.solver.termination_condition in {
                tc.optimal,
                tc.locallyOptimal,
                tc.feasible,
            }:
                config.logger.info(
                    self.log_formatter.format(
                        self.fp_iter,
                        'FP-NLP',
                        value(fp_nlp.MindtPy_utils.fp_nlp_obj),
                        self.primal_bound,
                        self.dual_bound,
                        self.rel_gap,
                        get_main_elapsed_time(self.timing),
                    )
                )
                self.handle_fp_subproblem_optimal(fp_nlp)
            elif fp_nlp_result.solver.termination_condition in {
                tc.infeasible,
                tc.noSolution,
            }:
                config.logger.error('Feasibility pump NLP subproblem infeasible')
                self.should_terminate = True
                self.results.solver.status = SolverStatus.error
                return
            elif fp_nlp_result.solver.termination_condition is tc.maxIterations:
                config.logger.error(
                    'Feasibility pump NLP subproblem failed to converge within iteration limit.'
                )
                self.should_terminate = True
                self.results.solver.status = SolverStatus.error
                return
            else:
                raise ValueError(
                    'MindtPy unable to handle NLP subproblem termination '
                    'condition of {}'.format(fp_nlp_result.solver.termination_condition)
                )
            # Call the NLP post-solve callback
            config.call_after_subproblem_solve(fp_nlp)
            self.fp_iter += 1
        self.mip.MindtPy_utils.del_component('fp_mip_obj')

        if config.fp_main_norm == 'L1':
            self.mip.MindtPy_utils.del_component('L1_obj')
        elif config.fp_main_norm == 'L_infinity':
            self.mip.MindtPy_utils.del_component('L_infinity_obj')

        # deactivate the improving_objective_cut
        self.mip.MindtPy_utils.cuts.del_component('improving_objective_cut')
        if not config.fp_transfercuts:
            for c in self.mip.MindtPy_utils.cuts.oa_cuts:
                c.deactivate()
            for c in self.mip.MindtPy_utils.cuts.no_good_cuts:
                c.deactivate()
        if config.fp_projcuts:
            self.working_model.MindtPy_utils.cuts.del_component('fp_orthogonality_cuts')

    def initialize_mip_problem(self):
        '''Deactivate the nonlinear constraints to create the MIP problem.'''
        # if single tree is activated, we need to add bounds for unbounded variables in nonlinear constraints to avoid unbounded main problem.
        config = self.config
        if config.single_tree:
            add_var_bound(self.working_model, config)

        self.mip = self.working_model.clone()
        next(self.mip.component_data_objects(Objective, active=True)).deactivate()
        if hasattr(self.mip, 'dual') and isinstance(self.mip.dual, Suffix):
            self.mip.del_component('dual')
        # Deactivate extraneous IMPORT/EXPORT suffixes
        if config.nlp_solver in {'ipopt', 'cyipopt'}:
            getattr(self.mip, 'ipopt_zL_out', _DoNothing()).deactivate()
            getattr(self.mip, 'ipopt_zU_out', _DoNothing()).deactivate()

        MindtPy = self.mip.MindtPy_utils
        if len(MindtPy.grey_box_list) > 0:
            for grey_box in MindtPy.grey_box_list:
                grey_box.deactivate()

        if config.init_strategy == 'FP':
            MindtPy.cuts.fp_orthogonality_cuts = ConstraintList(
                doc='Orthogonality cuts in feasibility pump'
            )
            if config.fp_projcuts:
                self.working_model.MindtPy_utils.cuts.fp_orthogonality_cuts = (
                    ConstraintList(doc='Orthogonality cuts in feasibility pump')
                )

        self.fixed_nlp = self.working_model.clone()
        TransformationFactory('core.fix_integer_vars').apply_to(self.fixed_nlp)
        initialize_feas_subproblem(self.fixed_nlp, config)

    def initialize_subsolvers(self):
        """Initialize and set options for MIP and NLP subsolvers."""
        config = self.config
        if config.mip_solver == 'gurobi_persistent' and config.single_tree:
            self.mip_opt = GurobiPersistent4MindtPy()
        else:
            self.mip_opt = SolverFactory(config.mip_solver)
        self.nlp_opt = SolverFactory(config.nlp_solver)
        self.feasibility_nlp_opt = SolverFactory(config.nlp_solver)
        if config.mip_regularization_solver is not None:
            self.regularization_mip_opt = SolverFactory(
                config.mip_regularization_solver
            )

        self.check_subsolver_validity()
        if config.mip_solver == 'gams':
            self.mip_opt.options['add_options'] = []
        if config.nlp_solver == 'gams':
            self.nlp_opt.options['add_options'] = []
            self.feasibility_nlp_opt.options['add_options'] = []
        set_solver_mipgap(self.mip_opt, config.mip_solver, config)

        set_solver_constraint_violation_tolerance(
            self.nlp_opt, config.nlp_solver, config
        )
        set_solver_constraint_violation_tolerance(
            self.feasibility_nlp_opt, config.nlp_solver, config
        )

        self.set_appsi_solver_update_config()

        if config.mip_solver == 'gurobi_persistent' and config.single_tree:
            # PreCrush: Controls presolve reductions that affect user cuts
            # You should consider setting this parameter to 1 if you are using callbacks to add your own cuts.
            self.mip_opt.options['PreCrush'] = 1
            self.mip_opt.options['LazyConstraints'] = 1

        # set threads
        if config.threads > 0:
            self.mip_opt.options['threads'] = config.threads
        # regularization solver
        if config.mip_regularization_solver is not None:
            set_solver_mipgap(
                self.regularization_mip_opt, config.mip_regularization_solver, config
            )
            if config.mip_regularization_solver == 'gams':
                self.regularization_mip_opt.options['add_options'] = []
            if config.regularization_mip_threads > 0:
                self.regularization_mip_opt.options[
                    'threads'
                ] = config.regularization_mip_threads
            else:
                self.regularization_mip_opt.options['threads'] = config.threads

            if config.mip_regularization_solver in {
                'cplex',
                'appsi_cplex',
                'cplex_persistent',
            }:
                if config.solution_limit is not None:
                    self.regularization_mip_opt.options[
                        'mip_limits_solutions'
                    ] = config.solution_limit
                # We don't need to solve the regularization problem to optimality.
                # We will choose to perform aggressive node probing during presolve.
                self.regularization_mip_opt.options['mip_strategy_presolvenode'] = 3
                # When using ROA method to solve convex MINLPs, the Hessian of the Lagrangean is always positive semidefinite,
                # and the regularization subproblems are always convex.
                # However, due to numerical accuracy, the regularization problem ended up nonconvex for a few cases,
                # e.g., the smallest eigenvalue of the Hessian was slightly negative.
                # Therefore, we set the optimalitytarget parameter to 3 to enable CPLEX to solve nonconvex MIQPs in the ROA-L2 and ROA-∇2L methods.
                if config.add_regularization in {'hess_lag', 'hess_only_lag'}:
                    self.regularization_mip_opt.options['optimalitytarget'] = 3
            elif config.mip_regularization_solver == 'gurobi':
                if config.solution_limit is not None:
                    self.regularization_mip_opt.options[
                        'SolutionLimit'
                    ] = config.solution_limit
                # Same reason as mip_strategy_presolvenode.
                self.regularization_mip_opt.options['Presolve'] = 2

    def set_appsi_solver_update_config(self):
        """Set update config for APPSI solvers."""
        config = self.config
        if config.mip_solver in {'appsi_cplex', 'appsi_gurobi', 'appsi_highs'}:
            # mip main problem
            self.mip_opt.update_config.check_for_new_or_removed_constraints = True
            self.mip_opt.update_config.check_for_new_or_removed_vars = True
            self.mip_opt.update_config.check_for_new_or_removed_params = False
            self.mip_opt.update_config.check_for_new_objective = True
            self.mip_opt.update_config.update_constraints = True
            self.mip_opt.update_config.update_vars = True
            self.mip_opt.update_config.update_params = False
            self.mip_opt.update_config.update_named_expressions = False
            self.mip_opt.update_config.update_objective = False
            self.mip_opt.update_config.treat_fixed_vars_as_params = True

        if config.nlp_solver == 'appsi_ipopt':
            # fixed-nlp
            self.nlp_opt.update_config.check_for_new_or_removed_constraints = False
            self.nlp_opt.update_config.check_for_new_or_removed_vars = False
            self.nlp_opt.update_config.check_for_new_or_removed_params = False
            self.nlp_opt.update_config.check_for_new_objective = False
            self.nlp_opt.update_config.update_constraints = True
            self.nlp_opt.update_config.update_vars = True
            self.nlp_opt.update_config.update_params = False
            self.nlp_opt.update_config.update_named_expressions = False
            self.nlp_opt.update_config.update_objective = False
            self.nlp_opt.update_config.treat_fixed_vars_as_params = False

            self.feasibility_nlp_opt.update_config.check_for_new_or_removed_constraints = (
                False
            )
            self.feasibility_nlp_opt.update_config.check_for_new_or_removed_vars = False
            self.feasibility_nlp_opt.update_config.check_for_new_or_removed_params = (
                False
            )
            self.feasibility_nlp_opt.update_config.check_for_new_objective = False
            self.feasibility_nlp_opt.update_config.update_constraints = False
            self.feasibility_nlp_opt.update_config.update_vars = True
            self.feasibility_nlp_opt.update_config.update_params = False
            self.feasibility_nlp_opt.update_config.update_named_expressions = False
            self.feasibility_nlp_opt.update_config.update_objective = False
            self.feasibility_nlp_opt.update_config.treat_fixed_vars_as_params = False

    def solve(self, model, **kwds):
        """Solve the model.

        Parameters
        ----------
        model : Pyomo model
            The MINLP model to be solved.

        Returns
        -------
        results : SolverResults
            Results from solving the MINLP problem by MindtPy.
        """
        config = self.config = self.CONFIG(
            kwds.pop('options', {}), preserve_implicit=True
        )
        config.set_value(kwds)
        self.set_up_logger()
        new_logging_level = logging.INFO if config.tee else None
        with lower_logger_level_to(config.logger, new_logging_level):
            self.check_config()

        self.set_up_solve_data(model)

        if config.integer_to_binary:
            TransformationFactory('contrib.integer_to_binary').apply_to(
                self.working_model
            )

        self.create_utility_block(self.working_model, 'MindtPy_utils')
        with time_code(self.timing, 'total', is_main_timer=True), lower_logger_level_to(
            config.logger, new_logging_level
        ):
            self._log_solver_intro_message()
            self.initialize_subsolvers()

            # Validate the model to ensure that MindtPy is able to solve it.
            if not self.model_is_valid():
                return

            MindtPy = self.working_model.MindtPy_utils

            setup_results_object(self.results, self.original_model, config)

            # Reformulate the objective function.
            self.objective_reformulation()

            # Save model initial values.
            self.initial_var_values = list(v.value for v in MindtPy.variable_list)

            # TODO: if the MindtPy solver is defined once and called several times to solve models. The following two lines are necessary. It seems that the solver class will not be init every time call.
            # For example, if we remove the following two lines. test_RLPNLP_L1 will fail.
            self.best_solution_found = None
            self.best_solution_found_time = None
            self.initialize_mip_problem()

            # Initialization
            with time_code(self.timing, 'initialization'):
                self.MindtPy_initialization()

            # Algorithm main loop
            with time_code(self.timing, 'main loop'):
                self.MindtPy_iteration_loop()

            # Load solution
            if self.best_solution_found is not None:
                self.load_solution()

            # Get integral info
            self.get_integral_info()

            config.logger.info(
                ' {:<25}:   {:>7.4f} '.format(
                    'Primal-dual gap integral', self.primal_dual_gap_integral
                )
            )

        # Update result
        self.update_result()
        if config.single_tree:
            self.results.solver.num_nodes = self.nlp_iter - (
                1 if config.init_strategy == 'rNLP' else 0
            )

        return self.results

    def objective_reformulation(self):
        # In the process_objective function, as long as the objective function is nonlinear, it will be reformulated and the variable/constraint/objective lists will be updated.
        # For OA/GOA/LP-NLP algorithm, if the objective function is linear, it will not be reformulated as epigraph constraint.
        # If the objective function is linear, it will be reformulated as epigraph constraint only if the Feasibility Pump or ROA/RLP-NLP algorithm is activated. (move_objective = True)
        # In some cases, the variable/constraint/objective lists will not be updated even if the objective is epigraph-reformulated.
        # In Feasibility Pump, since the distance calculation only includes discrete variables and the epigraph slack variables are continuous variables, the Feasibility Pump algorithm will not affected even if the variable list are updated.
        # In ROA and RLP/NLP, since the distance calculation does not include these epigraph slack variables, they should not be added to the variable list. (update_var_con_list = False)
        # In the process_objective function, once the objective function has been reformulated as epigraph constraint, the variable/constraint/objective lists will not be updated only if the MINLP has a linear objective function and regularization is activated at the same time.
        # This is because the epigraph constraint is very "flat" for branching rules. The original objective function will be used for the main problem and epigraph reformulation will be used for the projection problem.
        self.process_objective(update_var_con_list=True)

    def handle_main_mip_termination(self, main_mip, main_mip_results):
        should_terminate = False
        if main_mip_results is not None:
            if not self.config.single_tree:
                if main_mip_results.solver.termination_condition is tc.optimal:
                    self.handle_main_optimal(main_mip)
                elif main_mip_results.solver.termination_condition is tc.infeasible:
                    self.handle_main_infeasible()
                    self.last_iter_cuts = True
                    should_terminate = True
                elif main_mip_results.solver.termination_condition is tc.unbounded:
                    temp_results = self.handle_main_unbounded(main_mip)
                elif (
                    main_mip_results.solver.termination_condition
                    is tc.infeasibleOrUnbounded
                ):
                    temp_results = self.handle_main_unbounded(main_mip)
                    if temp_results.solver.termination_condition is tc.infeasible:
                        self.handle_main_infeasible()
                elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
                    self.handle_main_max_timelimit(main_mip, main_mip_results)
                    self.results.solver.termination_condition = tc.maxTimeLimit
                elif main_mip_results.solver.termination_condition is tc.feasible or (
                    main_mip_results.solver.termination_condition is tc.other
                    and main_mip_results.solution.status is SolutionStatus.feasible
                ):
                    # load the solution and suppress the warning message by setting
                    # solver status to ok.
                    MindtPy = main_mip.MindtPy_utils
                    self.config.logger.info(
                        'MILP solver reported feasible solution, '
                        'but not guaranteed to be optimal.'
                    )
                    copy_var_list_values(
                        main_mip.MindtPy_utils.variable_list,
                        self.fixed_nlp.MindtPy_utils.variable_list,
                        self.config,
                        skip_fixed=False,
                    )
                    self.update_suboptimal_dual_bound(main_mip_results)
                    self.config.logger.info(
                        self.log_formatter.format(
                            self.mip_iter,
                            'MILP',
                            value(MindtPy.mip_obj.expr),
                            self.primal_bound,
                            self.dual_bound,
                            self.rel_gap,
                            get_main_elapsed_time(self.timing),
                        )
                    )
                else:
                    raise ValueError(
                        'MindtPy unable to handle MILP main termination condition '
                        'of %s. Solver message: %s'
                        % (
                            main_mip_results.solver.termination_condition,
                            main_mip_results.solver.message,
                        )
                    )
        else:
            self.config.logger.info('Algorithm should terminate here.')
            should_terminate = True
            # break
        return should_terminate

    # iterate.py
    def MindtPy_iteration_loop(self):
        """Main loop for MindtPy Algorithms.

        This is the outermost function for the Outer Approximation algorithm in this package; this function controls the progress of
        solving the model.

        Raises
        ------
        ValueError
            The strategy value is not correct or not included.
        """
        config = self.config
        while self.mip_iter < config.iteration_limit:
            # solve MIP main problem
            with time_code(self.timing, 'main'):
                main_mip, main_mip_results = self.solve_main()
            if self.handle_main_mip_termination(main_mip, main_mip_results):
                break
            # Call the MIP post-solve callback
            with time_code(self.timing, 'Call after main solve'):
                config.call_after_main_solve(main_mip)

            # Regularization is activated after the first feasible solution is found.
            if config.add_regularization is not None:
                if not config.single_tree:
                    self.add_regularization()

                # In R-LP/NLP, we might end up with an integer combination that hasn't been explored.
                # Therefore, we need to solve fixed NLP subproblem one more time.
                if config.single_tree:
                    self.curr_int_sol = get_integer_solution(self.mip, string_zero=True)
                    copy_var_list_values(
                        main_mip.MindtPy_utils.variable_list,
                        self.fixed_nlp.MindtPy_utils.variable_list,
                        config,
                        skip_fixed=False,
                    )
                    if self.curr_int_sol not in set(self.integer_list):
                        fixed_nlp, fixed_nlp_result = self.solve_subproblem()
                        self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)

            if self.algorithm_should_terminate(check_cycling=True):
                self.last_iter_cuts = False
                break

            if not config.single_tree:  # if we don't use lazy callback, i.e. LP_NLP
                # Solve NLP subproblem
                # The constraint linearization happens in the handlers
                if not config.solution_pool:
                    fixed_nlp, fixed_nlp_result = self.solve_subproblem()
                    self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)

                    # Call the NLP post-solve callback
                    with time_code(self.timing, 'Call after subproblem solve'):
                        config.call_after_subproblem_solve(fixed_nlp)

                    if self.algorithm_should_terminate(check_cycling=False):
                        self.last_iter_cuts = True
                        break
                else:
                    solution_name_obj = self.get_solution_name_obj(main_mip_results)
                    for index, (name, _) in enumerate(solution_name_obj):
                        # the optimal solution of the main problem has been added to integer_list above
                        # so we should skip checking cycling for the first solution in the solution pool
                        if index > 0:
                            copy_var_list_values_from_solution_pool(
                                self.mip.MindtPy_utils.variable_list,
                                self.fixed_nlp.MindtPy_utils.variable_list,
                                config,
                                solver_model=main_mip_results._solver_model,
                                var_map=main_mip_results._pyomo_var_to_solver_var_map,
                                solution_name=name,
                            )
                            self.curr_int_sol = get_integer_solution(self.fixed_nlp)
                            if self.curr_int_sol in set(self.integer_list):
                                config.logger.info(
                                    'The same combination has been explored and will be skipped here.'
                                )
                                continue
                            else:
                                self.integer_list.append(self.curr_int_sol)
                        fixed_nlp, fixed_nlp_result = self.solve_subproblem()
                        self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)

                        # Call the NLP post-solve callback
                        with time_code(self.timing, 'Call after subproblem solve'):
                            config.call_after_subproblem_solve(fixed_nlp)

                        if self.algorithm_should_terminate(check_cycling=False):
                            self.last_iter_cuts = True
                            break  # TODO: break two loops.

        # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
        # we correct it after the iteration.
        if (
            (config.add_no_good_cuts or config.use_tabu_list)
            and not self.should_terminate
            and config.add_regularization is None
        ):
            self.fix_dual_bound(self.last_iter_cuts)
        config.logger.info(
            ' ==============================================================================================='
        )

    def get_solution_name_obj(self, main_mip_results):
        if self.config.mip_solver == 'cplex_persistent':
            solution_pool_names = (
                main_mip_results._solver_model.solution.pool.get_names()
            )
        elif self.config.mip_solver == 'gurobi_persistent':
            solution_pool_names = list(range(main_mip_results._solver_model.SolCount))
        # list to store the name and objective value of the solutions in the solution pool
        solution_name_obj = []
        for name in solution_pool_names:
            if self.config.mip_solver == 'cplex_persistent':
                obj = main_mip_results._solver_model.solution.pool.get_objective_value(
                    name
                )
            elif self.config.mip_solver == 'gurobi_persistent':
                main_mip_results._solver_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, name
                )
                obj = main_mip_results._solver_model.PoolObjVal
            solution_name_obj.append([name, obj])
        solution_name_obj.sort(
            key=itemgetter(1), reverse=self.objective_sense == maximize
        )
        solution_name_obj = solution_name_obj[: self.config.num_solution_iteration]
        return solution_name_obj

    def add_regularization(self):
        if self.best_solution_found is not None:
            # The main problem might be unbounded, regularization is activated only when a valid bound is provided.
            if self.dual_bound != self.dual_bound_progress[0]:
                with time_code(self.timing, 'regularization main'):
                    (
                        regularization_main_mip,
                        regularization_main_mip_results,
                    ) = self.solve_regularization_main()
                self.handle_regularization_main_tc(
                    regularization_main_mip, regularization_main_mip_results
                )

    def bounds_converged(self):
        # Check bound convergence
        if self.abs_gap <= self.config.absolute_bound_tolerance:
            self.config.logger.info(
                'MindtPy exiting on bound convergence. '
                'Absolute gap: {} <= absolute tolerance: {} \n'.format(
                    self.abs_gap, self.config.absolute_bound_tolerance
                )
            )
            self.results.solver.termination_condition = tc.optimal
            return True
        # Check relative bound convergence
        if self.best_solution_found is not None:
            if self.rel_gap <= self.config.relative_bound_tolerance:
                self.config.logger.info(
                    'MindtPy exiting on bound convergence. '
                    'Relative gap : {} <= relative tolerance: {} \n'.format(
                        self.rel_gap, self.config.relative_bound_tolerance
                    )
                )
                self.results.solver.termination_condition = tc.optimal
                return True
        return False

    def reached_iteration_limit(self):
        # Check iteration limit
        if self.mip_iter >= self.config.iteration_limit:
            self.config.logger.info(
                'MindtPy unable to converge bounds '
                'after {} main iterations.'.format(self.mip_iter)
            )
            self.config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.format(
                    self.primal_bound, self.dual_bound
                )
            )
            if self.config.single_tree:
                self.results.solver.termination_condition = tc.feasible
            else:
                self.results.solver.termination_condition = tc.maxIterations
            return True
        else:
            return False

    def reached_time_limit(self):
        if get_main_elapsed_time(self.timing) >= self.config.time_limit:
            self.config.logger.info(
                'MindtPy unable to converge bounds '
                'before time limit of {} seconds. '
                'Elapsed: {} seconds'.format(
                    self.config.time_limit, get_main_elapsed_time(self.timing)
                )
            )
            self.config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.format(
                    self.primal_bound, self.dual_bound
                )
            )
            self.results.solver.termination_condition = tc.maxTimeLimit
            return True
        else:
            return False

    def reached_stalling_limit(self):
        config = self.config
        if len(self.primal_bound_progress) >= config.stalling_limit:
            if (
                abs(
                    self.primal_bound_progress[-1]
                    - self.primal_bound_progress[-config.stalling_limit]
                )
                <= config.zero_tolerance
            ):
                config.logger.info(
                    'Algorithm is not making enough progress. '
                    'Exiting iteration loop.'
                )
                config.logger.info(
                    'Final bound values: Primal Bound: {}  Dual Bound: {}'.format(
                        self.primal_bound, self.dual_bound
                    )
                )
                if self.best_solution_found is not None:
                    self.results.solver.termination_condition = tc.feasible
                else:
                    # TODO: Is it correct to set self.working_model as the best_solution_found?
                    # In function copy_var_list_values, skip_fixed is set to True in default.
                    self.best_solution_found = self.working_model.clone()
                    config.logger.warning(
                        'Algorithm did not find a feasible solution. '
                        'Returning best bound solution. Consider increasing stalling_limit or absolute_bound_tolerance.'
                    )
                    self.results.solver.termination_condition = tc.noSolution
                return True
        return False

    def iteration_cycling(self):
        config = self.config
        if config.cycling_check or config.use_tabu_list:
            self.curr_int_sol = get_integer_solution(self.mip)
            if config.cycling_check and self.mip_iter >= 1:
                if self.curr_int_sol in set(self.integer_list):
                    config.logger.info(
                        'Cycling happens after {} main iterations. '
                        'The same combination is obtained in iteration {} '
                        'This issue happens when the NLP subproblem violates constraint qualification. '
                        'Convergence to optimal solution is not guaranteed.'.format(
                            self.mip_iter,
                            self.integer_list.index(self.curr_int_sol) + 1,
                        )
                    )
                    config.logger.info(
                        'Final bound values: Primal Bound: {}  Dual Bound: {}'.format(
                            self.primal_bound, self.dual_bound
                        )
                    )
                    # TODO determine self.primal_bound, self.dual_bound is inf or -inf.
                    self.results.solver.termination_condition = tc.feasible
                    return True
            self.integer_list.append(self.curr_int_sol)
        return False
