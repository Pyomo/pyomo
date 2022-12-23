
# -*- coding: utf-8 -*-

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
from pyomo.contrib.gdpopt.util import (time_code, lower_logger_level_to, SuppressInfeasibleWarning, get_main_elapsed_time)
from pyomo.contrib.mindtpy.util import set_up_logger,setup_results_object, add_var_bound, calc_jacobians, set_solver_options, get_integer_solution
from pyomo.core import TransformationFactory, Objective, ConstraintList, value
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_ECP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts
from pyomo.opt import TerminationCondition as tc


@SolverFactory.register(
    'mindtpy.ecp',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPy_OA_Solver(_MindtPyAlgorithm):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver 
    applies a variety of decomposition-based approaches to solve Mixed-Integer 
    Nonlinear Programming (MINLP) problems. 
    These approaches include:

    - Outer approximation (OA)
    - Global outer approximation (GOA)
    - Regularized outer approximation (ROA)
    - LP/NLP based branch-and-bound (LP/NLP)
    - Global LP/NLP based branch-and-bound (GLP/NLP)
    - Regularized LP/NLP based branch-and-bound (RLP/NLP)
    - Feasibility pump (FP)

    This solver implementation has been developed by David Bernal <https://github.com/bernalde>
    and Zedong Peng <https://github.com/ZedongPeng> as part of research efforts at the Grossmann
    Research Group (http://egon.cheme.cmu.edu/) at the Department of Chemical Engineering at 
    Carnegie Mellon University.
    """
    CONFIG = _get_MindtPy_ECP_config()

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
        config = self.config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)  # TODO: do we need to set preserve_implicit=True?
        config.set_value(kwds)
        set_up_logger(config)
        new_logging_level = logging.INFO if config.tee else None
        with lower_logger_level_to(config.logger, new_logging_level):
            self.check_config()

        self.set_up_solve_data(model, config)

        if config.integer_to_binary:
            TransformationFactory('contrib.integer_to_binary'). \
                apply_to(self.working_model)

        self.create_utility_block(self.working_model, 'MindtPy_utils')

        with time_code(self.timing, 'total', is_main_timer=True), \
                lower_logger_level_to(config.logger, new_logging_level):
            config.logger.info(
                '---------------------------------------------------------------------------------------------\n'
                '              Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo (MindtPy)               \n'
                '---------------------------------------------------------------------------------------------\n'
                'For more information, please visit https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html')

            # Validate the model to ensure that MindtPy is able to solve it.
            if not self.model_is_valid():
                return

            MindtPy = self.working_model.MindtPy_utils
            setup_results_object(self.results, self.original_model, config)
            # In the process_objective function, as long as the objective function is nonlinear, it will be reformulated and the variable/constraint/objective lists will be updated.
            # For OA/GOA/LP-NLP algorithm, if the objective funtion is linear, it will not be reformulated as epigraph constraint.
            # If the objective function is linear, it will be reformulated as epigraph constraint only if the Feasibility Pump or ROA/RLP-NLP algorithm is activated. (move_objective = True)
            # In some cases, the variable/constraint/objective lists will not be updated even if the objective is epigraph-reformulated.
            # In Feasibility Pump, since the distance calculation only includes discrete variables and the epigraph slack variables are continuous variables, the Feasibility Pump algorithm will not affected even if the variable list are updated.
            # In ROA and RLP/NLP, since the distance calculation does not include these epigraph slack variables, they should not be added to the variable list. (update_var_con_list = False)
            # In the process_objective function, once the objective function has been reformulated as epigraph constraint, the variable/constraint/objective lists will not be updated only if the MINLP has a linear objective function and regularization is activated at the same time.
            # This is because the epigraph constraint is very "flat" for branching rules. The original objective function will be used for the main problem and epigraph reformulation will be used for the projection problem.
            # TODO: The logic here is too complicated, can we simplify it?
            self.process_objective(config,
                                   move_objective=config.move_objective,
                                   use_mcpp=config.use_mcpp,
                                   update_var_con_list=True,
                                   partition_nonlinear_terms=config.partition_obj_nonlinear_terms,
                                   obj_handleable_polynomial_degree=self.mip_objective_polynomial_degree,
                                   constr_handleable_polynomial_degree=self.mip_constraint_polynomial_degree)

            # Save model initial values.
            self.initial_var_values = list(v.value for v in MindtPy.variable_list)
            self.initialize_mip_problem()

            # Initialization
            with time_code(self.timing, 'initialization'):
                self.MindtPy_initialization(config)

            # Algorithm main loop
            with time_code(self.timing, 'main loop'):
                self.MindtPy_iteration_loop(config)

            # Load solution
            if self.best_solution_found is not None:
                self.load_solution()

            # Get integral info
            self.get_integral_info()

            config.logger.info(' {:<25}:   {:>7.4f} '.format(
                'Primal-dual gap integral', self.primal_dual_gap_integral))

        # Update result
        self.update_result()

        return self.results


    def MindtPy_iteration_loop(self, config):
        """Main loop for MindtPy Algorithms.

        This is the outermost function for the algorithms in this package; this function controls the progression of
        solving the model.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            The strategy value is not correct or not included.
        """
        last_iter_cuts = False
        while self.mip_iter < config.iteration_limit:

            self.mip_subiter = 0
            # solve MILP main problem
            main_mip, main_mip_results = self.solve_main(config)
            if main_mip_results is not None:
                if not config.single_tree:
                    if main_mip_results.solver.termination_condition is tc.optimal:
                        self.handle_main_optimal(main_mip, config)
                    elif main_mip_results.solver.termination_condition is tc.infeasible:
                        self.handle_main_infeasible(main_mip, config)
                        last_iter_cuts = True
                        break
                    else:
                        self.handle_main_other_conditions(
                            main_mip, main_mip_results, config)
                    # Call the MILP post-solve callback
                    with time_code(self.timing, 'Call after main solve'):
                        config.call_after_main_solve(main_mip)
            else:
                config.logger.info('Algorithm should terminate here.')
                break

            if self.algorithm_should_terminate(config):
                last_iter_cuts = False
                break

            add_ecp_cuts(self.mip, self.jacobians, config, self.timing)

        # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
        # we correct it after the iteration.
        if (config.add_no_good_cuts or config.use_tabu_list) and not self.should_terminate:
            self.fix_dual_bound(config, last_iter_cuts)
        config.logger.info(
            ' ===============================================================================================')

    def check_config(self):
        config = self.config
        if config.init_strategy is None:
            config.init_strategy = 'max_binary'
        # if ecp tolerance is not provided use bound tolerance
        if config.ecp_tolerance is None:
            config.ecp_tolerance = config.absolute_bound_tolerance
        super().check_config()

    def initialize_mip_problem(self):
        ''' Deactivate the nonlinear constraints to create the MIP problem.
        '''
        config = self.config

        m = self.mip = self.working_model.clone()
        next(self.mip.component_data_objects(
            Objective, active=True)).deactivate()

        MindtPy = m.MindtPy_utils
        if config.calculate_dual_at_solution:
            m.dual.deactivate()

        self.jacobians = calc_jacobians(self.mip, config)  # preload jacobians
        MindtPy.cuts.ecp_cuts = ConstraintList(doc='Extended Cutting Planes')

        if config.init_strategy == 'FP':
            MindtPy.cuts.fp_orthogonality_cuts = ConstraintList(
                doc='Orthogonality cuts in feasibility pump')
            if config.fp_projcuts:
                self.working_model.MindtPy_utils.cuts.fp_orthogonality_cuts = ConstraintList(
                    doc='Orthogonality cuts in feasibility pump')


    def init_rNLP(self, config):
        """Initialize the problem by solving the relaxed NLP and then store the optimal variable
        values obtained from solving the rNLP.

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the relaxed NLP.
        """
        m = self.working_model.clone()
        config.logger.debug(
            'Relaxed NLP: Solve relaxed integrality')
        MindtPy = m.MindtPy_utils
        TransformationFactory('core.relax_integer_vars').apply_to(m)
        nlp_args = dict(config.nlp_solver_args)
        nlpopt = SolverFactory(config.nlp_solver)
        set_solver_options(nlpopt, self.timing, config, solver_type='nlp')
        with SuppressInfeasibleWarning():
            results = nlpopt.solve(m,
                                tee=config.nlp_solver_tee, 
                                load_solutions=False,
                                **nlp_args)
            if len(results.solution) > 0:
                m.solutions.load_from(results)
        subprob_terminate_cond = results.solver.termination_condition
        if subprob_terminate_cond in {tc.optimal, tc.feasible, tc.locallyOptimal}:
            main_objective = MindtPy.objective_list[-1]
            if subprob_terminate_cond == tc.optimal:
                self.update_dual_bound(value(main_objective.expr))
            else:
                config.logger.info(
                    'relaxed NLP is not solved to optimality.')
                self.update_suboptimal_dual_bound(results)
            config.logger.info(self.log_formatter.format('-', 'Relaxed NLP', value(main_objective.expr),
                                                            self.primal_bound, self.dual_bound, self.rel_gap,
                                                            get_main_elapsed_time(self.timing)))
        elif subprob_terminate_cond in {tc.infeasible, tc.noSolution}:
            # TODO fail? try something else?
            config.logger.info(
                'Initial relaxed NLP problem is infeasible. '
                'Problem may be infeasible.')
        elif subprob_terminate_cond is tc.maxTimeLimit:
            config.logger.info(
                'NLP subproblem failed to converge within time limit.')
            self.results.solver.termination_condition = tc.maxTimeLimit
        elif subprob_terminate_cond is tc.maxIterations:
            config.logger.info(
                'NLP subproblem failed to converge within iteration limit.')
        else:
            raise ValueError(
                'MindtPy unable to handle relaxed NLP termination condition '
                'of %s. Solver message: %s' %
                (subprob_terminate_cond, results.solver.message))


    def algorithm_should_terminate(self, config):
        """Checks if the algorithm should terminate at the given point.

        This function determines whether the algorithm should terminate based on the solver options and progress.
        (Sets the self.results.solver.termination_condition to the appropriate condition, i.e. optimal,
        maxIterations, maxTimeLimit).

        Parameters
        ----------
        config : ConfigBlock
            The specific configurations for MindtPy.
        check_cycling : bool
            Whether to check for a special case that causes the discrete variables to loop through the same values.

        Returns
        -------
        bool
            True if the algorithm should terminate, False otherwise.
        """
        if self.should_terminate:
            if self.primal_bound == self.primal_bound_progress[0]:
                self.results.solver.termination_condition = tc.noSolution
            else:
                self.results.solver.termination_condition = tc.feasible
            return True

        # Check bound convergence
        if self.abs_gap <= config.absolute_bound_tolerance:
            config.logger.info(
                'MindtPy exiting on bound convergence. '
                'Absolute gap: {} <= absolute tolerance: {} \n'.format(
                    self.abs_gap, config.absolute_bound_tolerance))
            self.results.solver.termination_condition = tc.optimal
            return True
        # Check relative bound convergence
        if self.best_solution_found is not None:
            if self.rel_gap <= config.relative_bound_tolerance:
                config.logger.info(
                    'MindtPy exiting on bound convergence. '
                    'Relative gap : {} <= relative tolerance: {} \n'.format(
                        self.rel_gap, config.relative_bound_tolerance))

        # Check iteration limit
        if self.mip_iter >= config.iteration_limit:
            config.logger.info(
                'MindtPy unable to converge bounds '
                'after {} main iterations.'.format(self.mip_iter))
            config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                format(self.primal_bound, self.dual_bound))
            if config.single_tree:
                self.results.solver.termination_condition = tc.feasible
            else:
                self.results.solver.termination_condition = tc.maxIterations
            return True

        # Check time limit
        if get_main_elapsed_time(self.timing) >= config.time_limit:
            config.logger.info(
                'MindtPy unable to converge bounds '
                'before time limit of {} seconds. '
                'Elapsed: {} seconds'
                .format(config.time_limit, get_main_elapsed_time(self.timing)))
            config.logger.info(
                'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                format(self.primal_bound, self.dual_bound))
            self.results.solver.termination_condition = tc.maxTimeLimit
            return True

        # Check if algorithm is stalling
        if len(self.primal_bound_progress) >= config.stalling_limit:
            if abs(self.primal_bound_progress[-1] - self.primal_bound_progress[-config.stalling_limit]) <= config.zero_tolerance:
                config.logger.info(
                    'Algorithm is not making enough progress. '
                    'Exiting iteration loop.')
                config.logger.info(
                    'Final bound values: Primal Bound: {}  Dual Bound: {}'.
                    format(self.primal_bound, self.dual_bound))
                if self.best_solution_found is not None:
                    self.results.solver.termination_condition = tc.feasible
                else:
                    # TODO: Is it correct to set self.working_model as the best_solution_found?
                    # In function copy_var_list_values, skip_fixed is set to True in default.
                    self.best_solution_found = self.working_model.clone()
                    config.logger.warning(
                        'Algorithm did not find a feasible solution. '
                        'Returning best bound solution. Consider increasing stalling_limit or absolute_bound_tolerance.')
                    self.results.solver.termination_condition = tc.noSolution
                return True

        # check to see if the nonlinear constraints are satisfied
        MindtPy = self.working_model.MindtPy_utils
        nonlinear_constraints = [c for c in MindtPy.nonlinear_constraint_list]
        for nlc in nonlinear_constraints:
            if nlc.has_lb():
                try:
                    lower_slack = nlc.lslack()
                except (ValueError, OverflowError):
                    # Set lower_slack (upper_slack below) less than -config.ecp_tolerance in this case.
                    lower_slack = -10*config.ecp_tolerance
                if lower_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc))
                    return False
            if nlc.has_ub():
                try:
                    upper_slack = nlc.uslack()
                except (ValueError, OverflowError):
                    upper_slack = -10*config.ecp_tolerance
                if upper_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc))
                    return False
        # For ECP to know whether to know which bound to copy over (primal or dual)
        self.primal_bound = self.dual_bound
        config.logger.info(
            'MindtPy-ECP exiting on nonlinear constraints satisfaction. '
            'Primal Bound: {} Dual Bound: {}\n'.format(self.primal_bound, self.dual_bound))

        self.best_solution_found = self.working_model.clone()
        self.results.solver.termination_condition = tc.optimal
        return True
