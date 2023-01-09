
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
import math
from pyomo.contrib.gdpopt.util import time_code, lower_logger_level_to, get_main_elapsed_time
from pyomo.contrib.mindtpy.util import set_up_logger, setup_results_object, get_integer_solution, copy_var_list_values_from_solution_pool, add_var_bound, add_feas_slacks
from pyomo.core import TransformationFactory, minimize, maximize, Objective, ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_GOA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from operator import itemgetter
from pyomo.contrib.mindtpy.cut_generation import add_affine_cuts


@SolverFactory.register(
    'mindtpy.goa',
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
    CONFIG = _get_MindtPy_GOA_config()


    def MindtPy_iteration_loop(self, config):
        """Main loop for MindtPy Algorithms.

        This is the outermost function for the Global Outer Approximation algorithm in this package; this function controls the progression of
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

            if self.algorithm_should_terminate(config, check_cycling=True):
                last_iter_cuts = False
                break

            if not config.single_tree:  # if we don't use lazy callback, i.e. LP_NLP
                # Solve NLP subproblem
                # The constraint linearization happens in the handlers
                if not config.solution_pool:
                    fixed_nlp, fixed_nlp_result = self.solve_subproblem(config)
                    self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, config)

                    # Call the NLP post-solve callback
                    with time_code(self.timing, 'Call after subproblem solve'):
                        config.call_after_subproblem_solve(fixed_nlp)

                    if self.algorithm_should_terminate(config, check_cycling=False):
                        last_iter_cuts = True
                        break
                else:
                    if config.mip_solver == 'cplex_persistent':
                        solution_pool_names = main_mip_results._solver_model.solution.pool.get_names()
                    elif config.mip_solver == 'gurobi_persistent':
                        solution_pool_names = list(
                            range(main_mip_results._solver_model.SolCount))
                    # list to store the name and objective value of the solutions in the solution pool
                    solution_name_obj = []
                    for name in solution_pool_names:
                        if config.mip_solver == 'cplex_persistent':
                            obj = main_mip_results._solver_model.solution.pool.get_objective_value(
                                name)
                        elif config.mip_solver == 'gurobi_persistent':
                            main_mip_results._solver_model.setParam(
                                gurobipy.GRB.Param.SolutionNumber, name)
                            obj = main_mip_results._solver_model.PoolObjVal
                        solution_name_obj.append([name, obj])
                    solution_name_obj.sort(
                        key=itemgetter(1), reverse=self.objective_sense == maximize)
                    counter = 0
                    for name, _ in solution_name_obj:
                        # the optimal solution of the main problem has been added to integer_list above
                        # so we should skip checking cycling for the first solution in the solution pool
                        if counter >= 1:
                            copy_var_list_values_from_solution_pool(self.mip.MindtPy_utils.variable_list,
                                                                    self.fixed_nlp.MindtPy_utils.variable_list,
                                                                    config, solver_model=main_mip_results._solver_model,
                                                                    var_map=main_mip_results._pyomo_var_to_solver_var_map,
                                                                    solution_name=name)
                            self.curr_int_sol = get_integer_solution(
                                self.working_model)
                            if self.curr_int_sol in set(self.integer_list):
                                config.logger.info(
                                    'The same combination has been explored and will be skipped here.')
                                continue
                            else:
                                self.integer_list.append(self.curr_int_sol)
                        counter += 1
                        fixed_nlp, fixed_nlp_result = self.solve_subproblem(config)
                        self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result, config)

                        # Call the NLP post-solve callback
                        with time_code(self.timing, 'Call after subproblem solve'):
                            config.call_after_subproblem_solve(fixed_nlp)

                        if self.algorithm_should_terminate(config, check_cycling=False):
                            last_iter_cuts = True
                            break

                        if counter >= config.num_solution_iteration:
                            break

        # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
        # we correct it after the iteration.
        if (config.add_no_good_cuts or config.use_tabu_list) and config.strategy != 'FP' and not self.should_terminate and config.add_regularization is None:
            self.fix_dual_bound(config, last_iter_cuts)
        config.logger.info(
            ' ===============================================================================================')


    def check_config(self):
        config = self.config
        config.add_slack = False
        config.use_mcpp = True
        config.equality_relaxation = False
        config.use_fbbt = True
        # add_no_good_cuts is Ture by default in GOA
        if not config.add_no_good_cuts and not config.use_tabu_list:
            config.add_no_good_cuts = True
            config.use_tabu_list = False
        # Set default initialization_strategy
        if config.single_tree:
            config.logger.info('Single-tree implementation is activated.')
            config.iteration_limit = 1
            config.add_slack = False
            if config.mip_solver not in {'cplex_persistent', 'gurobi_persistent'}:
                raise ValueError("Only cplex_persistent and gurobi_persistent are supported for LP/NLP based Branch and Bound method."
                                "Please refer to https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html#lp-nlp-based-branch-and-bound.")
            if config.threads > 1:
                config.threads = 1
                config.logger.info(
                    'The threads parameter is corrected to 1 since lazy constraint callback conflicts with multi-threads mode.')

        super().check_config()


    def initialize_mip_problem(self):
        ''' Deactivate the nonlinear constraints to create the MIP problem.
        '''
        super().initialize_mip_problem()
        self.mip.MindtPy_utils.cuts.aff_cuts = ConstraintList(doc='Affine cuts')


    def update_primal_bound(self, bound_value):
        """Update the primal bound.

        Call after solve fixed NLP subproblem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the primal bound.
        """
        super().update_primal_bound(bound_value)
        self.primal_bound_progress_time.append(get_main_elapsed_time(self.timing))
        if self.primal_bound_improved:
            self.num_no_good_cuts_added.update(
                    {self.primal_bound: len(self.mip.MindtPy_utils.cuts.no_good_cuts)})


    def add_cuts(self,
                 dual_values=None,
                 linearize_active=True,
                 linearize_violated=True,
                 cb_opt=None):
        add_affine_cuts(self.mip, self.config, self.timing)


    def deactivate_no_good_cuts_when_fixing_bound(self, no_good_cuts):
        try:
            valid_no_good_cuts_num = self.num_no_good_cuts_added[self.primal_bound]
            if self.config.add_no_good_cuts:
                for i in range(valid_no_good_cuts_num+1, len(no_good_cuts)+1):
                    no_good_cuts[i].deactivate()
            if self.config.use_tabu_list:
                self.integer_list = self.integer_list[:valid_no_good_cuts_num]
        except KeyError:
            self.config.logger.info('No-good cut deactivate failed.')
