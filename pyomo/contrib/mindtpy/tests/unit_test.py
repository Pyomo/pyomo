#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Unit tests for the MindtPy solver."""
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import \
    EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.environ import SolverFactory, maximize
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QCP_simple import QCP_simple
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_config, check_config
from pyomo.contrib.mindtpy.util import get_primal_integral, get_dual_integral, set_up_solve_data, add_feas_slacks, set_solver_options
from pyomo.contrib.mindtpy.nlp_solve import handle_subproblem_other_termination, handle_feasibility_subproblem_tc, solve_subproblem, handle_nlp_subproblem_tc
from pyomo.core.base import TransformationFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.gdpopt.util import create_utility_block, time_code, process_objective, setup_results_object
from pyomo.contrib.mindtpy.initialization import MindtPy_initialize_main, init_rNLP
from pyomo.contrib.mindtpy.feasibility_pump import generate_norm_constraint, handle_fp_main_tc
from pyomo.core import Block, ConstraintList
from pyomo.contrib.mindtpy.mip_solve import solve_main, handle_main_other_conditions
from pyomo.opt import SolutionStatus, SolverStatus
from pyomo.core import (Constraint, Objective,
                        TransformationFactory, minimize, Var, RangeSet, NonNegativeReals)
from pyomo.contrib.mindtpy.iterate import algorithm_should_terminate

nonconvex_model_list = [EightProcessFlowsheet(convex=False)]

LP_model = LP_unbounded()
LP_model._generate_model()

QCP_model = QCP_simple()
QCP_model._generate_model()
extreme_model_list = [LP_model.model, QCP_model.model]

required_solvers = ('ipopt', 'glpk')
# required_solvers = ('gams', 'gams')
if all(SolverFactory(s).available() for s in required_solvers):
    subsolvers_available = True
else:
    subsolvers_available = False


@unittest.skipIf(not subsolvers_available,
                 'Required subsolvers %s are not available'
                 % (required_solvers,))
class TestMindtPy(unittest.TestCase):
    """Tests for the MindtPy solver plugin."""

    def test_handle_termination_condition(self):
        """Test the outer approximation decomposition algorithm."""
        model = SimpleMINLP()
        config = _get_MindtPy_config()
        solve_data = set_up_solve_data(model, config)
        with time_code(solve_data.timing, 'total', is_main_timer=True), \
                create_utility_block(solve_data.working_model, 'MindtPy_utils', solve_data):

            MindtPy = solve_data.working_model.MindtPy_utils

            MindtPy = solve_data.working_model.MindtPy_utils
            setup_results_object(solve_data, config)
            process_objective(solve_data, config,
                              move_linear_objective=(config.init_strategy == 'FP'
                                                     or config.add_regularization is not None),
                              use_mcpp=config.use_mcpp,
                              update_var_con_list=config.add_regularization is None
                              )
            feas = MindtPy.feas_opt = Block()
            feas.deactivate()
            feas.feas_constraints = ConstraintList(
                doc='Feasibility Problem Constraints')

            lin = MindtPy.cuts = Block()
            lin.deactivate()

            if config.feasibility_norm == 'L1' or config.feasibility_norm == 'L2':
                feas.nl_constraint_set = RangeSet(len(MindtPy.nonlinear_constraint_list),
                                                  doc='Integer index set over the nonlinear constraints.')
                # Create slack variables for feasibility problem
                feas.slack_var = Var(feas.nl_constraint_set,
                                     domain=NonNegativeReals, initialize=1)
            else:
                feas.slack_var = Var(domain=NonNegativeReals, initialize=1)

            # no-good cuts exclude particular discrete decisions
            lin.no_good_cuts = ConstraintList(doc='no-good cuts')

            fixed_nlp = solve_data.working_model.clone()
            TransformationFactory('core.fix_integer_vars').apply_to(fixed_nlp)

            MindtPy_initialize_main(solve_data, config)

            # test handle_subproblem_other_termination
            termination_condition = tc.maxIterations
            config.add_no_good_cuts = True
            handle_subproblem_other_termination(fixed_nlp, termination_condition,
                                                solve_data, config)
            self.assertEqual(
                len(solve_data.mip.MindtPy_utils.cuts.no_good_cuts), 1)

            # test handle_main_other_conditions
            main_mip, main_mip_results = solve_main(solve_data, config)
            main_mip_results.solver.termination_condition = tc.infeasible
            handle_main_other_conditions(
                solve_data.mip, main_mip_results, solve_data, config)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.feasible)

            main_mip_results.solver.termination_condition = tc.unbounded
            handle_main_other_conditions(
                solve_data.mip, main_mip_results, solve_data, config)
            self.assertIn(main_mip.MindtPy_utils.objective_bound,
                          main_mip.component_data_objects(ctype=Constraint))

            main_mip.MindtPy_utils.del_component('objective_bound')
            main_mip_results.solver.termination_condition = tc.infeasibleOrUnbounded
            handle_main_other_conditions(
                solve_data.mip, main_mip_results, solve_data, config)
            self.assertIn(main_mip.MindtPy_utils.objective_bound,
                          main_mip.component_data_objects(ctype=Constraint))

            main_mip_results.solver.termination_condition = tc.maxTimeLimit
            handle_main_other_conditions(
                solve_data.mip, main_mip_results, solve_data, config)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.maxTimeLimit)

            main_mip_results.solver.termination_condition = tc.other
            main_mip_results.solution.status = SolutionStatus.feasible
            handle_main_other_conditions(
                solve_data.mip, main_mip_results, solve_data, config)
            for v1, v2 in zip(main_mip.MindtPy_utils.variable_list, solve_data.working_model.MindtPy_utils.variable_list):
                self.assertEqual(v1.value, v2.value)

            # test handle_feasibility_subproblem_tc
            feas_subproblem = solve_data.working_model.clone()
            add_feas_slacks(feas_subproblem, config)
            MindtPy = feas_subproblem.MindtPy_utils
            MindtPy.feas_opt.activate()
            if config.feasibility_norm == 'L1':
                MindtPy.feas_obj = Objective(
                    expr=sum(s for s in MindtPy.feas_opt.slack_var[...]),
                    sense=minimize)
            elif config.feasibility_norm == 'L2':
                MindtPy.feas_obj = Objective(
                    expr=sum(s*s for s in MindtPy.feas_opt.slack_var[...]),
                    sense=minimize)
            else:
                MindtPy.feas_obj = Objective(
                    expr=MindtPy.feas_opt.slack_var,
                    sense=minimize)

            handle_feasibility_subproblem_tc(
                tc.optimal, MindtPy, solve_data, config)
            handle_feasibility_subproblem_tc(
                tc.infeasible, MindtPy, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(solve_data.results.solver.status, SolverStatus.error)

            solve_data.should_terminate = False
            solve_data.results.solver.status = None
            handle_feasibility_subproblem_tc(
                tc.maxIterations, MindtPy, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(solve_data.results.solver.status, SolverStatus.error)

            solve_data.should_terminate = False
            solve_data.results.solver.status = None
            handle_feasibility_subproblem_tc(
                tc.solverFailure, MindtPy, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(solve_data.results.solver.status, SolverStatus.error)

            # test NLP subproblem infeasible
            solve_data.working_model.Y[1].value = 0
            solve_data.working_model.Y[2].value = 0
            solve_data.working_model.Y[3].value = 0
            fixed_nlp, fixed_nlp_results = solve_subproblem(solve_data, config)
            solve_data.working_model.Y[1].value = None
            solve_data.working_model.Y[2].value = None
            solve_data.working_model.Y[3].value = None

            # test handle_nlp_subproblem_tc
            fixed_nlp_results.solver.termination_condition = tc.maxTimeLimit
            handle_nlp_subproblem_tc(
                fixed_nlp, fixed_nlp_results, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.maxTimeLimit)

            fixed_nlp_results.solver.termination_condition = tc.maxEvaluations
            handle_nlp_subproblem_tc(
                fixed_nlp, fixed_nlp_results, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.maxEvaluations)

            fixed_nlp_results.solver.termination_condition = tc.maxIterations
            handle_nlp_subproblem_tc(
                fixed_nlp, fixed_nlp_results, solve_data, config)
            self.assertIs(solve_data.should_terminate, True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.maxEvaluations)

            # test handle_fp_main_tc
            config.init_strategy = 'FP'
            solve_data.fp_iter = 1
            init_rNLP(solve_data, config)
            feas_main, feas_main_results = solve_main(
                solve_data, config, fp=True)
            feas_main_results.solver.termination_condition = tc.optimal
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, False)

            feas_main_results.solver.termination_condition = tc.maxTimeLimit
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.maxTimeLimit)

            feas_main_results.solver.termination_condition = tc.infeasible
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, True)

            feas_main_results.solver.termination_condition = tc.unbounded
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, True)

            feas_main_results.solver.termination_condition = tc.other
            feas_main_results.solution.status = SolutionStatus.feasible
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, False)

            feas_main_results.solver.termination_condition = tc.solverFailure
            fp_should_terminate = handle_fp_main_tc(
                feas_main_results, solve_data, config)
            self.assertIs(fp_should_terminate, True)

            # test generate_norm_constraint
            fp_nlp = solve_data.working_model.clone()
            config.fp_main_norm = 'L1'
            generate_norm_constraint(fp_nlp, solve_data, config)
            self.assertIsNotNone(fp_nlp.MindtPy_utils.find_component(
                'L1_norm_constraint'))

            config.fp_main_norm = 'L2'
            generate_norm_constraint(fp_nlp, solve_data, config)
            self.assertIsNotNone(fp_nlp.find_component('norm_constraint'))

            fp_nlp.del_component('norm_constraint')
            config.fp_main_norm = 'L_infinity'
            generate_norm_constraint(fp_nlp, solve_data, config)
            self.assertIsNotNone(fp_nlp.find_component('norm_constraint'))

            # test set_solver_options
            config.mip_solver = 'gams'
            config.threads = 1
            opt = SolverFactory(config.mip_solver)
            set_solver_options(opt, solve_data, config,
                               'mip', regularization=False)

            config.mip_solver = 'gurobi'
            config.mip_regularization_solver = 'gurobi'
            config.regularization_mip_threads = 1
            opt = SolverFactory(config.mip_solver)
            set_solver_options(opt, solve_data, config,
                               'mip', regularization=True)

            config.nlp_solver = 'gams'
            config.nlp_solver_args['solver'] = 'ipopt'
            set_solver_options(opt, solve_data, config,
                               'nlp', regularization=False)

            config.nlp_solver_args['solver'] = 'ipopth'
            set_solver_options(opt, solve_data, config,
                               'nlp', regularization=False)

            config.nlp_solver_args['solver'] = 'conopt'
            set_solver_options(opt, solve_data, config,
                               'nlp', regularization=False)

            config.nlp_solver_args['solver'] = 'msnlp'
            set_solver_options(opt, solve_data, config,
                               'nlp', regularization=False)

            config.nlp_solver_args['solver'] = 'baron'
            set_solver_options(opt, solve_data, config,
                               'nlp', regularization=False)

            # test algorithm_should_terminate
            solve_data.should_terminate = True
            solve_data.primal_bound = float('inf')
            self.assertIs(algorithm_should_terminate(
                solve_data, config, check_cycling=False), True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.noSolution)

            solve_data.primal_bound = 100
            self.assertIs(algorithm_should_terminate(
                solve_data, config, check_cycling=False), True)
            self.assertIs(
                solve_data.results.solver.termination_condition, tc.feasible)

            solve_data.primal_bound_progress = [float('inf'), 5, 4, 3, 2, 1]
            solve_data.primal_bound_progress_time = [1, 2, 3, 4, 5, 6]
            solve_data.primal_bound = 1
            self.assertEqual(get_primal_integral(solve_data, config), 14.5)

            solve_data.dual_bound_progress = [float('-inf'), 1, 2, 3, 4, 5]
            solve_data.dual_bound_progress_time = [1, 2, 3, 4, 5, 6]
            solve_data.dual_bound = 5
            self.assertEqual(get_dual_integral(solve_data, config), 14.1)

            # test check_config
            config.add_regularization = 'level_L1'
            config.regularization_mip_threads = 0
            config.threads = 8
            check_config(config)
            self.assertEqual(config.regularization_mip_threads, 8)

            config.mip_solver = 'cplex'
            config.single_tree = True
            check_config(config)
            self.assertEqual(config.mip_solver, 'cplex_persistent')
            self.assertEqual(config.threads, 1)

            config.add_slack = True
            config.max_slack == 0.0
            check_config(config)
            self.assertEqual(config.add_slack, False)

            config.strategy = 'GOA'
            config.add_slack = True
            config.use_mcpp = False
            config.equality_relaxation = True
            config.use_fbbt = False
            config.add_no_good_cuts = False
            config.use_tabu_list = False
            check_config(config)
            self.assertTrue(config.use_mcpp)
            self.assertTrue(config.use_fbbt)
            self.assertFalse(config.add_slack)
            self.assertFalse(config.equality_relaxation)
            self.assertTrue(config.add_no_good_cuts)
            self.assertFalse(config.use_tabu_list)
            
            config.single_tree = False
            config.strategy = 'FP'
            config.init_strategy = 'rNLP'
            config.iteration_limit = 100
            config.add_no_good_cuts = False
            config.use_tabu_list = True
            check_config(config)
            self.assertIs(config.init_strategy, 'FP')
            self.assertEqual(config.iteration_limit, 0)
            self.assertEqual(config.add_no_good_cuts, True)
            self.assertEqual(config.use_tabu_list, False)

if __name__ == '__main__':
    unittest.main()
