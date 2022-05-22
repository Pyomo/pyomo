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

from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigValue
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs, _add_nlp_solver_configs, _add_tolerance_configs,
    _add_OA_configs, _add_BB_configs)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_util_block, add_disjunct_list, add_boolean_variable_lists,
    add_algebraic_variable_list)
from pyomo.contrib.gdpopt.util import (
    get_main_elapsed_time, solve_continuous_problem, time_code)
from pyomo.core.base import Objective, value, minimize, maximize
from pyomo.core.staleflag import StaleFlagManager
from pyomo.opt import SolverResults
from pyomo.opt import TerminationCondition as tc
from pyomo.util.model_size import build_model_size_report

class _GDPoptAlgorithm():
    def __init__(self, parent, kwds=None):
        """
        This is a common init method for all the GDPopt algorithms, so that we
        correctly set up the config arguments based on what the 'parent' class
        (GDPoptSolver) already knows. It has parsed the common arguments, such
        as time limit and iteration limit, but the config block on this class
        is about to become the ground truth for as long as the algorithm is the
        same.

        Then, we initialize the generic parts of the algorithm state.
        """
        # Transfer the parent config info: we create it if it is not there, and
        # overwrite the values if it is already there. The parent defers to what
        # is in this class during solve. We bypass the parent config's getter
        # because that's an infinite loop: we know it doesn't have an impl,
        # we're *setting* it's impl.
        for kwd, val in parent._CONFIG.items():
            if kwd not in self.CONFIG:
                self.CONFIG.declare(kwd, ConfigValue(default=val))
            else:
                self.CONFIG[kwd] = val
        # kwds might be a superset of what we saw in the previous loop if there
        # are algorithm-specific options that the parent ignored in it's pass
        # through what was specified, so we go through again to find them.
        if kwds is not None:
            for kwd, val in kwds.items():
                if kwd not in self.CONFIG and kwd in parent._CONFIG:
                    self.CONFIG.declare(kwd, ConfigValue(default=val))
                else:
                    self.CONFIG[kwd] = val

        # Store the solver version since it will get put on the results object
        # later.
        self.version = parent.version()

        # We store bounds, timing info, iteration count, incumbent, and the
        # expression of the original (possibly nonlinear) objective function.
        self.LB = float('-inf')
        self.UB = float('inf')
        self.unbounded = False
        self.timing = Bunch()
        self.iteration = 0

        self.incumbent_boolean_soln = None
        self.incumbent_continuous_soln = None

        self.original_obj = None

        self.log_formatter = ('{:>9}   {:>15}   {:>11.5f}   {:>11.5f}   '
                              '{:>8.2%}   {:>7.2f}  {}')

    def _solve_gdp(self, original_model, config):
        # To be implemented by the algorithms
        pass

    def _log_header(self, logger):
        logger.info(
            '================================================================='
            '============================')
        logger.info(
            '{:^9} | {:^15} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration', 'Subproblem Type', 'Lower Bound', 'Upper Bound',
                ' Gap ', 'Time(s)'))

    def _gather_problem_info_and_solve_non_gdps(self, model, config):
        """Solve the model.

        Args:
            model (Block): a Pyomo model or block to be solved

        """
        # set up the logger so that we will have a pretty log printed
        logger = config.logger

        self.pyomo_results = self._get_pyomo_results_object_with_problem_info(
            model, config)
        self.objective_sense = self.pyomo_results.problem.sense

        # Check if this problem actually has any discrete decisions. If not,
        # just solve it.
        problem = self.pyomo_results.problem
        if (problem.number_of_binary_variables == 0 and
            problem.number_of_integer_variables == 0 and
            problem.number_of_disjunctions == 0):
            cont_results = solve_continuous_problem(model, config)
            problem.lower_bound = cont_results.problem.lower_bound
            problem.upper_bound = cont_results.problem.upper_bound
            # Just put the info from the MIP solver
            self.pyomo_results.solver = cont_results.solver

            return self.pyomo_results

        # This class assumes that we have a util_block with an algebraic
        # variable list and a boolean variable list, so that we can transfer the
        # solution onto the original model. Everything else will be up to the
        # algorithm, but we make sure we have those here.

        # Make a block where we will store some component lists so that after we
        # clone we know who's who
        util_block = self.original_util_block = add_util_block(model)
        # Needed for finding indicator_vars mainly
        add_disjunct_list(util_block)
        add_boolean_variable_lists(util_block)
        # To transfer solutions between cloned models
        add_algebraic_variable_list(util_block)

    def _call_main_loop(self, model, config):
        with time_code(self.timing, 'total', is_main_timer=True):
            results = self._gather_problem_info_and_solve_non_gdps(model,
                                                                   config)
            # If it wasn't disjunctive, we solved it
            if results:
                return results
            else:
                # main loop implemented by each algorithm
                self._solve_gdp(model, config)

        self._get_final_pyomo_results_object()
        self._log_termination_message(config.logger)
        if (self.pyomo_results.solver.termination_condition not in
            {tc.infeasible, tc.unbounded}):
            self._transfer_incumbent_to_original_model(config.logger)
        self._delete_original_model_util_block()
        return self.pyomo_results

    def _update_bounds_after_solve(self, subprob_nm, primal=None, dual=None,
                                   logger=None):
        primal_improved = self._update_bounds(primal, dual)
        if logger is not None:
            self._log_current_state(logger, subprob_nm, primal_improved)

        return primal_improved

    def _update_bounds(self, primal=None, dual=None, force_update=False):
        """
        Update bounds correctly depending on objective sense.

        primal: bound from solving subproblem with fixed master solution
        dual: bound from solving master problem (relaxation of original problem)
        force_update: flag so this function will set the bound even if it's
                      not an improvement. (Used at termination if the bounds
                      cross.)
        """
        oldLB = self.LB
        oldUB = self.UB
        primal_bound_improved = False

        if self.objective_sense is minimize:
            if primal is not None and (primal < oldUB or force_update):
                self.UB = primal
                primal_bound_improved = primal < oldUB
            if dual is not None and (dual > oldLB or force_update):
                self.LB = dual
        else:
            if primal is not None and (primal > oldLB or force_update):
                self.LB = primal
                primal_bound_improved = primal > oldLB
            if dual is not None and (dual < oldUB or force_update):
                self.UB = dual

        return primal_bound_improved

    def relative_gap(self):
        """Returns current relative optimality gap.

        Note that this gap is not necessarily monotonically decreasing if at
        some point the primal bound changes signs.
        """
        absolute_gap = abs(self.UB - self.LB)
        return absolute_gap/(abs(self.primal_bound() + 1e-10))

    def _log_current_state(self, logger, subproblem_type,
                           primal_improved=False):
        star = "*" if primal_improved else ""
        logger.info(self.log_formatter.format(
            self.iteration, subproblem_type, self.LB,
            self.UB, self.relative_gap(),
            get_main_elapsed_time(self.timing), star))

    def _log_termination_message(self, logger):
        logger.info(
            '\nSolved in {} iterations and {:.5f} seconds\n'
            'Optimal objective value {:.10f}\n'
            'Relative optimality gap {:.5%}'.format(
                self.iteration, get_main_elapsed_time(self.timing),
                self.primal_bound(), self.relative_gap()))

    def primal_bound(self):
        if self.objective_sense is minimize:
            return self.UB
        else:
            return self.LB

    def update_incumbent(self, util_block):
        self.incumbent_continuous_soln = [v.value for v in
                                          util_block.algebraic_variable_list]
        self.incumbent_boolean_soln = [
            v.value for v in util_block.transformed_boolean_variable_list]

    def _update_bounds_after_master_problem_solve(self, mip_termination,
                                                  obj_expr, logger):
        if mip_termination is tc.optimal:
            self._update_bounds_after_solve('master', dual=value(obj_expr),
                                            logger=logger)
        elif mip_termination is tc.infeasible:
            # Master problem was infeasible.
            self._update_dual_bound_to_infeasible()
        elif mip_termination is tc.feasible or tc.unbounded:
            # we won't update the bound, because we didn't solve to
            # optimality. (And in the unbounded case, we wouldn't be here if we
            # didn't find a solution, so we're going to keep going, but we don't
            # have any info in terms of a dual bound.)
            pass
        else:
            raise DeveloperError("Unrecognized termination condition %s when "
                                 "updating the dual bound." % mip_termination)

    def _update_dual_bound_to_infeasible(self):
        # set optimistic bound to infinity
        if self.objective_sense == minimize:
            self._update_bounds(dual=float('inf'))
        else:
            self._update_bounds(dual=float('-inf'))

    def _update_primal_bound_to_unbounded(self):
        if self.objective_sense == minimize:
            self._update_bounds(primal=float('-inf'))
        else:
            self._update_bounds(primal=float('inf'))
        self.unbounded = True

    def bounds_converged(self, config):
        if self.unbounded:
            config.logger.info('GDPopt exiting--GDP is unbounded.')
            self.pyomo_results.solver.termination_condition = tc.unbounded
            return True
        elif self.LB + config.bound_tolerance >= self.UB:
            if self.LB == float('inf') and self.UB == float('inf'):
                config.logger.info('GDPopt exiting--problem is infeasible.')
                self.pyomo_results.solver.termination_condition = tc.infeasible
            elif self.LB == float('-inf') and self.UB == float('-inf'):
                config.logger.info('GDPopt exiting--problem is infeasible.')
                self.pyomo_results.solver.termination_condition = tc.infeasible
            else:
                # if they've crossed, then the gap is actually 0: Update the
                # dual (master problem) bound to be equal to the primal
                # (subproblem) bound
                if self.LB + config.bound_tolerance > self.UB:
                    self._update_bounds(dual=self.primal_bound(),
                                        force_update=True)
                self._log_current_state(config.logger, '')
                config.logger.info(
                    'GDPopt exiting--bounds have converged or crossed.')
                self.pyomo_results.solver.termination_condition = tc.optimal

            return True
        return False

    def reached_iteration_limit(self, config):
        if self.iteration >= config.iterlim:
            config.logger.info(
                'GDPopt unable to converge bounds within iteration limit of '
                '{} iterations.'.format(config.iterlim))
            self.pyomo_results.solver.termination_condition = tc.maxIterations
            return True
        return False

    def reached_time_limit(self, config):
        elapsed = get_main_elapsed_time(self.timing)
        if elapsed >= config.time_limit:
            config.logger.info(
                'GDPopt exiting--Did not converge bounds '
                'before time limit of {} seconds. '.format(config.time_limit))
            self.pyomo_results.solver.termination_condition = tc.maxTimeLimit
            return True
        return False

    def any_termination_criterion_met(self, config):
        return (self.bounds_converged(config) or
                self.reached_iteration_limit(config) or
                self.reached_time_limit(config))

    def _get_pyomo_results_object_with_problem_info(self, original_model,
                                                    config):
        """
        Initialize a results object with results.problem information
        """
        results = SolverResults()

        results.solver.name = 'GDPopt %s - %s' % (self.version,
                                                  config.algorithm)

        prob = results.problem
        prob.name = original_model.name
        prob.number_of_nonzeros = None  # TODO

        num_of = build_model_size_report(original_model)

        # Get count of constraints and variables
        prob.number_of_constraints = num_of.activated.constraints
        prob.number_of_disjunctions = num_of.activated.disjunctions
        prob.number_of_variables = num_of.activated.variables
        prob.number_of_binary_variables = num_of.activated.binary_variables
        prob.number_of_continuous_variables = num_of.activated.\
                                              continuous_variables
        prob.number_of_integer_variables = num_of.activated.integer_variables

        config.logger.info(
            "Original model has %s constraints (%s nonlinear) "
            "and %s disjunctions, "
            "with %s variables, of which %s are binary, %s are integer, "
            "and %s are continuous." %
            (num_of.activated.constraints,
             num_of.activated.nonlinear_constraints,
             num_of.activated.disjunctions,
             num_of.activated.variables,
             num_of.activated.binary_variables,
             num_of.activated.integer_variables,
             num_of.activated.continuous_variables))

        # Handle missing or multiple objectives, and get sense
        active_objectives = list(original_model.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        number_of_objectives = len(active_objectives)
        if number_of_objectives == 0:
            config.logger.warning(
                'Model has no active objectives. Adding dummy objective.')
            main_obj = Objective(expr=1)
            original_model.add_component(unique_component_name(original_model,
                                                               'dummy_obj'),
                                         main_obj)
        elif number_of_objectives > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = active_objectives[0]
        prob.sense = minimize if main_obj.sense == 1 else maximize

        return results

    def _transfer_incumbent_to_original_model(self, logger):
        StaleFlagManager.mark_all_as_stale(delayed=False)
        if self.incumbent_boolean_soln is None:
            assert self.incumbent_continuous_soln is None
            # we don't have a solution to transfer
            logger.info("No feasible solutions found.")
            return
        for var, soln in zip(self.original_util_block.algebraic_variable_list,
                             self.incumbent_continuous_soln):
            var.set_value(soln, skip_validation=True)
        for var, soln in zip(
                self.original_util_block.boolean_variable_list,
                self.incumbent_boolean_soln):
            # If there is an associated binary, make sure it's in sync too
            original_binary = var.get_associated_binary()
            if soln is None:
                var.set_value(soln, skip_validation=True)
                if original_binary is not None:
                    original_binary.set_value(None, skip_validation=True)
                continue
            elif soln > 0.5:
                var.set_value(True)
            else:
                var.set_value(False)

            if original_binary is not None:
                original_binary.set_value(soln)
        StaleFlagManager.mark_all_as_stale(delayed=True)

    def _delete_original_model_util_block(self):
        """For cleaning up after a solve--we want the original model to be
        untouched except for the solution being loaded"""
        blk = self.original_util_block
        m = blk.model()
        m.del_component(blk)
        # We just deleted the linearized objective if we had one, so restore the
        # prior one.
        if self.original_obj is not None:
            self.original_obj.activate()

    def _get_final_pyomo_results_object(self):
        """
        Fill in the results.solver information onto the results object
        """
        results = self.pyomo_results
        # Finalize results object
        results.problem.lower_bound = self.LB
        results.problem.upper_bound = self.UB
        results.solver.iterations = self.iteration
        results.solver.timing = self.timing
        results.solver.user_time = self.timing.total
        results.solver.wallclock_time = self.timing.total

        return results
