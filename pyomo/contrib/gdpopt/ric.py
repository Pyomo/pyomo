#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.opt.base import SolverFactory
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs, _add_nlp_solver_configs, _add_tolerance_configs,
    _add_OA_configs)
from pyomo.contrib.gdpopt.oa_algorithm_utils import _get_master_and_subproblem
from pyomo.contrib.gdpopt.util import (
    time_code, lower_logger_level_to, fix_master_solution_in_subproblem,
    move_nonlinear_objective_to_constraints)
from pyomo.core import Objective, Expression, value
from pyomo.opt import TerminationCondition
from pyomo.contrib.gdpopt.mip_solve import solve_linear_GDP
from pyomo.contrib.gdpopt.nlp_solve import solve_subproblem
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
import logging

# TODO: If we have a direct interface to cplex or gurobi, we should get the
# integer solutions several-at-a-time with a solution pool or something.

@SolverFactory.register(
    '_relaxation_with_integer_cuts',
    doc='GDP Relaxation with Integer Cuts (RIC) solver')
class GDP_RIC_Solver(_GDPoptAlgorithm):
    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_OA_configs(CONFIG)

    def __init__(self, **kwds):
        self.CONFIG = self.CONFIG(kwds)
        super(GDP_RIC_Solver, self).__init__()

    def solve(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)
        
        super().solve(model, config)
        min_logging_level = logging.INFO if config.tee else None
        with time_code(self.timing, 'total', is_main_timer=True), \
            lower_logger_level_to(config.logger, min_logging_level):
            return self._solve_gdp_with_ric(model, config)

    def _solve_gdp_with_ric(self, original_model, config):
        logger = config.logger

        (master_util_block,
         subproblem_util_block) = _get_master_and_subproblem(
             original_model, config, self, constraint_list=False)
        master = master_util_block.model()
        subproblem = subproblem_util_block.model()
        master_obj = next(master.component_data_objects(Objective, active=True,
                                                        descend_into=True))

        # main loop
        while self.master_iteration < config.iterlim:
            # Set iteration counters for new master iteration.
            self.master_iteration += 1
            self.mip_iteration = 0
            self.nlp_iteration = 0

            # print line for visual display
            logger.info('---GDPopt Master Iteration %s---' % 
                        self.master_iteration)

            # solve linear master problem
            with time_code(self.timing, 'mip'):
                mip_feasible = solve_linear_GDP(master_util_block, config,
                                                self.timing)
                self._update_bounds_after_master_problem_solve(mip_feasible,
                                                               master_obj,
                                                               logger)
                # TODO: No idea what args these callbacks should actually take
                config.call_after_master_solve(master_util_block, self)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

            with time_code(self.timing, 'nlp'):
                with fix_master_solution_in_subproblem(
                        master_util_block, 
                        subproblem_util_block,
                        config,
                        make_subproblem_continuous=config.force_subproblem_nlp):
                    nlp_feasible = solve_subproblem(subproblem_util_block,
                                                    config, self.timing)
                    if nlp_feasible:
                        new_primal = value(subproblem_util_block.obj.expr)
                        primal_improved = self._update_bounds(primal=new_primal,
                                                              logger=logger)
                        if primal_improved:
                            self.update_incumbent(subproblem_util_block)

            # Add integer cut
            with time_code(self.timing, "integer cut generation"):
                added = add_no_good_cut(master_util_block, config)
                if not added:
                    # We've run out of discrete solutions, so we're done.
                    self._update_dual_bound_to_infeasible(logger)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

        self._get_final_pyomo_results_object()
        if not self.pyomo_results.solver.termination_condition == \
           TerminationCondition.infeasible:
            self._transfer_incumbent_to_original_model()
        return self.pyomo_results
