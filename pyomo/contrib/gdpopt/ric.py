#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.config import add_docstring_list
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs, _add_nlp_solver_configs, _add_tolerance_configs,
    _add_OA_configs)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_master_and_subproblem)
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.mip_solve import solve_MILP_master_problem
from pyomo.contrib.gdpopt.oa_algorithm_utils import (
    _fix_master_soln_solve_subproblem_and_add_cuts)
from pyomo.contrib.gdpopt.util import (
    time_code, lower_logger_level_to, move_nonlinear_objective_to_constraints)
from pyomo.core import Objective, Expression, value
from pyomo.opt import TerminationCondition
from pyomo.opt.base import SolverFactory

import logging

# ESJ: In the future, if we have a direct interface to cplex or gurobi, we
# should get the integer solutions several-at-a-time with a solution pool or
# something of the like...

@SolverFactory.register(
    '_relaxation_with_integer_cuts',
    doc='GDP Relaxation with Integer Cuts (RIC) solver')
class GDP_RIC_Solver(_GDPoptAlgorithm):
    """The GDPopt (Generalized Disjunctive Programming optimizer) relaxation
    with integer cuts (RIC) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For non-convex problems, RIC
    will not be exact unless the NLP subproblems are solved globally.
    """
    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_OA_configs(CONFIG)

    def __init__(self, **kwds):
        self.CONFIG = self.CONFIG(kwds)
        super(GDP_RIC_Solver, self).__init__()

    def solve(self, model, **kwds):
        """Solve the model with RIC

        Args:
            model (Block): a Pyomo model or block to be solved.

        """
        config = self.CONFIG(kwds.pop('options', {}), preserve_implicit=True)
        config.set_value(kwds)

        with time_code(self.timing, 'total', is_main_timer=True), \
            lower_logger_level_to(config.logger, config.tee):
            results = super().solve(model, config)
            if results:
                return results
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

        self._log_header(logger)

        # main loop
        while self.iteration < config.iterlim:
            self.iteration += 1

            # solve linear master problem
            with time_code(self.timing, 'mip'):
                mip_feasible = solve_MILP_master_problem(master_util_block,
                                                         config, self.timing)
                self._update_bounds_after_master_problem_solve(mip_feasible,
                                                               master_obj,
                                                               logger)

            # Check termination conditions
            if self.any_termination_criterion_met(config):
                break

            with time_code(self.timing, 'nlp'):
                _fix_master_soln_solve_subproblem_and_add_cuts(
                    master_util_block, subproblem_util_block, config, self)

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
        self._log_termination_message(logger)
        if self.pyomo_results.solver.termination_condition not in \
           {TerminationCondition.infeasible, TerminationCondition.unbounded}:
            self._transfer_incumbent_to_original_model()
        return self.pyomo_results

    def _add_cuts_to_master_problem(self, subproblem_util_block,
                                    master_util_block, objective_sense, config,
                                    timing):
        # Nothing to do here
        pass

GDP_RIC_Solver.solve.__doc__ = add_docstring_list(
    GDP_RIC_Solver.solve.__doc__, GDP_RIC_Solver.CONFIG, indent_by=8)
