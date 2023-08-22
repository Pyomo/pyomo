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

from pyomo.contrib.gdpopt.util import time_code, get_main_elapsed_time
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_ECP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts
from pyomo.opt import TerminationCondition as tc


@SolverFactory.register(
    'mindtpy.ecp', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPy_ECP_Solver(_MindtPyAlgorithm):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    This class includes:

    - Extended Cutting Plane (ECP)
    """

    CONFIG = _get_MindtPy_ECP_config()

    def MindtPy_iteration_loop(self):
        """Main loop for MindtPy Algorithms.

        This is the outermost function for the Extended Cutting Plane algorithm in this package; this function controls the progress of
        solving the model.

        Raises
        ------
        ValueError
            The strategy value is not correct or not included.
        """
        while self.mip_iter < self.config.iteration_limit:
            # solve MIP main problem
            main_mip, main_mip_results = self.solve_main()

            if self.handle_main_mip_termination(main_mip, main_mip_results):
                break

            # Call the MIP post-solve callback
            with time_code(self.timing, 'Call after main solve'):
                self.config.call_after_main_solve(main_mip)

            if self.algorithm_should_terminate():
                self.last_iter_cuts = False
                break

            add_ecp_cuts(self.mip, self.jacobians, self.config, self.timing)

        # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
        # we correct it after the iteration.
        if (
            self.config.add_no_good_cuts or self.config.use_tabu_list
        ) and not self.should_terminate:
            self.fix_dual_bound(self.last_iter_cuts)
        self.config.logger.info(
            ' ==============================================================================================='
        )

    def check_config(self):
        config = self.config
        # if ecp tolerance is not provided use bound tolerance
        if config.ecp_tolerance is None:
            config.ecp_tolerance = config.absolute_bound_tolerance
        super().check_config()

    def initialize_mip_problem(self):
        '''Deactivate the nonlinear constraints to create the MIP problem.'''
        super().initialize_mip_problem()
        self.jacobians = calc_jacobians(self.mip, self.config)  # preload jacobians
        self.mip.MindtPy_utils.cuts.ecp_cuts = ConstraintList(
            doc='Extended Cutting Planes'
        )

    def init_rNLP(self):
        """Initialize the problem by solving the relaxed NLP and then store the optimal variable
        values obtained from solving the rNLP.

        Raises
        ------
        ValueError
            MindtPy unable to handle the termination condition of the relaxed NLP.
        """
        super().init_rNLP(add_oa_cuts=False)

    def algorithm_should_terminate(self):
        """Checks if the algorithm should terminate at the given point.

        This function determines whether the algorithm should terminate based on the solver options and progress.
        (Sets the self.results.solver.termination_condition to the appropriate condition, i.e. optimal,
        maxIterations, maxTimeLimit).

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

        return (
            self.bounds_converged()
            or self.reached_iteration_limit()
            or self.reached_time_limit()
            or self.reached_stalling_limit()
            or self.all_nonlinear_constraint_satisfied()
        )

    def all_nonlinear_constraint_satisfied(self):
        # check to see if the nonlinear constraints are satisfied
        config = self.config
        MindtPy = self.mip.MindtPy_utils
        nonlinear_constraints = [c for c in MindtPy.nonlinear_constraint_list]
        for nlc in nonlinear_constraints:
            if nlc.has_lb():
                try:
                    lower_slack = nlc.lslack()
                except (ValueError, OverflowError) as e:
                    # Set lower_slack (upper_slack below) less than -config.ecp_tolerance in this case.
                    config.logger.error(e)
                    lower_slack = -10 * config.ecp_tolerance
                if lower_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc)
                    )
                    return False
            if nlc.has_ub():
                try:
                    upper_slack = nlc.uslack()
                except (ValueError, OverflowError) as e:
                    config.logger.error(e)
                    upper_slack = -10 * config.ecp_tolerance
                if upper_slack < -config.ecp_tolerance:
                    config.logger.debug(
                        'MindtPy-ECP continuing as {} has not met the '
                        'nonlinear constraints satisfaction.'
                        '\n'.format(nlc)
                    )
                    return False
        # For ECP to know whether to know which bound to copy over (primal or dual)
        self.primal_bound = self.dual_bound
        config.logger.info(
            'MindtPy-ECP exiting on nonlinear constraints satisfaction. '
            'Primal Bound: {} Dual Bound: {}\n'.format(
                self.primal_bound, self.dual_bound
            )
        )

        self.best_solution_found = self.mip.clone()
        self.results.solver.termination_condition = tc.optimal
        return True
