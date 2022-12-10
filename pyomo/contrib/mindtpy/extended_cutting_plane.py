
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
from pyomo.contrib.gdpopt.util import (time_code, lower_logger_level_to)
from pyomo.contrib.mindtpy.util import set_up_logger,setup_results_object, add_var_bound, calc_jacobians
from pyomo.core import TransformationFactory, Objective, ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_ECP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts
from pyomo.opt import TerminationCondition as tc

__version__ = (0, 1, 0)

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

    def available(self, exception_flag=True):
        """Check if solver is available.
        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        """Return a 3-tuple describing the solver version."""
        return __version__

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
            
            # Update result
            self.update_result()

            config.logger.info(' {:<25}:   {:>7.4f} '.format(
                'Primal-dual gap integral', self.results.solver.primal_dual_gap_integral))

            if config.single_tree:
                self.results.solver.num_nodes = self.nlp_iter - \
                    (1 if config.init_strategy == 'rNLP' else 0)

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

            if self.algorithm_should_terminate(config, check_cycling=True):
                last_iter_cuts = False
                break

            add_ecp_cuts(self.mip, self.jacobians, config, self.timing)

        # if add_no_good_cuts is True, the bound obtained in the last iteration is no reliable.
        # we correct it after the iteration.
        if (config.add_no_good_cuts or config.use_tabu_list) and not self.should_terminate:
            self.fix_dual_bound(config, last_iter_cuts)
        config.logger.info(
            ' ===============================================================================================')

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

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
