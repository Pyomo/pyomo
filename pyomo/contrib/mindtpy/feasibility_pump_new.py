
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
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_FP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.opt import TerminationCondition as tc
from pyomo.core import minimize, Constraint, TransformationFactory, value, Objective, ConstraintList
from pyomo.contrib.mindtpy.feasibility_pump import generate_norm_constraint, fp_converged, add_orthogonality_cuts
from pyomo.contrib.mindtpy.util import generate_norm2sq_objective_function, set_solver_options, set_up_logger, setup_results_object, add_var_bound, calc_jacobians
from pyomo.opt import SolverFactory, SolverResults, SolutionStatus, SolverStatus
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, copy_var_list_values, get_main_elapsed_time, time_code, lower_logger_level_to
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts


@SolverFactory.register(
    'mindtpy.fp',
    doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo')
class MindtPy_FP_Solver(_MindtPyAlgorithm):
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
    CONFIG = _get_MindtPy_FP_config()

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
            self.initial_var_values = list(
                v.value for v in MindtPy.variable_list)
            self.initialize_mip_problem()

            # Initialization
            with time_code(self.timing, 'initialization'):
                self.MindtPy_initialization(config)

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


    def check_config(self):
        # feasibility pump alone will lead to iteration_limit = 0, important!
        self.config.init_strategy = 'FP'
        self.config.iteration_limit = 0
        self.config.move_objective = True
        super().check_config()

    #
    # Support 'with' statements.
    #
    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def initialize_mip_problem(self):
        ''' Deactivate the nonlinear constraints to create the MIP problem.
        '''
        # if single tree is activated, we need to add bounds for unbounded variables in nonlinear constraints to avoid unbounded main problem.
        config = self.config
        if config.single_tree:
            add_var_bound(self.working_model, config)

        m = self.mip = self.working_model.clone()
        next(self.mip.component_data_objects(
            Objective, active=True)).deactivate()

        MindtPy = m.MindtPy_utils
        if config.calculate_dual_at_solution:
            m.dual.deactivate()

        self.jacobians = calc_jacobians(self.mip, config)  # preload jacobians
        MindtPy.cuts.oa_cuts = ConstraintList(doc='Outer approximation cuts')
        MindtPy.cuts.fp_orthogonality_cuts = ConstraintList(
            doc='Orthogonality cuts in feasibility pump')
        if config.fp_projcuts:
            self.working_model.MindtPy_utils.cuts.fp_orthogonality_cuts = ConstraintList(
                doc='Orthogonality cuts in feasibility pump')


    def add_cuts(self,
                 dual_values,
                 linearize_active=True,
                 linearize_violated=True,
                 cb_opt=None):
        add_oa_cuts(self.mip, 
                    dual_values,
                    self.jacobians,
                    self.objective_sense,
                    self.mip_constraint_polynomial_degree,
                    self.mip_iter,
                    self.config,
                    self.timing,
                    cb_opt,
                    linearize_active,
                    linearize_violated)
