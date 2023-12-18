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

from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_oa_cuts_for_grey_box


@SolverFactory.register(
    'mindtpy.oa', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPy_OA_Solver(_MindtPyAlgorithm):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    This class includes:

    - Outer approximation (OA)
    - Regularized outer approximation (ROA)
    - LP/NLP based branch-and-bound (LP/NLP)
    - Regularized LP/NLP based branch-and-bound (RLP/NLP)
    """

    CONFIG = _get_MindtPy_OA_config()

    def check_config(self):
        config = self.config
        if config.add_regularization is not None:
            if config.add_regularization in {
                'grad_lag',
                'hess_lag',
                'hess_only_lag',
                'sqp_lag',
            }:
                config.calculate_dual_at_solution = True
            if config.regularization_mip_threads == 0 and config.threads > 0:
                config.regularization_mip_threads = config.threads
                config.logger.info('Set regularization_mip_threads equal to threads')
            if config.single_tree:
                config.add_cuts_at_incumbent = True
            if config.mip_regularization_solver is None:
                config.mip_regularization_solver = config.mip_solver
        if config.single_tree:
            config.logger.info('Single-tree implementation is activated.')
            config.iteration_limit = 1
            config.add_slack = False
            if config.mip_solver not in {'cplex_persistent', 'gurobi_persistent'}:
                raise ValueError(
                    "Only cplex_persistent and gurobi_persistent are supported for LP/NLP based Branch and Bound method."
                    "Please refer to https://pyomo.readthedocs.io/en/stable/contributed_packages/mindtpy.html#lp-nlp-based-branch-and-bound."
                )
            if config.threads > 1:
                config.threads = 1
                config.logger.info(
                    'The threads parameter is corrected to 1 since lazy constraint callback conflicts with multi-threads mode.'
                )
        if config.heuristic_nonconvex:
            config.equality_relaxation = True
            config.add_slack = True
        if config.equality_relaxation:
            config.calculate_dual_at_solution = True
        if config.init_strategy == 'FP' or config.add_regularization is not None:
            config.move_objective = True
        if config.add_regularization is not None:
            if config.add_regularization in {
                'level_L1',
                'level_L_infinity',
                'grad_lag',
            }:
                self.regularization_mip_type = 'MILP'
            elif config.add_regularization in {
                'level_L2',
                'hess_lag',
                'hess_only_lag',
                'sqp_lag',
            }:
                self.regularization_mip_type = 'MIQP'
        _MindtPyAlgorithm.check_config(self)

    def initialize_mip_problem(self):
        '''Deactivate the nonlinear constraints to create the MIP problem.'''
        super().initialize_mip_problem()
        self.jacobians = calc_jacobians(self.mip, self.config)  # preload jacobians
        self.mip.MindtPy_utils.cuts.oa_cuts = ConstraintList(
            doc='Outer approximation cuts'
        )

    def add_cuts(
        self,
        dual_values,
        linearize_active=True,
        linearize_violated=True,
        cb_opt=None,
        nlp=None,
    ):
        add_oa_cuts(
            self.mip,
            dual_values,
            self.jacobians,
            self.objective_sense,
            self.mip_constraint_polynomial_degree,
            self.mip_iter,
            self.config,
            self.timing,
            cb_opt,
            linearize_active,
            linearize_violated,
        )
        if len(self.mip.MindtPy_utils.grey_box_list) > 0:
            add_oa_cuts_for_grey_box(
                self.mip, nlp, self.config, self.objective_sense, self.mip_iter, cb_opt
            )

    def deactivate_no_good_cuts_when_fixing_bound(self, no_good_cuts):
        # Only deactivate the last OA cuts may not be correct.
        # Since integer solution may also be cut off by OA cuts due to calculation approximation.
        if self.config.add_no_good_cuts:
            no_good_cuts[len(no_good_cuts)].deactivate()
        if self.config.use_tabu_list:
            self.integer_list = self.integer_list[:-1]

    def objective_reformulation(self):
        # In the process_objective function, as long as the objective function is nonlinear, it will be reformulated and the variable/constraint/objective lists will be updated.
        # For OA/GOA/LP-NLP algorithm, if the objective function is linear, it will not be reformulated as epigraph constraint.
        # If the objective function is linear, it will be reformulated as epigraph constraint only if the Feasibility Pump or ROA/RLP-NLP algorithm is activated. (move_objective = True)
        # In some cases, the variable/constraint/objective lists will not be updated even if the objective is epigraph-reformulated.
        # In Feasibility Pump, since the distance calculation only includes discrete variables and the epigraph slack variables are continuous variables, the Feasibility Pump algorithm will not affected even if the variable list are updated.
        # In ROA and RLP/NLP, since the distance calculation does not include these epigraph slack variables, they should not be added to the variable list. (update_var_con_list = False)
        # In the process_objective function, once the objective function has been reformulated as epigraph constraint, the variable/constraint/objective lists will not be updated only if the MINLP has a linear objective function and regularization is activated at the same time.
        # This is because the epigraph constraint is very "flat" for branching rules. The original objective function will be used for the main problem and epigraph reformulation will be used for the projection problem.
        MindtPy = self.working_model.MindtPy_utils
        config = self.config
        self.process_objective(update_var_con_list=config.add_regularization is None)
        # The epigraph constraint is very "flat" for branching rules.
        # If ROA/RLP-NLP is activated and the original objective function is linear, we will use the original objective for the main mip.
        if (
            MindtPy.objective_list[0].expr.polynomial_degree()
            in self.mip_objective_polynomial_degree
            and config.add_regularization is not None
        ):
            MindtPy.objective_list[0].activate()
            MindtPy.objective_constr.deactivate()
            MindtPy.objective.deactivate()
