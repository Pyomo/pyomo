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


from pyomo.contrib.gdpopt.util import get_main_elapsed_time
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_GOA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_affine_cuts


@SolverFactory.register(
    'mindtpy.goa', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPy_GOA_Solver(_MindtPyAlgorithm):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    This class includes:

    - Global outer approximation (GOA)
    - Global LP/NLP based branch-and-bound (GLP/NLP)
    """

    CONFIG = _get_MindtPy_GOA_config()

    def check_config(self):
        config = self.config
        config.add_slack = False
        config.use_mcpp = True
        config.equality_relaxation = False
        config.use_fbbt = True
        # add_no_good_cuts is True by default in GOA
        if not config.add_no_good_cuts and not config.use_tabu_list:
            config.add_no_good_cuts = True
            config.use_tabu_list = False
        # Set default initialization_strategy
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

        super().check_config()

    def initialize_mip_problem(self):
        '''Deactivate the nonlinear constraints to create the MIP problem.'''
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
                {self.primal_bound: len(self.mip.MindtPy_utils.cuts.no_good_cuts)}
            )

    def add_cuts(
        self,
        dual_values=None,
        linearize_active=True,
        linearize_violated=True,
        cb_opt=None,
        nlp=None,
    ):
        add_affine_cuts(self.mip, self.config, self.timing)

    def deactivate_no_good_cuts_when_fixing_bound(self, no_good_cuts):
        try:
            valid_no_good_cuts_num = self.num_no_good_cuts_added[self.primal_bound]
            if self.config.add_no_good_cuts:
                for i in range(valid_no_good_cuts_num + 1, len(no_good_cuts) + 1):
                    no_good_cuts[i].deactivate()
            if self.config.use_tabu_list:
                self.integer_list = self.integer_list[:valid_no_good_cuts_num]
        except KeyError as e:
            self.config.logger.error(str(e) + '\nDeactivating no-good cuts failed.')
