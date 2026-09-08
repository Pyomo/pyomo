# -*- coding: utf-8 -*-

# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________


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
                    "Please refer to https://pyomo.readthedocs.io/en/stable/explanation/solvers/mindtpy.html#lp-nlp-based-branch-and-bound."
                )
            if config.threads > 1:
                config.threads = 1
                config.logger.info(
                    'The threads parameter is corrected to 1 since lazy constraint callback conflicts with multi-threads mode.'
                )

        super().check_config()

    def initialize_mip_problem(self):
        """Deactivate the nonlinear constraints to create the MIP problem."""
        super().initialize_mip_problem()
        self.mip.MindtPy_utils.cuts.aff_cuts = ConstraintList(doc='Affine cuts')

    def add_cuts(
        self,
        dual_values=None,
        linearize_active=True,
        linearize_violated=True,
        cb_opt=None,
        nlp=None,
    ):
        add_affine_cuts(self.mip, self.config, self.timing)
