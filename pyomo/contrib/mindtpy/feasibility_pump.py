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
from pyomo.core import ConstraintList
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts


@SolverFactory.register(
    'mindtpy.fp', doc='MindtPy: Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo'
)
class MindtPy_FP_Solver(_MindtPyAlgorithm):
    """
    Decomposition solver for Mixed-Integer Nonlinear Programming (MINLP) problems.

    The MindtPy (Mixed-Integer Nonlinear Decomposition Toolbox in Pyomo) solver
    applies a variety of decomposition-based approaches to solve Mixed-Integer
    Nonlinear Programming (MINLP) problems.
    This class includes:

    - Feasibility pump (FP)
    """

    CONFIG = _get_MindtPy_FP_config()

    def check_config(self):
        # feasibility pump alone will lead to iteration_limit = 0, important!
        self.config.iteration_limit = 0
        self.config.move_objective = True
        super().check_config()

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

    def MindtPy_iteration_loop(self):
        pass
