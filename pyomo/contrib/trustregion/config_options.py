#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.config import ( 
    ConfigDict, ConfigValue, 
    PositiveInt, PositiveFloat, In)

logger = logging.getLogger('pyomo.contrib.trustregion')

def get_TRF_config():
    CONFIG = ConfigDict('Trust Region')

    CONFIG.declare('solver', ConfigValue(
        default='ipopt',
        description='Solver to use. Defaults to ipopt.',
        doc = 'Specify solver to use for Trust region method. Defaults to ipopt.'))

    CONFIG.declare('solver_options', ConfigDict(
        implicit=True,
        description='Options to pass to the subproblem solver.',
        doc = 'Initialize to pass options to the subproblem solver.'))

    # ================================================
    # === Options common to all solvers
    # ================================================
    CONFIG.declare('keepfiles', ConfigValue(
        default=False,
        domain=bool, description="Optional. Default = False. Whether or not to write files of sub-problems for use in debugging. "
                                 "Must be paired with a writable directory supplied via ``subproblem_file_directory``."))

    CONFIG.declare('tee', ConfigValue(
        default=False,
        domain=bool, description="Optional. Default = False. Sets the ``tee`` for all sub-solvers utilized."))

    # ================================================
    # === Optional user inputs
    # ================================================
    CONFIG.declare('trust radius', ConfigValue(
        default = 1.0,
        domain = PositiveFloat,
        description = 'Initial trust region radius (type: float). Default is 1.0.',
        doc = 'Initialize a positive float trust region radius. Default is 1.0.'))

    CONFIG.declare('sample radius', ConfigValue(
        default = CONFIG.trust_radius / 2.0,
        domain = PositiveFloat,
        description = 'Initial sample radius (type: float). Default is half the initial trust radius.',
        doc = 'Initialize a positive float sample radius. Default is half the initial trust radius.'))

    CONFIG.declare('max radius', ConfigValue(
        default = 1000.0 * CONFIG.trust_radius,
        domain = PositiveFloat,
        description = 'Maximum trust radius. Default is 1000x the initial trust region radius.',
        doc = 'This value protects the trust region radius from becoming too large.'
              'The default is 1000x the initial trust region radius.'))

    # Termination tolerances / parameters
    CONFIG.declare('epsilon delta', ConfigValue(
        default = 1e-5,
        domain = PositiveFloat,
        description = 'Termination tolerance for trust region method (epsilon_delta).',
        doc = 'Initialize a positive float termination tolerance (epsilon_delta).'))

    CONFIG.declare('epsilon s', ConfigValue(
        default = 1e-5,
        domain = PositiveFloat,
        description = 'Termination tolerance for step size s_k (epsilon_s).',
        doc = 'Initialize a positive float termination tolerance (epsilon_s) for step size s_k.'))

    CONFIG.declare('delta min', ConfigValue(
        default = 1e-6,
        domain = PositiveFloat,
        description = 'Minimum trust region. Default is 1e-6.',
        doc = 'Initialize a minimum trust region size. Ensure delta_min <= epsilon_delta.'))

    CONFIG.declare('max iterations', ConfigValue(
        default = 20,
        domain = PositiveInt,
        description = 'Maximum allowed iterations of trust region method. Default is 20.',
        doc = 'Initialize a maximum number of iterations for trust region method. Default is 20.'))

    # Switching Condition parameters
    CONFIG.declare('kappa theta', ConfigValue(
        default = 0.1,
        domain = PositiveFloat,
        description = 'Switching condition parameter between 0 and 1.',
        doc = 'Initialize switching condition parameter (kappa_theta) within the open set (0, 1).'))

    CONFIG.declare('gamma s', ConfigValue(
        default = 2.0,
        domain = PositiveFloat,
        description = 'Switching condition parameter greater than the ratio (1/(1+mu)).',
        doc = 'Initialize switching condition parameter (gamma_s) that satisfies'
              'the condition gamma_s > (1/(1+mu)), with mu within the set (0, 1].'))

    CONFIG.declare('theta min', ConfigValue(
        default = 1e-4,
        domain = PositiveFloat,
        description = 'Switching condition parameter to help determine classification of steps in theta-type or f-type.',
        doc = 'Initialize switching condition parameter (theta_min) for determination of step type.'))

    # Trust region update parameters
    CONFIG.declare('gamma c', ConfigValue(
        default = 0.5,
        domain = PositiveFloat,
        description = 'Trust region update parameter between 0 and 1.',
        doc = 'Initialize first trust region update parameter within the open set (0, 1).'))

    CONFIG.declare('gamma e', ConfigValue(
        default = 2.5,
        domain = PositiveFloat,
        description = 'Trust region update parameter greater than or equal to 1.',
        doc = 'Initialize second trust region update parameter greater than or equal to 1.'))

    # Ratio test parameters (for theta steps)
    CONFIG.declare('eta1', ConfigValue(
        default = 0.05,
        domain = PositiveFloat,
        description = 'Lower ratio test parameter.',
        doc = 'Initialize lower ratio test parameter (eta_1) subject to constraint 0 < eta_1 <= eta_2 < 1.'))

    CONFIG.declare('eta2', ConfigValue(
        default = 0.2,
        domain = PositiveFloat,
        description = 'Upper ratio test parameter.',
        doc = 'Initialize upper ratio test parameter (eta_2) subject to constraint 0 < eta_1 <= eta_2 < 1.'))

    # Filter parameters
    CONFIG.declare('gamma f', ConfigValue(
        default = 0.01,
        domain = PositiveFloat,
        description = 'Filter parameter between 0 and 1.',
        doc = 'Initialize filter parameter (gamma_f) within the open set (0, 1).'))

    CONFIG.declare('gamma theta', ConfigValue(
        default = 0.01,
        domain = PositiveFloat,
        description = 'Filter parameter between 0 and 1.',
        doc = 'Initialize filter parameter (gamma_theta) within the open set (0, 1).'))

    # CONFIG.declare('theta max', ConfigValue(
    #     default = 50,
    #     domain = PositiveInt,
    #     description = '',
    #     doc = ''))

    # Output level (replace with real printlevels!!!)
    CONFIG.declare('verbosity', ConfigValue(
        default = 1,
        domain = PositiveInt,
        description = 'Set a verbosity level. Default is 0.',
        doc = 'Set the verbosity level to get more information at each iteration.'
              'Default is 0. Low is 1. Medium is 2. High is 3.'))

    # # Sample Radius reset parameter
    # CONFIG.declare('sample radius adjust', ConfigValue(
    #     default = 0.5,
    #     domain = PositiveFloat,
    #     description = '',
    #     doc = ''))

    # Default RM type
    CONFIG.declare('reduced model type', ConfigValue(
        default = 1,
        domain = In([0,1]),
        description = '0 = Linear, 1 = Quadratic',
        doc = ''))
