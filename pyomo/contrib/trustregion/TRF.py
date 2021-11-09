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

from pyomo.common.dependencies import numpy, numpy_available
if numpy_available:
    from numpy.linalg import norm

from pyomo.core.base.range import NumericRange
from pyomo.common.config import (ConfigDict, ConfigValue,
                                 Bool, PositiveInt,
                                 PositiveFloat, In)
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import (
    copyVector, minIgnoreNone, maxIgnoreNone, IterationLogger
    )
from pyomo.opt.base import SolverFactory

logger = logging.getLogger('pyomo.contrib.trustregion')

__version__ = '0.2.0'


def trust_region_method(model, config, ext_fcn_surrogate_map_rule):
    """
    Main driver of the Trust Region algorithm.
    """
    # Grab values from the configuration options
    trust_radius = config.trust_radius
    min_radius = config.minimum_radius
    max_radius = config.maximum_radius
    max_its = config.maximum_iterations
    feasibility_term = config.feasibility_termination
    step_size_term = config.step_size_termination
    min_feasibility = config.minimum_feasibility
    max_feasibility = config.maximum_feasibility
    switch_cond_kappa_theta = config.kappa_theta
    switch_cond_gamma_s = config.gamma_s
    trf_update_param_gamma_c = config.gamma_c
    trf_update_param_gamma_e = config.gamma_e
    ratio_test_param_eta_1 = config.eta_1
    ratio_test_param_eta_2 = config.eta_2
    filter_param_gamma_theta = config.gamma_theta
    filter_param_gamma_f = config.gamma_f

    # Initialize necessary TRF methods
    TRFLogger = IterationLogger()
    TRFilter = Filter()
    interface = TRFInterface(model, config, ext_fcn_surrogate_map_rule)

    # Initialize the problem
    rebuildSM = False
    obj, feasibility, SM = interface.initializeProblem() # TODO
    # Initialize step_norm_k to a bogus value to enable termination check
    step_norm_k = 1

    iteration = 0
    while iteration < max_its:
        iteration += 1

        # Log relevant information
        TRFLogger.logIteration()

        # Generate suggorate model r_k(w)
        if rebuildSM:
            interface.buildSM() # TODO

        feasibility_k = feasibility
        obj_k = obj
        # Check termination conditions
        if (feasibility_k <= feasibility_term) and (step_norm_k <= step_size_term):
            print('EXIT: Optimal solution found.')
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if ((trust_radius <= min_radius) and
            (feasibility_k < feasibility_term)):
            if subopt_flag:
                print('WARNING: Insufficient progress.')
                print('EXIT: Feasible solution found.')
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        success, obj_k = interface.TRSP() # TODO
        if not success:
            raise Exception('EXIT: Subproblem TRSP_k solve failed.\n')

        step_norm_k = norm() # Difference between init and new input/outputs
        # Check filter acceptance
        feasibility_k = norm() # feasibility(x) = norm(y - d(w))_1
        filterElement = FilterElement(feasibility_k, obj_k)
        if not TRFilter.isAcceptable(filterElement, max_feasibility):
            # Reject the step
            TRFLogger.IterationRecord.rejected = True
            trust_radius = max(min_radius,
                               step_norm_k*trf_update_param_gamma_c)
            rebuildSM = False
            interface.reset() # TODO
            continue

        # Switching condition: Eq. (7) in Yoshio/Biegler (2020)
        if ((obj - obj_k >=
             switch_cond_kappa_theta*pow(feasibility, switch_cond_gamma_s))
            and (feasibility <= min_feasibility)):
            # f-type step
            TRFLogger.IterationRecord.fStep = True
            trust_radius = min(max(step_norm_k*trf_update_param_gamma_e,
                                   trust_radius),
                               max_radius)
        else:
            # theta-type step
            TRFLogger.IterationRecord.thetaStep = True
            filterElement = FilterElement(obj_k - filter_param_gamma_f*feasibility_k,
                                          (1 - filter_param_gamma_theta)*feasibility_k)
            TRFilter.addToFilter(filterElement)
            # Calculate ratio: Eq. (10) in Yoshio/Biegler (2020)
            rho_k = ((feasibility - feasibility_k + feasibility_term) /
                     max(feasibility, feasibility_term))
            # Ratio tests: Eq. (8) in Yoshio/Biegler (2020)
            # If rho_k is between eta_1 and eta_2, trust radius stays same
            if ((rho_k < ratio_test_param_eta_1) or
                (feasibility > min_feasibility)):
                trust_radius = max(min_radius,
                                   trf_update_param_gamma_c*step_norm_k)
            elif ((rho_k >= ratio_test_param_eta_2) and
                  (feasibility <= min_feasibility)):
                trust_radius = max(trust_radius,
                                   max_radius,
                                   trf_update_param_gamma_e*step_norm_k)
        # Accept step and reset for next iteration
        rebuildSM = True
        interface.reset() # TODO
        feasibility = feasibility_k
        obj = obj_k

    if iteration > max_its:
        print('EXIT: Maximum iterations reached: {}.'.format(max_its))


def _trf_config():
    CONFIG = ConfigDict('TrustRegion')

    ### Solver options
    CONFIG.declare('solver', ConfigValue(
        default='ipopt',
        description='Solver to use. Default = ipopt.'
    ))
    CONFIG.declare('keepfiles', ConfigValue(
        default=False,
        domain=Bool,
        description="Optional. Default = False. Whether or not to "
                    "write files of sub-problems for use in debugging. "
                    "Must be paired with a writable directory "
                    "supplied via ``subproblem_file_directory``."
    ))
    CONFIG.declare('tee', ConfigValue(
        default=False,
        domain=Bool,
        description="Optional. Default = False. Sets the ``tee`` "
                    "for sub-solver(s) utilized."
    ))
    CONFIG.declare('load_solution', ConfigValue(
        default=True,
        domain=Bool,
        description="Optional. Default = True. "
                    "Whether or not to load the final solution of "
                    "Trust Region into the model object."
    ))
    ### Trust Region specific options
    CONFIG.declare('trust radius', ConfigValue(
        default=1.0,
        domain=PositiveFloat,
        description="Initial trust region radius (delta_0). "
                    "Default = 1.0."
    ))
    CONFIG.declare('minimum radius', ConfigValue(
        default=CONFIG.trust_radius / 2.0,
        domain=PositiveFloat,
        description="Minimum allowed trust region radius (delta_min). "
                    "Default = trust_radius / 2."
    ))
    CONFIG.declare('maximum radius', ConfigValue(
        default=CONFIG.trust_radius * 1000,
        domain=PositiveFloat,
        description="Maximum allowed trust region radius. If trust region "
                    "radius reaches maximum allowed, solver will exit. "
                    "Default = 1000 * trust_radius."
    ))
    CONFIG.declare('maximum iterations', ConfigValue(
        default=50,
        domain=PositiveInt,
        description="Maximum allowed number of iterations. "
                    "Default = 50."
    ))
    ### Termination options
    CONFIG.declare('feasibility termination', ConfigValue(
        default=1e-5,
        domain=PositiveFloat,
        description="Feasibility measure termination tolerance (epsilon_theta). "
                    "Default = 1e-5."
    ))
    CONFIG.declare('step size termination', ConfigValue(
        default=CONFIG.feasibility_termination,
        domain=PositiveFloat,
        description="Step size termination tolerance (epsilon_s). "
                    "Matches the feasibility termination tolerance by default."
    ))
    ### Switching Condition options
    CONFIG.declare('minimum feasibility', ConfigValue(
        default=1e-4,
        domain=PositiveFloat,
        description="Minimum feasibility measure (theta_min). "
                    "Default = 1e-4."
    ))
    CONFIG.declare('kappa theta', ConfigValue(
        default=0.1,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Switching condition parameter (kappa_theta). "
                    "Contained in open set (0, 1). "
                    "Default = 0.1."
    ))
    CONFIG.declare('gamma s', ConfigValue(
        default=2.0,
        domain=PositiveFloat,
        description="Switching condition parameter (gamma_s). "
                    "Must satisfy: gamma_s > 1/(1+mu) where mu "
                    "is contained in set (0, 1]. "
                    "Default = 2.0."
    ))
    ### Trust region update/ratio test parameters
    CONFIG.declare('gamma c', ConfigValue(
        default=0.5,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Lower trust region update parameter (gamma_c). "
                    "Default = 0.5."
    ))
    CONFIG.declare('gamma e', ConfigValue(
        default=2.5,
        domain=In(NumericRange(1, None, 0)),
        description="Upper trust region update parameter (gamma_e). "
                    "Default = 2.5."
    ))
    CONFIG.declare('eta_1', ConfigValue(
        default = 0.05,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Lower ratio test parameter (eta_1). "
                    "Must satisfy: 0 < eta_1 <= eta_2 < 1. "
                    "Default = 0.05."
    ))
    CONFIG.declare('eta_2', ConfigValue(
        default = 0.25,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Lower ratio test parameter (eta_2). "
                    "Must satisfy: 0 < eta_1 <= eta_2 < 1. "
                    "Default = 0.25."
    ))
    ### Filter
    CONFIG.declare('maximum feasibility', ConfigValue(
        default=50.0,
        domain=PositiveFloat,
        description="Maximum allowable feasibility measure (theta_max). "
                    "Parameter for use in filter method."
                    "Default = 50.0."
    ))
    CONFIG.declare('gamma theta', ConfigValue(
        default=0.01,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Fixed filter parameter (gamma_theta) within (0, 1). "
                    "Default = 0.01"
    ))
    CONFIG.declare('gamma f', ConfigValue(
        default=0.01,
        domain=In(NumericRange(0, 1, 0, (False, False))),
        description="Fixed filter parameter (gamma_f) within (0, 1). "
                    "Default = 0.01"
    ))

    return CONFIG


@SolverFactory.register(
    'trustregion',
    doc='Trust region algorithm "solver" for black box/glass box optimization')
class TrustRegionSolver(object):
    """
    The Trust Region Solver is a 'solver' based on the 2016/2018/2020 AiChE
    papers by Eason (2016/2018), Yoshio (2020), and Biegler.
    """
    CONFIG = _trf_config()

    def available(self, exception_flag=False):
        """
        Check if solver is available.
        """
        return True

    def version(self):
        """
        Return a 3-tuple describing the solver version.
        """
        return __version__

    def license_is_valid(self):
        """
        License for using Trust Region solver.
        """
        return True

    def __enter__(self):
        pass

    def __exit(self, et, ev, tb):
        pass

    def solve(self, model, ext_fcn_surrogate_map_rule, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        if ext_fcn_surrogate_map_rule is None:
            # If the user does not pass us a "basis" function,
            # we default to 0.
            ext_fcn_surrogate_map_rule = lambda comp,ef: 0
        trust_region_method(config, model, ext_fcn_surrogate_map_rule)
