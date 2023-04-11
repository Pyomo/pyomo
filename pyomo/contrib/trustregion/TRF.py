#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________


import logging

from pyomo.core.base.range import NumericRange
from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    Bool,
    PositiveInt,
    PositiveFloat,
    In,
    document_kwargs_from_configdict,
)
from pyomo.contrib.trustregion.filter import Filter, FilterElement
from pyomo.contrib.trustregion.interface import TRFInterface
from pyomo.contrib.trustregion.util import IterationLogger
from pyomo.opt import SolverFactory

logger = logging.getLogger('pyomo.contrib.trustregion')

__version__ = '0.2.0'


def trust_region_method(model, decision_variables, ext_fcn_surrogate_map_rule, config):
    """
    The main driver of the Trust Region algorithm method.

    Parameters
    ----------
    model : ConcreteModel
        The user's model to be solved.
    degrees_of_freedom_variables : List of Vars
        User-supplied input. The user must provide a list of vars which
        are the degrees of freedom or decision variables within
        the model.
    ext_fcn_surrogate_map_rule : Function, optional
        In the 2020 Yoshio/Biegler paper, this is referred to as
        the basis function `b(w)`.
        This is the low-fidelity model with which to solve the original
        process model problem and which is integrated into the
        surrogate model.
        The default is 0 (i.e., no basis function rule.)
    config : ConfigDict
        This holds the solver and TRF-specific configuration options.

    """

    # Initialize necessary TRF methods
    TRFLogger = IterationLogger()
    TRFilter = Filter()
    interface = TRFInterface(
        model, decision_variables, ext_fcn_surrogate_map_rule, config
    )

    # Initialize the problem
    rebuildSM = False
    obj_val, feasibility = interface.initializeProblem()
    # Initialize first iteration feasibility/objective value to enable
    # termination check
    feasibility_k = feasibility
    obj_val_k = obj_val
    # Initialize step_norm_k to a bogus value to enable termination check
    step_norm_k = 0
    # Initialize trust region radius
    trust_radius = config.trust_radius

    iteration = 0

    TRFLogger.newIteration(
        iteration, feasibility_k, obj_val_k, trust_radius, step_norm_k
    )
    TRFLogger.logIteration()
    if config.verbose:
        TRFLogger.printIteration()
    while iteration < config.maximum_iterations:
        iteration += 1

        # Check termination conditions
        if (feasibility_k <= config.feasibility_termination) and (
            step_norm_k <= config.step_size_termination
        ):
            print('EXIT: Optimal solution found.')
            interface.model.display()
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if (trust_radius <= config.minimum_radius) and (
            abs(feasibility_k - feasibility) < config.feasibility_termination
        ):
            if subopt_flag:
                logger.warning('WARNING: Insufficient progress.')
                print('EXIT: Feasible solution found.')
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        # Set bounds to enforce the trust region
        interface.updateDecisionVariableBounds(trust_radius)
        # Generate suggorate model r_k(w)
        if rebuildSM:
            interface.updateSurrogateModel()

        # Solve the Trust Region Subproblem (TRSP)
        obj_val_k, step_norm_k, feasibility_k = interface.solveModel()

        TRFLogger.newIteration(
            iteration, feasibility_k, obj_val_k, trust_radius, step_norm_k
        )

        # Check filter acceptance
        filterElement = FilterElement(obj_val_k, feasibility_k)
        if not TRFilter.isAcceptable(filterElement, config.maximum_feasibility):
            # Reject the step
            TRFLogger.iterrecord.rejected = True
            trust_radius = max(
                config.minimum_radius, step_norm_k * config.radius_update_param_gamma_c
            )
            rebuildSM = False
            interface.rejectStep()
            # Log iteration information
            TRFLogger.logIteration()
            if config.verbose:
                TRFLogger.printIteration()
            continue

        # Switching condition: Eq. (7) in Yoshio/Biegler (2020)
        if (obj_val - obj_val_k) >= (
            config.switch_condition_kappa_theta
            * pow(feasibility, config.switch_condition_gamma_s)
        ) and (feasibility <= config.minimum_feasibility):
            # f-type step
            TRFLogger.iterrecord.fStep = True
            trust_radius = min(
                max(step_norm_k * config.radius_update_param_gamma_e, trust_radius),
                config.maximum_radius,
            )
        else:
            # theta-type step
            TRFLogger.iterrecord.thetaStep = True
            filterElement = FilterElement(
                obj_val_k - config.param_filter_gamma_f * feasibility_k,
                (1 - config.param_filter_gamma_theta) * feasibility_k,
            )
            TRFilter.addToFilter(filterElement)
            # Calculate ratio: Eq. (10) in Yoshio/Biegler (2020)
            rho_k = (
                feasibility - feasibility_k + config.feasibility_termination
            ) / max(feasibility, config.feasibility_termination)
            # Ratio tests: Eq. (8) in Yoshio/Biegler (2020)
            # If rho_k is between eta_1 and eta_2, trust radius stays same
            if (rho_k < config.ratio_test_param_eta_1) or (
                feasibility > config.minimum_feasibility
            ):
                trust_radius = max(
                    config.minimum_radius,
                    (config.radius_update_param_gamma_c * step_norm_k),
                )
            elif rho_k >= config.ratio_test_param_eta_2:
                trust_radius = min(
                    config.maximum_radius,
                    max(
                        trust_radius, (config.radius_update_param_gamma_e * step_norm_k)
                    ),
                )

        TRFLogger.updateIteration(trustRadius=trust_radius)
        # Accept step and reset for next iteration
        rebuildSM = True
        feasibility = feasibility_k
        obj_val = obj_val_k
        # Log iteration information
        TRFLogger.logIteration()
        if config.verbose:
            TRFLogger.printIteration()

    if iteration >= config.maximum_iterations:
        logger.warning(
            'EXIT: Maximum iterations reached: {}.'.format(config.maximum_iterations)
        )

    return interface.model


def _trf_config():
    """
    Generate the configuration dictionary.
    The user may change the configuration options during the instantiation
    of the trustregion solver:
        >>> optTRF = SolverFactory('trustregion',
        ...                        solver='ipopt',
        ...                        maximum_iterations=50,
        ...                        minimum_radius=1e-5,
        ...                        verbose=True)

    The user may also update the configuration after instantiation:
        >>> optTRF = SolverFactory('trustregion')
        >>> optTRF._CONFIG.trust_radius = 0.5

    The user may also update the configuration as part of the solve call:
        >>> optTRF = SolverFactory('trustregion')
        >>> optTRF.solve(model, decision_variables, trust_radius=0.5)
    Returns
    -------
    CONFIG : ConfigDict
        This holds all configuration options to be passed to the TRF solver.

    """
    CONFIG = ConfigDict('TrustRegion')

    ### Solver options
    CONFIG.declare(
        'solver',
        ConfigValue(default='ipopt', description='Solver to use. Default = ``ipopt``.'),
    )
    CONFIG.declare(
        'keepfiles',
        ConfigValue(
            default=False,
            domain=Bool,
            description="Optional. Whether or not to "
            "write files of sub-problems for use in debugging. "
            "Default = False.",
        ),
    )
    CONFIG.declare(
        'tee',
        ConfigValue(
            default=False,
            domain=Bool,
            description="Optional. Sets the ``tee`` "
            "for sub-solver(s) utilized. "
            "Default = False.",
        ),
    )

    ### Trust Region specific options
    CONFIG.declare(
        'verbose',
        ConfigValue(
            default=False,
            domain=Bool,
            description="Optional. When True, print each "
            "iteration's relevant information to the console "
            "as well as to the log. "
            "Default = False.",
        ),
    )
    CONFIG.declare(
        'trust_radius',
        ConfigValue(
            default=1.0,
            domain=PositiveFloat,
            description="Initial trust region radius ``delta_0``. Default = 1.0.",
        ),
    )
    CONFIG.declare(
        'minimum_radius',
        ConfigValue(
            default=1e-6,
            domain=PositiveFloat,
            description="Minimum allowed trust region radius ``delta_min``. "
            "Default = 1e-6.",
        ),
    )
    CONFIG.declare(
        'maximum_radius',
        ConfigValue(
            default=CONFIG.trust_radius * 100,
            domain=PositiveFloat,
            description="Maximum allowed trust region radius. If trust region "
            "radius reaches maximum allowed, solver will exit. "
            "Default = 100 * trust_radius.",
        ),
    )
    CONFIG.declare(
        'maximum_iterations',
        ConfigValue(
            default=50,
            domain=PositiveInt,
            description="Maximum allowed number of iterations. Default = 50.",
        ),
    )
    ### Termination options
    CONFIG.declare(
        'feasibility_termination',
        ConfigValue(
            default=1e-5,
            domain=PositiveFloat,
            description="Feasibility measure termination tolerance ``epsilon_theta``. "
            "Default = 1e-5.",
        ),
    )
    CONFIG.declare(
        'step_size_termination',
        ConfigValue(
            default=CONFIG.feasibility_termination,
            domain=PositiveFloat,
            description="Step size termination tolerance ``epsilon_s``. "
            "Matches the feasibility termination tolerance by default.",
        ),
    )
    ### Switching Condition options
    CONFIG.declare(
        'minimum_feasibility',
        ConfigValue(
            default=1e-4,
            domain=PositiveFloat,
            description="Minimum feasibility measure ``theta_min``. Default = 1e-4.",
        ),
    )
    CONFIG.declare(
        'switch_condition_kappa_theta',
        ConfigValue(
            default=0.1,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Switching condition parameter ``kappa_theta``. "
            "Contained in open set (0, 1). "
            "Default = 0.1.",
        ),
    )
    CONFIG.declare(
        'switch_condition_gamma_s',
        ConfigValue(
            default=2.0,
            domain=PositiveFloat,
            description="Switching condition parameter ``gamma_s``. "
            "Must satisfy: ``gamma_s > 1/(1+mu)`` where ``mu`` "
            "is contained in set (0, 1]. "
            "Default = 2.0.",
        ),
    )
    ### Trust region update/ratio test parameters
    CONFIG.declare(
        'radius_update_param_gamma_c',
        ConfigValue(
            default=0.5,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Lower trust region update parameter ``gamma_c``. "
            "Default = 0.5.",
        ),
    )
    CONFIG.declare(
        'radius_update_param_gamma_e',
        ConfigValue(
            default=2.5,
            domain=In(NumericRange(1, None, 0)),
            description="Upper trust region update parameter ``gamma_e``. "
            "Default = 2.5.",
        ),
    )
    CONFIG.declare(
        'ratio_test_param_eta_1',
        ConfigValue(
            default=0.05,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Lower ratio test parameter ``eta_1``. "
            "Must satisfy: ``0 < eta_1 <= eta_2 < 1``. "
            "Default = 0.05.",
        ),
    )
    CONFIG.declare(
        'ratio_test_param_eta_2',
        ConfigValue(
            default=0.2,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Lower ratio test parameter ``eta_2``. "
            "Must satisfy: ``0 < eta_1 <= eta_2 < 1``. "
            "Default = 0.2.",
        ),
    )
    ### Filter
    CONFIG.declare(
        'maximum_feasibility',
        ConfigValue(
            default=50.0,
            domain=PositiveFloat,
            description="Maximum allowable feasibility measure ``theta_max``. "
            "Parameter for use in filter method."
            "Default = 50.0.",
        ),
    )
    CONFIG.declare(
        'param_filter_gamma_theta',
        ConfigValue(
            default=0.01,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Fixed filter parameter ``gamma_theta`` within (0, 1). "
            "Default = 0.01",
        ),
    )
    CONFIG.declare(
        'param_filter_gamma_f',
        ConfigValue(
            default=0.01,
            domain=In(NumericRange(0, 1, 0, (False, False))),
            description="Fixed filter parameter ``gamma_f`` within (0, 1). "
            "Default = 0.01",
        ),
    )

    return CONFIG


@SolverFactory.register(
    'trustregion',
    doc='Trust region algorithm "solver" for black box/glass box optimization',
)
class TrustRegionSolver(object):
    """
    The Trust Region Solver is a 'solver' based on the 2016/2018/2020 AiChE
    papers by Eason (2016/2018), Yoshio (2020), and Biegler.

    """

    CONFIG = _trf_config()

    def __init__(self, **kwds):
        self.config = self.CONFIG(kwds)

    def available(self, exception_flag=True):
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
        return self

    def __exit__(self, et, ev, tb):
        pass

    @document_kwargs_from_configdict(CONFIG)
    def solve(
        self,
        model,
        degrees_of_freedom_variables,
        ext_fcn_surrogate_map_rule=None,
        **kwds
    ):
        """
        This method calls the TRF algorithm.

        Parameters
        ----------
        model : ConcreteModel
            The model to be solved using the Trust Region Framework.
        degrees_of_freedom_variables : List[Var]
            User-supplied input. The user must provide a list of vars which
            are the degrees of freedom or decision variables within
            the model.
        ext_fcn_surrogate_map_rule : Function, optional
            In the 2020 Yoshio/Biegler paper, this is referred to as
            the basis function `b(w)`.
            This is the low-fidelity model with which to solve the original
            process model problem and which is integrated into the
            surrogate model.
            The default is 0 (i.e., no basis function rule.)

        """
        config = self.config(kwds.pop('options', {}))
        config.set_value(kwds)
        if ext_fcn_surrogate_map_rule is None:
            # If the user does not pass us a "basis" function,
            # we default to 0.
            ext_fcn_surrogate_map_rule = lambda comp, ef: 0
        result = trust_region_method(
            model, degrees_of_freedom_variables, ext_fcn_surrogate_map_rule, config
        )
        return result
