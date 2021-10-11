#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from math import pow 
from numpy import inf, concatenate
from numpy.linalg import norm 

from pyomo.contrib.trustregion.filter import (
    FilterElement, Filter)
from pyomo.contrib.trustregion.utils import copyVector
from pyomo.contrib.trustregion.Logger import Logger
from pyomo.contrib.trustregion.interface import (
    PyomoInterface, RMType)

def TrustRegionMethod(m, efList, config):
    """
    The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    efList is a list of ExternalFunction objects that should be
    treated with the trust region

    config is the persistent set of variables defined 
    in the ConfigDict class object

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR".
    """

    logger = Logger()
    TrustRegionFilter = Filter()
    problem = PyomoInterface(m, efList, config)
    init_inputs, init_outputs, init_other = problem.getInitialValue()
    iteration = 0
    rmParams, rmOutputs = problem.buildRM(init_inputs,
                                          config.sample_radius)
    rebuildRM = False
    inputs_k, outputs_k, other_k = copyVector(init_inputs,
                                              init_outputs,
                                              init_other)
    theta = norm(rmOutputs - outputs_k, 1)
    obj = problem.evaluateObj(init_inputs, init_outputs, init_other)
    # Initialize stepNorm_k to a bogus value to ensure termination check is correct
    stepNorm_k = 1

    while iteration < config.max_iterations:
        if iteration > 0:
            logger.printIteration(iteration, config.verbosity)

        # Keep sample radius within Trust Region radius
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust*config.trust_radius,
                config.delta_min)
            rebuildRM = True

        # Generate Reduced Model r_k(x)
        # TODO: The better thing to do here is to check the coefficients for
        # quadratic - and if they are sufficiently small, switch to linear
        if rebuildRM:
            if config.trust_radius < 1e-3:
                problem.rmtype = RMType.linear
            else:
                problem.rmtype = config.reduced_model_type
            rmParams, rmOutputs = problem.buildRM(init_inputs,
                                                  config.sample_radius)

        # Set starter theta for kth iteration
        theta_k = theta
        # Set objective for kth iteration
        obj_k = obj
        # Log iteration information
        logger.newIteration(iteration, inputs_k, outputs_k, other_k,
                            theta_k, obj_k, config.verbosity,
                            problem.rmtype, rmParams)

        # Termination check
        if ((theta_k < config.epsilon_theta) and (stepNorm_k < config.epsilon_s)):
            print('EXIT: Optimal solution found.')
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if ((config.trust_radius <= config.delta_min) and
            (theta_k < config.epsilon_theta)):
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

        # Solve TRSP_k
        flag, obj_k = problem.TRSPk(init_inputs, init_outputs, init_other,
                                    inputs_k, outputs_k, other_k,
                                    rmParams, config.trust_radius)
        if not flag:
            raise Exception('EXIT: Subproblem TRSP_k solve failed.\n')

        cache_inputs = problem.getEFInputValues(init_inputs)
        efOutputs = problem.evaluateEF(init_inputs)
        problem.setEFInputValues(cache_inputs)

        stepNorm_k = norm(concatenate([init_inputs - inputs_k,
                                       init_outputs - outputs_k,
                                       init_other - other_k]), inf)
        logger.setCurrentIteration(trustRadius=config.trust_radius,
                                   sampleRadius=config.sample_radius,
                                   stepNorm=stepNorm_k)
        # Check filter acceptance
        theta_k = norm(efOutputs - init_outputs, 1)
        filterElement = FilterElement(obj_k, theta_k)

        if not TrustRegionFilter.checkElement(filterElement, config.theta_max):
            logger.iterlog.rejected = True
            config.trust_radius = max(config.delta_min,
                                      stepNorm_k*config.gamma_c)
            rebuildRM = False
            init_inputs, init_outputs, init_other = copyVector(inputs_k,
                                                               outputs_k,
                                                               other_k)
            # Reject step
            iteration += 1
            continue

        # Switching condition and Trust region update
        if (((obj - obj_k) >= config.kappa_theta*
             pow(theta, config.gamma_s))
            and (theta < config.theta_min)):
            # Conditions met: f-type step
            logger.iterlog.fStep = True
            # Make the trust region bigger - no bigger than the max allowed
            config.trust_radius = min(max(config.gamma_e*stepNorm_k),
                                      config.max_radius)
        else:
            # Conditions for f-type step NOT met
            # Theta-step
            logger.iterlog.thetaStep = True
            filterElement = FilterElement(obj_k - config.gamma_f*theta_k,
                                          (1-config.gamma_theta)*theta_k)
            TrustRegionFilter.addToFilter(filterElement)

            # Calculate ratio: Equation (10) in 2020 Paper
            rho_k = ((theta - theta_k + config.epsilon_theta) /
                     max(theta, config.epsilon_theta))
            # Ratio tests: Equation (8) in 2020 Paper
            # If rho_k is between eta_1 and eta_2, trust region stays same
            if ((rho_k < config.eta_1) or (theta > config.theta_min)):
                config.trust_radius = max(config.delta_min,
                                          config.gamma_c*stepNorm_k)
            elif ((rho_k >= config.eta_2) and (theta <= config.theta_min)):
                config.trust_radius = max(config.trust_radius,
                                          config.max_radius,
                                          config.gamma_e*stepNorm_k)
        # Accept step; reset for next iteration
        rebuildRM = True
        inputs_k, outputs_k, other_k = copyVector(init_inputs,
                                                  init_outputs,
                                                  init_other)
        theta = theta_k
        obj = obj_k
        iteration += 1

    if iteration > config.max_iterations:
        print('EXIT: Maximum iterations reached: {}.'.format(config.max_iterations))

