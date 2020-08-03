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
from numpy import inf
from numpy.linalg import norm
from pyomo.contrib.trustregion.filterMethod import (
    FilterElement, Filter)
from pyomo.contrib.trustregion.helper import (cloneXYZ, packXYZ)
from pyomo.contrib.trustregion.Logger import Logger
from pyomo.contrib.trustregion.PyomoInterface import (
    PyomoInterface, ROMType)

def TRF(m, eflist, config):
    """The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    eflist is a list of ExternalFunction objects that should be
    treated with the trust region

    config is the persistent set of variables defined 
    in the ConfigBlock class object

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR" TODO: reverse
    the transformation.
    """

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m, eflist, config)
    x, y, z = problem.getInitialValue()

    iteration = -1

    romParam, yr = problem.buildROM(x, config.sample_radius)
    #y = yr
    rebuildROM = False
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    thetak = norm(yr - yk,1)

    objk = problem.evaluateObj(x, y, z)

    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
            #print(xk)
        # increment iteration counter
        iteration = iteration + 1
        if iteration > config.max_it:
            print("EXIT: Maxmium iterations\n")
            break

        ######  Why is this here ###########
        if iteration == 1:
            config.sample_region = False
        ################################

        # Keep Sample Region within Trust Region
        if config.trust_radius < config.sample_radius:
            config.sample_radius = max(
                config.sample_radius_adjust*config.trust_radius,
                config.delta_min)
            rebuildROM = True

        #Generate a RM r_k (x) that is kappa-fully linear on sigma k
        if(rebuildROM):
            #TODO: Ask Jonathan what variable 1e-3 should be
            if config.trust_radius < 1e-3:
                problem.romtype = ROMType.linear
            else:
                problem.romtype = config.reduced_model_type

            romParam, yr = problem.buildROM(x, config.sample_radius)
            #print(romParam)
            #print(config.sample_radius)



        # Criticality Check
        if iteration > 0:
            flag, chik = problem.criticalityCheck(x, y, z, romParam)
            if (not flag):
                raise Exception("Criticality Check fails!\n")

        # Save the iteration information to the logger
        logger.newIter(iteration,xk,yk,zk,thetak,objk,chik,
                       config.print_variables)

        # Check for Termination
        if (thetak < config.ep_i and
            chik < config.ep_chi and
            config.sample_radius < config.ep_delta):
            print("EXIT: OPTIMAL SOLUTION FOUND")
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if (config.trust_radius <= config.delta_min and thetak < config.ep_i):
            if subopt_flag:
                print("EXIT: FEASIBLE SOLUTION FOUND ")
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare
            # the boolean subopt_flag
            subopt_flag = False

        # New criticality phase
        if not config.sample_region:
            config.sample_radius = config.trust_radius/2.0
            if config.sample_radius > chik * config.criticality_check:
                config.sample_radius = config.sample_radius/10.0
            config.trust_radius = config.sample_radius*2
        else:
            config.sample_radius = max(min(config.sample_radius,
                                   chik*config.criticality_check),
                               config.delta_min)

        logger.setCurIter(trustRadius=config.trust_radius,
                          sampleRadius=config.sample_radius)

        # Compatibility Check (Definition 2)
        # radius=max(kappa_delta*config.trust_radius*min(1,kappa_mu*config.trust_radius**mu),
        #            delta_min)
        radius = max(config.kappa_delta*config.trust_radius *
                     min(1,
                         config.kappa_mu*pow(config.trust_radius,config.mu)),
                     config.delta_min)

        try:
            flag, obj = problem.compatibilityCheck(
                x, y, z, xk, yk, zk, romParam, radius,
                config.compatibility_penalty)
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")


        theNorm = norm(x - xk, 2)**2 + norm(z - zk, 2)**2
        if (obj - config.compatibility_penalty*theNorm >
            config.ep_compatibility):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            theta = norm(yr - y, 1)

            logger.iterlog.restoration = True

            fe = FilterElement(
                objk - config.gamma_f*thetak,
                (1 - config.gamma_theta)*thetak)
            filteR.addToFilter(fe)

            rhok = 1 - ((theta - config.ep_i)/max(thetak, config.ep_i))
            if rhok < config.eta1:
                config.trust_radius = max(config.gamma_c*config.trust_radius,
                                  config.delta_min)
            elif rhok >= config.eta2:
                config.trust_radius = min(config.gamma_e*config.trust_radius,
                                  config.radius_max)

            obj = problem.evaluateObj(x, y, z)

            stepNorm = norm(packXYZ(x-xk, y-yk, z-zk), inf)
            logger.setCurIter(stepNorm=stepNorm)

        else:

            # Solve TRSP_k
            flag, obj = problem.TRSPk(x, y, z, xk, yk, zk,
                                      romParam, config.trust_radius)
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)

            stepNorm = norm(packXYZ(x-xk, y-yk, z-zk), inf)
            logger.setCurIter(stepNorm=stepNorm)

            theta = norm(yr - y, 1)
            fe = FilterElement(obj, theta)

            if not filteR.checkAcceptable(fe, config.theta_max) and iteration > 0:
                logger.iterlog.rejected = True
                config.trust_radius = max(config.gamma_c*stepNorm,
                                  config.delta_min)
                rebuildROM = False
                x, y, z = cloneXYZ(xk, yk, zk)
                continue

            # Switching Condition and Trust Region update
            if (((objk - obj) >= config.kappa_theta*
                 pow(thetak, config.gamma_s))
                and
                (thetak < config.theta_min)):
                logger.iterlog.fStep = True

                config.trust_radius = min(
                    max(config.gamma_e*stepNorm, config.trust_radius),
                    config.radius_max)

            else:
                logger.iterlog.thetaStep = True

                fe = FilterElement(
                    obj - config.gamma_f*theta,
                    (1 - config.gamma_theta)*theta)
                filteR.addToFilter(fe)

                # Calculate rho for theta step trust region update
                rhok = 1 - ((theta - config.ep_i) /
                            max(thetak, config.ep_i))
                if rhok < config.eta1:
                    config.trust_radius = max(config.gamma_c*stepNorm,
                                      config.delta_min)
                elif rhok >= config.eta2:
                    config.trust_radius = min(
                        max(config.gamma_e*stepNorm, config.trust_radius),
                        config.radius_max)

        # Accept step
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj


    logger.printVectors()
#    problem.reverseTransform()
