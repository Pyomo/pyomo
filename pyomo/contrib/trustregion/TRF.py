import numpy as np
from math import pow
from numpy import inf
from numpy.linalg import norm
from pyomo.contrib.trustregion.param import *
from pyomo.contrib.trustregion.filterMethod import (
    FilterElement, Filter)
from pyomo.contrib.trustregion.helper import (cloneXYZ, packXYZ,
    minIgnoreNone, maxIgnoreNone)
from pyomo.contrib.trustregion.Logger import (IterLog, Logger)
from pyomo.contrib.trustregion.PyomoInterface import (
    PyomoInterface, ROMType)

def TRF(m,eflist):
    """The main function of the Trust Region Filter algorithm
    m is a PyomoModel containing ExternalFunction() objects Model
    requirements: m is a nonlinear program, with exactly one active
    objective function.

    eflist is a list of ExternalFunction objects that should be
    treated with the trust region

    Return: 
    model is solved, variables are at optimal solution or
    other exit condition.  model is left in reformulated form, with
    some new variables introduced in a block named "tR" TODO: reverse
    the transformation.
    """

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m, eflist)
    x, y, z = problem.getInitialValue()

    trustRadius = CONFIG.trust_radius
    sampleRadius = CONFIG.sample_radius
    sampleregion_yn = CONFIG.sample_region

    iteration = -1

    romParam, yr = problem.buildROM(x, sampleRadius)
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
        if iteration > CONFIG.max_it:
            print("EXIT: Maxmium iterations\n")
            break

        ######  Why is this here ###########
        if iteration == 1:
            sampleregion_yn = False
        ################################

        # Keep Sample Region within Trust Region
        if trustRadius < sampleRadius:
            sampleRadius = max(
                CONFIG.sample_radius_adjust*trustRadius,
                CONFIG.delta_min)
            rebuildROM = True

        #Generate a RM r_k (x) that is kappa-fully linear on sigma k
        if(rebuildROM):
            #TODO: Ask Jonathan what variable 1e-3 should be
            if trustRadius < 1e-3:
                problem.romtype = ROMType.linear
            else:
                problem.romtype = CONFIG.reduced_model_type

            romParam, yr = problem.buildROM(x, sampleRadius)
            #print(romParam)
            #print(sampleRadius)



        # Criticality Check
        if iteration > 0:
            flag, chik = problem.criticalityCheck(x, y, z, romParam)
            if (not flag):
                raise Exception("Criticality Check fails!\n")

        # Save the iteration information to the logger
        logger.newIter(iteration,xk,yk,zk,thetak,objk,chik)

        # Check for Termination
        if (thetak < CONFIG.ep_i and
            chik < CONFIG.ep_chi and
            sampleRadius < CONFIG.ep_delta):
            print("EXIT: OPTIMAL SOLUTION FOUND")
            break

        # If trust region very small and no progress is being made,
        # terminate. The following condition must hold for two
        # consecutive iterations.
        if (trustRadius <= CONFIG.delta_min and thetak < CONFIG.ep_i):
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
        if not sampleregion_yn:
            sampleRadius = trustRadius/2.0
            if sampleRadius > chik * CONFIG.criticality_check:
                sampleRadius = sampleRadius/10.0
            trustRadius = sampleRadius*2
        else:
            sampleRadius = max(min(sampleRadius,
                                   chik*CONFIG.criticality_check),
                               CONFIG.delta_min)

        logger.setCurIter(trustRadius=trustRadius,
                          sampleRadius=sampleRadius)

        # Compatibility Check (Definition 2)
        # radius=max(kappa_delta*trustRadius*min(1,kappa_mu*trustRadius**mu),
        #            delta_min)
        radius = max(CONFIG.kappa_delta*trustRadius *
                     min(1,
                         CONFIG.kappa_mu*pow(trustRadius,CONFIG.mu)),
                     CONFIG.delta_min)

        try:
            flag, obj = problem.compatibilityCheck(
                x, y, z, xk, yk, zk, romParam, radius,
                CONFIG.compatibility_penalty)
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")


        theNorm = norm(x - xk, 2)**2 + norm(z - zk, 2)**2
        if (obj - CONFIG.compatibility_penalty*theNorm >
            CONFIG.ep_compatibility):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            theta = norm(yr - y, 1)

            logger.iterlog.restoration = True

            fe = FilterElement(
                objk - CONFIG.gamma_f*thetak,
                (1 - CONFIG.gamma_theta)*thetak)
            filteR.addToFilter(fe)

            rhok = 1 - ((theta - CONFIG.ep_i)/max(thetak, CONFIG.ep_i))
            if rhok < CONFIG.eta1:
                trustRadius = max(CONFIG.gamma_c*trustRadius,
                                  CONFIG.delta_min)
            elif rhok >= CONFIG.eta2:
                trustRadius = min(CONFIG.gamma_e*trustRadius,
                                  CONFIG.radius_max)

            obj = problem.evaluateObj(x, y, z)

            stepNorm = norm(packXYZ(x-xk, y-yk, z-zk), inf)
            logger.setCurIter(stepNorm=stepNorm)

        else:

            # Solve TRSP_k
            flag, obj = problem.TRSPk(x, y, z, xk, yk, zk,
                                      romParam, trustRadius)
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)

            stepNorm = norm(packXYZ(x-xk, y-yk, z-zk), inf)
            logger.setCurIter(stepNorm=stepNorm)

            theta = norm(yr - y, 1)
            fe = FilterElement(obj, theta)

            if not filteR.checkAcceptable(fe) and iteration > 0:
                logger.iterlog.rejected = True
                trustRadius = max(CONFIG.gamma_c*stepNorm,
                                  CONFIG.delta_min)
                rebuildROM = False
                x, y, z = cloneXYZ(xk, yk, zk)
                continue

            # Switching Condition and Trust Region update
            if (((objk - obj) >= CONFIG.kappa_theta*
                 pow(thetak, CONFIG.gamma_s))
                and
                (thetak < CONFIG.theta_min)):
                logger.iterlog.fStep = True

                trustRadius = min(
                    max(CONFIG.gamma_e*stepNorm, trustRadius),
                    CONFIG.radius_max)

            else:
                logger.iterlog.thetaStep = True

                fe = FilterElement(
                    obj - CONFIG.gamma_f*theta,
                    (1 - CONFIG.gamma_theta)*theta)
                filteR.addToFilter(fe)

                # Calculate rho for theta step trust region update
                rhok = 1 - ((theta - CONFIG.ep_i) /
                            max(thetak, CONFIG.ep_i))
                if rhok < CONFIG.eta1:
                    trustRadius = max(CONFIG.gamma_c*stepNorm,
                                      CONFIG.delta_min)
                elif rhok >= CONFIG.eta2:
                    trustRadius = min(
                        max(CONFIG.gamma_e*stepNorm, trustRadius),
                        CONFIG.radius_max)

        # Accept step
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj


    logger.printVectors()
#    problem.reverseTransform()
