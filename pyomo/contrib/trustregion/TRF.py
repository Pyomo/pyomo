#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import pow
from numpy import inf
from numpy.linalg import norm
from pyomo.contrib.trustregion.param import *
from pyomo.contrib.trustregion.filterMethod import FilterElement, Filter
from pyomo.contrib.trustregion.helper import cloneXYZ, packXYZ, minIgnoreNone, maxIgnoreNone
from pyomo.contrib.trustregion.Logger import IterLog, Logger
from pyomo.contrib.trustregion.PyomoInterface import PyomoInterface, ROMType



def TRF(m,eflist):
    """
    The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects
    Model requirements: m is a nonlinear program, with exactly one active objective function.

    eflist is a list of ExternalFunction objects that should be treated with the
    trust region


    Return:
    model is solved, variables are at optimal solution or other exit condition.
    model is left in reformulated form, with some new variables introduced
    in a block named "tR" TODO: reverse the transformation.
    """


    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m,eflist)
    x, y, z = problem.getInitialValue()


    trustRadius = TRUST_RADIUS
    sampleRadius = SAMPLE_RADIUS
    sampleregion_yn = SAMPLEREGION_YN
    iteration = -1

    romParam, yr = problem.buildROM(x, sampleRadius)
    #y = yr
    rebuildROM = False
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    thetak = norm(yr - yk,1)

    objk = problem.evaluateObj(x, y, z)

    while True:
        if(iteration>=0):
            logger.printIteration(iteration)
            #print(xk)
        # increment iteration counter
        iteration = iteration + 1
        if(iteration > MAXIT):
            print("EXIT: Maxmium iterations\n")
            break

        ######  Why is this here ###########
        if iteration == 1:
            sampleregion_yn = False
        ################################

        # Keep Sample Region within Trust Region
        if trustRadius < sampleRadius:
            sampleRadius = max(SR_ADJUST * trustRadius, DELTMIN)
            rebuildROM = True

        #Generate a RM r_k (x) that is κ-fully linear on sigma k
        if(rebuildROM):
            if trustRadius < 1e-3:
                problem.romtype = ROMType.linear
            else:
                problem.romtype = DEFAULT_ROMTYPE

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
        if thetak < EP_I and chik < EP_CHI and sampleRadius < EP_DELT:
            print("EXIT: OPTIMAL SOLUTION FOUND")
            break

        # If trust region very small and no progress is being made, terminate
        # The following condition must hold for two consecutive iterations.
        if trustRadius <= DELTMIN and thetak < EP_I:
            if subopt_flag:
                print("EXIT: FEASIBLE SOLUTION FOUND ")
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare the boolean subopt_flag
            subopt_flag = False


        # New criticality phase
        if not sampleregion_yn:
            sampleRadius = trustRadius/2.0
            if sampleRadius > chik * CRITICALITY_CHECK:
                sampleRadius = sampleRadius/10.0
            trustRadius = sampleRadius*2
        else:
            sampleRadius = max(min(sampleRadius,chik*CRITICALITY_CHECK),DELTMIN)

        logger.setCurIter(trustRadius=trustRadius,sampleRadius=sampleRadius)

        # Compatibility Check
        radius = max(KAPPA_DELTA * trustRadius * \
            min(1, KAPPA_MU * pow(trustRadius, MU)),DELTMIN)

        try:
            flag, obj = problem.compatibilityCheck(
                x, y, z, xk, yk, zk, romParam, radius, COMPAT_PENALTY)
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")



        if(obj - COMPAT_PENALTY * (norm(x - xk, 2)**2 + norm(z - zk, 2)**2) > EP_COMPAT):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            theta = norm(yr - y, 1)

            logger.iterlog.restoration = True

            fe = FilterElement(objk - GAMMA_F * thetak, (1 - GAMMA_THETA) * thetak)
            filteR.addToFilter(fe)

            rhok = 1 - (theta - EP_I)/max(thetak,EP_I)
            if(rhok < ETA1):
                trustRadius = max(GAMMA_C * trustRadius, DELTMIN)
            elif (rhok >= ETA2):
                trustRadius = min(GAMMA_E * trustRadius, RADIUS_MAX)

            obj = problem.evaluateObj(x, y, z)

            stepNorm = norm(packXYZ(x-xk,y-yk,z-zk),inf)
            logger.setCurIter(stepNorm=stepNorm)

        else:

            # Solve TRSP_k
            flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, romParam, trustRadius)
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)

            stepNorm = norm(packXYZ(x-xk,y-yk,z-zk),inf)
            logger.setCurIter(stepNorm=stepNorm)

            theta = norm(yr - y, 1)
            fe = FilterElement(obj, theta)

            if not filteR.checkAcceptable(fe) and iteration>0:
                logger.iterlog.rejected = True
                trustRadius = max(GAMMA_C * stepNorm, DELTMIN)
                rebuildROM = False
                x, y, z = cloneXYZ(xk, yk, zk)
                continue

            # Switching Condition and Trust Region update
            if((objk - obj) >= KAPPA_THETA * pow(thetak, GAMMA_S) and thetak < THETA_MIN):
                logger.iterlog.fStep = True

                trustRadius = min(max(GAMMA_E*stepNorm,trustRadius),RADIUS_MAX)

            else:
                logger.iterlog.thetaStep = True

                fe = FilterElement(
                    obj - GAMMA_F * theta, (1 - GAMMA_THETA) * theta)
                filteR.addToFilter(fe)

                # Calculate rho for theta step trust region update
                rhok = 1 - (theta - EP_I)/max(thetak,EP_I)
                if(rhok < ETA1):
                    trustRadius = max(GAMMA_C * stepNorm, DELTMIN)
                elif (rhok >= ETA2):
                    trustRadius = min(max(GAMMA_E * stepNorm,trustRadius), RADIUS_MAX)



        # Accept step
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj


    logger.printVectors()
#    problem.reverseTransform()
