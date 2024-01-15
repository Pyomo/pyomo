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

"""
Common utilities for Trust Region Framework
"""

import logging

logger = logging.getLogger('pyomo.contrib.trustregion')


def minIgnoreNone(a, b):
    """
    Return the min of two numbers, ignoring None
    """
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return a
    return b


def maxIgnoreNone(a, b):
    """
    Return the max of two numbers, ignoring None
    """
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return b
    return a


class IterationRecord:
    """
    Record relevant information at each individual iteration
    """

    def __init__(
        self,
        iteration,
        feasibility=None,
        objectiveValue=None,
        trustRadius=None,
        stepNorm=None,
    ):
        self.iteration = iteration
        self.fStep, self.thetaStep, self.rejected = [False] * 3
        if feasibility is not None:
            self.feasibility = feasibility
        if objectiveValue is not None:
            self.objectiveValue = objectiveValue
        if trustRadius is not None:
            self.trustRadius = trustRadius
        if stepNorm is not None:
            self.stepNorm = stepNorm

    def detailLogger(self):
        """
        Print information about the iteration to the log.
        """
        logger.info("****** Iteration %d ******" % self.iteration)
        logger.info("trustRadius = %s" % self.trustRadius)
        logger.info("feasibility = %s" % self.feasibility)
        logger.info("objectiveValue = %s" % self.objectiveValue)
        logger.info("stepNorm = %s" % self.stepNorm)
        if self.fStep:
            logger.info("INFO: f-type step")
        if self.thetaStep:
            logger.info("INFO: theta-type step")
        if self.rejected:
            logger.info("INFO: step rejected")

    def verboseLogger(self):
        """
        Print information about the iteration to the console
        """
        print("****** Iteration %d ******" % self.iteration)
        print("objectiveValue = %s" % self.objectiveValue)
        print("feasibility = %s" % self.feasibility)
        print("trustRadius = %s" % self.trustRadius)
        print("stepNorm = %s" % self.stepNorm)
        if self.fStep:
            print("INFO: f-type step")
        if self.thetaStep:
            print("INFO: theta-type step")
        if self.rejected:
            print("INFO: step rejected")
        print(25 * '*')


class IterationLogger:
    """
    Log (and print) information for all iterations
    """

    def __init__(self):
        self.iterations = []

    def newIteration(
        self, iteration, feasibility, objectiveValue, trustRadius, stepNorm
    ):
        """
        Add a new iteration to the list of iterations
        """
        self.iterrecord = IterationRecord(
            iteration,
            feasibility=feasibility,
            objectiveValue=objectiveValue,
            trustRadius=trustRadius,
            stepNorm=stepNorm,
        )
        self.iterations.append(self.iterrecord)

    def updateIteration(
        self, feasibility=None, objectiveValue=None, trustRadius=None, stepNorm=None
    ):
        """
        Update values in current record
        """
        if feasibility is not None:
            self.iterrecord.feasibility = feasibility
        if objectiveValue is not None:
            self.iterrecord.objectiveValue = objectiveValue
        if trustRadius is not None:
            self.iterrecord.trustRadius = trustRadius
        if stepNorm is not None:
            self.iterrecord.stepNorm = stepNorm

    def logIteration(self):
        """
        Log detailed information about the iteration to the log
        """
        self.iterrecord.detailLogger()

    def printIteration(self):
        """
        Print information to the screen
        """
        self.iterrecord.verboseLogger()
