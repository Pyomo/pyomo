#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Common utilities for Trust Region Framework
"""

import logging

logger = logging.getLogger('pyomo.contrib.trustregion')


class IterationRecord:
    """
    Record relevant information at each individual iteration
    """

    def __init__(self, iteration, feasibility=None, objectiveValue=None,
                 trustRadius=None, stepNorm=None):
        self.iteration = iteration
        self.fStep, self.thetaStep, self.rejected = [False]*3
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
        logger.info("**** Iteration %d ****" % self.iteration)
        logger.info("feasibility = %s" % self.feasibility)
        logger.info("objectiveValue = %s" % self.objectiveValue)
        logger.info("trustRadius = %s" % self.trustRadius)
        logger.info("stepNorm = %s" % self.stepNorm)
        if self.fStep:
            logger.info("INFO: f-type step")
        if self.thetaStep:
            logger.info("INFO: theta-type step")
        if self.rejected:
            logger.info("INFO: step rejected")


class IterationLogger:
    """
    Log (and print) information for all iterations
    """
    def __init__(self):
        self.iterations = []

    def newIteration(self, iteration, feasibility, objectiveValue,
                     trustRadius, stepNorm):
        """
        Add a new iteration to the list of iterations
        """
        self.iterrecord = IterationRecord(iteration,
                                          feasibility=feasibility,
                                          objectiveValue=objectiveValue,
                                          trustRadius=trustRadius,
                                          stepNorm=stepNorm)
        self.iterations.append(self.iterrecord)

    def logIteration(self):
        """
        Log detailed information about the iteration to the log
        """
        self.iterrecord.detailLogger()