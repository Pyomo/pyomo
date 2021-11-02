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
from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available
)
if numpy_available:
    from numpy import array

logger = logging.getLogger('pyomo.contrib.trustregion')

def copyVector(x, y, z):
    """
    This function is to create a hard copy of vector x, y, z.
    """
    return array(x), array(y), array(z)

def minIgnoreNone(a, b):
    """
    Return minimum between two values, ignoring None unless both are None
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
    Return maximum between two values, ignoring None unless both are None
    """
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return b
    return a

def getVarDict(m, dict_keys=None):
    """
    Returns a dictionary of variables
    """
    return {name: m.find_component(name) for name in dict_keys}


class IterationRecord:
    """
    Record relevant information at each individual iteration
    """

    def __init__(self, iteration, inputs, outputs, other, rmtype, params):
        self.iteration = iteration
        self.inputs = inputs
        self.outputs = outputs
        self.other = other
        self.rmtype = rmtype
        self.rmParams = params
        self.thetak = None
        self.objk = None
        self.trustRadius = None
        self.sampleRadius = None
        self.stepNorm = None
        self.fStep, self.thetaStep, self.rejected = [False]*3

    def setRelatedValue(self, thetak=None, objk=None,
                        trustRadius=None,
                        sampleRadius=None, stepNorm=None):
        if thetak is not None:
            self.thetak = thetak
        if objk is not None:
            self.objk = objk
        if trustRadius is not None:
            self.trustRadius = trustRadius
        if sampleRadius is not None:
            self.sampleRadius = sampleRadius
        if stepNorm is not None:
            self.stepNorm = stepNorm

    def fprint(self):
        """
        Print information about the iteration to the log.
        """
        logger.info("**** Iteration %d ****" % self.iteration)
        logger.info(np.concatenate([self.inputs, self.outputs, self.other]))
        logger.info("Reduced Model Type: %s" % self.rmtype)
        logger.info("thetak = %s" % self.thetak)
        logger.info("objk = %s" % self.objk)
        logger.info("Reduced model parameters: %s" %self.rmParams)
        logger.info("trustRadius = %s" % self.trustRadius)
        logger.info("sampleRadius = %s" % self.sampleRadius)
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

    def newIteration(self, iteration, inputs, outputs, other,
                     thetak, objk, rmtype, params):
        self.iterrecord = IterationRecord(iteration, inputs, outputs,
                                       other, rmtype,
                                       params)
        self.iterrecord.setRelatedValue(thetak=thetak, objk=objk)
        self.iterations.append(self.iterrecord)

    def setCurrentIteration(self, trustRadius=None,
                            sampleRadius=None,
                            stepNorm=None):
        self.iterrecord.setRelatedValue(trustRadius=trustRadius,
                                     sampleRadius=sampleRadius,
                                     stepNorm=stepNorm)

    def printIteration(self, iteration):
        if (iteration < len(self.iterations)):
            self.iterations[iteration].fprint()
