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

import numpy as np
from numpy.linalg import norm
from numpy import array

def copyVector(x, y, z):
    """
    This function is to create a hard copy of vector x, y, z.
    """
    return array(x), array(y), array(z)

def minIgnoreNone(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return a
    return b

def maxIgnoreNone(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a < b:
        return b
    return a


class IterationLog:
    """
    Log relevant information at each individual iteration
    """

    def __init__(self, iteration, inputs, outputs, other, verbosity, rmtype, params):
        self.iteration = iteration
        self.inputs = inputs
        self.outputs = outputs
        self.other = other
        self.verbosity = verbosity
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

    def fprint(self, verbosity):
        """
        Print information about the iteration.
        
        verbosity parameter:
            None (0): No verbosity / print nothing
            Low (1): Print iteration and variables
            Medium (2): Print iteration, variables, and param values
            High (3): Print all available information
        """
        if verbosity >= 1:
            print("\n**************************************")
            print("Iteration %d:" % self.iteration)
            print(np.concatenate([self.inputs, self.outputs, self.other]))
        if verbosity >= 2:
            print("Reduced Model Type: %s" % self.rmtype)
            print("thetak = %s" % self.thetak)
            print("objk = %s" % self.objk)
        if verbosity >= 3:
            print("Reduced model parameters: %s" %self.rmParams)
            print("trustRadius = %s" % self.trustRadius)
            print("sampleRadius = %s" % self.sampleRadius)
            print("stepNorm = %s" % self.stepNorm)
            if self.fStep:
                print("INFO: f-type step")
            if self.thetaStep:
                print("INFO: theta-type step")
            if self.rejected:
                print("INFO: step rejected")
        if verbosity != 0:
            print("**************************************\n")


class Logger:

    iterations = []

    def newIteration(self, iteration, inputs, outputs, other,
                     thetak, objk, verbosity, rmtype, params):
        self.iterlog = IterationLog(iteration, inputs, outputs,
                                    other, verbosity, rmtype)
        self.iterlog.setRelatedValue(thetak=thetak, objk=objk)
        self.iterations.append(self.iterlog)

    def setCurrentIteration(self, trustRadius=None,
                            sampleRadius=None,
                            stepNorm=None):
        self.iterlog.setRelatedValue(trustRadius=trustRadius,
                                     sampleRadius=sampleRadius,
                                     stepNorm=stepNorm)

    def printIteration(self, iteration, verbosity):
        if(iteration < len(self.iterations)):
            self.iterations[iteration].fprint(verbosity)

    def printVectors(self):
        for iteration in self.iterations:
            dis = norm(np.concatenate([iteration.inputs - self.iterlog.inputs,
                                       iteration.outputs - self.iterlog.outputs,
                                       iteration.other - self.iterlog.other]),
                       np.inf)
            print(iteration.iteration, iteration.thetak, iteration.objk,
                  iteration.trustRadius, iteration.sampleRadius,
                  iteration.stepNorm, dis, sep='\t')
