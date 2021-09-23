#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import numpy as np
from numpy.linalg import norm

class IterationLog:
    """
    Log relevant information at each individual iteration
    """
    # # Todo: Include the following in high printlevel
    #     for i in range(problem.ly):
    #         printmodel(romParam[i],problem.lx,problem.romtype)
    #
    # # Include the following in medium printlevel
    # print("romtype = ", problem.romtype)
    # print(romParam)
    # stepNorm

    def __init__(self, iteration, xk, yk, zk, verbosity):
        self.iteration = iteration
        self.xk = xk
        self.yk = yk
        self.zk = zk
        self.verbosity = verbosity
        # self.rmtype = None
        self.thetak = None
        self.objk = None
        self.trustRadius = None
        self.sampleRadius = None
        self.stepNorm = None
        self.fStep, self.thetaStep, self.rejected = [False]*3

    def setRelatedValue(self,thetak=None, objk=None,
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
            Low (1): Print iteration and variables
            Medium (2): Print iteration, variables, and param values
            High (3): Print all available information
        """
        if verbosity >= 1:
            print("\n**************************************")
            print("Iteration %d:" % self.iteration)
            print(np.concatenate([self.xk, self.yk, self.zk]))
        if verbosity >= 2:
            # print("Reduced Model Type: %s" % self.rmtype)
            print("thetak = %s" % self.thetak)
            print("objk = %s" % self.objk)
        if verbosity >= 3:
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

    def newIteration(self,iteration, xk, yk, zk,
                thetak, objk, verbosity):
        self.iterlog = IterationLog(iteration, xk, yk, zk, verbosity)
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
            dis = norm(np.concatenate([iteration.xk - self.iterlog.xk,
                                       iteration.yk - self.iterlog.yk,
                                       iteration.zk - self.iterlog.zk]), np.inf)
            print(iteration.iteration, iteration.thetak, iteration.objk,
                  iteration.trustRadius, iteration.sampleRadius,
                  iteration.stepNorm, dis, sep='\t')
            # print(str(x.iteration)+"\t"+str(x.thetak)+"\t"+str(x.objk)+"\t"+str(x.chik)+"\t"+str(x.trustRadius)+"\t"+str(x.sampleRadius)+"\t"+str(x.stepNorm)+"\t"+str(dis))
