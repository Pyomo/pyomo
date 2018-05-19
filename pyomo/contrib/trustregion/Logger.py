import numpy as np
from numpy.linalg import norm
from pyomo.contrib.trustregion.helper import *
from pyomo.contrib.trustregion.param import *

class IterLog:
    # # Todo: Include the following in high printlevel
    #     for i in range(problem.ly):
    #         printmodel(romParam[i],problem.lx,problem.romtype)
    #
    # # Include the following in medium printlevel
    # print("romtype = ", problem.romtype)
    # print(romParam)
    # stepNorm

    def __init__(self, iteration,xk,yk,zk):
        self.iteration = iteration
        self.xk = xk
        self.yk = yk
        self.zk = zk
        self.thetak = None
        self.objk = None
        self.chik = None
        self.trustRadius = None
        self.sampleRadius = None
        self.stepNorm = None
        self.fStep, self.thetaStep, self.rejected, self.restoration, self.criticality = [False]*5


    def setRelatedValue(self,thetak=None,objk=None,chik=None,trustRadius=None,sampleRadius=None,stepNorm=None):
        if thetak is not None:
            self.thetak = thetak
        if objk is not None:
            self.objk = objk
        if chik is not None:
            self.chik = chik
        if trustRadius is not None:
            self.trustRadius = trustRadius
        if sampleRadius is not None:
            self.sampleRadius = sampleRadius
        if stepNorm is not None:
            self.stepNorm = stepNorm


    def fprint(self):
        """
        TODO: set a PrintLevel param to control the print level.
        """
        print("\n**************************************")
        print("Iteration %d:" % self.iteration)
        if PRINT_VARS:
            print(packXYZ(self.xk, self.yk, self.zk))
        print("thetak = %e" % self.thetak)
        print("objk = %e" % self.objk)
        print("trustRadius = %e" % self.trustRadius)
        print("sampleRadius = %f" % self.sampleRadius)
        print("stepNorm = %f" % self.stepNorm)
        print("chi = %e" % self.chik)
        if self.fStep:
            print("f-type step")
        if self.thetaStep:
            print("theta-type step")
        if self.rejected:
            print("step rejected")
        if self.restoration:
            print("RESTORATION")
        if self.criticality:
            print("criticality test update")
        print("**************************************\n")


class Logger:
    iters = []
    def newIter(self,iteration,xk,yk,zk,thetak,objk,chik):
        self.iterlog = IterLog(iteration,xk,yk,zk)
        self.iterlog.setRelatedValue(thetak=thetak,objk=objk,chik=chik)
        self.iters.append(self.iterlog)
    def setCurIter(self,trustRadius=None,sampleRadius=None,stepNorm=None):
        self.iterlog.setRelatedValue(trustRadius=trustRadius,sampleRadius=sampleRadius,stepNorm=stepNorm)
    def printIteration(self,iteration):
        if(iteration<len(self.iters)):
            self.iters[iteration].fprint()
    def printVectors(self):
        for x in self.iters:
            dis = norm(packXYZ(x.xk-self.iterlog.xk,x.yk-self.iterlog.yk,x.zk-self.iterlog.zk),np.inf)
            print(str(x.iteration)+"\t"+str(x.thetak)+"\t"+str(x.objk)+"\t"+str(x.chik)+"\t"+str(x.trustRadius)+"\t"+str(x.sampleRadius)+"\t"+str(x.stepNorm)+"\t"+str(dis))
