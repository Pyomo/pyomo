#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import numpy
import time
#x + W*x + 0.5*R*(x-XB)**2
#x >= L
#
#x + W*x + 0.5*R*(x-XB)**2 + l*(x-L)
#
#1 + W + R*(x-XB) + l = 0
#l*(x-L) = 0

L = 0
U = 10

def Faug(c, x, weight, rho, xbar):
    return c.dot(x) + weight.dot(x) + 0.5*rho*(x-xbar).dot((x-xbar))

def Fwaug(c, x, weight):
    return c.dot(x) + weight.dot(x)

def F(c, x):
    return c.dot(x)

def argminFaug(c,weight, rho, xbar):
    minx = xbar-((c+weight)/rho)
    return numpy.array([min(max(el,L),U) for el in minx])

def argminFwaug(c, weight):
    k = c+weight
    return numpy.array([U if (ki < 0) else L for ki in k])

def argminF(c):
    return numpy.array([U if (ci < 0) else L for ci in c])

def compute_xbar(x,a):
    return x.dot(a)

def update_weights(x, xbar, rho, w):
    return numpy.add(w,rho*(x-xbar))

c = numpy.array([1,-1,1])
#a = numpy.array([0.25,0.49,0.26])
a = numpy.array([1.0/3,1.0/3,1.0/3])

w = numpy.array([0,0,0])
rho = 1.0
print "RHO:", rho
x = argminF(c)
#x = numpy.array([0,0,0])
print "X:", x
while(1):
    xbar = compute_xbar(x,a)
    print "XBAR:", xbar
    w = update_weights(x, xbar, rho, w)
    print "W:", w
    faug = Faug(c, x, w, rho, xbar)
    print "FAUG:", faug
    fwaug = Fwaug(c, x, w)
    print "FWAUG:", fwaug
    f = F(c,x)
    print "F:", f
    print
    x = argminFaug(c, w, rho, xbar)
    print "X:", x
    #print "X(w)", argminFwaug(c, w)
    time.sleep(0.5)
