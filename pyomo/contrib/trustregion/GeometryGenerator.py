#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is an auto geometry generator for quadratic ROM
import numpy as np

MAX_CONDITION=10000
NUM_ROM_SEEDS=5000
MAX_LX = 24

for lx in range(16,MAX_LX):
    dim =  (lx*lx+lx*3)/2 + 1
    x1 = np.zeros(lx)
    condOpt = MAX_CONDITION
    psetOpt = None
    matOpt = None
    for i in range(0,NUM_ROM_SEEDS):
        pset = np.random.multivariate_normal(x1,np.eye(lx),dim-1)
        for j in range(dim-1):
            pset[j] = pset[j]/np.linalg.norm(pset[j])
        pset = np.append(pset,[x1],axis=0)
        mat = []
        for p in pset:
            basisValue = [1]
            for i1 in range(lx):
                basisValue.append(p[i1])
            for i1 in range(lx):
                for i2 in range(i1,lx):
                    basisValue.append(p[i1]*p[i2])
            mat.append(basisValue)
        cond = np.linalg.cond(np.mat(mat))
        if(cond<condOpt):
            condOpt = cond
            psetOpt = pset
            matOpt = mat
    if(condOpt == MAX_CONDITION):
        print "Warning: lx = %d failed in initialization!\n" % lx
    else:
        np.savetxt('QradROMGeo/geo%d.out'% lx, psetOpt)
        print "Condition number: lx = %d is %f\n" % (lx,condOpt)




