#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from lpsolve55 import *
from cutstock_util import (
    getCutCount,
    getPatCount,
    getCuts,
    getPatterns,
    getSheetsAvail,
    getCutDemand,
    getPriceSheetData,
    getCutsInPattern,
)


# Reading in Data using the cutstock_util
cutcount = getCutCount()
patcount = getPatCount()
Cuts = getCuts()
Patterns = getPatterns()
PriceSheet = getPriceSheetData()
SheetsAvail = getSheetsAvail()
CutDemand = getCutDemand()
CutsInPattern = getCutsInPattern()
########################################

varcount = cutcount + patcount + 1 + 1
PatCountStart = 2

# Objective Coeff Array
ObjCoeff = range(varcount)
for i in range(varcount):
    if i == 0:
        ObjCoeff[i] = PriceSheet
    else:
        ObjCoeff[i] = 0

# Arrays for constraints
TotCostB = range(varcount)
for i in TotCostB:
    TotCostB[i] = 0
TotCostB[0] = -PriceSheet
TotCostB[1] = 1

RawAvailB = range(varcount)
for i in RawAvailB:
    RawAvailB[i] = 0
RawAvailB[0] = 1

SheetsB = range(varcount)
for i in SheetsB:
    SheetsB[i] = 0
SheetsB[0] = 1
for i in range(patcount):
    SheetsB[i + PatCountStart] = -1

CutReqB = [[0 for col in range(varcount)] for row in range(cutcount)]
for i in range(cutcount):
    for j in range(patcount):
        CutReqB[i][j + PatCountStart] = CutsInPattern[i][j]
    CutReqB[i][patcount + PatCountStart + i] = -1
###################################################

lp = lpsolve('make_lp', 0, varcount)
ret = lpsolve('set_lp_name', lp, 'CutStock')
lpsolve('set_verbose', 'CutStock', IMPORTANT)

# Define Objective
ret = lpsolve('set_obj_fn', 'CutStock', ObjCoeff)

# Define Constraints
ret = lpsolve('add_constraint', 'CutStock', TotCostB, EQ, 0)
ret = lpsolve('add_constraint', 'CutStock', RawAvailB, LE, SheetsAvail)
ret = lpsolve('add_constraint', 'CutStock', SheetsB, EQ, 0)
for i in range(cutcount):
    ret = lpsolve('add_constraint', 'CutStock', CutReqB[i], EQ, CutDemand[i])

lpsolve('solve', 'CutStock')
# ret = lpsolve('write_lp', 'CutStock', 'cutstock.lp')
lpsolve('solve', 'CutStock')
statuscode = lpsolve('get_status', 'CutStock')
print(lpsolve('get_statustext', 'CutStock', statuscode))
print(lpsolve('get_objective', 'CutStock'))
print(lpsolve('get_variables', 'CutStock')[0])
