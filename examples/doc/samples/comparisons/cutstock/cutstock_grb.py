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

from gurobipy import *
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

m = Model("CutStock")

# Defining Variables
SheetsCut = m.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "SheetsCut")
TotalCost = m.addVar(0, GRB.INFINITY, 1, GRB.CONTINUOUS, "TotCost")

PatternCount = []
for i in range(patcount):
    newvar = m.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, Patterns[i])
    PatternCount += [newvar]

ExcessCuts = []
for j in range(cutcount):
    newvar = m.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, Cuts[j])
    ExcessCuts += [newvar]

# Objective Sense
m.ModelSense = 1

# Update model to integrate variables
m.update()

# Defining Constraints
m.addConstr(LinExpr(PriceSheet, SheetsCut), GRB.EQUAL, TotalCost, "TotCostCalc")
m.addConstr(LinExpr(1, SheetsCut), GRB.LESS_EQUAL, SheetsAvail, "RawAvail")

sheetsB = LinExpr()
for i in range(patcount):
    sheetsB.addTerms(1, PatternCount[i])
m.addConstr(sheetsB, GRB.EQUAL, SheetsCut, "Sheets")

for c in range(cutcount):
    cutReqB = LinExpr()
    cutReqB.addTerms(-1, ExcessCuts[c])
    for p in range(patcount):
        cutReqB.addTerms(CutsInPattern[c][p], PatternCount[p])
    m.addConstr(cutReqB, GRB.EQUAL, CutDemand[c], "CutReq_")

m.update()
# m.write("CutStock.lp")
m.optimize()

print(m.ObjVal)
