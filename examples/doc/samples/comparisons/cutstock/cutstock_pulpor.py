#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pulp import *
from cutstock_util import *

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

# Dictionary for PulpOR
CutsInPattern = makeDict([Cuts, Patterns], CutsInPattern)
CutDemand = makeDict([Cuts], CutDemand)

prob = LpProblem("CutStock Problem", LpMinimize)

# Defining Variables
SheetsCut = LpVariable("SheetsCut", 0)
TotalCost = LpVariable("TotalCost", 0)
PatternCount = LpVariable.dicts("PatternCount", Patterns, lowBound=0)
ExcessCuts = LpVariable.dicts("ExcessCuts", Cuts, lowBound=0)

# objective
prob += TotalCost, ""

# Constraints
prob += TotalCost == PriceSheet * SheetsCut, "TotCost"
prob += SheetsCut <= SheetsAvail, "RawAvail"
prob += PatternCount == SheetsCut, "Sheets"
for c in Cuts:
    prob += lpSum(
        [CutsInPattern[c][p] * PatternCount[p] for p in Patterns]
    ) == CutDemand[c] + ExcessCuts[c], "CutReq" + str(c)


# prob.writeLP("CutStock.lp")
prob.solve()
print("Status:", LpStatus[prob.status])
print("Minimum total cost:", prob.objective.value())
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
