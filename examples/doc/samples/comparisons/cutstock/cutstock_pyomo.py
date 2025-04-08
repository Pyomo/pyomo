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

import pyomo.environ as pyo
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
# CutsInPattern = makeDict([Cuts,Patterns],CutsInPattern)
tmp = {}
for i in range(len(Cuts)):
    tmp[Cuts[i]] = {}
    for j in range(len(CutsInPattern[i])):
        tmp[Cuts[i]][Patterns[j]] = CutsInPattern[i][j]
CutsInPattern = tmp
########################################
# CutDemand = makeDict([Cuts],CutDemand)
tmp = {}
for i in range(len(Cuts)):
    tmp[Cuts[i]] = CutDemand[i]
CutDemand = tmp

model = pyo.ConcreteModel(name="CutStock Problem")

# Defining Variables
model.SheetsCut = pyo.Var()
model.TotalCost = pyo.Var()
model.PatternCount = pyo.Var(Patterns, bounds=(0, None))
model.ExcessCuts = pyo.Var(Cuts, bounds=(0, None))

# objective
model.objective = pyo.Objective(expr=1.0 * model.TotalCost)

# Constraints
model.TotCost = pyo.Constraint(expr=model.TotalCost == PriceSheet * model.SheetsCut)
model.RawAvail = pyo.Constraint(expr=model.SheetsCut <= SheetsAvail)
model.Sheets = pyo.Constraint(expr=pyo.summation(model.PatternCount) == model.SheetsCut)
model.CutReq = pyo.Constraint(Cuts)
for c in Cuts:
    model.CutReq.add(
        c,
        expr=sum(CutsInPattern[c][p] * model.PatternCount[p] for p in Patterns)
        == CutDemand[c] + model.ExcessCuts[c],
    )

instance = model.create()
opt = pyo.SolverFactory('glpk')
results = opt.solve(instance)
instance.load(results)

print("Status:", results.solver.status)
print("Minimum total cost:", pyo.value(instance.objective))
for v in instance.variables():
    var = instance.variable(v)
    if pyo.value(var) > 0:
        print(v, "=", pyo.value(var))
