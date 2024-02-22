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

import cplex
from cutstock_util import *
from cplex.exceptions import CplexSolverError

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

indA = range(patcount + 1)
indA[0] = "SheetsCut"
for i in range(patcount):
    indA[i + 1] = Patterns[i]
valA = [1] + [-1 for i in range(patcount)]


indP = range(cutcount)
valP = range(cutcount)
for c in range(cutcount):
    count = 0
    for p in range(patcount):
        if CutsInPattern[c][p] >= 1:
            count += 1
    indP[c] = range(count + 1)
    valP[c] = range(count + 1)
    count = 0
    for p in range(patcount):
        if CutsInPattern[c][p] >= 1:
            indP[c][count] = Patterns[p]
            valP[c][count] = CutsInPattern[c][p]
            count += 1
    indP[c][count] = Cuts[c]
    valP[c][count] = -1

cpx = cplex.Cplex()

# Variable definition
cpx.variables.add(names=["SheetsCut"], lb=[0], ub=[cplex.infinity])
cpx.variables.add(names=["TotalCost"], lb=[0], ub=[cplex.infinity], obj=[1])
cpx.variables.add(names=Patterns)
cpx.variables.add(names=Cuts)

# objective
cpx.objective.set_sense(cpx.objective.sense.minimize)

# Constraints
cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(ind=["SheetsCut", "TotalCost"], val=[-PriceSheet, 1.0])],
    senses=["E"],
    rhs=[0],
)
cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(ind=["SheetsCut"], val=[1.0])],
    senses=["L"],
    rhs=[SheetsAvail],
)
cpx.linear_constraints.add(
    lin_expr=[cplex.SparsePair(ind=indA, val=valA)], senses=["E"], rhs=[0]
)
for c in range(cutcount):
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=indP[c], val=valP[c])],
        senses=["E"],
        rhs=[CutDemand[c]],
    )

# cpx.write("CutStock.lp")
cpx.solve()
numcols = cpx.variables.get_num()
x = cpx.solution.get_values()
print(cpx.solution.status[cpx.solution.get_status()])
print("Objective value  = ", cpx.solution.get_objective_value())
for j in range(numcols):
    if x[j] >= 1:
        print("Var:", j, "Value=", x[j])
