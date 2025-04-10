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
from pyomo.repn import generate_standard_repn
import time

# @runtime
M = pyo.ConcreteModel()
M.A = pyo.RangeSet(100000)
M.p = pyo.Param(M.A, mutable=True, initialize=1)
M.x = pyo.Var(M.A)

start = time.time()
e = sum((M.x[i] - 1) ** M.p[i] for i in M.A)
print("sum:      %f" % (time.time() - start))

start = time.time()
generate_standard_repn(e)
print("repn:     %f" % (time.time() - start))

start = time.time()
e = pyo.quicksum((M.x[i] - 1) ** M.p[i] for i in M.A)
print("quicksum: %f" % (time.time() - start))

start = time.time()
generate_standard_repn(e)
print("repn:     %f" % (time.time() - start))

# @runtime
