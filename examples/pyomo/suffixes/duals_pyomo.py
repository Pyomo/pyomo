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

# A Suffix example for the collection of duals.

import pyomo.environ as pyo

### Create a trivial and infeasible example model
model = pyo.ConcreteModel()
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=model.x)
model.con = pyo.Constraint(expr=model.x >= 1)
###

# Declare an IMPORT Suffix to store the dual information that will
# be returned by the solver. When pyo.Suffix components are declared
# with an IMPORT direction, Pyomo solver interfaces will attempt to collect
# this named information from a solver solution.
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
