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

import pyomo.environ as pyo
import scont

model = scont.model

# @xfrm:
xfrm = pyo.TransformationFactory('gdp.bigm')
xfrm.apply_to(model)

solver = pyo.SolverFactory('glpk')
status = solver.solve(model)
# @:xfrm

print(status)
import verify_scont

verify_scont.verify_model(model)
