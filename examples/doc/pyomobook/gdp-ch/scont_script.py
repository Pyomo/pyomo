from pyomo.environ import *
import scont

model = scont.model

# @xfrm:
xfrm = TransformationFactory('gdp.bigm')
xfrm.apply_to(model)

solver = SolverFactory('glpk')
status = solver.solve(model)
# @:xfrm

print(status)
import verify_scont
verify_scont.verify_model(model)
