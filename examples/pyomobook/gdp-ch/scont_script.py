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
