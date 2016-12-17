# @function:
import pyomo.environ as aml

def inventory_model(c, b, h, d):
    model = aml.ConcreteModel()
    model.x = aml.Var(bounds=(0,None))
    model.t = aml.Var()
    model.o = aml.Objective(expr= model.t)
    model.c1 = aml.Constraint(expr=
                model.t >= (c - b)*model.x + b*d)
    model.c2 = aml.Constraint(expr=
                model.t >= (c + h)*model.x + h*d)
    return model

solver = aml.SolverFactory("glpk")
for d in [25.0, 50.0, 75.0]:
    model = inventory_model(1.0, 1.5, 0.1, d)
    status = solver.solve(model)
    print("Objective: %s" % (model.o()))
# @:function

del solver
del model
del status

# @ef:
# build the extensive form model
ef = aml.ConcreteModel()
ef.o = aml.Objective(expr=0)
ef.x = aml.Var()
ef.c = aml.ConstraintList()
# @:ef

# @scenarios:
import random
N = 100
d_mean = 50.0
d_std = 25.0
for i in range(N):
    # create a random scenario
    d = random.normalvariate(d_mean, d_std)
    scenario = inventory_model(1.0, 1.5, 0.1, d)

    # add the scenario objective to the
    # ef objective with weight 1/N
    ef.o.expr += (1.0/N)*scenario.o

    # add binding constraint for x
    ef.c.add(scenario.x == ef.x)

    # add the scenario to the ef model
    name = "scenario"+str(i+1)
    setattr(ef, name, scenario)

    # deactivate objective on the scenario
    scenario.o.deactivate()
# @:scenarios

# @solve:
# solve the extensive form model
solver = aml.SolverFactory("glpk")
status = solver.solve(ef)
print("EF Objective: %s" % (ef.o()))
print("EF Solution: %s" % (ef.x()))

scenario1 = ef.scenario1
print("Scenario1 Objective: %s" % (scenario1.o()))
scenario1.o.activate()
status = solver.solve(scenario1)
print("Scenario1 Objective: %s" % (scenario1.o()))
# @:solve
