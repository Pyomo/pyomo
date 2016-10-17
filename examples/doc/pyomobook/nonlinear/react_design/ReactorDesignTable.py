from pyomo.environ import *

# create the concrete model
model = ConcreteModel()

# set the data (native python data)
k1 = 5.0/6.0     # min^-1
k2 = 5.0/3.0     # min^-1
k3 = 1.0/6000.0  # m^3/(gmol min)
caf = 10000.0    # gmol/m^3

# create the variables
model.sv = Var(initialize = 1.0, within=PositiveReals)
model.ca = Var(initialize = 5000.0, within=PositiveReals)
model.cb = Var(initialize = 2000.0, within=PositiveReals)
model.cc = Var(initialize = 2000.0, within=PositiveReals)
model.cd = Var(initialize = 1000.0, within=PositiveReals)

# create the objective
model.obj = Objective(expr = model.cb, sense=maximize)

# create the constraints
model.ca_bal = Constraint(expr = (0 == model.sv * caf \
                 - model.sv * model.ca - k1 * model.ca \
                 -  2.0 * k3 * model.ca ** 2.0))

model.cb_bal = Constraint(expr=(0 == -model.sv * model.cb \
                 + k1 * model.ca - k2 * model.cb))

model.cc_bal = Constraint(expr=(0 == -model.sv * model.cc \
                 + k2 * model.cb))

model.cd_bal = Constraint(expr=(0 == -model.sv * model.cd \
                 + k3 * model.ca ** 2.0))

# run the sequence of square problems
solver = SolverFactory('ipopt')
model.sv.fixed = True
sv_values = [1.0 + v * 0.05 for v in range(1, 20)]
print("   %s %s" % (str('sv'.rjust(10)), str('cb'.rjust(10))))
for sv_value in sv_values:
    model.sv = sv_value
    solver.solve(model)
    print("    %s %s" %(str(model.sv.value).rjust(10),\
        str(model.cb.value).rjust(15)))
