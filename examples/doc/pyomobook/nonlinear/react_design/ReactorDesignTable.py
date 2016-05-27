import pyomo.environ
from pyomo.util import PyomoAPIData
from pyomo.core import *
import pyomo.scripting
from pyutilib.misc import Options

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

# setup the solver options
data = PyomoAPIData()
data.options = Options()
data.options.runtime = Options()
data.options.solvers = [Options()]
data.options.solvers[0]['solver_name'] = 'ipopt'
data.options.solvers[0]['suffixes'] = []
data.options.solvers[0]['options'] = Options()
data.options.postsolve = Options()
data.options.model = Options()
data.options.quiet = True

# run the sequence of square problems
instance = model
instance.sv.fixed = True
sv_values = [1.0 + v * 0.05 for v in range(1, 20)]
print("   %s %s" % (str('sv'.rjust(10)), str('cb'.rjust(10))))
for sv_value in sv_values:
    instance.sv = sv_value
    output = pyomo.scripting.util.apply_optimizer(data, instance=instance)
    #instance.load(output.results)
    print("    %s %s" %(str(instance.sv.value).rjust(10),\
        str(instance.cb.value).rjust(15)))
