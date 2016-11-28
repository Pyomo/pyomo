from pyomo.environ import *

model = ConcreteModel()
model.T = RangeSet(5)    # time periods
model.S = RangeSet(5)

i0 = 5.0           # initial inventory
c = 4.6            # setup cost
h_pos = 0.7        # inventory holding cost
h_neg = 1.2        # shortage cost
P = 5.0            # maximum production amount

# demand during period t
d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

# @vars:
# define the variables
model.y = Var(domain=Binary)
model.x = Var(domain=NonNegativeReals)
model.i = Var()
model.i_pos = Var(domain=NonNegativeReals)
model.i_neg = Var(domain=NonNegativeReals)
# @:vars

model.pprint()
