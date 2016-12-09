from pyomo.environ import *

model = ConcreteModel()
model.T = RangeSet(5)    # time periods

i0 = 5.0           # initial inventory
c = 4.6            # setup cost
h_pos = 0.7        # inventory holding cost
h_neg = 1.2        # shortage cost
P = 5.0            # maximum production amount

# demand during period t
d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

# @vars:
# define the variables
model.y = Var(model.T, domain=Binary)
model.x = Var(model.T, domain=NonNegativeReals)
model.i = Var(model.T)
model.i_pos = Var(model.T, domain=NonNegativeReals)
model.i_neg = Var(model.T, domain=NonNegativeReals)
# @:vars

# define the inventory relationships
def inventory_rule(m, t):
    if t == m.T.first():
        return m.i[t] == i0 + m.x[t] - d[t]
    return m.i[t] == m.i[t-1] + m.x[t] - d[t]
model.inventory = Constraint(model.T, rule=inventory_rule)

def pos_neg_rule(m, t):
    return m.i[t] == m.i_pos[t] - m.i_neg[t]
model.pos_neg = Constraint(model.T, rule=pos_neg_rule)

# create the big-M constraint for the production indicator variable
def prod_indicator_rule(m,t):
    return m.x[t] <= P*m.y[t]
model.prod_indicator = Constraint(model.T, rule=prod_indicator_rule)

# define the cost function
def obj_rule(m):
    return sum(c*m.y[t] + h_pos*m.i_pos[t] + h_neg*m.i_neg[t] for t in m.T)
model.obj = Objective(rule=obj_rule)

# solve the problem
solver = SolverFactory('glpk')
solver.solve(model)

# print the results
for t in model.T:
    print('Period: {0}, Prod. Amount: {1}'.format(t, value(model.x[t]))) 

