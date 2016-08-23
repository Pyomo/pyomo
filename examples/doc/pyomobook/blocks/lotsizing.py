from pyomo.environ import *

model = ConcreteModel()
model.T = RangeSet(5)    # time periods

i_pos0 = 5.0       # initial inventory
i_neg0 = 0.0       # initial backlog
c = 4.6            # setup cost
h_pos = 0.7        # inventory holding cost
h_neg = 1.2        # shortage cost
P = 5.0            # maximum production amount

# demand during period t
d = {1: 5.0, 2:7.0, 3:6.2, 4:3.1, 5:1.7}

### create the model
# indicator of production in period t
model.y = Var(model.T, domain=Binary)
# amount to be produced in period t
model.x = Var(model.T, domain=NonNegativeReals)
# ending positive inventory in period t
model.i_pos = Var(model.T, domain=NonNegativeReals)
# ending negative inventory (backlogged) in period t
model.i_neg = Var(model.T, domain=NonNegativeReals)

# define the inventory relationships
def inventory_rule(m, t):
    if t == m.T.first():
        return m.i_pos[t] - m.i_neg[t] == i_pos0 - i_neg0 + m.x[t] - d[t]
    return m.i_pos[t] - m.i_neg[t] == m.i_pos[t-1] - m.i_neg[t-1] + m.x[t] - d[t]
model.inventory = Constraint(model.T, rule=inventory_rule)

# create the big-M constraint for the production indicator variable
def production_indicator_rule(m,t):
    return m.x[t] <= P*m.y[t]
model.production_indicator = Constraint(model.T, rule=production_indicator_rule)

# define the cost function
def obj_rule(m):
    return sum(c*m.y[t] + h_pos*m.i_pos[t] + h_neg*m.i_neg[t] for t in model.T)
model.obj = Objective(rule=obj_rule)

### solve the problem
solver = SolverFactory('glpk')
solver.solve(model)

### print the results
for v in model.y[:]:
    print(v.cname(), value(v))
    
