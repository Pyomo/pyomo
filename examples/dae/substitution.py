from pyomo.environ import *

from pyomo.core.base import _ConstraintData, _ObjectiveData

from pyomo.core.base.expr import clone_expression

m = ConcreteModel()

m.s = Set(initialize=[1,2,3,4])

def _p_init(m, i):
    return i**2
m.p = Param(m.s, initialize= _p_init, mutable=True)

m.v = Var(m.s)
m.x = Var(m.s)

m.p2 = Param(initialize=3, mutable=True)
m.v2 = Var()

def _con(m, i):
    return m.p[i]*m.x[i] + m.p[i] == 10 + m.x[i]**2 
m.con = Constraint(m.s, rule=_con)

def _con2(m):
    return m.p2*m.x[1] + m.x[2]**2 == 10
m.con2 = Constraint(rule=_con2)

m.obj = Objective(expr=sum(m.x[i]*m.p[i] for i in m.s))

print('******* Before substitution')
m.pprint()

# Param Substitution map
# Params to be substituted must be mutable!!!
paramsublist = [m.p, m.p2]
paramsubmap = {id(m.p):m.v, id(m.p2):m.v2}

# Loop through components to build substitution map
variable_substitution_map = {}
for parameter in paramsublist:
    # Loop over each ParamData in the parameter (will this work on sparse params?)
    for k in parameter:
        p = parameter[k]
        print(k, p)
        variable_substitution_map[id(p)] = paramsubmap[id(parameter)][k]

# substitute the objectives/constraints
for component in m.component_objects(ctype=(Constraint,Objective), descend_into=True):
    for k in component:
        c = component[k]
        print(c.name)
        if isinstance(c, _ConstraintData):
            
            # Have to be really careful with upper and lower bounds of a constraint
            c._body = clone_expression(c._body, substitute=variable_substitution_map)
                    
        elif isinstance(c, _ObjectiveData):
            c.expr = clone_expression(c.expr, substitute=variable_substitution_map)

print('******* After substitution')
m.pprint()
