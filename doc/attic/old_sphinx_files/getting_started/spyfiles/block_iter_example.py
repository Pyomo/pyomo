# written by jds, adapted for doc by dlw
from pyomo.environ import *

# simple way to get arbitrary, unique values for each thing 
val_iter = 0
def get_val(*args, **kwds):
    global val_iter
    val_iter += 1
    return val_iter

model = ConcreteModel()
model.I = RangeSet(3)
model.x = Var(initialize=get_val)
model.y = Var(model.I, initialize=get_val)

model.b = Block()
model.b.a = Var(initialize=get_val)
model.b.b = Var(model.I, initialize=get_val)

def c_rule(b,i):
    b.c = Var(initialize=get_val)
    b.d = Var(b.model().I, initialize=get_val)
model.c = Block([1,2], rule=c_rule)

model.pprint()

# @compprintloop
for v in model.component_objects(Var, descend_into=True):
    print("FOUND VAR:" + v.cname(True))
    v.pprint()

for v_data in model.component_data_objects(Var, descend_into=True):
    print("Found: "+v_data.cname(True)+", value = "+str(value(v_data)))
# @compprintloop
