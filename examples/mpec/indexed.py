import coopr.environ
from coopr.pyomo import *
from coopr.mpec import *

n = 10

model = ConcreteModel()

model.x = Var(RangeSet(1,n))

model.f = Objective(expr=sum(i*(model.x[i]-1)**2 
                             for i in range(1,n+1)))

def compl_(model, i):
    return complements(model.x[i] >= 0, model.x[i+1] >= 0)
model.compl = Complementarity(RangeSet(1,n-1), rule=compl_)


model.pprint()
