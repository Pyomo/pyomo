# ex1b.py
from pyomo.environ import *
from pyomo.mpec import *

n = 5

model = ConcreteModel()

model.x = Var( range(1,n+1) )

model.f = Objective(expr=sum(i*(model.x[i]-1)**2 
                    for i in range(1,n+1)) )

model.compl = ComplementarityList()
model.compl.add(complements(model.x[1]>=0, model.x[2]>=0))
model.compl.add(complements(model.x[2]>=0, model.x[3]>=0))
model.compl.add(complements(model.x[3]>=0, model.x[4]>=0))
model.compl.add(complements(model.x[4]>=0, model.x[5]>=0))
