from pyomo.environ import *
from pyomo.gdp import Disjunct, Disjunction
from six import iteritems

m= ConcreteModel()
m.x1= Var(domain=NonNegativeReals, bounds=(0, 8))
m.x2= Var(domain=NonNegativeReals, bounds=(0, 8))
m.c= Var(domain= NonNegativeReals, bounds=(1, 3))

m.y1= Disjunct(); m.y2= Disjunct(); m.y3= Disjunct()
m.y1.constr1 = Constraint(expr= m.x1**2 + m.x2**2 -1 <=0)
m.y1.constr2 = Constraint(expr= m.c==2)
m.y2.constr1 = Constraint(expr= (m.x1-4)**2+(m.x2-1)**2-1<=0)
m.y2.constr2 = Constraint(expr= m.c==1)
m.y3.constr1 = Constraint(expr= (m.x1-2)**2+(m.x2-4)**2-1<=0)
m.y3.constr2 = Constraint(expr= m.c==3)
m.GPD123 = Disjunction(expr=[m.y1, m.y2, m.y3])

m.obj= Objective(expr=(m.x1-3)**2 + (m.x2-2)**2 + m.c, sense = minimize)

# Option 1: Solve with Logic based OA
#opt = SolverFactory('gdpopt')
#result= opt.solve(m, strategy='LOA', 
##          mip='cbc',
#          tee=True
#          )

## Option 2: Tranform in MINLP and solve with solver
#TransformationFactory('gdp.chull').apply_to(m)
#opt = SolverFactory('gams', solver_io='direct')
#io_options = dict()
#opt.solve(m, 
##          mip='cbc',
#          tee=True,
#          solver='dicopt'
#          )
TransformationFactory('gdp.chull').apply_to(m)
m.pprint()
m.write('nonlinearGDP.gms', io_options=io_options)


