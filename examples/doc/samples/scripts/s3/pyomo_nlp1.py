#
# pyomo_nlp1.py
#
# Adapted from the OpenOpt nlp_1.py example.
#
"""
Example:
(x0-5)^2 + (x2-5)^2 + ... +(x149-5)^2 -> min

subjected to

# lb<= x <= ub:
-6 <= x4 <= 4
8  <= x5 <= 15
  else
-6 <= x* <= 6

# Ax <= b
x9 + x19 <= 7
x10+x11 <= 9

# Aeq x = beq
x100+x101 = 11

# c(x) <= 0
2*x0^4-32 <= 0
x1^2+x2^2-8 <= 0

# h(x) = 0
(x[149]-1)**6 = 0
(x[148]-1.5)**6 = 0
"""


from pyomo.core import *
N = 150

model = ConcreteModel()

def x_init(model, i):
    return 8*cos(i)
model.x = Var(RangeSet(0, N-1), initialize=x_init,  bounds=(-6,6))
model.x[4].ub = 4
model.x[5].lb = 8
model.x[5].ub = 15

model.o = Objective(expr=sum((model.x[i]-5)**2 for i in model.x))

model.Aleq = ConstraintList()
model.Aleq.add( model.x[9] + model.x[19] <= 7 )
model.Aleq.add( model.x[10] + model.x[11] <= 9 )

model.Aeq = ConstraintList()
model.Aeq.add( model.x[100]+model.x[101] == 11)

model.c = ConstraintList()
model.c.add( 2*model.x[0]**4 - 32 <= 0 )
model.c.add( model.x[1]**2 + model.x[2]**2 - 8 <= 0 )

model.h = ConstraintList()
model.h.add( (model.x[149]-1)**6 == 0 )
model.h.add( (model.x[148]-1.5)**6 == 0 )


if __name__ == '__main__':
    #
    # Execute this when running this script interactively
    #
    instance = model
    S = Pyomo2FuncDesigner(instance)
    #
    r = S.minimize(S.f, S.initial_point, solver='ralg', contol=1e-7, gtol=1e-7, maxFunEvals=1e7, maxIter=1e7, iprint=50)
    tmp = {}
    for i in r.xf.keys():
        tmp[ int(i.name[1:]) ] = r.xf[i]
    for i in sorted(tmp.keys()):
        print(str(i)+' '+str(tmp[i]))

