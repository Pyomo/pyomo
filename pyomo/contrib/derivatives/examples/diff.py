import pyomo.environ as pe
from pyomo.contrib.derivatives.differentiate import reverse_ad, reverse_sd

m = pe.ConcreteModel()
m.x = pe.Var(initialize=2)
m.y = pe.Var(initialize=3)
m.p = pe.Param(initialize=0.5, mutable=True)

e = pe.exp(m.x**m.p + 0.1*m.y)
derivs = reverse_ad(e)
print('dfdx: ', derivs[m.x])
print('dfdy: ', derivs[m.y])
print('dfdp: ', derivs[m.p])
derivs = reverse_sd(e)
print('dfdx: ', derivs[m.x])
print('dfdy: ', derivs[m.y])
print('dfdp: ', derivs[m.p])
