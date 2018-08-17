from pyomo.contrib.pynumero.interfaces import PyomoNLP
import matplotlib.pylab as plt
import pyomo.environ as aml
import numpy as np


def create_basic_model():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], domain=aml.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = aml.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = aml.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = aml.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = aml.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = aml.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = aml.Objective(expr=m.x[2]**2)
    return m

model = create_basic_model()
solver = aml.SolverFactory('ipopt')
solver.solve(model, tee=True)

# build nlp initialized at the solution
nlp = PyomoNLP(model)

# get initial point
print(nlp.variable_order())
x0 = nlp.x_init

# compression matrices on x
Pxl = nlp.matrix_pxl()
Pxu = nlp.matrix_pxu()

# vectors of finite lower and upper bounds
xl = Pxl * nlp.xl
xu = Pxu * nlp.xu

# lower and upper inequalities residual
res_xl = Pxl * x0 - xl
res_xu = xu - Pxu * x0
print("Residuals lower bounds x-xl:", res_xl)
print("Residuals upper bounds xu-x:", res_xu)

# evaluate residual of equality constraints
print(nlp.constraint_order())
res_c = nlp.evaluate_c(x0)
print("Residuals equality constraints c(x):", res_c)
# evaluate residual of inequality constraints
d = nlp.evaluate_d(x0)

# compression matrices
Pdl = nlp.matrix_pdl()
Pdu = nlp.matrix_pdu()

# vectors of finite lower and upper bounds
dl = Pdl*nlp.dl
du = Pdu*nlp.du

# lower and upper inequalities residual
res_dl = Pdl*d - dl
res_du = du - Pdu*d
print("Residuals lower bounds d-dl:", res_dl)
print("Residuals upper bounds du-d:", res_du)

feasible = False
if np.all(res_xl>=0) and np.all(res_xu>=0) \
    and np.all(res_dl>=0) and np.all(res_du>=0) and \
    np.allclose(res_c, np.zeros(nlp.nc), atol=1e-5):
    feasible = True

print("Is x0 feasible:", feasible)

