from pyomo.contrib.pynumero.interfaces import PyomoNLP
import pyomo.environ as aml


def create_model():
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m

model = create_model()
nlp = PyomoNLP(model)

# initial guesses
x = nlp.x_init()
y = nlp.y_init()

# NLP function evaluations
f = nlp.objective(x)
print("Objective Function\n", f)
df = nlp.grad_objective(x)
print("Gradient of Objective Function:\n", df)
jac_g = nlp.jacobian_g(x)
print("Jacobian of Constraints:\n", jac_g.toarray())
hess_lag = nlp.hessian_lag(x, y)
print("Hessian of Lagrangian\n", hess_lag.toarray())