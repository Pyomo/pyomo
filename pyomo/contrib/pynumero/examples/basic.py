#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import pyomo.environ as pyo


def create_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=4.0)
    m.c = pyo.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = pyo.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    m.o = pyo.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


model = create_model()
nlp = PyomoNLP(model)

# initial guesses
x = nlp.init_primals()
lam = nlp.init_duals()

nlp.set_primals(x)
nlp.set_duals(lam)

# NLP function evaluations
f = nlp.evaluate_objective()
print("Objective Function\n", f)
df = nlp.evaluate_grad_objective()
print("Gradient of Objective Function:\n", df)
c = nlp.evaluate_constraints()
print("Constraint Values:\n", c)
c_eq = nlp.evaluate_eq_constraints()
print("Equality Constraint Values:\n", c_eq)
c_ineq = nlp.evaluate_ineq_constraints()
print("Inequality Constraint Values:\n", c_ineq)
jac = nlp.evaluate_jacobian()
print("Jacobian of Constraints:\n", jac.toarray())
jac_eq = nlp.evaluate_jacobian_eq()
print("Jacobian of Equality Constraints:\n", jac_eq.toarray())
jac_ineq = nlp.evaluate_jacobian_ineq()
print("Jacobian of Inequality Constraints:\n", jac_ineq.toarray())
hess_lag = nlp.evaluate_hessian_lag()
print("Hessian of Lagrangian\n", hess_lag.toarray())
