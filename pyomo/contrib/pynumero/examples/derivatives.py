#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.sparse import BlockSymMatrix
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import matplotlib.pylab as plt
import pyomo.environ as aml
import pyomo.dae as dae


def create_problem(begin, end):

    m = aml.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(begin, end))

    m.x = aml.Var([1, 2], m.t, initialize=1.0)
    m.u = aml.Var(m.t, bounds=(None, 0.8), initialize=0)

    m.xdot = dae.DerivativeVar(m.x)

    def _x1dot(M, i):
        if i == M.t.first():
            return aml.Constraint.Skip
        return M.xdot[1, i] == (1-M.x[2, i] ** 2) * M.x[1, i] - M.x[2, i] + M.u[i]

    m.x1dotcon = aml.Constraint(m.t, rule=_x1dot)

    def _x2dot(M, i):
        if i == M.t.first():
            return aml.Constraint.Skip
        return M.xdot[2, i] == M.x[1, i]

    m.x2dotcon = aml.Constraint(m.t, rule=_x2dot)

    def _init(M):
        t0 = M.t.first()
        yield M.x[1, t0] == 0
        yield M.x[2, t0] == 1
        yield aml.ConstraintList.End

    m.init_conditions = aml.ConstraintList(rule=_init)

    def _int_rule(M, i):
        return M.x[1, i] ** 2 + M.x[2, i] ** 2 + M.u[i] ** 2

    m.integral = dae.Integral(m.t, wrt=m.t, rule=_int_rule)

    m.obj = aml.Objective(expr=m.integral)

    m.init_condition_names = ['init_conditions']
    return m

instance = create_problem(0.0, 10.0)
# Discretize model using Orthogonal Collocation
discretizer = aml.TransformationFactory('dae.collocation')
discretizer.apply_to(instance, nfe=100, ncp=3, scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(instance, var=instance.u, ncp=1, contset=instance.t)

# Interface pyomo model with nlp
nlp = PyomoNLP(instance)
x = nlp.create_vector_x()
lam = nlp.create_vector_y()

# Evaluate jacobian
jac_c = nlp.jacobian_g(x)
plt.spy(jac_c)
plt.title('Jacobian of the constraints\n')
plt.show()

# Evaluate hessian of the lagrangian
hess_lag = nlp.hessian_lag(x, lam)
plt.spy(hess_lag)
plt.title('Hessian of the Lagrangian function\n')
plt.show()

# Build KKT matrix
kkt = BlockSymMatrix(2)
kkt[0, 0] = hess_lag
kkt[1, 0] = jac_c
plt.spy(kkt.tocoo())
plt.title('KKT system\n')
plt.show()

