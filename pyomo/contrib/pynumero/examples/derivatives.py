#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.pynumero.sparse import BlockMatrix
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import matplotlib.pylab as plt
import pyomo.environ as pyo
import pyomo.dae as dae


def create_problem(begin, end):
    m = pyo.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(begin, end))

    m.x = pyo.Var([1, 2], m.t, initialize=1.0)
    m.u = pyo.Var(m.t, bounds=(None, 0.8), initialize=0)

    m.xdot = dae.DerivativeVar(m.x)

    def _x1dot(M, i):
        if i == M.t.first():
            return pyo.Constraint.Skip
        return M.xdot[1, i] == (1-M.x[2, i] ** 2) * M.x[1, i] - M.x[2, i] + M.u[i]

    m.x1dotcon = pyo.Constraint(m.t, rule=_x1dot)

    def _x2dot(M, i):
        if i == M.t.first():
            return pyo.Constraint.Skip
        return M.xdot[2, i] == M.x[1, i]

    m.x2dotcon = pyo.Constraint(m.t, rule=_x2dot)

    def _init(M):
        t0 = M.t.first()
        yield M.x[1, t0] == 0
        yield M.x[2, t0] == 1
        yield pyo.ConstraintList.End

    m.init_conditions = pyo.ConstraintList(rule=_init)

    def _int_rule(M, i):
        return M.x[1, i] ** 2 + M.x[2, i] ** 2 + M.u[i] ** 2

    m.integral = dae.Integral(m.t, wrt=m.t, rule=_int_rule)

    m.obj = pyo.Objective(expr=m.integral)

    m.init_condition_names = ['init_conditions']
    return m


instance = create_problem(0.0, 10.0)
# Discretize model using Orthogonal Collocation
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(instance, nfe=100, ncp=3, scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(instance,
                                      var=instance.u,
                                      ncp=1,
                                      contset=instance.t)

# Interface pyomo model with nlp
nlp = PyomoNLP(instance)
x = nlp.create_new_vector('primals')
x.fill(1.0)
nlp.set_primals(x)

lam = nlp.create_new_vector('duals')
lam.fill(1.0)
nlp.set_duals(lam)

# Evaluate jacobian
jac = nlp.evaluate_jacobian()
plt.spy(jac)
plt.title('Jacobian of the constraints\n')
plt.show()

# Evaluate hessian of the lagrangian
hess_lag = nlp.evaluate_hessian_lag()
plt.spy(hess_lag)
plt.title('Hessian of the Lagrangian function\n')
plt.show()

# Build KKT matrix
kkt = BlockMatrix(2, 2)
kkt.set_block(0, 0, hess_lag)
kkt.set_block(1, 0, jac)
kkt.set_block(0, 1, jac.transpose())
plt.spy(kkt.tocoo())
plt.title('KKT system\n')
plt.show()
