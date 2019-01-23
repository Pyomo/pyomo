#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.interfaces import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_compositions import TwoStageStochasticNLP
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockVector
import matplotlib.pylab as plt
import pyomo.environ as aml
import numpy as np
import os

def create_basic_dense_qp(G, A, b, c):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.con_ids = range(nl)
    model.x = aml.Var(model.var_ids, initialize=0.0)
    
    def equality_constraint_rule(m, i):
        return sum(A[i, j] * m.x[j] for j in m.var_ids) == b[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def objective_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(G[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * c[j] for j in m.var_ids)
        return accum

    model.obj = aml.Objective(rule=objective_rule, sense=aml.minimize)

    return model

G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
A = np.array([[1, 0, 1], [0, 1, 1]])
b = np.array([3, 0])
c = np.array([-8, -3, -3])

models = []
scenarios = dict()
coupling_vars = dict()
n_scenarios = 5
np.random.seed(seed=985739465)
bs = [b+np.random.normal(scale=2.0, size=1) for i in range(n_scenarios)]

for i in range(n_scenarios):
    instance = create_basic_dense_qp(G,
                                     A,
                                     bs[i],
                                     c)

    nlp = PyomoNLP(instance)
    models.append(instance)
    scenario_name = "s{}".format(i)
    scenarios[scenario_name] = nlp
    coupling_vars[scenario_name] = [nlp.variable_idx(instance.x[0])]

nlp = TwoStageStochasticNLP(scenarios, coupling_vars)

x = nlp.x_init()
y = nlp.y_init()

jac_c = nlp.jacobian_c(x)
plt.spy(jac_c)
plt.title('Jacobian of the constraints\n')
plt.show()

hess_lag = nlp.hessian_lag(x, y)
plt.spy(hess_lag.tocoo())
plt.title('Hessian of the Lagrangian function\n')
plt.show()

kkt = BlockSymMatrix(2)
kkt[0, 0] = hess_lag
kkt[1, 0] = jac_c

plt.spy(kkt.tocoo())
plt.title('KKT system\n')
plt.show()
