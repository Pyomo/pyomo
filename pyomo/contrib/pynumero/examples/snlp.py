from pynumero.interfaces import TwoStageStochasticNLP
from pynumero.sparse import BlockMatrix, BlockSymMatrix
import matplotlib.pylab as plt
import pyomo.environ as aml
import numpy as np


def create_basic_dense_qp(G, A, b, c, complicated_var_ids):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.complicated_var_ids = complicated_var_ids
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)
    model.z = aml.Var(model.complicated_var_ids, initialize=0.0)
    model.hessian_f = aml.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = aml.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = aml.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = aml.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def fixing_constraints_rule(m, i):
        return m.z[i] == m.x[i]
    model.fixing_constraints = aml.Constraint(model.complicated_var_ids, rule=fixing_constraints_rule)

    def second_stage_cost_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.FirstStageCost = aml.Expression(expr=0.0)
    model.SecondStageCost = aml.Expression(rule=second_stage_cost_rule)

    model.obj = aml.Objective(expr=model.FirstStageCost + model.SecondStageCost,
                             sense=aml.minimize)


    return model

# Hessian
G = np.array([[36, 17, 19, 12, 8, 15],
               [17, 33, 18, 11, 7, 14],
               [19, 18, 43, 13, 8, 16],
               [12, 11, 13, 18, 6, 11],
               [8, 7, 8, 6, 9, 8],
               [15, 14, 16, 11, 8, 29]])

# jacobian
A = np.array([[7, 1, 8, 3, 3, 3],
               [5, 0, 5, 1, 5, 8],
               [2, 6, 7, 1, 1, 8],
               [1, 5, 0, 6, 1, 0]])

b = np.array([84, 62, 65, 1])
c = np.array([20, 15, 21, 18, 29, 24])

complicated_vars_ids = [4, 5]

scenarios = dict()
n_scenarios = 3
np.random.seed(seed=985739465)
bs = [b+np.random.normal(scale=20.0, size=1) for i in range(n_scenarios)]
for i in range(n_scenarios):
    instance = create_basic_dense_qp(G,
                                     A,
                                     bs[i],
                                     c,
                                     complicated_vars_ids)
    scenarios["s{}".format(i)] = instance


nlp = TwoStageStochasticNLP(scenarios, ["z"])

x = nlp.x_init
y = nlp.y_init
print("x vector")
print(x)
print("y vector")
print(y)

jac_g = nlp.jacobian_g(x)
print("Jacobian")
print(jac_g)
plt.title('Jacobian of the general constraints\n')
plt.spy(jac_g)
plt.show()

hess_lag = nlp.hessian_lag(x, y)
print("Hessian")
print(hess_lag)
plt.spy(hess_lag.tofullmatrix())
plt.title('Hessian of the Lagrangian function\n')
plt.show()

# Build KKT matrix
kkt = BlockSymMatrix(2)
kkt[0, 0] = hess_lag
kkt[1, 0] = jac_g
print("KKT")
print(kkt)
full_kkt = kkt.tofullmatrix()
plt.spy(full_kkt)
plt.title('Karush-Kuhn-Tucker Matrix\n')
plt.show()

