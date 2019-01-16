#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import CyIpoptSolver
from pyomo.contrib.pynumero.interfaces.nlp_compositions import DynoptNLP
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockVector
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import matplotlib.pylab as plt
import pyomo.environ as aml
import pyomo.dae as dae
import numpy as np
import os


def create_problem(begin, end, nfe, ncp, with_ic=True):

    m = aml.ConcreteModel()
    m.t = dae.ContinuousSet(bounds=(begin, end))

    m.x = aml.Var([1, 2], m.t, initialize=0.0)
    m.u = aml.Var(m.t, bounds=(None, 0.8), initialize=0)

    m.xdot = dae.DerivativeVar(m.x)

    def _x1dot(M, i):
        if i == M.t.first():
            return aml.Constraint.Skip
        return M.xdot[1, i] == (1-M.x[2, i] ** 2) * 1.0 * M.x[1, i] - M.x[2, i] + M.u[i]

    m.x1dotcon = aml.Constraint(m.t, rule=_x1dot)

    def _x2dot(M, i):
        if i == M.t.first():
            return aml.Constraint.Skip
        return M.xdot[2, i] == M.x[1, i]

    m.x2dotcon = aml.Constraint(m.t, rule=_x2dot)

    def _fake_bound(M, i):
        return M.x[2, i] + M.x[1, i] >= -100.0
    m.general_inequality = aml.Constraint(m.t, rule=_fake_bound)

    if with_ic:
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

    # Discretize model using Orthogonal Collocation
    discretizer = aml.TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=nfe, ncp=ncp, scheme='LAGRANGE-RADAU')
    discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

    return m

start = 0.0
end = 12.0
total_nfe = 24
ncp = 1
instance = create_problem(start, end, total_nfe, ncp)

opt = aml.SolverFactory("ipopt")
opt.solve(instance, tee=True)

times = sorted([t for t in instance.t])
x1 = [aml.value(instance.x[1, t]) for t in times]
x2 = [aml.value(instance.x[2, t]) for t in times]
plt.plot(times, x1, 'g')
plt.plot(times, x2, 'r')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot()
plt.show()

n_blocks = 3
nfe = total_nfe / n_blocks

blocks = list()
init_state_vars = list()
end_state_vars = list()
initial_conditions = [0.0, 1.0]
dt = (end - start) / float(n_blocks)
for i in range(n_blocks):
    begin = start + i * dt
    stop = start + (i + 1) * dt
    blk = create_problem(begin, stop, nfe, ncp, with_ic=False)
    t0 = blk.t.first()
    tf = blk.t.last()
    block_nlp = PyomoNLP(blk)
    blocks.append(block_nlp)
    init_states = [block_nlp.variable_idx(blk.x[1, t0]),
                   block_nlp.variable_idx(blk.x[2, t0])]
    init_state_vars.append(init_states)

    end_states = [block_nlp.variable_idx(blk.x[1, tf]),
                  block_nlp.variable_idx(blk.x[2, tf])]
    end_state_vars.append(end_states)

nlp = DynoptNLP(blocks, init_state_vars, end_state_vars, initial_conditions)

x = nlp.x_init()
y = nlp.y_init()

jac_g = nlp.jacobian_g(x)
plt.spy(jac_g.tocoo())
plt.title('Jacobian of the constraints\n')
plt.grid(True)
plt.show()

hess = nlp.hessian_lag(x, y)
plt.spy(hess.tocoo())
plt.title('Hessian of the Lagrangian function\n')
plt.grid(True)
plt.show()

kkt = BlockSymMatrix(2)
kkt[0, 0] = hess
kkt[1, 0] = jac_g

plt.spy(kkt.tocoo())
plt.title('KKT system\n')
plt.show()

opt = CyIpoptSolver(nlp)
x, info = opt.solve(tee=True)




