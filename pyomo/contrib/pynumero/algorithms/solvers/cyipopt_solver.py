#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

try:
    import ipopt
except ImportError:
    raise ImportError('ipopt solver relies on cyipopt. Install cyipopt'+
                      ' https://github.com/matthias-k/cyipopt.git')
import numpy as np
from pyomo.contrib.pynumero.interfaces import PyomoNLP
import pyomo.environ as aml


class _CyIpoptProblem(object):

    def __init__(self, nlp):
        self._nlp = nlp

        if hasattr(nlp, 'nblocks'):
            raise NotImplementedError('Not supported for composite nlps yet')

        x = nlp.x_init()
        y = nlp.y_init()
        y.fill(1.0)

        # get structures
        self._df = nlp.grad_objective(x)
        self._g = nlp.evaluate_g(x, substract=False)
        self._jac_g = nlp.jacobian_g(x)
        self._hess_lag = nlp.hessian_lag(x, y)

    def objective(self, x):
        return self._nlp.objective(x)

    def gradient(self, x):
        self._nlp.grad_objective(x, out=self._df)
        return self._df

    def constraints(self, x):
        self._nlp.evaluate_g(x, out=self._g)
        return self._g

    def jacobian(self, x):
        self._nlp.jacobian_g(x, out=self._jac_g)
        return self._jac_g.data

    def hessianstructure(self):
        return self._hess_lag.row, self._hess_lag.col

    def jacobianstructure(self):
        return self._jac_g.row, self._jac_g.col

    def hessian(self, x, lagrange, obj_factor):
        self._nlp.hessian_lag(x, lagrange, out=self._hess_lag, eval_f_c=False)
        return self._hess_lag.data

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
    ):
        pass


class IpoptSolver(object):

    def __init__(self, nlp, options=None):

        self._nlp = nlp

        if hasattr(nlp, 'nblocks'):
            raise NotImplementedError('Not supported for composite nlps yet')

        self._problem = _CyIpoptProblem(nlp)
        self._options = options

    def solve(self):

        cyipopt_solver = ipopt.problem(n=nlp.nx,
                                       m=nlp.ng,
                                       problem_obj=self._problem,
                                       lb=nlp.xl(),
                                       ub=nlp.xu(),
                                       cl=nlp.gl(),
                                       cu=nlp.gu()
                                       )
        x0 = nlp.x_init()

        # this is needed until NLP hessian takes obj_factor as an input
        cyipopt_solver.addOption('nlp_scaling_method', 'none')
        x, info = cyipopt_solver.solve(x0)
        return x, info


def create_model1():
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], initialize=4.0)
    m.c = aml.Constraint(expr=m.x[3] ** 2 + m.x[1] == 25)
    m.d = aml.Constraint(expr=m.x[2] ** 2 + m.x[1] <= 18.0)
    # m.d = aml.Constraint(expr=aml.inequality(-18, m.x[2] ** 2 + m.x[1],  28))
    m.o = aml.Objective(expr=m.x[1] ** 4 - 3 * m.x[1] * m.x[2] ** 3 + m.x[3] ** 2 - 8.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model2():
    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2], initialize=4.0)
    m.d = aml.Constraint(expr=m.x[1] + m.x[2] <= 5)
    m.o = aml.Objective(expr=m.x[1] ** 2 + 4 * m.x[2] ** 2 - 8 * m.x[1] - 16 * m.x[2])
    m.x[1].setub(3.0)
    m.x[1].setlb(0.0)
    m.x[2].setlb(0.0)

    return m


def create_model4():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2], initialize=1.0)
    m.c1 = aml.Constraint(expr=m.x[1] + m.x[2] - 1 == 0)
    m.obj = aml.Objective(expr=2 * m.x[1] ** 2 + m.x[2] ** 2)
    return m


def create_model6():
    model = aml.ConcreteModel()

    model.S = [1, 2]
    model.x = aml.Var(model.S, initialize=1.0)

    def f(model):
        return model.x[1] ** 4 + (model.x[1] + model.x[2]) ** 2 + (-1.0 + aml.exp(model.x[2])) ** 2

    model.f = aml.Objective(rule=f)
    return model


def create_model9():

    # clplatea OXR2-MN-V-0
    model = aml.ConcreteModel()

    p = 71
    wght = -0.1
    hp2 = 0.5 * p ** 2

    model.x = aml.Var(aml.RangeSet(1, p), aml.RangeSet(1, p), initialize=0.0)

    def f(model):
        return sum(0.5 * (model.x[i, j] - model.x[i, j - 1]) ** 2 + \
                   0.5 * (model.x[i, j] - model.x[i - 1, j]) ** 2 + \
                   hp2 * (model.x[i, j] - model.x[i, j - 1]) ** 4 + \
                   hp2 * (model.x[i, j] - model.x[i - 1, j]) ** 4 \
                   for i in range(2, p + 1) for j in range(2, p + 1)) + (wght * model.x[p, p])

    model.f = aml.Objective(rule=f)

    for j in range(1, p + 1):
        model.x[1, j] = 0.0
        model.x[1, j].fixed = True

    return model

if __name__ == "__main__":

    model = create_model1()
    solver = aml.SolverFactory('ipopt')
    solver.options['nlp_scaling_method'] = 'none'
    #solver.solve(model, tee=True)

    nlp = PyomoNLP(model)
    print(nlp.x_init())
    solver = IpoptSolver(nlp)
    #solver.solve()
    x = nlp.x_init()
    jac = nlp.jacobian_g(x)
    print(jac.todense())
    jac+=jac
    print(jac.todense())




