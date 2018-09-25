from pyomo.contrib.pynumero.interfaces import PyomoNLP
from pyomo.contrib.pynumero.algorithms import NewtonSolver
import pyomo.environ as aml
import numpy as np


class MontecarloAnalysis(object):

    def __init__(self):
        pass

    def run(self, nlp, fixing_constraints, stds, n_samples, callback, options=None):
        """

        Parameters
        ----------
        nlp: PyomoNLP
        fixing_constraints: list
            List of constraints that fix parameters of montecarlo analysis
        stds: list
            List of standard deviations
        callback: callable
            function f(x) to evaluate in every sampling point
        options: dict
            dictionary with options for NewtonSolver

        """

        if nlp.nd > 0:
            raise RuntimeError("MontecarloAnalysis does not support problems with inequalities")

        if nlp.nx != nlp.nc:
            raise RuntimeError("MontecarloAnalysis only supports square systems")

        # find indices of constraints
        rhs_indices = list()
        for c in fixing_constraints:
            if not c.is_indexed():
                rhs_indices.append(nlp.constraint_idx(c))
            else:
                for cc in c:
                    rhs_indices.append(nlp.constraint_idx(cc))

        rhs_indices = np.array(rhs_indices)

        if len(rhs_indices) != len(stds):
            raise RuntimeError('Dimension missmatch')

        mean_values = np.copy(nlp._g_rhs[rhs_indices])
        covariance = np.diag(np.square(np.array(stds)))
        x0 = nlp.x_init()
        solver = NewtonSolver()
        results = list()
        for i in range(n_samples):
            nlp._g_rhs[rhs_indices] = np.random.multivariate_normal(mean_values,
                                                                    covariance)

            solution = solver.solve(nlp.jacobian_g,
                                    nlp.evaluate_g,
                                    x0)
            if solution[2] == 0:
                results.append(callback(solution[0]))
            else:
                print("Warning: sample did not converge. Ignoring it")
        return results

if __name__ == "__main__":

    import pyomo.environ as pe
    import pyomo.dae as dae
    model = pe.ConcreteModel()

    # process parameters
    model.R = pe.Param(initialize=5.5, mutable=True)
    model.t_end = pe.Param(initialize=100.0, mutable=True)

    # uncertain parameters
    #model.k1 = pe.Param(initialize=0.31051, mutable=True)
    #model.k2 = pe.Param(initialize=0.02665, mutable=True)

    model.k1 = pe.Var(initialize=0.31051)
    model.k2 = pe.Var(initialize=0.02665)

    # fixed parameters
    model.ca0 = pe.Param(initialize=5.3e-1)
    model.cc0 = pe.Param(initialize=0.0)
    model.cd0 = pe.Param(initialize=0.0)
    model.ce0 = pe.Param(initialize=0.0)

    model.t = dae.ContinuousSet(bounds=(0, 1.0))

    # Define variables
    model.ca = pe.Var(model.t)
    model.cb = pe.Var(model.t)
    model.cc = pe.Var(model.t)
    model.cd = pe.Var(model.t)
    model.ce = pe.Var(model.t)

    model.dca = dae.DerivativeVar(model.ca, wrt=model.t)
    model.dcb = dae.DerivativeVar(model.cb, wrt=model.t)
    model.dcc = dae.DerivativeVar(model.cc, wrt=model.t)
    model.dcd = dae.DerivativeVar(model.cd, wrt=model.t)
    model.dce = dae.DerivativeVar(model.ce, wrt=model.t)

    model.r1 = pe.Var(model.t)
    model.r2 = pe.Var(model.t)


    # model equations
    def _reaction1(m, i):
        return m.r1[i] == m.k1 * m.ca[i] * m.cb[i]


    model.r1_con = pe.Constraint(model.t, rule=_reaction1)


    def _reaction2(m, i):
        return m.r2[i] == m.k2 * m.cc[i]


    model.r2_con = pe.Constraint(model.t, rule=_reaction2)


    def _dcAdt(m, i):
        if i == m.t.first():
            return pe.Constraint.Skip
        return m.dca[i] == -m.r1[i] * m.t_end


    model.cA_con = pe.Constraint(model.t, rule=_dcAdt)


    def _dcBdt(m, i):
        if i == m.t.first():
            return pe.Constraint.Skip
        return m.dcb[i] == -m.r1[i] * m.t_end


    model.cB_con = pe.Constraint(model.t, rule=_dcBdt)


    def _dcCdt(m, i):
        if i == m.t.first():
            return pe.Constraint.Skip
        return m.dcc[i] == (m.r1[i] - m.r2[i]) * m.t_end


    model.cC_con = pe.Constraint(model.t, rule=_dcCdt)


    def _dcDdt(m, i):
        if i == m.t.first():
            return pe.Constraint.Skip
        return m.dcd[i] == m.r2[i] * m.t_end


    model.cD_con = pe.Constraint(model.t, rule=_dcDdt)


    def _dcEdt(m, i):
        if i == m.t.first():
            return pe.Constraint.Skip
        return m.dce[i] == m.r2[i] * m.t_end


    model.cE_con = pe.Constraint(model.t, rule=_dcEdt)


    def _init(m):
        yield m.ca[0] == m.ca0
        yield m.cb[0] == m.R * 5.3e-1
        yield m.cc[0] == m.cc0
        yield m.cd[0] == m.cd0
        yield m.ce[0] == m.ce0


    model.init_conditions = pe.ConstraintList(rule=_init)

    model.k1_constraint = pe.Constraint(expr=model.k1 == 0.31051)
    model.k2_constraint = pe.Constraint(expr=model.k2 == 0.02665)

    # Objective
    model.obj = pe.Objective(expr=0.0)

    tf = model.t.last()
    t0 = model.t.first()
    model.validate1 = pe.Constraint(expr=model.cd[tf] / (model.ca[t0] - model.ca[tf]) >= 0.9)
    model.validate1.deactivate()

    # Discretize the model
    discretizer = pe.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(model, nfe=500, wrt=model.t, scheme='BACKWARD')

    opt = pe.SolverFactory('ipopt')
    #opt.solve(model, tee=True)

    nlp = PyomoNLP(model)
    solver = NewtonSolver()
    sol = solver.solve(nlp.jacobian_g,
                       nlp.evaluate_g,
                       nlp.x_init(),
                       tee=True)
    x = sol[0]
    print(np.linalg.norm(x))
    fixing_c = [model.k1_constraint,
                model.k2_constraint]

    mc = MontecarloAnalysis()
    def f(x):
        return np.linalg.norm(x)
    results = mc.run(nlp, fixing_c, [1e-3, 1e-5], 10, callback=f)
    print(results)