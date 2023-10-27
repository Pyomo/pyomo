import pyomo.environ as pe
import coramin


def create_nlp(a, b):
    # Create the nlp
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-20.0, 20.0))
    m.y = pe.Var(bounds=(-20.0, 20.0))

    m.objective = pe.Objective(expr=(a - m.x)**2 + b*(m.y - m.x**2)**2)

    return m


def create_relaxation(a, b):
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-20.0, 20.0))
    m.x_sq = pe.Var()
    m.y = pe.Var(bounds=(-20.0, 20.0))
    m.z = pe.Var()

    m.objective = pe.Objective(expr=(a - m.x)**2 + b*m.z**2)
    m.con1 = pe.Constraint(expr=m.z == m.y - m.x_sq)
    m.x_sq_con = coramin.relaxations.PWXSquaredRelaxation()
    m.x_sq_con.build(x=m.x, aux_var=m.x_sq, use_linear_relaxation=True)

    return m


def main():
    a = 1
    b = 1
    nlp = create_nlp(a, b)
    rel = create_relaxation(a, b)

    nlp_opt = pe.SolverFactory('ipopt')
    rel_opt = pe.SolverFactory('gurobi_direct')

    res = nlp_opt.solve(nlp, tee=False)
    assert res.solver.termination_condition == pe.TerminationCondition.optimal
    ub = pe.value(nlp.objective)

    res = rel_opt.solve(rel, tee=False)
    assert res.solver.termination_condition == pe.TerminationCondition.optimal
    lb = pe.value(rel.objective)

    print('lb: ', lb)
    print('ub: ', ub)

    print('nlp results:')
    print('--------------------------')
    nlp.x.pprint()
    nlp.y.pprint()

    print('relaxation results:')
    print('--------------------------')
    rel.x.pprint()
    rel.y.pprint()


if __name__ == '__main__':
    main()
