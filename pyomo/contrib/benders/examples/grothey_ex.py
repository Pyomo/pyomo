from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import pyomo.environ as pe
import math


def create_master():
    m = pe.ConcreteModel()
    m.y = pe.Var(bounds=(1, None))
    m.eta = pe.Var(bounds=(-10, None))
    m.obj = pe.Objective(expr=m.y**2 + m.eta)
    return m


def create_subproblem(master):
    m = pe.ConcreteModel()
    m.x1 = pe.Var()
    m.x2 = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=-m.x2)
    m.c1 = pe.Constraint(expr=(m.x1 - 1)**2 + m.x2**2 <= pe.log(m.y))
    m.c2 = pe.Constraint(expr=(m.x1 + 1)**2 + m.x2**2 <= pe.log(m.y))

    complicating_vars_map = pe.ComponentMap()
    complicating_vars_map[master.y] = m.y

    return m, complicating_vars_map


def main():
    m = create_master()
    master_vars = [m.y]
    m.benders = BendersCutGenerator()
    m.benders.set_input(master_vars=master_vars, subproblem_solver='ipopt', tol=1e-8)
    m.benders.add_subproblem(subproblem_fn=create_subproblem, subproblem_fn_kwargs={'master': m}, master_eta=m.eta)
    opt = pe.SolverFactory('gurobi_direct')

    for i in range(30):
        res = opt.solve(m, tee=False)
        num_cuts_added = m.benders.generate_cut()
        print(num_cuts_added, m.y.value, m.eta.value)
        if num_cuts_added == 0:
            break


if __name__ == '__main__':
    main()
