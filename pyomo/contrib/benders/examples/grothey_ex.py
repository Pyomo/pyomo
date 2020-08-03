#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, ComponentMap, SolverFactory, log


def create_master():
    m =  ConcreteModel()
    m.y =  Var(bounds=(1, None))
    m.eta =  Var(bounds=(-10, None))
    m.obj =  Objective(expr=m.y**2 + m.eta)
    return m


def create_subproblem(master):
    m =  ConcreteModel()
    m.x1 =  Var()
    m.x2 =  Var()
    m.y =  Var()
    m.obj =  Objective(expr=-m.x2)
    m.c1 =  Constraint(expr=(m.x1 - 1)**2 + m.x2**2 <=  log(m.y))
    m.c2 =  Constraint(expr=(m.x1 + 1)**2 + m.x2**2 <=  log(m.y))

    complicating_vars_map =  ComponentMap()
    complicating_vars_map[master.y] = m.y

    return m, complicating_vars_map


def main():
    m = create_master()
    master_vars = [m.y]
    m.benders = BendersCutGenerator()
    m.benders.set_input(master_vars=master_vars, tol=1e-8)
    m.benders.add_subproblem(subproblem_fn=create_subproblem,
                             subproblem_fn_kwargs={'master': m},
                             master_eta=m.eta,
                             subproblem_solver='ipopt', )
    opt =  SolverFactory('gurobi_direct')

    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        print(len(cuts_added), m.y.value, m.eta.value)
        if len(cuts_added) == 0:
            break


if __name__ == '__main__':
    main()
