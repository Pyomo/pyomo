# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# === Required imports ===
import pyomo.environ as pyo
import pyomo.devel.initialization as ini
from pyomo.contrib.solver.common.factory import SolverFactory


def build_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-20, 20), initialize=-3.6)
    m.c = pyo.Constraint(expr=(m.x + 7) * (m.x + 5) * (m.x - 4) + 200 == 0)
    return m


def lp_init_ex():
    m = build_model()
    nlp_solver = SolverFactory('ipopt')
    lp_solver = SolverFactory('highs')
    results = ini.initialize_with_LP_approximation(
        nlp=m, nlp_solver=nlp_solver, lp_solver=lp_solver, seed=0
    )

    return results.solution_status, m.x.value


def pwl_init_ex():
    m = build_model()
    nlp_solver = SolverFactory('ipopt')
    mip_solver = SolverFactory('highs')
    results = ini.initialize_with_piecewise_linear_approximation(
        nlp=m, nlp_solver=nlp_solver, mip_solver=mip_solver
    )

    return results.solution_status, m.x.value


def global_init_ex():
    m = build_model()
    nlp_solver = SolverFactory('ipopt')
    global_solver = SolverFactory('scip_direct')
    results = ini.initialize_with_global_opt(
        nlp=m, nlp_solver=nlp_solver, global_solver=global_solver
    )

    return results.solution_status, m.x.value


if __name__ == '__main__':
    # stat, x = lp_init_ex()
    # stat, x = pwl_init_ex()
    stat, x = global_init_ex()
    print(stat, round(x, 4))
