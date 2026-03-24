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
    m.x = pyo.Var(bounds=(-20, 20))
    m.c = pyo.Constraint(expr=(m.x+7)*(m.x+5)*(m.x-4) + 200 == 0)
    return m


def main(method: ini.InitializationMethod):
    m = build_model()
    nlp_solver = SolverFactory('ipopt')
    global_solver = SolverFactory('scip_direct')
    mip_solver = SolverFactory('scip_direct')
    results = ini.initialize_nlp(
        nlp=m,
        nlp_solver=nlp_solver,
        mip_solver=mip_solver,
        global_solver=global_solver,
        method=method,
    )

    return results.solution_status, m.x.value


if __name__ == '__main__':
    stat, x = main(ini.InitializationMethod.global_opt)
    print(stat)
    print(round(x, 4))
