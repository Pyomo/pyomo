#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m

"""
This example uses an iteration callback to halt the solver
"""


def iteration_callback(
    nlp,
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
    ls_trials,
):
    if iter_count >= 4:
        return False
    return True


def main():
    solver = pyo.SolverFactory('cyipopt')
    status = solver.solve(m, tee=False, intermediate_callback=iteration_callback)
    return status


if __name__ == '__main__':
    status = main()
    print(status)
