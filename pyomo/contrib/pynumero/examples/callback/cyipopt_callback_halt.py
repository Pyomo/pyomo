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
