import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
import logging

"""
This example uses an interation callback to print the values
of the constraint residuals at each iteration of the CyIpopt
solver
"""
def iteration_callback(nlp, alg_mod, iter_count, obj_value,
                       inf_pr, inf_du, mu, d_norm, regularization_size,
                       alpha_du, alpha_pr, ls_trials):
    logger = logging.getLogger('pyomo')
    constraint_names = nlp.constraint_names()
    residuals = nlp.evaluate_constraints()
    logger.info('      ...Residuals for iteration {}'.format(iter_count))
    for i,nm in enumerate(constraint_names):
        logger.info('      ...{}: {}'.format(nm, residuals[i]))

def main():
    solver = pyo.SolverFactory('cyipopt')
    status, nlp = solver.solve(m, tee=False, return_nlp=True,
                           intermediate_callback=iteration_callback)

if __name__ == '__main__':
    logging.getLogger('pyomo').setLevel(logging.INFO)
    main()


