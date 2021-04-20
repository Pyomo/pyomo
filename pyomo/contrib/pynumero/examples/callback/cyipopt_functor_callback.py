import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
import pandas as pd

"""
This example uses an interation callback with a functor to store
values from each iteration in a class
"""
class ResidualsTableCallback(object):
    def __init__(self):
        self._residuals = None

    def __call__(self, nlp, alg_mod, iter_count, obj_value,
                 inf_pr, inf_du, mu, d_norm, regularization_size,
                 alpha_du, alpha_pr, ls_trials):
        constraint_names = nlp.constraint_names()
        if self._residuals is None:
            self._residuals = {nm: [] for nm in constraint_names}
            self._residuals['iter'] = []
        residuals = nlp.evaluate_constraints()
        for i,nm in enumerate(constraint_names):
            self._residuals[nm].append(residuals[i])
        self._residuals['iter'].append(iter_count)

    def get_residual_dataframe(self):
        return pd.DataFrame(self._residuals)

def main():
    solver = pyo.SolverFactory('cyipopt')
    resid_table_by_iter = ResidualsTableCallback()
    status = solver.solve(m, tee=False,
                          intermediate_callback=resid_table_by_iter)
    return resid_table_by_iter.get_residual_dataframe()

if __name__ == '__main__':
    df = main()
    print(df)
    

