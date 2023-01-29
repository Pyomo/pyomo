#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import numpy as np

class CyIpoptCallback(object):

    def __init__(
        self,
        primal_inf_threshold=1e8,
        dual_inf_threshold=1e8,
        n_residuals=5,
    ):
        self._primal_inf_threshold = primal_inf_threshold
        self._dual_inf_threshold = dual_inf_threshold
        self._n_residuals = n_residuals

        self._callback_count = 0
        self._prev_grad_obj = None
        self._prev_con_resids = None
        self._prev_jac_eq = None
        # prev_primals?

    def __call__(
        self,
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
        self._callback_count += 1
        if iter_count == 0:
            self._prev_grad_obj = nlp.evaluate_grad_objective()
            self._prev_con_resids = nlp.evaluate_constraints()
            self._prev_jac_eq = nlp.evaluate_jacobian_eq()
            return
        tab = "    "
        #if (
        #    inf_pr > self._primal_inf_threshold
        #    or inf_du > self._dual_inf_threshold
        #):
        #    print()
        #else:
        #    return
        print(
            "%sCallback at iter %s: inf_pr = %s, inf_du = %s"
            % (tab, iter_count, inf_pr, inf_du)
        )
        #if inf_pr > self._primal_inf_threshold:
        constraints = nlp.get_pyomo_constraints()
        residuals = self._prev_con_resids
        con_resids = list(zip(constraints, residuals))
        sorted_resids = sorted(con_resids, key=lambda item: abs(item[1]), reverse=True)
        top_n_resids = sorted_resids[:self._n_residuals]
        print("%s  %s largest primal residuals:" % (tab, self._n_residuals))
        for con, val in top_n_resids:
            print("%s    %s: %s" % (tab, con.name, val))

        #grad_lag = get_gradient_lagrangian(nlp)
        duals_eq = nlp.get_duals_eq()
        grad_lag = (
            self._prev_grad_obj
            + self._prev_jac_eq.transpose()*duals_eq
        )
        #if inf_du > self._dual_inf_threshold:
        variables = nlp.get_pyomo_variables()
        var_resids = list(zip(variables, grad_lag))
        sorted_resids = sorted(var_resids, key=lambda item: abs(item[1]), reverse=True)
        top_n_resids = sorted_resids[:self._n_residuals]
        print("%s  %s largest dual residuals:" % (tab, self._n_residuals))
        print("   ", max(abs(nlp.get_duals_eq())))
        for var, val in top_n_resids:
            print("%s    %s: %s" % (tab, var.name, val))

        #if self._prev_duals_eq is not None:
        #    grad_lag = get_gradient_lagrangian(nlp, duals_eq=self._prev_duals_eq)
        #    var_resids = list(zip(variables, np.abs(grad_lag)))
        #    sorted_resids = sorted(var_resids, key=lambda item: abs(item[1]), reverse=True)
        #    top_n_resids = sorted_resids[:self._n_residuals]
        #    print("%s  %s largest dual residuals (with previous duals):" % (tab, self._n_residuals))
        #    print("   ", max(abs(nlp.get_duals_eq())))
        #    for var, val in top_n_resids:
        #        print("%s    %s: %s" % (tab, var.name, val))

        self._prev_grad_obj = nlp.evaluate_grad_objective()
        self._prev_con_resids = nlp.evaluate_constraints()
        self._prev_jac_eq = nlp.evaluate_jacobian_eq()
        print()


def get_gradient_lagrangian(nlp, duals_eq=None):
    # Could primals and duals be getting out of sync here?
    # Primals are "correct" in the sense that they produce the inf_pr that
    # I expect. But duals are not necessarily coming from the same iteration.
    grad_obj = nlp.evaluate_grad_objective()
    jac_eq = nlp.evaluate_jacobian_eq()

    if duals_eq is None:
        duals_eq = nlp.get_duals_eq()

    grad_lag = (
        grad_obj
        + jac_eq.transpose()*duals_eq
    )
    return grad_lag
