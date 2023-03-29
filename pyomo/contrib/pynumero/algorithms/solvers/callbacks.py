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


class CyIpoptIntermediateCallbackBase(object):
    """A base class for CyIpopt intermediate callbacks that are compatible
    with CyIpoptProblemInterface

    Implementing callbacks with callable classes has the advantages of allowing
    easily configurable callbacks and caching computed information that may be
    useful in multiple calls.

    """

    def __call__(
        self,
        nlp,
        ipopt_problem,
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
        raise NotImplementedError("Subclasses must define the __call__ method")


class InfeasibilityCallback(CyIpoptIntermediateCallbackBase):
    """An intermediate callback for displaying the constraints and variables
    with the largest primal and dual infeasibilities at each iteration of
    an Ipopt solve

    """

    def __init__(
        self, infeasibility_threshold=1e8, n_residuals=5, scaled=False
    ):
        self._infeasibility_threshold = infeasibility_threshold
        self._n_residuals = n_residuals
        self._scaled = scaled
        self._variable_names = None
        self._constraint_names = None

        # TODO: Allow a custom file to write callback information to, and handle
        # output through a logger. This should probably go in the base class

    def _get_header(self):
        column_width = 8
        infeas_label = " infeas "
        name_label = "  name  "
        return infeas_label + name_label

    def __call__(
        self,
        nlp,
        ipopt_problem,
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
        infeas = ipopt_problem.get_current_violations(scaled=self._scaled)

        x_L_viol = infeas["x_L_violation"]
        x_U_viol = infeas["x_U_violation"]
        compl_x_L = infeas["compl_x_L"]
        compl_x_U = infeas["compl_x_U"]
        dual_infeas = infeas["grad_lag_x"]
        primal_infeas = infeas["g_violation"]
        compl_g = infeas["compl_g"]

        nx = len(x_L_viol)
        ng = len(primal_infeas)
        sorted_coords_x_L_viol = sorted(
            range(nx), key=lambda i: abs(x_L_viol[i]), reverse=True
        )
        sorted_coords_x_U_viol = sorted(
            range(nx), key=lambda i: abs(x_U_viol[i]), reverse=True
        )
        sorted_coords_compl_x_L = sorted(
            range(nx), key=lambda i: abs(compl_x_L[i]), reverse=True
        )
        sorted_coords_compl_x_U = sorted(
            range(nx), key=lambda i: abs(compl_x_U[i]), reverse=True
        )
        sorted_coords_dual_infeas = sorted(
            range(nx), key=lambda i: abs(dual_infeas[i]), reverse=True
        )
        sorted_coords_primal_infeas = sorted(
            range(ng), key=lambda i: abs(primal_infeas[i]), reverse=True
        )
        sorted_coords_compl_g = sorted(
            range(ng), key=lambda i: abs(compl_g[i]), reverse=True
        )

        threshold = self._infeasibility_threshold

        if self._variable_names is None:
            self._variable_names = nlp.primals_names()
        if self._constraint_names is None:
            self._constraint_names = nlp.constraint_names()

        # Print new line to clearly separate this information from the
        # previous iteration.
        print()

        # TODO: Reduce repeated code here
        i_max_xL = sorted_coords_x_L_viol[0]
        if abs(x_L_viol[i_max_xL]) >= threshold:
            print("Lower bound violation")
            print(self._get_header())
            for i in sorted_coords_x_L_viol[:self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(x_L_viol[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)

        i_max_xU = sorted_coords_x_U_viol[0]
        if abs(x_U_viol[i_max_xU]) >= threshold:
            print("Uppper bound violation")
            print(self._get_header())
            for i in sorted_coords_x_U_viol[:self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(x_U_viol[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)

        i_max_compl_xL = sorted_coords_compl_x_L[0]
        if abs(x_L_viol[i_max_compl_xL]) >= threshold:
            print("Lower bound complementarity")
            print(self._get_header())
            for i in sorted_coords_compl_x_L[:self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(compl_x_L[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)

        i_max_compl_xU = sorted_coords_compl_x_U[0]
        if abs(x_U_viol[i_max_xU]) >= threshold:
            print("Upper bound complementarity")
            print(self._get_header())
            for i in sorted_coords_compl_x_U[:self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(compl_x_U[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)

        i_max_primal = sorted_coords_primal_infeas[0]
        if abs(primal_infeas[i_max_primal]) >= threshold:
            print("Primal infeasibility")
            print(self._get_header())
            for i in sorted_coords_primal_infeas[:self._n_residuals]:
                name = self._constraint_names[i]
                infeas = abs(primal_infeas[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)

        i_max_dual = sorted_coords_dual_infeas[0]
        if abs(dual_infeas[i_max_dual]) >= threshold:
            print("Dual infeasibility")
            print(self._get_header())
            for i in sorted_coords_dual_infeas[:self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(dual_infeas[i])
                infeas_str = f"{infeas:.2e}"
                print(infeas_str, name)
