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

import logging
from collections import namedtuple
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint


logger = logging.getLogger("pyomo")


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


InfeasibilityTuple = namedtuple(
    "InfeasibilityTuple",
    [
        "x_L_viol",
        "x_U_viol",
        "compl_x_L",
        "compl_x_U",
        "dual_infeas",
        "primal_infeas",
        "compl_g",
    ],
)


class TrackingCallback(CyIpoptIntermediateCallbackBase):

    def __init__(self, components, scaled=False):
        self._variables = [comp for comp in components if comp.ctype is Var]
        self._constraints = [comp for comp in components if comp.ctype is Constraint]
        self._scaled = scaled
        self._var_to_idx = None
        self._con_to_idx = None
        self.iter_data = []

        self.infeasibilities = InfeasibilityTuple(
            *[
                ComponentMap((var, []) for var in self._variables),
                ComponentMap((var, []) for var in self._variables),
                ComponentMap((var, []) for var in self._variables),
                ComponentMap((var, []) for var in self._variables),
                ComponentMap((var, []) for var in self._variables),
                ComponentMap((con, []) for con in self._constraints),
                ComponentMap((con, []) for con in self._constraints),
            ]
        )


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

        if self._var_to_idx is None:
            self._var_to_idx = ComponentMap(
                (var, i) for i, var in enumerate(nlp.get_pyomo_variables())
            )
        if self._con_to_idx is None:
            self._con_to_idx = ComponentMap(
                (con, i) for i, con in enumerate(nlp.get_pyomo_constraints())
            )

        # TODO: Named tuple for basic iteration data
        self.iter_data.append(
            (
                alg_mod,
                iter_count,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            )
        )

        for var in self._variables:
            self.infeasibilities.x_L_viol[var].append(x_L_viol[self._var_to_idx[var]])
            self.infeasibilities.x_U_viol[var].append(x_U_viol[self._var_to_idx[var]])
            self.infeasibilities.compl_x_L[var].append(compl_x_L[self._var_to_idx[var]])
            self.infeasibilities.compl_x_U[var].append(compl_x_U[self._var_to_idx[var]])
            self.infeasibilities.dual_infeas[var].append(dual_infeas[self._var_to_idx[var]])
        for con in self._constraints:
            self.infeasibilities.primal_infeas[con].append(primal_infeas[self._con_to_idx[con]])
            self.infeasibilities.compl_g[con].append(compl_g[self._con_to_idx[con]])


class InfeasibilityCallback(CyIpoptIntermediateCallbackBase):
    """An intermediate callback for displaying the constraints and variables
    with the largest primal and dual infeasibilities at each iteration of
    an Ipopt solve

    """

    def __init__(self, infeasibility_threshold=1e8, n_residuals=5, scaled=False):
        self._infeasibility_threshold = infeasibility_threshold
        self._n_residuals = n_residuals
        self._scaled = scaled
        self._nlp = None
        self._variable_names = None
        self._constraint_names = None

        # These may be useful to access outside of a callback
        self._x_L_viol = None
        self._x_U_viol = None
        self._compl_x_L = None
        self._compl_x_U = None
        self._dual_infeas = None
        self._primal_infeas = None
        self._compl_g = None

        # Store dicts mapping names to the number of times implicated in each
        # infeasibility category.
        self.count = InfeasibilityTuple({}, {}, {}, {}, {}, {}, {})
        # Each item in these lists will be a tuple (iter, name, infeas)
        self.infeasibilities = InfeasibilityTuple([], [], [], [], [], [], [])

        # This is getting to be quite a lot to cache. Do we want to enforce that
        # we don't re-use this for different models?

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

        self._x_L_viol = x_L_viol
        self._x_U_viol = x_U_viol
        self._compl_x_L = compl_x_L
        self._compl_x_U = compl_x_U
        self._dual_infeas = dual_infeas
        self._primal_infeas = primal_infeas
        self._compl_g = compl_g

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

        # TODO: Be more explicit about caching nlp-specific information
        if self._variable_names is None or nlp is not self._nlp:
            self._variable_names = nlp.primals_names()
        if self._constraint_names is None or nlp is not self._nlp:
            self._constraint_names = nlp.constraint_names()
        if nlp is not self._nlp:
            # Re-set if we're solving a new model.
            self._nlp = nlp

        # Print new line to clearly separate this information from the
        # previous iteration.
        logger.info("")
        logger.info(f"INFEASIBILITIES FOR ITERATION {iter_count}")
        logger.info("==============================" + "=" * len(str(iter_count)))

        # TODO: Reduce repeated code here
        i_max_xL = sorted_coords_x_L_viol[0]
        if abs(x_L_viol[i_max_xL]) >= threshold:
            logger.info("Lower bound violation")
            logger.info("---------------------")
            logger.info(self._get_header())
            for i in sorted_coords_x_L_viol[: self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(x_L_viol[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.x_L_viol.append((iter_count, name, infeas))
            logger.info("---------------------")

        i_max_xU = sorted_coords_x_U_viol[0]
        if abs(x_U_viol[i_max_xU]) >= threshold:
            logger.info("Upper bound violation")
            logger.info("---------------------")
            logger.info(self._get_header())
            for i in sorted_coords_x_U_viol[: self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(x_U_viol[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.x_U_viol.append((iter_count, name, infeas))
            logger.info("---------------------")

        i_max_compl_xL = sorted_coords_compl_x_L[0]
        if abs(compl_x_L[i_max_compl_xL]) >= threshold:
            logger.info("Lower bound complementarity")
            logger.info("---------------------------")
            logger.info(self._get_header())
            for i in sorted_coords_compl_x_L[: self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(compl_x_L[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.compl_x_L.append((iter_count, name, infeas))
            logger.info("---------------------------")

        i_max_compl_xU = sorted_coords_compl_x_U[0]
        if abs(compl_x_U[i_max_compl_xU]) >= threshold:
            logger.info("Upper bound complementarity")
            logger.info("---------------------------")
            logger.info(self._get_header())
            for i in sorted_coords_compl_x_U[: self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(compl_x_U[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.compl_x_U.append((iter_count, name, infeas))
            logger.info("---------------------------")

        i_max_primal = sorted_coords_primal_infeas[0]
        if abs(primal_infeas[i_max_primal]) >= threshold:
            logger.info("Primal infeasibility")
            logger.info("--------------------")
            logger.info(self._get_header())
            for i in sorted_coords_primal_infeas[: self._n_residuals]:
                name = self._constraint_names[i]
                infeas = abs(primal_infeas[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.primal_infeas.append((iter_count, name, infeas))
            logger.info("--------------------")

        i_max_dual = sorted_coords_dual_infeas[0]
        if abs(dual_infeas[i_max_dual]) >= threshold:
            logger.info("Dual infeasibility")
            logger.info("------------------")
            logger.info(self._get_header())
            for i in sorted_coords_dual_infeas[: self._n_residuals]:
                name = self._variable_names[i]
                infeas = abs(dual_infeas[i])
                infeas_str = f"{infeas:.2e}"
                msg = infeas_str + " " + name
                logger.info(msg)
                self.infeasibilities.dual_infeas.append((iter_count, name, infeas))
            logger.info("------------------")

        logger.info("==============================" + "=" * len(str(iter_count)))

    def infeas_by_component(self):
        # Initialize an empty dict for every infeasibility category
        init = tuple({} for _ in self.infeasibilities)
        by_comp = InfeasibilityTuple(*init)
        for i, categ in enumerate(self.infeasibilities):
            for itr, name, infeas in categ:
                if name in by_comp[i]:
                    by_comp[i][name].append((itr, infeas))
                else:
                    by_comp[i][name] = [(itr, infeas)]
        return by_comp
