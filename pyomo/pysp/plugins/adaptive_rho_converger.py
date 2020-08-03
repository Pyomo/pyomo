#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import copy
import math

from pyomo.common.plugin import implements, alias, SingletonPlugin
from pyomo.pysp import phextension
from pyomo.pysp.phutils import indexToString

logger = logging.getLogger('pyomo.pysp')

class _AdaptiveRhoBase(object):

    def __init__(self):

        self._tol = 1e-5
        self._required_converged_before_decrease = 0
        self._rho_converged_residual_decrease = 1.1
        self._rho_feasible_decrease = 1.25
        self._rho_decrease = 2.0
        self._rho_increase = 2.0
        self._log_rho_norm_convergence_tolerance = 1.0
        self._converged_count = 0
        self._last_adjusted_iter = -1
        self._stop_iter_rho_update = None
        self._prev_avg = None
        self._primal_residual_history = []
        self._dual_residual_history = []
        self._rho_norm_history = []

    def _compute_rho_norm(self, ph):
        rho_norm = 0.0
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                node_rho_norm = 0.0
                for variable_id in tree_node._standard_variable_ids:
                    name, index = tree_node._variable_ids[variable_id]
                    # rho is really a node parameter, so we only check
                    # one scenario
                    node_rho_norm += tree_node._scenarios[0].\
                                     _rho[tree_node._name][variable_id]
                rho_norm += node_rho_norm * tree_node._probability
        return rho_norm

    def _snapshot_avg(self, ph):
        self._prev_avg = {}
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                self._prev_avg[tree_node._name] = \
                    copy.deepcopy(tree_node._averages)

    def _compute_primal_residual_norm(self, ph):
        primal_resid = {}
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                node_primal_resid = primal_resid[tree_node._name] = \
                    dict((variable_id,0.0) for variable_id \
                         in tree_node._standard_variable_ids)
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        node_primal_resid[variable_id] += \
                            scenario._probability * \
                            (scenario._x[tree_node._name][variable_id] - \
                             tree_node._averages[variable_id])**2
        return primal_resid

    def _compute_dual_residual_norm(self, ph):
        dual_resid = {}
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                node_prev_avg = self._prev_avg[tree_node._name]
                dual_resid[tree_node._name] = \
                    dict((variable_id,
                          tree_node._scenarios[0].\
                          _rho[tree_node._name][variable_id]**2 * \
                          (tree_node._averages[variable_id] - \
                           node_prev_avg[variable_id])**2) \
                         for variable_id in tree_node._standard_variable_ids)
        return dual_resid

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self, ph):
        pass

    def post_ph_initialization(self, ph):
        self._stop_iter_rho_update = int(ph._max_iterations/2)

    def post_iteration_0_solves(self, ph):
        pass

    def post_iteration_0(self, ph):
        pass

    def pre_iteration_k_solves(self, ph):

        if (ph._current_iteration > self._stop_iter_rho_update) and \
           all(not _converger.isConverged(ph) for _converger in ph._convergers):
            return

        converged = any(_converger.isConverged(ph) for _converger in ph._convergers)

        rho_updated = False
        adjust_rho = 0
        if self._prev_avg is None:
            self._snapshot_avg(ph)
        else:
            self._primal_residual_history.append(
                self._compute_primal_residual_norm(ph))
            self._dual_residual_history.append(
                self._compute_dual_residual_norm(ph))
            self._snapshot_avg(ph)
            first_line = ("Updating Rho Values:\n%21s %25s %16s %16s %16s"
                          % ("Action",
                             "Variable",
                             "Primal Residual",
                             "Dual Residual",
                             "New Rho"))
            first = True
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    primal_resid = \
                        math.sqrt(sum(self._primal_residual_history[-1]\
                                      [tree_node._name].values()))
                    dual_resid = \
                        math.sqrt(sum(self._dual_residual_history[-1]\
                                      [tree_node._name].values()))
                    for variable_id in tree_node._standard_variable_ids:
                        name, index = tree_node._variable_ids[variable_id]
                        primal_resid = \
                            math.sqrt(self._primal_residual_history[-1]\
                                      [tree_node._name][variable_id])
                        dual_resid = \
                            math.sqrt(self._dual_residual_history[-1]\
                                      [tree_node._name][variable_id])

                        action = None
                        rho = tree_node._scenarios[0]._rho[tree_node._name][variable_id]
                        if (primal_resid > 10*dual_resid) and (primal_resid > self._tol):
                            rho *= self._rho_increase
                            action = "Increasing"
                        elif ((dual_resid > 10*primal_resid) and (dual_resid > self._tol)):
                            if self._converged_count >= self._required_converged_before_decrease:
                                rho /= self._rho_decrease
                                action = "Decreasing"
                        elif converged:
                            rho /= self._rho_feasible_decrease
                            action = "Feasible, Decreasing"
                        elif (primal_resid < self._tol) and (dual_resid < self._tol):
                            rho /= self._rho_converged_residual_decrease
                            action = "Converged, Decreasing"
                        if action is not None:
                            if first:
                                first = False
                                print(first_line)
                            print("%21s %25s %16g %16g %16g"
                                  % (action, name+indexToString(index),
                                     primal_resid, dual_resid, rho))
                            for scenario in tree_node._scenarios:
                                scenario._rho[tree_node._name][variable_id] = rho

        self._rho_norm_history.append(self._compute_rho_norm(ph))
        if rho_updated:
            print("log(|rho|) = "+repr(math.log(self._rho_norm_history[-1])))

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def ph_convergence_check(self, ph):

        self._converged_count += 1

        log_rho_norm = math.log(self._compute_rho_norm(ph))
        print("log(|rho|) = "+repr(log_rho_norm))
        if log_rho_norm <= self._log_rho_norm_convergence_tolerance:
            print("Adaptive Rho Convergence Check Passed")
            return True
        print("Adaptive Rho Convergence Check Failed "
              "(requires log(|rho|) < "+
              repr(self._log_rho_norm_convergence_tolerance)+")")
        print("Continuing PH with updated Rho")
        return False

    def post_ph_execution(self, ph):
        pass

class admm(SingletonPlugin,
           _AdaptiveRhoBase):

    implements(phextension.IPHExtension)

    alias("admm")

    def __init__(self):
        _AdaptiveRhoBase.__init__(self)

    def pre_ph_initialization(self,ph):
        _AdaptiveRhoBase.pre_ph_initialization(self, ph)

    def post_instance_creation(self, ph):
        _AdaptiveRhoBase.post_instance_creation(self, ph)

    def post_ph_initialization(self, ph):
        _AdaptiveRhoBase.post_ph_initialization(self, ph)

    def post_iteration_0_solves(self, ph):
        _AdaptiveRhoBase.post_iteration_0_solves(self, ph)

    def post_iteration_0(self, ph):
        _AdaptiveRhoBase.post_iteration_0(self, ph)

    def pre_iteration_k_solves(self, ph):
        _AdaptiveRhoBase.pre_iteration_k_solves(self, ph)

    def post_iteration_k_solves(self, ph):
        _AdaptiveRhoBase.post_iteration_k_solves(self, ph)

    def post_iteration_k(self, ph):
        _AdaptiveRhoBase.post_iteration_k(self, ph)

    def ph_convergence_check(self, ph):
        return _AdaptiveRhoBase.ph_convergence_check(self, ph)

    def post_ph_execution(self, ph):
        _AdaptiveRhoBase.post_ph_execution(self, ph)
