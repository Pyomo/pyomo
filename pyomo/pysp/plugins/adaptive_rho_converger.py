import logging
import copy
import math

import pyomo.util.plugin
from pyomo.pysp import phextension
from pyomo.pysp.phutils import indexToString

from pyomo.pysp.plugins.phboundextension import _PHBoundExtensionImpl
from pyomo.pysp.plugins.phhistoryextension import _dump_to_history, \
    extract_convergence, extract_scenario_tree_structure, \
    extract_scenario_solutions, extract_node_solutions

logger = logging.getLogger('pyomo.pysp')

class _AdaptiveRhoBase(object):

    def __init__(self):

        self._required_converged_before_decrease = 0
        self._rho_converged_residual_decrease = 1.0
        self._rho_feasible_decrease = 1.1
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
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    adjust_rho = 0
                    primal_resid = \
                        math.sqrt(sum(self._primal_residual_history[-1]\
                                      [tree_node._name].values()))
                    dual_resid = \
                        math.sqrt(sum(self._dual_residual_history[-1]\
                                      [tree_node._name].values()))
                    #print(primal_resid, min(self._primal_residual_history[-1][tree_node._name].values()), max(self._primal_residual_history[-1][tree_node._name].values()))
                    #print(dual_resid, min(self._dual_residual_history[-1][tree_node._name].values()), max(self._dual_residual_history[-1][tree_node._name].values()))
                    for variable_id in tree_node._standard_variable_ids:
                        name, index = tree_node._variable_ids[variable_id]
                        primal_resid = \
                            math.sqrt(self._primal_residual_history[-1]\
                                      [tree_node._name][variable_id])
                        dual_resid = \
                            math.sqrt(self._dual_residual_history[-1]\
                                      [tree_node._name][variable_id])
                        tol = 1e-5
                        #if converged:
                        #    adjust_rho = -2
                        #    rho_updated = True
                        if (primal_resid > 10*dual_resid) and (primal_resid > tol):
                            adjust_rho = 1
                            print("INCREASE %s, %s, %s" % (name+indexToString(index),
                                                           primal_resid, dual_resid))
                            rho_updated = True
                        elif ((dual_resid > 10*primal_resid) and (dual_resid > tol)):
                            if self._converged_count >= self._required_converged_before_decrease:
                                adjust_rho = -1
                                print("DECREASE %s, %s, %s" % (name+indexToString(index),
                                                               primal_resid, dual_resid))
                                rho_updated = True
                        elif converged:
                            adjust_rho = -2
                            rho_updated = True
                            print("FEASIBLE, DECREASING %s, %s, %s" % (name+indexToString(index),
                                                                       primal_resid, dual_resid))
                        elif (primal_resid < tol) and (dual_resid < tol):
                            adjust_rho = -3
                            rho_updated = True
                            #print("RESIDUALS CONVERGED %s, %s" % (primal_resid, dual_resid))
                        else:
                            print("NO CHANGE %s, %s, %s" % (name+indexToString(index),
                                                           primal_resid, dual_resid))
                        for scenario in tree_node._scenarios:
                            rho = scenario._rho[tree_node._name][variable_id]
                            rho_old = rho
                            if adjust_rho == -3:
                                rho /= self._rho_converged_residual_decrease
                            if adjust_rho == -2:
                                rho /= self._rho_feasible_decrease
                            elif adjust_rho == 1:
                                rho *= self._rho_increase
                            elif adjust_rho == -1:
                                rho /= self._rho_decrease
                            #scenario._rho[tree_node._name][variable_id] = 0.5*rho_old + 0.5*rho
                            scenario._rho[tree_node._name][variable_id] = rho
                        #if ph._verbose:
                        #    print(name+indexToString(index)+" rho updated: "+
                        #          repr(tree_node._scenarios[0].\
                        #               _rho[tree_node._name][variable_id]))
            #self._rho_factor += (1.0 - self._rho_factor)/25
        self._rho_norm_history.append(self._compute_rho_norm(ph))
        if rho_updated:
            print("log(|rho|) = "+repr(math.log(self._rho_norm_history[-1])))

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def ph_convergence_check(self, ph):

        self._converged_count += 1
        #if abs(ph._incumbent_cost_history[ph._current_iteration]-ph._bound_history[ph._current_iteration-1])/(1e-10+abs(0.5*ph._incumbent_cost_history[ph._current_iteration])) < 0.0001:
        #    return True
        return False

        print("log(|rho|) = "+repr(math.log(rho_norm)))
        if rho_norm <= self._log_rho_norm_convergence_tolerance:
            print("Adaptive Rho Convergence Check Passed")
            return True
        print("Adaptive Rho Convergence Check Failed (requires log(|rho|) < "+
              repr(math.log(self._log_rho_norm_convergence_tolerance))+")")
        print("Continuing PH with updated Rho")
        return False

    def post_ph_execution(self, ph):
        pass

class admm(pyomo.util.plugin.SingletonPlugin,
           _PHBoundExtensionImpl,
           _AdaptiveRhoBase):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("admm")

    def __init__(self):
        _PHBoundExtensionImpl.__init__(self)
        _AdaptiveRhoBase.__init__(self)
        self.save_filename = "admm.db"
        self._history_started = False

    def _prepare_history_file(self, ph):
        if not self._history_started:
            data = extract_scenario_tree_structure(ph._scenario_tree)
            _dump_to_history(self.save_filename,
                             data,
                             'scenario tree',
                             first=True)
            self._history_started = True

    def _snapshot_all(self, ph):
        data = {}
        data['convergence'] = extract_convergence(ph)
        data['scenario solutions'] = \
            extract_scenario_solutions(ph._scenario_tree,
                                       include_ph_objective_parameters=True,
                                       include_leaf_stage_vars=False)
        data['node solutions'] = \
            extract_node_solutions(ph._scenario_tree,
                                   include_ph_objective_parameters=True,
                                   include_variable_statistics=True,
                                   include_leaf_stage_vars=False)

        node_sol = data['node solutions'][ph._scenario_tree.findRootNode()._name]
        node_sol['outer bound history'] = self._outer_bound_history
        node_sol['inner bound history'] = self._inner_bound_history
        node_sol['incumbent cost history'] = ph._incumbent_cost_history
        node_sol['primal residual history'] = self._primal_residual_history
        node_sol['dual residual history'] = self._dual_residual_history
        node_sol['rho norm history'] = self._rho_norm_history

        return data

    def pre_ph_initialization(self,ph):
        _PHBoundExtensionImpl.pre_ph_initialization(self, ph)
        _AdaptiveRhoBase.pre_ph_initialization(self, ph)

    def post_instance_creation(self, ph):
        _PHBoundExtensionImpl.post_instance_creation(self, ph)
        _AdaptiveRhoBase.post_instance_creation(self, ph)

    def post_ph_initialization(self, ph):
        _PHBoundExtensionImpl.post_ph_initialization(self, ph)
        _AdaptiveRhoBase.post_ph_initialization(self, ph)

    def post_iteration_0_solves(self, ph):
        _PHBoundExtensionImpl.post_iteration_0_solves(self, ph)
        _AdaptiveRhoBase.post_iteration_0_solves(self, ph)

    def post_iteration_0(self, ph):
        _PHBoundExtensionImpl.post_iteration_0(self, ph)
        _AdaptiveRhoBase.post_iteration_0(self, ph)

    def pre_iteration_k_solves(self, ph):
        _PHBoundExtensionImpl.pre_iteration_k_solves(self, ph)
        _AdaptiveRhoBase.pre_iteration_k_solves(self, ph)

        self._prepare_history_file(ph)
        key = str(ph._current_iteration - 1)
        data = self._snapshot_all(ph)
        _dump_to_history(self.save_filename, data, key)

    def post_iteration_k_solves(self, ph):
        _PHBoundExtensionImpl.post_iteration_k_solves(self, ph)
        _AdaptiveRhoBase.post_iteration_k_solves(self, ph)

    def post_iteration_k(self, ph):
        _PHBoundExtensionImpl.post_iteration_k(self, ph)
        _AdaptiveRhoBase.post_iteration_k(self, ph)

    def ph_convergence_check(self, ph):
        #_PHBoundExtensionImpl.ph_convergence_check(self, ph)
        _AdaptiveRhoBase.ph_convergence_check(self, ph)

    def post_ph_execution(self, ph):
        _PHBoundExtensionImpl.post_ph_execution(self, ph)
        _AdaptiveRhoBase.post_ph_execution(self, ph)

        self._prepare_history_file(ph)
        key = str(ph._current_iteration)
        data = self._snapshot_all(ph)
        _dump_to_history(self.save_filename, data, key, last=True)
        print("Results saved to file: "+self.save_filename)
