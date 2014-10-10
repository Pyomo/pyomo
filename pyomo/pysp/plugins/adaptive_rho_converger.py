import logging
import copy
import math
import pprint

import coopr.core.plugin
from coopr.pysp import phextension
from coopr.pyomo.base import minimize
from coopr.opt import UndefinedData
from coopr.pysp.generators import scenario_tree_node_variables_generator_noinstances
from coopr.pysp.phutils import indexToString

from operator import itemgetter
from six import iteritems

from coopr.pysp.plugins.phboundextension import _PHBoundBase
from coopr.pyomo.base import Var
from coopr.pyomo import *
from coopr.opt import SolverFactory

from coopr.pysp.ef import create_ef_instance

logger = logging.getLogger('coopr.pysp')

class adaptive_rho_converger(coopr.core.plugin.SingletonPlugin, _PHBoundBase):

    coopr.core.plugin.implements(phextension.IPHExtension)

    coopr.core.plugin.alias("adaptive_rho_converger")

    def __init__(self):

        _PHBoundBase.__init__(self)

        self._rho_reduction_factor = 0.9
        self._rho_norm_convergence_tolerance = 1.0
        self._last_adjusted_iter = -1
        self._max_iters_without_change = 20

    def _compute_rho_norm(self, ph):
        rho_norm = 0.0
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                node_rho_norm = 0.0
                for variable_id in tree_node._standard_variable_ids:
                    name, index = tree_node._variable_ids[variable_id]
                    # rho is really a node parameter, so we only check on scenario
                    node_rho_norm += tree_node._scenarios[0]._rho[tree_node._name][variable_id]
                rho_norm += node_rho_norm * tree_node._probability
        return rho_norm

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self, ph):
        pass

    def post_ph_initialization(self, ph):
        pass

    def post_iteration_0_solves(self, ph):
        pass

    def post_iteration_0(self, ph):
        pass

    def pre_iteration_k_solves(self, ph):

        rho_updated = False
        #
        # Reduce rhos now that we've got a primal feasible solution
        #
        if ph._converger.isConverged(ph):
            rho_updated = True
            self._last_converged_iter = ph._current_iteration - 1
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    for variable_id in tree_node._standard_variable_ids:
                        name, index = tree_node._variable_ids[variable_id]
                        for scenario in tree_node._scenarios:
                            scenario._rho[tree_node._name][variable_id] *= self._rho_reduction_factor
                        if ph._verbose:
                            print(name+indexToString(index)+" rho updated: "+repr(tree_node._scenarios[0]._rho[tree_node._name][variable_id]))
        #
        # Give rhos's a jolt if primal convergence hasn't occured in a while
        #
        """
        elif (ph._current_iteration - self._last_adjusted_iter) > self._max_iters_without_change:
            self._last_adjusted_iter = ph._current_iteration
            rho_updated = True
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    for variable_id in tree_node._standard_variable_ids:
                        avg = 0.0
                        for scenario in tree_node._scenarios:
                            avg += math.fabs(scenario._w[tree_node._name][variable_id]) * \
                                   scenario._probability
                        name, index = tree_node._variable_ids[variable_id]
                        if ph._verbose:
                            print(name+indexToString(index)+" rho updated: "+repr(avg))
                        for scenario in tree_node._scenarios:
                            scenario._rho[tree_node._name][variable_id] = avg
        """
        if rho_updated:
            print("log(|rho|) = "+repr(math.log(self._compute_rho_norm(ph))))

    def post_iteration_k_solves(self, ph):
        pass

    def post_iteration_k(self, ph):
        pass

    def ph_convergence_check(self, ph):

        rho_norm = self._compute_rho_norm(ph)

        print("log(|rho|) = "+repr(math.log(rho_norm)))
        if rho_norm <= self._rho_norm_convergence_tolerance:
            print("Adaptive Rho Convergence Check Passed")
            return True
        print("Adaptive Rho Convergence Check Failed (requires log(|rho|) < "+repr(math.log(self._rho_norm_convergence_tolerance))+")")
        print("Continuing PH with updated Rho")
        return False

    def post_ph_execution(self, ph):
        pass
