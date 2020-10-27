#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
from math import fabs

from pyomo.pysp.generators import \
    scenario_tree_node_variables_generator_noinstances

from six import iterkeys

#
# This module contains a hierarchy of convergence "computers" for PH
# (or any other scenario-based decomposition strategy). Their basic
# function is to compute some measure of convergence among disparate
# scenario solutions, and to track the corresponding history of the
# metric. the sole inputs are a scenario tree and a (time-varying) set
# of instances (with solutions).
#

class ConvergenceBase(object):

    """ Constructor
        Arguments:
           convergence_threshold      numeric threshold at-or-below which a set
                                      of scenario solutions is considered
                                      converged. (must be >= 0.0)
    """
    def __init__(self, *args, **kwds):

        # human-readable name of the converger
        self._name = ""

        # key is the iteration number, passed in via the update()
        # method.
        self._metric_history = {}

        # the largest iteration key thus far - we assume continugous
        # values from 0.
        self._largest_iteration_key = 0

        # at what point do I consider the scenario solution pool
        # converged?
        self._convergence_threshold = 0.0

        # test for <= or >= the convergence threshold?
        self._test_for_le_threshold = True

        for key in kwds:
            if key == "convergence_threshold":
                self._convergence_threshold = kwds[key]
            elif key == "convergence_threshold_sense":
                if kwds[key] == True:
                    self._test_for_le_threshold = True
                else:
                    self._test_for_le_threshold = False
            else:
                print("Unknown option=" + key + " specified in "
                      "call to ConvergenceBase constructor")

    def reset(self):

        self._metric_history.clear()

    def lastMetric(self):

        if len(self._metric_history) == 0:
            raise RuntimeError("ConvergenceBase::lastMetric() "
                               "invoked with 0-length history")

        assert max(iterkeys(self._metric_history)) == \
            self._largest_iteration_key
        return self._metric_history[self._largest_iteration_key]

    def update(self, iteration_id, ph, scenario_tree, instances):

        current_value = self.computeMetric(ph, scenario_tree, instances)
        self._metric_history[iteration_id] = current_value
        self._largest_iteration_key = \
            max(self._largest_iteration_key, iteration_id)

    def computeMetric(self, ph, scenario_tree, solutions):

        raise NotImplementedError("ConvergenceBase::computeMetric() is "
                                  "an abstract method")

    def isConverged(self, ph):

        if self.lastMetric() == None:
            return False

        if self._test_for_le_threshold:
            return self.lastMetric() <= self._convergence_threshold
        else:
            return self.lastMetric() >= self._convergence_threshold

    def isImproving(self, iteration_lag):

        last_iteration = self._largest_iteration_key
        reference_iteration = \
            min(0,self._largest_iteration_key - iteration_lag)
        return self._metric_history[last_iteration] < \
            self._metric_history[reference_iteration]

    def pprint(self):

        print("Iteration    Metric Value")
        for key in sorted(iterkeys(self._metric_history)):
            val = self._metric_history[key]

            if val is None:
                print(' %5d       %12s' % (key, "None"))
            else:
                metric_format_string = ""
                if self._convergence_threshold >= 0.0001:
                    metric_format_string += "%14.4f"
                else:
                    metric_format_string += "%14.3e"
                print(' %5d       %12s' % (key, (metric_format_string % val)))

class PrimalDualResidualConvergence(ConvergenceBase):
    """ Constructor
        Arguments: None beyond those in the base class.
    """
    def __init__(self, *args, **kwds):
        super(PrimalDualResidualConvergence, self).__init__(*args, **kwds)
        self._name = "PrimalDual-Residual"
        self._previous_average = None

    @staticmethod
    def snapshot_average(ph):
        previous_average = {}
        for stage in ph.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                previous_average[tree_node.name] = \
                    copy.deepcopy(tree_node._averages)
        return previous_average

    @staticmethod
    def compute_residual_squared_norm(ph, previous_average):
        residual_squared_norm = 0.0
        for stage in ph.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                node_residual_squared_norm = 0.0
                node_previous_average = previous_average[tree_node.name]
                for scenario in tree_node.scenarios:
                    scenario_node_x = scenario._x[tree_node.name]
                    scenario_residual_squared_norm = 0.0
                    for variable_id in tree_node._standard_variable_ids:
                        scenario_residual_squared_norm += \
                            (scenario_node_x[variable_id] - \
                             node_previous_average[variable_id])**2
                    node_residual_squared_norm += \
                        scenario._probability * \
                        scenario_residual_squared_norm
                residual_squared_norm += \
                    tree_node.conditional_probability * \
                    node_residual_squared_norm
        return residual_squared_norm

    @staticmethod
    def compute_primal_residual_squared_norm(ph):
        primal_residual_squared_norm = 0.0
        for stage in ph.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                node_primal_residual_squared_norm = 0.0
                node_average = tree_node._averages
                for scenario in tree_node.scenarios:
                    scenario_node_x = scenario._x[tree_node.name]
                    scenario_primal_residual_squared_norm = 0.0
                    for variable_id in tree_node._standard_variable_ids:
                        scenario_primal_residual_squared_norm += \
                            (scenario_node_x[variable_id] - \
                             node_average[variable_id])**2
                    node_primal_residual_squared_norm += \
                        scenario._probability * \
                        scenario_primal_residual_squared_norm
                primal_residual_squared_norm += \
                    tree_node.conditional_probability * \
                    node_primal_residual_squared_norm
        return primal_residual_squared_norm

    @staticmethod
    def compute_dual_residual_squared_norm(ph, previous_average, rho_scaled=True):
        dual_residual_squared_norm = 0.0
        for stage in ph.scenario_tree.stages[:-1]:
            for tree_node in stage.nodes:
                node_dual_residual_squared_norm = 0.0
                node_previous_average = previous_average[tree_node.name]
                node_average = tree_node._averages
                node_rho = tree_node._scenarios[0]._rho[tree_node.name]
                for variable_id in tree_node._standard_variable_ids:
                    if rho_scaled:
                        node_dual_residual_squared_norm += \
                            node_rho[variable_id]**2 * \
                            (node_average[variable_id] - \
                             node_previous_average[variable_id])**2
                    else:
                        node_dual_residual_squared_norm += \
                            (node_average[variable_id] - \
                             node_previous_average[variable_id])**2
                dual_residual_squared_norm += \
                    tree_node.conditional_probability * \
                    node_dual_residual_squared_norm
        return dual_residual_squared_norm

    def computeMetric(self, ph, scenario_tree, instances):
        previous_average = self._previous_average
        self._previous_average = self.snapshot_average(ph)
        if previous_average is None:
            return None

        prsqn = self.compute_primal_residual_squared_norm(ph)
        drsqn = self.compute_dual_residual_squared_norm(ph,
                                                        previous_average)

        return prsqn+drsqn

#
# Implements the baseline "term-diff" metric from our submitted CMS
# paper.  For each variable, take the fabs of the difference from the
# mean at that node, and weight by scenario probability.
#

class TermDiffConvergence(ConvergenceBase):

    """ Constructor
        Arguments: None beyond those in the base class.

    """
    def __init__(self, *args, **kwds):

        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Term diff"

    def computeMetric(self, ph, scenario_tree, instances):

        term_diff = 0.0

        for stage, tree_node, variable_id, variable_values, is_fixed, is_stale \
            in scenario_tree_node_variables_generator_noinstances(
                scenario_tree,
                includeDerivedVariables=False,
                includeLastStage=False):

            if (not is_stale) or (is_fixed):
                
                for var_value, scenario_probability in variable_values:

                    term_diff += \
                        scenario_probability * \
                        fabs(var_value - tree_node._averages[variable_id])

        return term_diff


#
# Implements the normalized "term-diff" metric from our submitted CMS
# paper.  For each variable, take the fabs of the difference from the
# mean at that node, and weight by scenario probability - but
# normalize by the mean.  If I wasn't being lazy, this could be
# derived from the TermDiffConvergence class to avoid code replication
# :)
#

class NormalizedTermDiffConvergence(ConvergenceBase):

    """ Constructor
        Arguments: None beyond those in the base class.

    """
    def __init__(self, *args, **kwds):

        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Normalized term diff"

    def computeMetric(self, ph, scenario_tree, instances):

        normalized_term_diff = 0.0

        for stage, tree_node, variable_id, variable_values, is_fixed, is_stale \
            in scenario_tree_node_variables_generator_noinstances(
                scenario_tree,
                includeDerivedVariables=False,
                includeLastStage=False):

            average_value = tree_node._averages[variable_id]

            # should think about nixing the magic constant below (not
            # sure how to best pararamterize it).
            if ((not is_stale) or (is_fixed)) and \
                (fabs(average_value) > 0.0001):

                for var_value, scenario_probability in variable_values:
                    
                    normalized_term_diff += \
                        scenario_probability * \
                        fabs((var_value - average_value)/average_value)

        normalized_term_diff = \
            normalized_term_diff / \
            (ph._total_discrete_vars + ph._total_continuous_vars)

        return normalized_term_diff

#
# Implements a super-simple convergence criterion based on when a
# particular number of discrete variables are free (e.g., 20 or
# fewer).
#

class NumFixedDiscreteVarConvergence(ConvergenceBase):

    """ Constructor
        Arguments: None beyond those in the base class.

    """
    def __init__(self, *args, **kwds):

        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Number of fixed discrete variables"

    def computeMetric(self, ph, scenario_tree, instances):

        # the metric is brain-dead; just look at PH to see how many
        # free discrete variables there are!
        return ph._total_discrete_vars - ph._total_fixed_discrete_vars

# 
# Implements a convergence criterion that is based on exceeding 
# a threshold on the outer bound.
#

class OuterBoundConvergence(ConvergenceBase):

    """ Constructor
        Arguments: None beyond those in the base class.

    """
    def __init__(self, *args, **kwds):

        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Outer bound"

    def computeMetric(self, ph, scenario_tree, instances):

        return ph._best_reported_outer_bound
# 
# Implements a convergence criterion that is based on exceeding 
# the gap between the best inner and best outer bound
#

class InnerOuterConvergence(ConvergenceBase):

    """ Constructor
        Arguments: None beyond those in the base class.

    """
    def __init__(self, *args, **kwds):

        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Inner Outer gap bound"

    def computeMetric(self, ph, scenario_tree, instances):
        if ph._best_reported_outer_bound is None \
           or ph._best_reported_inner_bound is None:
            return float('inf')
        else:
            return abs(ph._best_reported_outer_bound \
                       - ph._best_reported_inner_bound)
