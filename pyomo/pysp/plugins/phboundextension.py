#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import os
import logging
import copy

import pyomo.util.plugin
from pyomo.pysp import phextension
from pyomo.core.base import minimize
from pyomo.opt import UndefinedData

from operator import itemgetter
from six import iteritems

logger = logging.getLogger('pyomo.pysp')

class _PHBoundBase(object):

    # Nothing interesting
    STATUS_NONE                      = 0b000
    # Used mipgap
    STATUS_MIPGAP                    = 0b001
    # Solution gap was not reported
    STATUS_GAP_NA                    = 0b010
    # Solution has nonzero
    # optimality gap
    STATUS_GAP_NONZERO               = 0b100

    WARNING_MESSAGE = {}

    # No mipgap detected, but a
    # nonzero solution gap was
    # found
    WARNING_MESSAGE[0b100] = \
        "** Possibly Conservative - Mipgap Unknown, And Nonzero Solution Gap Reported **"

    # Used mipgap and solver did
    # report a solution gap
    WARNING_MESSAGE[0b101] = \
        "** Possibly Conservative - Mipgap Detected, And Nonzero Solution Gap Reported **"

    # Used mipgap and solver did NOT
    # report a solution gap
    WARNING_MESSAGE[0b111] = \
        "** Extreme Caution - Mipgap Detected, But No Solution Gap Information Obtained - Bound May Be Invalid **"
    WARNING_MESSAGE[0b011] = \
        "** Extreme Caution - Mipgap Detected, But No Solution Gap Information Obtained - Bound May Be Invalid **"

    WARNING_MESSAGE[0b110] = \
        "** Caution - Solver Did Not Provide Solution Gap Information - Bound May Be Invalid **"
    WARNING_MESSAGE[0b010] = \
        "** Caution - Solver Did Not Provide Solution Gap Information - Bound May Be Invalid **"

    # Tags for operations for which we need to undo in order to return
    # ph to its original state
    SOLUTION_CACHING     = (1,)
    VARIABLE_FREEING     = (2,)
    DEACTIVATE_PROXIMAL  = (3,)
    CACHE_WEIGHTS        = (4,)
    VARIABLE_XBAR_FIXING = (5,)

    def __init__(self):

        # the interval for which bound computations are performed
        # during the ph iteration k solves.  A bound update is always
        # performed at iteration 0 with the (non-PH-augmented)
        # objective and the update interval starts with a bound update
        # at ph iteration 1.
        self._update_interval = 1

        # keys are ph iteration except for the trival bound whose key
        # is None
        self._bound_history = {}
        self._status_history = {}

        self._is_minimizing = True

        # stack of operations modifying ph
        self._stack = []

    def RestorePH(self, ph):

        while self._stack:

            op, op_data = self._stack.pop()

            if op == self.SOLUTION_CACHING:

                ph.restoreCachedSolutions(op_data, release_cache=True)

            elif op == self.VARIABLE_FREEING:

                ph_fixed, ph_fix_queue = op_data

                # Refix all previously fixed variables
                for tree_node in ph._scenario_tree._tree_nodes:
                    for variable_id, fix_value in iteritems(ph_fixed[tree_node._name]):
                        tree_node.fix_variable(variable_id, fix_value)

                # Push fixed variable statuses to instances (or
                # transmit to the phsolverservers)
                ph._push_fix_queue_to_instances()

                # Restore the fix_queue
                for tree_node in ph._scenario_tree._tree_nodes:
                    tree_node._fix_queue.update(
                        ph_fix_queue[tree_node._name])

            elif op == self.VARIABLE_XBAR_FIXING:

                ph_fixed, ph_fix_queue = op_data

                # Unfix all non-leaf variables that were fixed to xbar
                for stage in ph._scenario_tree._stages[:-1]:
                    for tree_node in stage._tree_nodes:
                        for variable_id in tree_node._standard_variable_ids:
                            tree_node.free_variable(variable_id)

                # Refix all previously fixed variables
                for tree_node in ph._scenario_tree._tree_nodes:
                    for variable_id, fix_value in iteritems(ph_fixed[tree_node._name]):
                        tree_node.fix_variable(variable_id, fix_value)

                # Push fixed variable statuses to instances (or
                # transmit to the phsolverservers)
                ph._push_fix_queue_to_instances()

                # Restore the fix_queue
                for tree_node in ph._scenario_tree._tree_nodes:
                    tree_node._fix_queue.update(
                        ph_fix_queue[tree_node._name])

            elif op == self.DEACTIVATE_PROXIMAL:

                assert op_data == None
                ph.activate_ph_objective_proximal_terms()

            elif op == self.CACHE_WEIGHTS:

                weights = op_data
                for scenario in ph._scenario_tree._scenarios:
                    cached_weights = weights[scenario._name]
                    scenario._w = copy.deepcopy(cached_weights)

            else:

                raise ValueError("Unexpected Operation Flag - "
                                 "This is a developer error")

    def CachePHSolution(self, ph):

        # Cache the current state of the ph instances and scenario
        # tree so we don't affect ph behavior by changing the solution
        # on the scenario tree / scenario instances prior to the next
        # solve
        cache_id = ph.cacheSolutions()

        self._stack.append((self.SOLUTION_CACHING, cache_id))

    def RelaxPHFixedVariables(self, ph):

        # Save the current fixed state and fix queue
        ph_fixed = dict((tree_node._name, copy.deepcopy(tree_node._fixed)) \
                             for tree_node in ph._scenario_tree._tree_nodes)

        ph_fix_queue = \
            dict((tree_node._name, copy.deepcopy(tree_node._fix_queue)) \
                 for tree_node in ph._scenario_tree._tree_nodes)

        # Free all currently fixed variables
        for tree_node in ph._scenario_tree._tree_nodes:
            tree_node.clear_fix_queue()
            for variable_id in ph_fixed[tree_node._name]:
                tree_node.free_variable(variable_id)

        # Push freed variable statuses on instances (or
        # transmit to the phsolverservers)
        ph._push_fix_queue_to_instances()

        self._stack.append((self.VARIABLE_FREEING,
                            (ph_fixed, ph_fix_queue)))

    def FixPHVariablesToXbar(self, ph):

        # Save the current fixed state and fix queue
        ph_fixed = dict((tree_node._name, copy.deepcopy(tree_node._fixed)) \
                             for tree_node in ph._scenario_tree._tree_nodes)

        ph_fix_queue = \
            dict((tree_node._name, copy.deepcopy(tree_node._fix_queue)) \
                 for tree_node in ph._scenario_tree._tree_nodes)

        # Fix everything to xbar
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                tree_node.clear_fix_queue()
                for variable_id in tree_node._standard_variable_ids:
                    tree_node.fix_variable(variable_id, tree_node._xbars[variable_id])

        # Push freed variable statuses on instances (or
        # transmit to the phsolverservers)
        ph._push_fix_queue_to_instances()

        self._stack.append((self.VARIABLE_XBAR_FIXING,
                            (ph_fixed, ph_fix_queue)))

    def DeactivatePHObjectiveProximalTerms(self, ph):

        ph.deactivate_ph_objective_proximal_terms()

        self._stack.append((self.DEACTIVATE_PROXIMAL, None))

    def CachePHWeights(self, ph):

        weights = {}
        for scenario in ph._scenario_tree._scenarios:
            weights[scenario._name] = \
                copy.deepcopy(scenario._w)

        self._stack.append((self.CACHE_WEIGHTS,weights))

    #
    # Calculates the probability weighted sum of all suproblem (or
    # bundle) objective functions, assuming the most recent solution
    # corresponds to a ph solve with the weight terms active and the
    # proximal terms inactive in the objective function.
    #
    def ComputeBound(self, ph, storage_key):

        bound_status = self.STATUS_NONE
        if (ph._mipgap is not None) and (ph._mipgap > 0):
            logger.warn("A nonzero mipgap was detected when using "
                        "the PH bound plugin. The bound "
                        "computation may as a result be conservative.")
            bound_status |= self.STATUS_MIPGAP

        objective_bound = 0.0

        # If we are bundling, we compute the objective bound in a way
        # such that we can still use solution gap information if it
        # is available.
        if ph._scenario_tree.contains_bundles():

            for scenario_bundle in ph._scenario_tree._scenario_bundles:

                bundle_gap = ph._gaps[scenario_bundle._name]
                bundle_objective_value = 0.0

                for scenario in scenario_bundle._scenario_tree._scenarios:
                    # The objective must be taken from the scenario
                    # objects on PH full scenario tree
                    scenario_objective = \
                        ph._scenario_tree.get_scenario(scenario._name)._objective
                    # And we need to make sure to use the
                    # probabilities assigned to scenarios in the
                    # compressed bundle scenario tree
                    bundle_objective_value += scenario_objective * \
                                              scenario._probability

                if not isinstance(bundle_gap, UndefinedData):
                    if bundle_gap > 0:
                        bound_status |= self.STATUS_GAP_NONZERO
                        if self._is_minimizing:
                            bundle_objective_value -= bundle_gap
                        else:
                            bundle_objective_value += bundle_gap
                else:
                    bound_status |= self.STATUS_GAP_NA

                objective_bound += bundle_objective_value * \
                                   scenario_bundle._probability

        else:

            for scenario in ph._scenario_tree._scenarios:

                this_objective_value = scenario._objective
                this_gap = ph._gaps[scenario._name]

                if not isinstance(this_gap, UndefinedData):
                    if this_gap > 0:
                        bound_status |= self.STATUS_GAP_NONZERO
                        if self._is_minimizing:
                            this_objective_value -= this_gap
                        else:
                            this_objective_value += this_gap
                else:
                    bound_status |= self.STATUS_GAP_NA

                objective_bound += (scenario._probability * this_objective_value)

        print("Computed objective lower bound=%12.4f\t%s"
              % (objective_bound,
                 self.WARNING_MESSAGE.get(bound_status,"")))

        self._status_history[storage_key] = bound_status
        self._bound_history[storage_key] = objective_bound

    def ReportBestBound(self):
        print("")
        best_bound = None
        if len(self._bound_history) > 0:
            if self._is_minimizing:
                best_bound_key, best_bound = max(self._bound_history.items(),
                                                 key=itemgetter(1))
            else:
                best_bound_key, best_bound = min(self._bound_history.items(),
                                                 key=itemgetter(1))
        print("Best Objective Bound: %15s\t\t%s"
              % (best_bound,
                 self.WARNING_MESSAGE.get(self._status_history[best_bound_key],"")))
        print("")
        output_filename = "phbestbound.txt"
        output_file = open(output_filename,"w")
        output_file.write("%.17g\n" % best_bound)
        output_file.close()
        print("Best Lower Bound written to file="+output_filename)

    def ReportBoundHistory(self):
        print("")
        print("Bound History")
        print("%15s %15s" % ("Iteration", "Bound"))
        output_filename = "phbound.txt"
        output_file = open(output_filename,"w")
        keys = list(self._bound_history.keys())
        if None in keys:
            keys.remove(None)
            print("%15s %15s\t\t%s"
                  % ("Trivial",
                     self._bound_history[None],
                     self.WARNING_MESSAGE.get(self._status_history[None],"")))
            output_file.write("Trivial: %.17g\n"
                              % (self._bound_history[None]))
        for key in sorted(keys):
            print("%15s %15s\t\t%s"
                  % (key,
                     self._bound_history[key],
                     self.WARNING_MESSAGE.get(self._status_history[key],"")))
            output_file.write("%d: %.17g\n"
                              % (key,
                                 self._bound_history[key]))
        print("")
        output_file.close()
        print("Lower bound history written to file="+output_filename)

class phboundextension(pyomo.util.plugin.SingletonPlugin, _PHBoundBase):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("phboundextension")

    def __init__(self):

        _PHBoundBase.__init__(self)

    def _iteration_k_bound_solves(self,ph, storage_key):

        # Caching the current set of ph solutions so we can restore
        # the original results. We modify the scenarios and re-solve -
        # which messes up the warm-start, which can seriously impact
        # the performance of PH. plus, we don't want lower bounding to
        # impact the primal PH in any way - it should be free of any
        # side effects.
        self.CachePHSolution(ph)

        # Save the current fixed state and fix queue
        self.RelaxPHFixedVariables(ph)

        # Assuming the weight terms are already active but proximal
        # terms need to be deactivated deactivate all proximal terms
        # and activate all weight terms
        self.DeactivatePHObjectiveProximalTerms(ph)

        # Weights have not been pushed to instance parameters (or
        # transmitted to the phsolverservers) at this point
        ph._push_w_to_instances()

        ph.solve_subproblems(warmstart=not ph._disable_warmstarts)

        if ph._verbose:
            print("Successfully completed PH bound extension "
                  "iteration %s solves\n"
                  "- solution statistics:\n" % (storage_key))
            if ph._scenario_tree.contains_bundles():
                ph._report_bundle_objectives()
            ph._report_scenario_objectives()

        # compute the bound
        self.ComputeBound(ph,storage_key)

        # Restore ph to its state prior to entering this method
        # (e.g., fixed variables, scenario solutions, proximal terms)
        self.RestorePH(ph)

    ############ Begin Callback Functions ##############

    def pre_ph_initialization(self,ph):
        """
        Called before PH initialization.
        """
        pass

    def post_instance_creation(self, ph):
        """
        Called after PH initialization has created the scenario
        instances, but before any PH-related
        weights/variables/parameters/etc are defined!
        """
        pass

    def post_ph_initialization(self, ph):
        """
        Called after PH initialization
        """

        if ph._verbose:
            print("Invoking post initialization callback in phboundextension")

        self._is_minimizing = True if (ph._objective_sense == minimize) else False
        # TODO: Check for ph options that may not be compatible with
        #       this plugin and warn / raise exception

        # grab the update interval from the environment variable, if
        # it exists.
        update_interval_variable_name = "PHBOUNDINTERVAL"
        if update_interval_variable_name in os.environ:
            self._update_interval = int(os.environ[update_interval_variable_name])
            print("phboundextension using update interval="
                  +str(self._update_interval)+", extracted from "
                  "environment variable="+update_interval_variable_name)
        else:
            print("phboundextension using default update "
                  "interval="+str(self._update_interval))

    def post_iteration_0_solves(self, ph):
        """
        Called after the iteration 0 solves
        """

        if ph._verbose:
            print("Invoking post iteration 0 solve callback in phboundextension")

        # Always compute a lower/upper bound here because it requires
        # no work.  The instances (or bundles) have already been
        # solved with the original (non-PH-augmented) objective and
        # are loaded with results.

        #
        # Note: We will still obtain a bound using the weights
        #       computed from PH iteration 0 in the
        #       pre_iteration_k_solves callback.
        #
        ph_iter = None

        # Note: It is important that the mipgap is not adjusted
        #       between the time after the subproblem solves
        #       and before now.
        self.ComputeBound(ph, ph_iter)

    def post_iteration_0(self, ph):
        """
        Called after the iteration 0 solves, averages computation, and weight computation
        """
        pass

    def pre_iteration_k_solves(self, ph):
        """
        Called immediately before the iteration k solves
        """

        #
        # Note: We invoke this callback pre iteration k in order to
        #       obtain a PH bound using weights computed from the
        #       PREVIOUS iteration's scenario solutions (including
        #       those of iteration zero).
        #
        ph_iter = ph._current_iteration-1

        if ph._verbose:
            print("Invoking pre iteration k solve callback in phboundextension")

        if (ph_iter % self._update_interval) != 0:
            return

        self._iteration_k_bound_solves(ph, ph_iter)

    def post_iteration_k_solves(self, ph):
        """
        Called after the iteration k solves!
        """
        pass

    def post_iteration_k(self, ph):
        """
        Called after the iteration k is finished, after weights have been updated!
        """
        pass

    def post_ph_execution(self, ph):
        """
        Called after PH has terminated!
        """

        if ph._verbose:
            print("Invoking post execution callback in phboundextension")

        #
        # Note: We invoke this callback in order to compute a bound
        #       using the weights obtained from the final PH
        #       iteration.
        #
        ph_iter = ph._current_iteration

        if (ph_iter % self._update_interval) == 0:

            self._iteration_k_bound_solves(ph, ph_iter)

        self.ReportBoundHistory()
        self.ReportBestBound()

