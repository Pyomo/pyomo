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
    STATUS_NONE                      = 0b0000
    # Used mipgap
    STATUS_MIPGAP                    = 0b0001
    # Solution gap was not reported
    STATUS_GAP_NA                    = 0b0010
    # Solution has nonzero
    # optimality gap
    STATUS_GAP_NONZERO               = 0b0100
    # One or more subproblems were infeasible
    STATUS_SOLVE_FAILED              = 0b1000

    WARNING_MESSAGE = {}

    # No mipgap detected, but a
    # nonzero solution gap was
    # found
    WARNING_MESSAGE[0b0100] = \
        "** Possibly Conservative - Mipgap Unknown, And Nonzero Solution Gap Reported **"

    # Used mipgap and solver did
    # report a solution gap
    WARNING_MESSAGE[0b0101] = \
        "** Possibly Conservative - Mipgap Detected, And Nonzero Solution Gap Reported **"

    # Used mipgap and solver did NOT
    # report a solution gap
    WARNING_MESSAGE[0b0111] = \
        "** Extreme Caution - Mipgap Detected, But No Solution Gap Information Obtained - Bound May Be Invalid **"
    WARNING_MESSAGE[0b0011] = \
        "** Extreme Caution - Mipgap Detected, But No Solution Gap Information Obtained - Bound May Be Invalid **"

    WARNING_MESSAGE[0b0110] = \
        "** Caution - Solver Did Not Provide Solution Gap Information - Bound May Be Invalid **"
    WARNING_MESSAGE[0b0010] = \
        "** Caution - Solver Did Not Provide Solution Gap Information - Bound May Be Invalid **"

    # Tags for operations for which we need to undo in order to return
    # ph to its original state
    SOLUTION_CACHING     = (1,)
    VARIABLE_FREEING     = (2,)
    DEACTIVATE_PROXIMAL  = (3,)
    DEACTIVATE_WEIGHT    = (4,)
    CACHE_WEIGHTS        = (5,)
    TREE_VARIABLE_FIXING = (6,)

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
        self._inner_bound_history = {}
        self._inner_status_history = {}

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

            elif op == self.TREE_VARIABLE_FIXING:

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

            elif op == self.DEACTIVATE_WEIGHT:

                assert op_data == None
                ph.activate_ph_objective_weight_terms()

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

    def FixScenarioTreeVariables(self, ph, fix_values):

        # Save the current fixed state and fix queue and clear the fix queue
        ph_fixed = {}
        ph_fix_queue = {}
        for tree_node in ph._scenario_tree._tree_nodes:
            ph_fixed[tree_node._name] = copy.deepcopy(tree_node._fixed)
            ph_fix_queue[tree_node._name] = copy.deepcopy(tree_node._fix_queue)
            tree_node.clear_fix_queue()

        # Fix everything in fix_values
        for node_name in fix_values:
            tree_node = ph._scenario_tree._tree_node_map[node_name]
            for variable_id, fix_val in iteritems(fix_values[node_name]):
                tree_node.fix_variable(variable_id, fix_val)

        # Push fixed variable statuses on instances (or
        # transmit to the phsolverservers)
        ph._push_fix_queue_to_instances()

        self._stack.append((self.TREE_VARIABLE_FIXING,
                            (ph_fixed, ph_fix_queue)))

    def DeactivatePHObjectiveProximalTerms(self, ph):

        ph.deactivate_ph_objective_proximal_terms()

        self._stack.append((self.DEACTIVATE_PROXIMAL, None))

    def DeactivatePHObjectiveWeightTerms(self, ph):

        ph.deactivate_ph_objective_weight_terms()

        self._stack.append((self.DEACTIVATE_WEIGHT, None))

    def CachePHWeights(self, ph):

        weights = {}
        for scenario in ph._scenario_tree._scenarios:
            weights[scenario._name] = \
                copy.deepcopy(scenario._w)

        self._stack.append((self.CACHE_WEIGHTS,weights))

    #
    # Calculates the probability weighted sum of all suproblem (or
    # bundle) objective functions. This function assumes the current
    # subproblem solutions, in particular the objective values stored
    # on the scenario objects, are appropriate for computing an outer
    # bound on the true optimal objective value (e.g., lower bound for
    # minimization problem). When a nonzero optimality gap is reported
    # in a scenario/bundle solution, the reported objective value for
    # that scenario/bundle will be relaxed accordingly before
    # including it in the average calculation.
    #
    def ComputeOuterBound(self, ph, storage_key):

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

        print("Computed objective %s bound=%12.4f\t%s"
              % (("lower" if self._is_minimizing else "upper"),
                 objective_bound,
                 self.WARNING_MESSAGE.get(bound_status,"")))

        return objective_bound, bound_status

    #
    # Calculates the probability weighted sum of all suproblem (or
    # bundle) objective functions. This function assumes the current
    # subproblem solutions represent valid, non-anticpative solutions
    # are appropriate for computing an inner bound on the optimal
    # objective value (e.g., upper bound for minimization problem).
    #
    def ComputeInnerBound(self, ph, storage_key):

        objective_bound = 0.0
        for scenario in ph._scenario_tree._scenarios:
            objective_bound += (scenario._probability * scenario._objective)

        print("Computed objective %s bound=%12.4f"
              % (("upper" if self._is_minimizing else "lower"),
                 objective_bound))

        return objective_bound, self.STATUS_NONE

    def ReportBestBound(self):

        print("")
        best_inner_bound = None
        if len(self._bound_history) > 0:
            if self._is_minimizing:
                best_inner_bound = min(self._inner_bound_history.values())
            else:
                best_inner_bound = max(self._inner_bound_history.values())
        print("Best Incumbent Bound: %15s"
              % (best_inner_bound))

        best_bound = None
        if len(self._bound_history) > 0:
            if self._is_minimizing:
                best_bound_key, best_bound = max(self._bound_history.items(),
                                                 key=itemgetter(1))
            else:
                best_bound_key, best_bound = min(self._bound_history.items(),
                                                 key=itemgetter(1))
        print("Best Dual Bound: %15s\t%s"
              % (best_bound,
                 self.WARNING_MESSAGE.get(self._status_history[best_bound_key],"")))

        print("Absolute Duality Gap: %15s"
              % abs(best_inner_bound - best_bound))
        relgap = float('inf')
        if (best_inner_bound != float('inf')) and \
           (best_inner_bound != float('-inf')) and \
           (best_bound != float('-inf')) and \
           (best_bound != float('-inf')):
            relgap = abs(best_inner_bound - best_bound) / \
                     (1e-10+abs(best_bound))

        print("Relative Gap: %15s %s" % (relgap*100.0,"%"))
        print("")
        output_filename = "phbestbound.txt"
        output_file = open(output_filename,"w")
        output_file.write("Incumbent: %.17g\n" % best_inner_bound)
        output_file.write("Dual: %.17g\n" % best_bound)
        output_file.close()
        print("Best bound written to file="+output_filename)

    def ReportBoundHistory(self):
        print("")
        print("Bound History")
        print("%15s %15s %15s" % ("Iteration", "Inner Bound", "Outer Bound"))
        output_filename = "phbound.txt"
        output_file = open(output_filename,"w")
        keys = list(self._bound_history.keys())
        if None in keys:
            keys.remove(None)
            print("%15s %15s %15s\t\t%s"
                  % ("Trivial",
                     "       -       ",
                     self._bound_history[None],
                     self.WARNING_MESSAGE.get(self._status_history[None],"")))
            output_file.write("Trivial: None, %.17g\n"
                              % (self._bound_history[None]))
        for key in sorted(keys):
            print("%15s %15s %15s\t\t%s"
                  % (key,
                     self._inner_bound_history[key],
                     self._bound_history[key],
                     self.WARNING_MESSAGE.get(self._status_history[key],"")))
            output_file.write("%d: %.17g, %.17g\n"
                              % (key,
                                 self._inner_bound_history[key],
                                 self._bound_history[key]))
        print("")
        output_file.close()
        print("Bound history written to file="+output_filename)

    def ExtractInternalNodeSolutionsWithDiscreteRounding(self, ph):

        node_solutions = {}
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                this_node_sol = node_solutions[tree_node._name] = {}
                xbars = tree_node._xbars
                for variable_id in tree_node._standard_variable_ids:
                    if not tree_node.is_variable_discrete(variable_id):
                        this_node_sol[variable_id] = xbars[variable_id]
                    else:
                        # rounded xbar, which has a MUCH
                        # better chance of being feasible
                        this_node_sol[variable_id] = \
                            int(round(xbars[variable_id]))

        return node_solutions

    def ExtractInternalNodeSolutionsWithDiscreteVoting(self, ph):

        node_solutions = {}
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                this_node_sol = node_solutions[tree_node._name] = {}
                xbars = tree_node._xbars
                for variable_id in tree_node._standard_variable_ids:
                    if not tree_node.is_variable_discrete(variable_id):
                        this_node_sol[variable_id] = xbars[variable_id]
                    else:
                        # for discrete variables use a weighted vote
                        # Note: for binary this can just be computed
                        #       by rounding the xbar (assuming it's an
                        #       average).  However, the following
                        #       works for binary and general integer,
                        #       where rounding the average is not
                        #       necessarily the same as a weighted vote
                        #       outcome.
                        vals = [int(round(scenario._x[tree_node._name][variable_id]))\
                                for scenario in tree_node._scenarios]
                        bins = list(set(vals))
                        vote = []
                        for val in bins:
                            vote.append(sum(scenario._probability \
                                            for scenario in tree_node._scenarios \
                                            if int(round(scenario._x[tree_node._name][variable_id])) == val))
                                               
                        # assign the vote outcome
                        this_node_sol[variable_id] = bins[vote.index(max(vote))]

        return node_solutions

class phboundextension(pyomo.util.plugin.SingletonPlugin, _PHBoundBase):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("phboundextension")

    def __init__(self):

        _PHBoundBase.__init__(self)

    def _iteration_k_bound_solves(self,ph, storage_key):

        # Extract a candidate solution to compute an upper bound
        #candidate_sol = self.ExtractInternalNodeSolutionsWithDiscreteRounding(ph)
        # ** Code uses the values stored in the scenario solutions
        #    to perform a weighted vote in the case of discrete
        #    variables, so it is important that we execute this
        #    before perform any new subproblem solves.
        candidate_sol = self.ExtractInternalNodeSolutionsWithDiscreteVoting(ph)
        # Caching the current set of ph solutions so we can restore
        # the original results. We modify the scenarios and re-solve -
        # which messes up the warm-start, which can seriously impact
        # the performance of PH. plus, we don't want lower bounding to
        # impact the primal PH in any way - it should be free of any
        # side effects.
        self.CachePHSolution(ph)

        # Save the current fixed state and fix queue.
        self.RelaxPHFixedVariables(ph)

        # Assuming the weight terms are already active but proximal
        # terms need to be deactivated deactivate all proximal terms
        # and activate all weight terms.
        self.DeactivatePHObjectiveProximalTerms(ph)

        # It is possible weights have not been pushed to instance
        # parameters (or transmitted to the phsolverservers) at this
        # point.
        ph._push_w_to_instances()

        failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts,
                                        exception_on_failure=False)

        if len(failures):

            print("Failed to compute duality-based bound due to "
                  "one or more solve failures")
            self._bound_history[storage_key] = \
                float('-inf') if self._is_minimizing else float('inf')
            self._status_history[storage_key] = self.STATUS_SOLVE_FAILED

        else:

            if ph._verbose:
                print("Successfully completed PH bound extension "
                      "weight-term only solves for iteration %s\n"
                      "- solution statistics:\n" % (storage_key))
                if ph._scenario_tree.contains_bundles():
                    ph._report_bundle_objectives()
                ph._report_scenario_objectives()

            # Compute the outer bound on the objective function.
            self._bound_history[storage_key], \
                self._status_history[storage_key] = \
                    self.ComputeOuterBound(ph, storage_key)

        # Deactivate the weight terms.
        self.DeactivatePHObjectiveWeightTerms(ph)

        # Fix all non-leaf stage variables involved
        # in non-anticipativity conditions to the most
        # recently computed xbar (or something like it)
        self.FixScenarioTreeVariables(ph, candidate_sol)

        failures = ph.solve_subproblems(warmstart=not ph._disable_warmstarts,
                                        exception_on_failure=False)
        if len(failures):

            print("Failed to compute bound at xbar due to "
                  "one or more solve failures")
            self._inner_bound_history[storage_key] = \
                float('inf') if self._is_minimizing else float('-inf')
            self._inner_status_history[storage_key] = self.STATUS_SOLVE_FAILED

        else:

            if ph._verbose:
                print("Successfully completed PH bound extension "
                      "fixed-to-xbar solves for iteration %s\n"
                      "- solution statistics:\n" % (storage_key))
                if ph._scenario_tree.contains_bundles():
                    ph._report_bundle_objectives()
                ph._report_scenario_objectives()

            # Compute the inner bound on the objective function.
            self._inner_bound_history[storage_key], \
                self._inner_status_history[storage_key] = \
                    self.ComputeInnerBound(ph, storage_key)

        # Restore ph to its state prior to entering this method (e.g.,
        # fixed variables, scenario solutions, proximal terms)
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
        self._bound_history[ph_iter], \
            self._status_history[ph_iter] = \
               self.ComputeOuterBound(ph, ph_iter)

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

