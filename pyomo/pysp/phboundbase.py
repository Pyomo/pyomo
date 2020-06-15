#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

import os
import logging
import copy

from six import iteritems
from operator import itemgetter

from pyomo.pysp.phutils import indexToString
from pyomo.opt import UndefinedData

logger = logging.getLogger('pyomo.pysp')

# ===== various ways to extract xhat =======
def ExtractInternalNodeSolutionsWithClosestScenarioNodebyNode(ph):
    # find the scenario closest to xbar at each node and return it

    node_solutions = {}

    def ScenXbarDist(scen, tree_node):
        # crude estimate of stdev to get a approx, truncated z score

        dist = 0
        xbars = tree_node._xbars
        mins = tree_node._minimums
        maxs = tree_node._maximums
        for variable_id in tree_node._standard_variable_ids:
            diff = scen._x[tree_node._name][variable_id] - xbars[variable_id]
            s_est = (maxs[variable_id] - mins[variable_id]) / 4.0 # close enough to stdev
            if s_est > ph._integer_tolerance:
                dist += min(3, abs(diff)/s_est) # truncated z score
        return dist

    ClosestScen = None
    ClosestScenDist = None
    for stage in ph._scenario_tree._stages[:-1]:
        for tree_node in stage._tree_nodes:
            this_node_sol = node_solutions[tree_node._name] = {}
            for scenario in tree_node._scenarios:
                if ClosestScenDist == 0:
                    break
                thisdist = ScenXbarDist(scenario, tree_node)
                if ClosestScenDist == None or thisdist < ClostScenDist:
                     ClosestScendist = thisdist
                     ClosestScen = scenario

            for variable_id in tree_node._standard_variable_ids:
                ## print ("extracting for "+str(variable_id)+" the value "+str(ClosestScen._x[tree_node._name][variable_id]))
                this_node_sol[variable_id] = ClosestScen._x[tree_node._name][variable_id]

    return node_solutions

def ExtractInternalNodeSolutionsWithDiscreteRounding(ph):

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

def ExtractInternalNodeSolutionsWithDiscreteVoting(ph):

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

def ExtractInternalNodeSolutionsWithSlamming(ph):
    from pyomo.pysp.plugins.wwphextension import _parse_yaml_file
    # Since it was a file, 
    #   assume that the argument was a json file with slamming instructions.
    # This will ignore suffixes we don't care about.
    # If there are no instructions use xbar.
    # Note: there is an implicit pecking order.
    print ("For x-hat, using slamming suffixes in",ph._xhat_method)
    slamdict = {}
    for suffix_name, suffix_value, variable_ids in \
          _parse_yaml_file(ph, ph._xhat_method):
        for node_name, node_variable_ids in iteritems(variable_ids):
            for variable_id in node_variable_ids:
                if variable_id not in slamdict:
                    slamdict[variable_id] = {}
                slamdict[variable_id][suffix_name] = suffix_value

    verbose = ph._verbose
    node_solutions = {}
    for stage in ph._scenario_tree._stages[:-1]:
        for tree_node in stage._tree_nodes:
            this_node_sol = node_solutions[tree_node._name] = {}
            xbars = tree_node._xbars
            mins = tree_node._minimums
            maxs = tree_node._maximums
            warnb = False  # did the user do something less than cool?

            for variable_id in tree_node._standard_variable_ids:
                if verbose:
                    variable_name, index = tree_node._variable_ids[variable_id]
                    full_variable_name = variable_name+indexToString(index)
                    print ("Setting x-hat for",full_variable_name)
                if variable_id not in slamdict or slamdict[variable_id]['CanSlamToAnywhere']:
                    if not tree_node.is_variable_discrete(variable_id):
                        this_node_sol[variable_id] = xbars[variable_id]
                        if verbose:
                            print ("   x-bar", this_node_sol[variable_id]) 
                    else:
                        this_node_sol[variable_id] = int(round(xbars[variable_id]))
                        if verbose:
                            print ("   rounded x-bar", this_node_sol[variable_id]) 
                elif slamdict[variable_id]['CanSlamToMin']:
                    this_node_sol[variable_id] = mins[variable_id]
                    if verbose:
                        print ("   min over scenarios", this_node_sol[variable_id]) 
                elif slamdict[variable_id]['CanSlamToMax']:
                    this_node_sol[variable_id] = maxs[variable_id]
                    if verbose:
                        print ("   max over scenarios", this_node_sol[variable_id]) 
                elif slamdict[variable_id]['CanSlamToLB']:
                    warnb = True
                    this_node_sol[variable_id] = mins[variable_id]
                    if verbose:
                        print ("   Lower Bound", this_node_sol[variable_id]) 
                elif slamdict[variable_id]['CanSlamToUB']:
                    warnb = True
                    this_node_sol[variable_id] = maxs[variable_id]
                    if verbose:
                        print ("   Upper Bound", this_node_sol[variable_id]) 
    if warnb:
        print ("Warning: for xhat determination from file %s, some variables had an upper or lower bound slam but not a corresponding min or max", ph._xhat_method)
    return node_solutions


#### call one ####

def ExtractInternalNodeSolutionsforInner(ph):
    # get xhat for the inner bound (upper bound for a min problem)
    if ph._xhat_method == "closest-scenario":
        return ExtractInternalNodeSolutionsWithClosestScenarioNodebyNode(ph)
    elif ph._xhat_method == "voting":
        return ExtractInternalNodeSolutionsWithDiscreteVoting(ph)
    elif ph._xhat_method == "rounding":
        return ExtractInternalNodeSolutionsWithDiscreteRounding(ph)
    elif os.path.isfile(ph._xhat_method):
        return ExtractInternalNodeSolutionsWithSlamming(ph)
    else:
        raise RuntimeError("Unknown x-hat determination method=%s specified in PH - unable to extract candidate solution for inner bound computation" % ph._xhat_method)

#============== end xhat code ==========

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
        self._outer_bound_history = {}
        self._outer_status_history = {}
        self._inner_bound_history = {}
        self._inner_status_history = {}

        self._is_minimizing = True

        # stack of operations modifying ph
        self._stack = []

    def RestoreLastPHChange(self, ph):
        if len(self._stack):
            tmp_restore_stack = [self._stack.pop()]
            real_stack = self._stack
            self._stack = tmp_restore_stack
            self.RestorePH(ph)
            self._stack = real_stack

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

        if ph._verbose:
            print("Computing objective %s bound" %
                  ("outer" if self._is_minimizing else "inner"))

        bound_status = self.STATUS_NONE
        if (ph._mipgap is not None) and (ph._mipgap > 0):
            logger.warning("A nonzero mipgap was detected when using "
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
        if ph._verbose:
            print("Computed objective %s bound=%12.4f\t%s"
                  % (("outer" if self._is_minimizing else "inner"),
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

        if ph._verbose:
            print("Computing objective %s bound" %
                  ("inner" if self._is_minimizing else "outer"))

        objective_bound = 0.0
        for scenario in ph._scenario_tree._scenarios:
            objective_bound += (scenario._probability * scenario._objective)

        if ph._verbose:
            print("Computed objective %s bound=%12.4f"
                  % (("inner" if self._is_minimizing else "outer"),
                     objective_bound))

        return objective_bound, self.STATUS_NONE

    def ReportBestBound(self):

        print("")
        best_inner_bound = None
        if len(self._inner_bound_history) > 0: # dlw May 2016
            if self._is_minimizing:
                best_inner_bound = min(self._inner_bound_history.values())
            else:
                best_inner_bound = max(self._inner_bound_history.values())
        print("Best Incumbent Bound: %15s"
              % (best_inner_bound))

        best_bound = None
        if len(self._outer_bound_history) > 0:
            if self._is_minimizing:
                best_bound_key, best_bound = max(self._outer_bound_history.items(),
                                                 key=itemgetter(1))
            else:
                best_bound_key, best_bound = min(self._outer_bound_history.items(),
                                                 key=itemgetter(1))
        print("Best Dual Bound: %15s\t%s"
              % (best_bound,
                 self.WARNING_MESSAGE.get(self._outer_status_history[best_bound_key],"")))

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
        keys = list(self._outer_bound_history.keys())
        if None in keys:
            keys.remove(None)
            print("%15s %15s %15s\t\t%s"
                  % ("Trivial",
                     "       -       ",
                     self._outer_bound_history[None],
                     self.WARNING_MESSAGE.get(self._outer_status_history[None],"")))
        for key in sorted(keys):
            print("%15s %15s %15s\t\t%s"
                  % (key,
                     self._inner_bound_history[key],
                     self._outer_bound_history[key],
                     self.WARNING_MESSAGE.get(self._outer_status_history[key],"")))
        print("")
        output_filename = "phbound.txt"
        with open(output_filename,"w") as output_file:
            output_file.write('Inner Bound:\n')
            for key in sorted(self._inner_bound_history.keys()):
                output_file.write("  %d: %.17g\n"
                                  % (key,
                                     self._inner_bound_history[key]))
            output_file.write('Outer Bound:\n')
            if None in self._outer_bound_history:
                output_file.write("  Trivial: %.17g\n"
                                  % (self._outer_bound_history[None]))
            for key in sorted(keys):
                output_file.write("  %d: %.17g\n"
                                  % (key,
                                     self._outer_bound_history[key]))
        print("Bound history written to file="+output_filename)

