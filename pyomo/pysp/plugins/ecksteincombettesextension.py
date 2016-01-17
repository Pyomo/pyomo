#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2015 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.util.plugin

from six import iteritems, itervalues

import random

from pyomo.pysp import phextension
from pyomo.pysp.convergence import ConvergenceBase

from pyomo.core.base import minimize, maximize

import math

# the converger for the class - everything (primal and dual) is 
# contained in the (u,v) vector of the Eckstein-Combettes extension.
class EcksteinCombettesConverger(ConvergenceBase):

    def __init__(self, *args, **kwds):
        ConvergenceBase.__init__(self, *args, **kwds)
        self._name = "Eckstein-Combettes (u,v) norm"        

        # the plugin computes the metric, so we'll just provide
        # it a place to stash the latest computed value.
        self._last_computed_uv_norm_value = None

    def computeMetric(self, ph, scenario_tree, instances):

        return self._last_computed_uv_norm_value 

# the primary Eckstein-Combettes extension class

class EcksteinCombettesExtension(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("ecksteincombettesextension")

    def __init__(self):

        self._check_output = False
        self._JName = "PhiSummary.csv"

        self._subproblems_to_queue = []

        # various configuration options.

        # if this is True, then the number of sub-problems 
        # returned may be less than the buffer length.
        self._queue_only_negative_subphi_subproblems = False

        # track the total number of projection steps performed (and, implicitly,
        # the current projection step) in addition to the last projection step
        # at which a scenario sub-problem was incorporated.
        self._total_projection_steps = 0
        self._projection_step_of_last_update = {} # maps scenarios to projection step number

        self._converger = None 

    def check_optimality_conditions(self, ph):

        print("Checking optimality conditions for Eckstein-Combettes plugin")
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    expected_y = 0.0
                    for scenario in tree_node._scenarios:
                        expected_y += ((scenario._y[variable_id] * scenario._probability) / tree_node._probability)
        # the expected value of the y vector should be 0 if the solution is optimal

    def compute_updates(self, ph, subproblems, scenario_solve_counts):

        self._total_projection_steps += 1
        print("Initiating projection step: %d" % self._total_projection_steps)

        print("Computing updates given solutions to the following sub-problems:")
        for subproblem in subproblems:
            print("%s" % subproblem)
        print("")

        for subproblem in subproblems:
            self._projection_step_of_last_update[subproblem] = self._total_projection_steps

        ########################################
        ##### compute y values and u values ####
        ##### these are scenario-based        ##
        ########################################

        # NOTE: z is initiaized to be xbar in the code above, but it is *not* xbar. 
        # NOTE: v is essentailly y bar
        # NOTE: lambda is 1/rho xxxxxxxxxxxxx so if you see 1/lamba in a latex file, use rho in the py file
        # ASSUME W is the Eckstein W, not the PH W

        for stage in ph._scenario_tree._stages[:-1]:

            for tree_node in stage._tree_nodes:

                if ph._dual_mode is True:
                    raise RuntimeError("***dual_mode not supported by compute_y in plugin ")
                tree_node_averages = tree_node._averages
                tree_node_zs = tree_node._z

                for scenario in tree_node._scenarios:

                    weight_values = scenario._w[tree_node._name]
                    rho_values = scenario._rho[tree_node._name]
                    var_values = scenario._x[tree_node._name]

                    for variable_id in tree_node._standard_variable_ids:
                        varval = var_values[variable_id]
                        if varval is not None:
                            if scenario._objective_sense == minimize:

                               if scenario._name in subproblems:
                                   # CRITICAL: Y depends on the z and weight values that were used when solving the scenario!
                                   z_for_solve = scenario._xbars_for_solve[tree_node._name][variable_id]
                                   w_for_solve = scenario._ws_for_solve[tree_node._name][variable_id]
                                   scenario._y[variable_id] = rho_values[variable_id] * (z_for_solve - varval) - w_for_solve

                               # check it!
                               #print("THIS %s SHOULD EQUAL THIS %s" % (varval + (1.0/rho_values[variable_id])*scenario._y[variable_id],z_for_solve-(1.0/rho_values[variable_id])*w_for_solve))
                               scenario._u[variable_id] = varval - tree_node_averages[variable_id]
                            else:
                                raise RuntimeError("***maximize not supported by compute_y in plugin ")

        if self._check_output:

            print("Y VALUES:")
            for scenario in ph._scenario_tree._scenarios:
                print(scenario._y)

            print("U VALUES:")
            for scenario in ph._scenario_tree._scenarios:
                print(scenario._u)

#        self.check_optimality_conditions(ph)

        ###########################################
        # compute v values - these are node-based #
        ###########################################

        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    expected_y = 0.0
                    for scenario in tree_node._scenarios:
                        expected_y += ((scenario._y[variable_id] * scenario._probability) / tree_node._probability)
                    tree_node._v[variable_id] = expected_y

        if self._check_output:

            print("V VALUES:")
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    print(tree_node._v)

        ###########################################
        # compute norms and test for convergence  #
        ###########################################

        p_unorm = 0.0
        p_vnorm = 0.0

        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        this_v_val = tree_node._v[variable_id]
                        p_vnorm += tree_node._probability * this_v_val * this_v_val
                        this_u_val = scenario._u[variable_id]
                        p_unorm += scenario._probability * this_u_val * this_u_val
                    
        p_unorm = math.sqrt(p_unorm)
        p_vnorm = math.sqrt(p_vnorm)

        scalarized_norm = math.sqrt(p_unorm*p_unorm + p_vnorm*p_vnorm)

        print("Computed separator norm: (%e,%e) - scalarized norm=%e" % (p_unorm, p_vnorm, scalarized_norm))

        self._converger._last_computed_uv_norm_value = scalarized_norm

#        if p_unorm < delta and p_vnorm < epsilon:
#            print("Separator norm dropped below threshold (%e,%e)" % (delta, epsilon))
#            return

        #####################################################
        # compute phi; if greater than zero, update z and w #
        #####################################################

        print("")
        print("Initiating projection calculations...")

        with open(self._JName,"a") as f:
             f.write("%10d" % (ph._current_iteration))

        phi = 0.0
        sub_phi_map = {}
        for scenario in tree_node._scenarios:
            for tree_node in scenario._node_list[:-1]:
                tree_node_zs = tree_node._z
                cumulative_sub_phi = 0.0
                for variable_id in tree_node._standard_variable_ids:
                        var_values = scenario._x[tree_node._name]
                        varval = var_values[variable_id]
                        weight_values = scenario._w[tree_node._name]
#                        print "WEIGHT VALUES=",weight_values[variable_id]
#                        print "TREE NODE ZS=",tree_node_zs[variable_id]
#                        print "YS=",scenario._y[variable_id]
#                        print "VAR VALUE=",varval
                        if varval is not None:
#                            print "COMPUTING SUB PHI", "Y=",scenario._y[variable_id],"W=",weight_values[variable_id]
                            sub_phi = scenario._probability * ((tree_node_zs[variable_id] - varval) * (scenario._y[variable_id] + weight_values[variable_id]))
                            cumulative_sub_phi += sub_phi
                            phi += sub_phi
                        else:
                            foobar
                with open(self._JName,"a") as f:
                    f.write(", %10f" % (cumulative_sub_phi))
                sub_phi_map[scenario._name] = cumulative_sub_phi

        with open(self._JName,"a") as f:
            for subproblem in subproblems:
                f.write(", %s" % subproblem)
            f.write("\n")

        print("Computed sub-phi values, by scenario:")
        for scenario_name in sorted(sub_phi_map.keys()):
            print("  %30s %16e" % (scenario_name, sub_phi_map[scenario_name]))

        print("Computed phi:    %16e" % phi)
        if phi > 0:
            tau = 1.0 # this is the over-relaxation parameter - we need to do something more useful
            # probability weighted norms are used below - this doesn't match the paper.
            theta = phi/(p_unorm*p_unorm + p_vnorm*p_vnorm) 
            print("Computed theta: %16e" % theta)
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    if self._check_output:
                        print("TREE NODE ZS BEFORE: %s" % tree_node._z)
                        print("TREE NODE VS BEFORE: %s" % tree_node._v)
                    tree_node_zs = tree_node._z
                    for variable_id in tree_node._standard_variable_ids:
                        for scenario in tree_node._scenarios:
                            rho_values = scenario._rho[tree_node._name]
                            weight_values = scenario._w[tree_node._name]
                            if self._check_output:
                                print("WEIGHT VALUE PRIOR TO MODIFICATION=",weight_values[variable_id])
                                print("U VALUE PRIOR TO MODIFICATION=",scenario._u[variable_id])
#                            print("SUBTRACTING TERM TO Z=%s" % (tau * theta * tree_node._v[variable_id]))
                            tree_node._z[variable_id] -= (tau * theta * tree_node._v[variable_id])
                            weight_values[variable_id] += (tau * theta * scenario._u[variable_id])
                            if self._check_output:
                                print("NEW WEIGHT FOR VARIABLE=",variable_id,"FOR SCENARIO=",scenario._name,"EQUALS",weight_values[variable_id])
#                    print("TREE NODE ZS AFTER: %s" % tree_node._z)
        elif phi == 0.0:
            print("***PHI WAS ZERO - NOT DOING ANYTHING - NO MOVES - DOING CHECK BELOW!")
            pass
        else:
            # WE MAY NOT BE SCREWED, BUT WE'LL ASSUME SO FOR NOW.
            print("***PHI IS NEGATIVE - NOT DOING ANYTHING")

        if self._check_output:

            print("Z VALUES:")
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:            
                    print("TREE NODE=%s",tree_node._name)
                    print("Zs:",tree_node._z)

        # CHECK HERE - PHI SHOULD BE 0 AT THIS POINT - THIS IS JUST A CHECK
        with open(self._JName,"a") as f:
             f.write("%10d" % (ph._current_iteration))

        # the z's have been updated - copy these to PH scenario tree xbar maps,
        # so they can be correctly transmitted to instances - this plugin is 
        # responsible for xbar updates.
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:            
                for variable_id in tree_node._z:
                    tree_node._xbars[variable_id] = tree_node._z[variable_id]

        print("")
        print("Initiating post-projection calculations...")

        phi = 0.0

        sub_phi_to_scenario_map = {}

        for scenario in tree_node._scenarios:
            for tree_node in scenario._node_list[:-1]:
                tree_node_zs = tree_node._z
                cumulative_sub_phi = 0.0
                for variable_id in tree_node._standard_variable_ids:
                    var_values = scenario._x[tree_node._name]
                    varval = var_values[variable_id]
                    weight_values = scenario._w[tree_node._name]
                    if varval is not None:
                        sub_phi = scenario._probability * ((tree_node_zs[variable_id] - varval) * (scenario._y[variable_id] + weight_values[variable_id]))
                        cumulative_sub_phi += sub_phi
                        phi += sub_phi
                    else:
                        foobar

                # HEY - shouldn't the following be moved out one level of indentation, to map to the scenario? it of course works for two-stage.
                    
                if not cumulative_sub_phi in sub_phi_to_scenario_map:
                    sub_phi_to_scenario_map[cumulative_sub_phi] = []
                sub_phi_to_scenario_map[cumulative_sub_phi].append(scenario._name)

                with open(self._JName,"a") as f:
                    f.write(", %10f" % (cumulative_sub_phi))

        print("Computed sub-phi values (scenario, phi, iters-since-last-incorporated):")
        for sub_phi in sorted(sub_phi_to_scenario_map.keys()):
            print("  %16e: " % sub_phi),
            for scenario_name in sub_phi_to_scenario_map[sub_phi]:
                print("%30s" % scenario_name),
                print(" %4d" % (self._total_projection_steps - self._projection_step_of_last_update[scenario_name])), # TBD - not handling multiple scenarios correctly here
            print("")

        print("Computed phi: %16e" % phi)
        with open(self._JName,"a") as f:
            f.write("\n")

        negative_sub_phis = [sub_phi for sub_phi in sub_phi_to_scenario_map if sub_phi < 0.0]

        if len(negative_sub_phis) == 0:
            print("**** YIKES! QUEUING SUBPROBLEMS AT RANDOM****")
            # TBD - THIS ASSUMES UNIQUE PHIS, WHICH IS NOT ALWAYS THE CASE.
            all_phis = sub_phi_to_scenario_map.keys()
            random.shuffle(all_phis)
            for phi in all_phis[0:ph._async_buffer_length]:
                scenario_name = sub_phi_to_scenario_map[phi][0]
                print("Queueing sub-problem=%s",scenario_name)
                self._subproblems_to_queue.append(scenario_name)

        else:
            if self._queue_only_negative_subphi_subproblems:
                print("Queueing sub-problems whose scenarios possess the most negative phi values:")
            else:
                print("Queueing sub-problems whose scenarios possess the smallest phi values:")
            sorted_phis = sorted(sub_phi_to_scenario_map.keys())
            for phi in sorted_phis[0:ph._async_buffer_length]:
                if ((self._queue_only_negative_subphi_subproblems) and (phi < 0.0)) or (not self._queue_only_negative_subphi_subproblems):
                    scenario_name = sub_phi_to_scenario_map[phi][0] 
                    print("%30s %16e" % (scenario_name,phi)),
                    self._subproblems_to_queue.append(scenario_name)
                    print("")

        print("")

    def reset(self, ph):
        self.__init__()

    def pre_ph_initialization(self, ph):
        """Called before PH initialization"""
        pass

    def post_instance_creation(self, ph):
        """Called after the instances have been created"""
        with open(self._JName,"w") as f:
            f.write("Phi Summary; generally two lines per iteration\n")
            f.write("Iteration ")
            for scenario in ph._scenario_tree._scenarios:
                f.write(", %10s" % (scenario._name))
            f.write(", Subproblems Returned")
            f.write("\n")

    def post_ph_initialization(self, ph):
        """Called after PH initialization"""
        
        # IMPORTANT: if the Eckstein-Combettes extension plugin is enabled,
        #            then make sure PH is in async mode - otherwise, nothing
        #            will work!
        if not ph._async:
            raise RuntimeError("PH is not in async mode - this is required for the Eckstein-Combettes extension")

        self._total_projection_steps = 0
        for scenario in ph._scenario_tree._scenarios:
            self._projection_step_of_last_update[scenario._name] = 0

        # NOTE: we don't yet have a good way to get keyword options into
        #       plugins - so this is mildy hack-ish. more hackish, but
        #       useful, would be to extract the value from an environment
        #       variable - similar to what is done in the bounds extension.

        # the convergence threshold should obviously be parameterized
        self._converger = EcksteinCombettesConverger(convergence_threshold=1e-5)
        ph._convergers.append(self._converger)

    ##########################################################
    # the following callbacks are specific to synchronous PH #
    ##########################################################

    def post_iteration_0_solves(self, ph):
        """Called after the iteration 0 solves"""

        # we want the PH estimates of the weights initially, but we'll compute them afterwards.
        ph._ph_weight_updates_enabled = False

        # we will also handle xbar updates (z).
        ph._ph_xbar_updates_enabled = False

    def post_iteration_0(self, ph):
        """Called after the iteration 0 solves, averages computation, and weight computation"""
        print("POST ITERATION 0 CALLBACK")

        # define y and u parameters for each non-leaf variable in each scenario.
        print("****ADDING Y, U, V, and Z PARAMETERS")

        for scenario in ph._scenario_tree._scenarios:

            scenario._y = {}
            scenario._u = {}

            # instance = scenario._instance

            for tree_node in scenario._node_list[:-1]:

                nodal_index_set = tree_node._standard_variable_ids
                assert nodal_index_set is not None

                scenario._y.update((variable_id, 0.0) for variable_id in nodal_index_set)
                scenario._u.update((variable_id, 0.0) for variable_id in nodal_index_set)
#                print "YS AFTER UPDATE:",scenario._y

        # define v and z parameters for each non-leaf variable in the tree.
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:

                nodal_index_set = tree_node._standard_variable_ids
                assert nodal_index_set is not None

                tree_node._v = dict((i,0) for i in nodal_index_set)
                tree_node._z = dict((i,tree_node._averages[i]) for i in nodal_index_set)

        # copy z to xbar in the scenario tree, as we've told PH we will be taking care of it.
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:

                nodal_index_set = tree_node._standard_variable_ids
                assert nodal_index_set is not None

                tree_node._xbars = dict((i,tree_node._z[i]) for i in nodal_index_set)

        # mainly to set up data structures.
        for scenario in ph._scenario_tree._scenarios:
            self.asynchronous_pre_scenario_queue(ph, scenario._name)

        # pick subproblems at random - we need a number equal to the async buffer length,
        # although we need all of them initially (PH does - not this particular plugin).
        async_buffer_length = ph._async_buffer_length
        all_subproblems = [scenario._name for scenario in ph._scenario_tree._scenarios]
        random.shuffle(all_subproblems)
        self._subproblems_to_queue = all_subproblems[0:ph._async_buffer_length]

    def pre_iteration_k_solves(self, ph):
        """Called before each iteration k solve"""
        pass

    def post_iteration_k_solves(self, ph):
        """Called after the iteration k solves"""
        pass

    def post_iteration_k(self, ph):
        """Called after the iteration k is finished"""
        pass

    ##########################################################

    ###########################################################
    # the following callbacks are specific to asynchronous PH #
    ###########################################################

    def pre_asynchronous_solves(self, ph):
        """Called before the asynchronous solve loop is executed"""
        pass

    def asynchronous_pre_scenario_queue(self, ph, scenario_name):
        """Called right before each scenario solve is been queued"""

        # we need to cache the z and w that were used when solving the input scenario.
        scenario = ph._scenario_tree.get_scenario(scenario_name)

        scenario._xbars_for_solve = {}
        for tree_node in scenario._node_list[:-1]:
            scenario._xbars_for_solve[tree_node._name] = dict((k,v) for k,v in iteritems(tree_node._z))

        scenario._ws_for_solve = {}
        for tree_node in scenario._node_list[:-1]:
            scenario._ws_for_solve[tree_node._name] = dict((k,v) for k,v in iteritems(scenario._w[tree_node._name]))

    def post_asynchronous_var_w_update(self, ph, subproblems, scenario_solve_counts):
        """Called after a batch of asynchronous sub-problems are solved and corresponding statistics are updated"""
        print("")
        print("Computing updates in Eckstein-Combettes extension")
        self.compute_updates(ph, subproblems, scenario_solve_counts)

    def post_asynchronous_solves(self, ph):
        """Called after the asynchronous solve loop is executed"""
        pass

    def asynchronous_subproblems_to_queue(self, ph):
        """Called after subproblems within buffer length window have been processed"""
        result = self._subproblems_to_queue
        self._subproblems_to_queue = []
        return result

    ###########################################################

    def post_ph_execution(self, ph):
        """Called after PH has terminated"""
        pass
