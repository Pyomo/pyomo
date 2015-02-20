#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2015 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.util.plugin

from six import iteritems

from pyomo.pysp import phextension

from pyomo.core.base import minimize, maximize

import math

class EcksteinCombettesExtension(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("ecksteincombettesextension")

    def compute_updates(self, ph):

        print "***WE ARE DOING STUFF***"

        ph.pprint(True,True,True,False,False)

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
                                # CRITICAL: Y depends on the z and weight values that were used when solving the scenario!
                                z_for_solve = scenario._xbars_for_solve[tree_node._name][variable_id]
                                w_for_solve = scenario._ws_for_solve[tree_node._name][variable_id]
                                scenario._y[variable_id] = rho_values[variable_id] * (z_for_solve - varval) - w_for_solve
                                # check it!
                                print "THIS",varval + (1.0/rho_values[variable_id])*scenario._y[variable_id],"SHOULD EQUAL THIS",z_for_solve-(1.0/rho_values[variable_id])*w_for_solve

                                scenario._u[variable_id] = varval - tree_node_averages[variable_id]
                            else:
                                raise RuntimeError("***maximize not supported by compute_y in plugin ")

        ###########################################
        # compute v values - these are node-based #
        ###########################################

        print "Y VALUES:"
        for scenario in ph._scenario_tree._scenarios:
            print scenario._y

        print "U VALUES:"
        for scenario in ph._scenario_tree._scenarios:
            print scenario._u

        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    expected_y = 0.0
                    for scenario in tree_node._scenarios:
                        expected_y += ((scenario._y[variable_id] * scenario._probability) / tree_node._probability)
                    tree_node._v[variable_id] = expected_y

        print "V VALUES:"
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                print tree_node._v

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

        print "U NORM=",p_unorm
        print "V NORM=",p_vnorm

        # TODO: make these real and configurable!
        delta = 1e-1
        epsilon = 1e-1

        if p_unorm < delta and p_vnorm < epsilon:
            print "***HEY -WE'RE DONE!!!***"
            foobar

        #####################################################
        # compute phi; if greater than zero, update z and w #
        #####################################################
        phi = 0.0
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                tree_node_zs = tree_node._z
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        var_values = scenario._x[tree_node._name]
                        varval = var_values[variable_id]
                        weight_values = scenario._w[tree_node._name]
                        print "WEIGHT VALUES=",weight_values[variable_id]
                        print "TREE NODE ZS=",tree_node_zs[variable_id]
                        print "YS=",scenario._y[variable_id]
                        print "VAR VALUE=",varval
                        if varval is not None:
                            phi += scenario._probability * ((tree_node_zs[variable_id] - varval) * (scenario._y[variable_id] + weight_values[variable_id]))
                        else:
                            foobar
                    print "PHI NOW=",phi,"VARIABLE ID=",variable_id

        print "PHI=",phi
        if phi > 0:
            tau = 1.0 # this is the over-relaxation parameter - we need to do something more useful
            # probability weighted norms are used below - this doesn't match the paper.
            theta = phi/(p_unorm*p_unorm + p_vnorm*p_vnorm) 
            print "THETA=",theta
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    print "TREE NODE ZS BEFORE:",tree_node._z
                    print "TREE NODE VS BEFORE:",tree_node._v
                    tree_node_zs = tree_node._z
                    for variable_id in tree_node._standard_variable_ids:
                        for scenario in tree_node._scenarios:
                            rho_values = scenario._rho[tree_node._name]
                            weight_values = scenario._w[tree_node._name]
                            print "SUBTRACTING TERM TO Z=",(tau * theta * tree_node._v[variable_id])
                            tree_node._z[variable_id] -= (tau * theta * tree_node._v[variable_id])
                            weight_values[variable_id] += (tau * theta * scenario._u[variable_id])
#                            print "NEW WEIGHT FOR VARIABLE=",variable_id,"FOR SCENARIO=",scenario._name,"EQUALS",weight_values[variable_id]
                    print "TREE NODE ZS AFTER:",tree_node._z
        elif phi == 0.0:
            print "***PHI WAS ZERO - NOT DOING ANYTHING"
            pass
        else:
            # WE MAY NOT BE SCREWED, BUT WE'LL ASSUME SO FOR NOW.
            print "***PHI IS NEGATIVE - BADNESS!"
            foobar

        # CHECK HERE - PHI SHOULD BE 0 AT THIS POINT
        phi = 0.0
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                tree_node_zs = tree_node._z
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        var_values = scenario._x[tree_node._name]
                        varval = var_values[variable_id]
                        weight_values = scenario._w[tree_node._name]
                        if varval is not None:
                            phi += scenario._probability * ((tree_node_zs[variable_id] - varval) * (scenario._y[variable_id] + weight_values[variable_id]))
                        else:
                            foobar

        print "NEW PHI=",phi
#        foobar

    def pre_ph_initialization(self,ph):
        """Called before PH initialization"""
        pass

    def post_instance_creation(self,ph):
        """Called after the instances have been created"""
        pass

    def post_ph_initialization(self, ph):
        """Called after PH initialization"""
        pass

    ##########################################################
    # the following callbacks are specific to synchronous PH #
    ##########################################################

    def post_iteration_0_solves(self, ph):
        """Called after the iteration 0 solves"""
        pass

    def post_iteration_0(self, ph):
        """Called after the iteration 0 solves, averages computation, and weight computation"""
        print "POST ITERATION 0 CALLBACK"

        # define y and u parameters for each non-leaf variable in each scenario.
        print "****ADDING Y, U, V, and Z PARAMETERS"

        for scenario in ph._scenario_tree._scenarios:

            scenario._y = {}
            scenario._u = {}

            instance = scenario._instance

            for tree_node in scenario._node_list[:-1]:

                nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
                nodal_index_set = instance.find_component(nodal_index_set_name)
                assert nodal_index_set is not None

                scenario._y.update(dict.fromkeys(scenario._y,0.0))
                scenario._u.update(dict.fromkeys(scenario._u,0.0))

        # define v and z parameters for each non-leaf variable in the tree.
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:

                nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
                nodal_index_set = instance.find_component(nodal_index_set_name)
                assert nodal_index_set is not None

                tree_node._v = dict((i,0) for i in nodal_index_set)
                tree_node._z = dict((i,tree_node._averages[i]) for i in nodal_index_set)

        # copy z to xbar in the scenario tree, as we've told PH we will be taking care of it.
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:

                nodal_index_set_name = "PHINDEX_"+str(tree_node._name)
                nodal_index_set = instance.find_component(nodal_index_set_name)
                assert nodal_index_set is not None

                tree_node._xbars = dict((i,tree_node._z[i]) for i in nodal_index_set)

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

        ph.pprint(True,True,True,False,False)

        # we want the PH estimates of the weights initially, but we'll compute them afterwards.
        ph._ph_weight_updates_enabled = False

        # we will also handle xbar updates (z).
        ph._ph_xbar_updates_enabled = False

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

    def post_asynchronous_var_w_update(self, ph):
        """Called after a batch of asynchronous sub-problems are solved and corresponding statistics are updated"""
        print "POST ASYCH VAR W CALLBACK"
        self.compute_updates(ph)

    def post_asynchronous_solves(self, ph):
        """Called after the asynchronous solve loop is executed"""
        pass

    ###########################################################

    def post_ph_execution(self, ph):
        """Called after PH has terminated"""
        pass
