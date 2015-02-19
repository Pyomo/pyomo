#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2015 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

import pyomo.util.plugin

from pyomo.pysp import phextension

from pyomo.core.base import minimize, maximize

import math

class EcksteinCombettesExtension(pyomo.util.plugin.SingletonPlugin):

    pyomo.util.plugin.implements(phextension.IPHExtension)

    pyomo.util.plugin.alias("ecksteincombettesextension")

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
        pass

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

        # define y and u parameters for each non-leaf variable in each scenario.
        print "****ADDING Y, U, and V PARAMETERS"

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

                tree_node._z = dict((i,tree_node._xbars[i]) for i in nodal_index_set)

    def post_asynchronous_var_w_update(self, ph):
        """Called after a batch of asynchronous sub-problems are solved and corresponding statistics are updated"""

        print "***WE SHOULD DO STUFF***"

        ########################################
        ##### compute y values and u values ####
        ##### these are scenario-based        ##
        ########################################

        # NOTE: z is xbar
        # NOTE: v is essentailly y bar
        # NOTE: lambda is 1/rho xxxxxxxxxxxxx so if you see 1/lamba in a latex file, use rho in the py file
        # ASSUME W is the Eckstein W, not the PH W

        for stage in ph._scenario_tree._stages[:-1]:

            for tree_node in stage._tree_nodes:

                if ph._dual_mode is True:
                    raise RuntimeError("***dual_mode not supported by compute_y in plugin ")
                tree_node_xbars = tree_node._averages

                for scenario in tree_node._scenarios:

                    weight_values = scenario._w[tree_node._name]
                    rho_values = scenario._rho[tree_node._name]
                    var_values = scenario._x[tree_node._name]

                    for variable_id in tree_node._standard_variable_ids:
                        varval = var_values[variable_id]
                        if varval is not None:
                            if scenario._objective_sense == minimize:
                                scenario._y[variable_id] = rho_values[variable_id] * (tree_node_xbars[variable_id] - varval) - weight_values[variable_id]
                                scenario._u[variable_id] = varval - tree_node_xbars[variable_id]
                            else:
                                raise RuntimeError("***maximize not supported by compute_y in plugin ")

        ###########################################
        # compute v values - these are node-based #
        ###########################################

        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                for variable_id in tree_node._standard_variable_ids:
                    expected_y = 0.0
                    for scenario in tree_node._scenarios:
                        expected_y += (scenario._y[variable_id] * scenario._probability)
                    tree_node._v[variable_id] = expected_y

        ###########################################
        # compute norms and test for convergence  #
        ###########################################

        p_unorm = 0.0
        p_vnorm = 0.0

        for scenario in ph._scenario_tree._scenarios:
            for tree_node in scenario._node_list[:-1]:
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        this_u_val = scenario._u[variable_id]
                        this_v_val = tree_node._z[variable_id]

                        p_unorm += scenario._probability * this_u_val * this_u_val
                        p_vnorm += scenario._probability * this_v_val * this_v_val
                    
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
        phi = 0
        for stage in ph._scenario_tree._stages[:-1]:
            for tree_node in stage._tree_nodes:
                tree_node_zs = tree_node._z
                for variable_id in tree_node._standard_variable_ids:
                    for scenario in tree_node._scenarios:
                        var_values = scenario._x[tree_node._name]
                        varval = var_values[variable_id]
                        weight_values = scenario._w[tree_node._name]
                        if varval is not None:
                            phi += scenario._probability * ((tree_node_zs[variable_id] - varval) * (scenario._y[variable_id] - weight_values[variable_id]))
                        else:
                            foobar

        print "PHI=",phi

        if phi > 0:
            tau = 1 # this is the over-relaxation parameter - we need to do something more useful
            # probability weighted norms are used below - this doesn't match the paper.
            theta = phi/(p_unorm*p_unorm + p_vnorm*p_vnorm) 
            for stage in ph._scenario_tree._stages[:-1]:
                for tree_node in stage._tree_nodes:
                    tree_node_zs = tree_node._z
                    for variable_id in tree_node._standard_variable_ids:
                        for scenario in tree_node._scenarios:
                            rho_values = scenario._rho[tree_node._name]
                            weight_values = scenario._w[tree_node._name]
                            tree_node._z[variable_id] -= (rho_values[variable_id] * theta * tree_node._v[variable_id])
                            weight_values[variable_id] += (rho_values[variable_id] * theta * scenario._u[variable_id])
                            print "NEW WEIGHT FOR VARIABLE=",variable_id,"FOR SCENARIO=",scenario._name,"EQUALS",weight_values[variable_id]

    def post_asynchronous_solves(self, ph):
        """Called after the asynchronous solve loop is executed"""
        pass

    ###########################################################

    def post_ph_execution(self, ph):
        """Called after PH has terminated"""
        pass
