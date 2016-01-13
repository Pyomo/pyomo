#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['IPHExtension','IPHSolverServerExtension']

from pyomo.util.plugin import Interface

# IMPORTANT: No variable fixing should occur until the post-iteration solves, following variable statistic updates.
#            Otherwise, variable statistics will not be correctly maintained.

class IPHExtension(Interface):

    def reset(self, ph):
        """Invoked to reset the state of a plugin to that of post-construction"""
        pass

    def pre_ph_initialization(self, ph):
        """Called before PH initialization"""
        pass

    def post_instance_creation(self, ph):
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
        # IMPT: This is only called once, and after iteration 0.
        pass

    def asynchronous_pre_scenario_queue(self, ph, scenario_name):
        """Called before the scenario solve has been queued"""
        pass

    def post_asynchronous_var_w_update(self, ph, subproblem_solve_counts):
        """Called after a batch of asynchronous sub-problems are solved and corresponding statistics are updated"""
        pass    

    def post_asynchronous_solves(self, ph):
        """Called after the asynchronous solve loop is executed"""
        # IMPT: This is only called once, after asychronous processing loop completes.
        pass

    ###########################################################

    def post_ph_execution(self, ph):
        """Called after PH has terminated"""
        pass


class IPHSolverServerExtension(Interface):

    def pre_ph_initialization(self, ph):
        """Called before PH initialization."""
        pass

    def post_instance_creation(self, ph):
        """Called after the instances have been created."""
        pass

    def post_ph_initialization(self, ph):
        """Called after PH initialization"""
        pass

    def pre_iteration_0_solve(self, ph):
        """Called before the iteration 0 solve begins"""
        pass

    def post_iteration_0_solve(self, ph):
        """Called after the iteration 0 solve is finished"""
        pass

    def pre_iteration_k_solve(self, ph):
        """Called before the iteration k solve begins"""
        pass

    def post_iteration_k_solve(self, ph):
        """Called after the iteration k solve is finished"""
        pass
