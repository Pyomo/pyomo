#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.util.plugin import *
from pyomo.pysp import phextension


class examplephextension(SingletonPlugin):

    implements (phextension.IPHExtension)

    def reset(self, ph):
        """Invoked to reset the state of a plugin to that of post-construction"""
        print("RESET PH CALLBACK INVOKED")
    
    def pre_ph_initialization(self,ph):
        """Called before PH initialization."""
        print("PRE INITIALIZATION PH CALLBACK INVOKED")

    def post_instance_creation(self, ph):
        """Called after PH initialization has created the scenario instances, but before any PH-related weights/variables/parameters/etc are defined!"""
        print("POST INSTANCE CREATION PH CALLBACK INVOKED")

    def post_ph_initialization(self, ph):
        """Called after PH initialization!"""
        print("POST INITIALIZATION PH CALLBACK INVOKED")

    def post_iteration_0_solves(self, ph):
        """Called after the iteration 0 solves!"""
        print("POST ITERATION 0 SOLVE PH CALLBACK INVOKED")

    def post_iteration_0(self, ph):
        """Called after the iteration 0 solves, averages computation, and weight computation"""
        print("POST ITERATION 0 PH CALLBACK INVOKED")

    def pre_iteration_k_solves(self, ph):
        """Called immediately before the iteration k solves!"""
        print("PRE ITERATION K SOLVE PH CALLBACK INVOKED")

    def post_iteration_k_solves(self, ph):
        """Called after the iteration k solves!"""
        print("POST ITERATION K SOLVE PH CALLBACK INVOKED")

    def post_iteration_k(self, ph):
        """Called after the iteration k is finished, after weights have been updated!"""
        print("POST ITERATION K PH CALLBACK INVOKED")

    def post_ph_execution(self, ph):
        """Called after PH has terminated!"""
        print("POST EXECUTION PH CALLBACK INVOKED")


class examplephsolverserverextension(SingletonPlugin):

    implements (phextension.IPHSolverServerExtension)

    def pre_ph_initialization(self,ph):
        """Called before PH initialization."""
        print("PRE INITIALIZATION PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def post_instance_creation(self,ph):
        """Called after the instances have been created."""
        print("POST INSTANCE CREATION PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def post_ph_initialization(self, ph):
        """Called after PH initialization!"""
        print("POST INITIALIZATION PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def pre_iteration_0_solve(self, ph):
        """Called before the iteration 0 solve begins!"""
        print("PRE ITERATION 0 SOLVE PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def post_iteration_0_solve(self, ph):
        """Called after the iteration 0 solve is finished!"""
        print("POST ITERATION 0 SOLVE PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def pre_iteration_k_solve(self, ph):
        """Called before the iteration k solve begins!"""
        print("PRE ITERATION K SOLVE PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)

    def post_iteration_k_solve(self, ph):
        """Called after the iteration k solve is finished!"""
        print("POST ITERATION K SOLVE PHSOLVERSERVER CALLBACK INVOKED ON WORKER: "+ph.WORKERNAME)
