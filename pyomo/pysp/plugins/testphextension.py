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


class testphextension(SingletonPlugin):

    implements(phextension.IPHExtension) 

    def reset(self, ph):
        pass

    def pre_ph_initialization(self, ph):
        pass

    def post_instance_creation(self, ph):
        pass

    def post_ph_initialization(self, ph):
        print("Called after PH initialization!")

    def post_iteration_0_solves(self, ph):
        print("Called after the iteration 0 solves!")

    def post_iteration_0(self, ph):
        print("Called after the iteration 0 solves, averages computation, and weight computation")

    def pre_iteration_k_solves(self, ph):
        # this one does not do anything
        pass

    def post_iteration_k_solves(self, ph):
        print("Called after the iteration k solves!")

    def post_iteration_k(self, ph):
        print("Called after an iteration k has finished!")

    def post_ph_execution(self, ph):
        print("Called after PH has terminated!")
