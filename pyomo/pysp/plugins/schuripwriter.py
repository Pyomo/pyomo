#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from pyutilib.misc import *

from pyomo.misc.plugin import *
from pyomo.pysp import phextension

from pyomo.core import *
import pyomo.opt

import os
import sys

from six import iteritems

# the purpose of this PH plugin is to write a PySP instance 
# to a directory for ingestion into the SchurIP solver. 
# this directory will contain a .nl file for each scenario, 
# and an lqm file describing the non-anticipative (binding)
# variables.

class schuripwriter(SingletonPlugin):

    implements (phextension.IPHExtension) 

    def pre_ph_initialization(self,ph):
        pass

    def post_instance_creation(self,ph):
        pass

    def post_ph_initialization(self, ph):
        print("Called after PH initialization!")

        print("Writing out PySP files for input to Schur IP")

        output_directory_name = "schurip"

        os.system("rm -rf "+output_directory_name)
        os.mkdir(output_directory_name)        

        nl_writer = pyomo.opt.WriterFactory('nl')

        root_node = ph._scenario_tree.findRootNode()

        scenario_number = 1

        for instance_name, instance in iteritems(ph._instances):

            # even though they are identical, SchurIP wants a .lqm file per scenario.
            # so tag the suffix data on a per-instance basis.

            instance.lqm = Suffix(direction=Suffix.LOCAL, default=-1)

            for variable_name, variable_indices in iteritems(root_node._variable_indices):
                variable = getattr(instance, variable_name)
                for index in variable_indices:
                    var_value = variable[index]
                    instance.lqm.setValue(var_value, 1)

            scenario_output_filename = output_directory_name + os.sep + "Scenario"+str(scenario_number)+".nl"

            result = nl_writer(instance, scenario_output_filename, lambda x: True, ph._symbolic_solver_labels)

            scenario_number += 1

        print("NL files for PySP instance written to output directory: "+output_directory_name)

        sys.exit(0)

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
