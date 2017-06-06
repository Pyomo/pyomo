#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This PySP example is setup to run as an independent python script
# that does the following:
#  (1) Constructs scenario instances for the farmer problem using the
#      serial scenario tree manager (non-distributed).
#  (2) Constructs the extensive form instance over all scenarios in the
#      scenario tree.
#  (3) Solves the extensive form instance and reports the dual values
#      associated with the non-anticipativity constraints on the
#      first-stage variables.

import os
import sys
from pyomo.environ import *
from pyomo.pysp.scenariotree.manager import \
    ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory

thisdir = os.path.dirname(os.path.abspath(__file__))
farmer_example_dir = os.path.join(os.path.dirname(thisdir), 'farmer')

options = ScenarioTreeManagerClientSerial.register_options()

# To see detailed information about options
#for name in options.keys():
#    print(options.about(name))

# To see a more compact display of options
#options.display()

options.model_location = \
    os.path.join(farmer_example_dir, 'models')
options.scenario_tree_location = \
    os.path.join(farmer_example_dir, 'scenariodata')

# using the 'with' block will automatically call
# manager.close() and gracefully shutdown
with ScenarioTreeManagerClientSerial(options) as manager:
    manager.initialize()

    ef_instance = create_ef_instance(manager.scenario_tree,
                                     verbose_output=options.verbose)

    ef_instance.dual = Suffix(direction=Suffix.IMPORT)

    with SolverFactory('cplex') as opt:

        opt.solve(ef_instance)

        #
        # Print duals of non-anticipaticity constraints
        #
        master_constraint_map = ef_instance.MASTER_CONSTRAINT_MAP
        print("%50s %20s" % ("Variable", "Dual"))
        for scenario in manager.scenario_tree.scenarios:
            instance = scenario._instance
            for i in instance.DevotedAcreage:
                print("%50s %20s" % (instance.DevotedAcreage[i],
                                     ef_instance.dual[master_constraint_map[
                                         instance.DevotedAcreage[i]]]))

            print("")
