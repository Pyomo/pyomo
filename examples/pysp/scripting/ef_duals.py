#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# This PySP example is setup to run as an indepedent python script
# that does the following.
#  (1) Constructs scenario instances over a distributed
#      Pyro-based scenario tree. A distrubuted scenario tree consists
#      of the following objects:
#        - A scenario tree manager (launched from this file)
#        - One or more scenario tree servers (launched using the
#          'scenariotreeserver' executable installed with PySP)
#        - One or more scenario tree workers managed by the
#          scenario tree servers. These will be setup by the scenario
#          tree manager.
#        - A dispatch server  (launched using the 'dispatch_srvr'
#          executable installed with PySP)
#  (2) Executes a function on each scenario of the distributed tree.
#      These function invocations must be transmitted via Pyro to the
#      scenario tree workers where the Pyomo scenario instances have
#      been constructed.

import os

from pyomo.environ import *
from pyomo.pysp.scenariotree.scenariotreemanager import \
    ScenarioTreeManagerSerial
from pyomo.pysp.ef import create_ef_instance
from pyomo.opt import SolverFactory

thisdir = os.path.dirname(os.path.abspath(__file__))
farmer_example_dir = os.path.join(os.path.dirname(thisdir), 'farmer')

options = ScenarioTreeManagerSerial.register_options()

options.model_location = os.path.join(farmer_example_dir, 'models')
options.scenario_tree_location = os.path.join(farmer_example_dir, 'scenariodata')

# using the 'with' block will automatically call
# manager.close() and gracefully shutdown
with ScenarioTreeManagerSerial(options) as manager:
    manager.initialize()

    ef_instance = create_ef_instance(manager._scenario_tree,
                                     verbose_output=options.verbose)

    ef_instance.dual = Suffix(direction=Suffix.IMPORT)
    
    with SolverFactory('cplex') as opt:

        opt.solve(ef_instance)

        ef_instance.dual.pprint()
