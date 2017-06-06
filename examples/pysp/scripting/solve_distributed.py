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
#  (1) Constructs scenario instances over a distributed
#      Pyro-based scenario tree. A distributed scenario tree consists
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

# *** How to run this example ***:
#
# In a separate shell launch
#   $ mpirun -np 1 pyomo_ns -n localhost : \
#            -np 1 dispatch_srvr -n localhost : \
#            -np 3 scenariotreeserver --pyro-host=localhost
#
# In this shell launch:
#   $ python solve_distributed.py

import os
import sys
from pyomo.environ import *
from pyomo.pysp.scenariotree.manager import \
    (ScenarioTreeManagerClientPyro,
     InvocationType)

# declare the number of scenarios over which to construct a simple
# two-stage scenario tree
num_scenarios = 3

#
# Define the scenario tree structure as well as stage
# costs and variables
#
def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model import \
        CreateConcreteTwoStageScenarioTreeModel

    st_model = CreateConcreteTwoStageScenarioTreeModel(num_scenarios)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    # First Stage
    st_model.StageCost[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('x')

    # Second Stage
    st_model.StageCost[second_stage] = 'SecondStageCost'
    st_model.StageVariables[second_stage].add('y')

    return st_model

#
# Define a PySP callback function that returns a constructed scenario
# instance for each scenario. Stochastic scenario data is created
# on the fly using the random module, so this script will likely
# produce different results with each execution.
# 
#
import random
random.seed(None)
def pysp_instance_creation_callback(scenario_name, node_names):

    model = ConcreteModel()
    model.x = Var(bounds=(-10,10))
    model.y = Var()
    model.FirstStageCost = Expression(expr=0.0)
    model.SecondStageCost = Expression(expr=model.y + 1)
    model.obj = Objective(expr=model.FirstStageCost + model.SecondStageCost)
    model.con1 = Constraint(expr=model.x >= model.y)
    model.con2 = Constraint(expr=model.y >= random.randint(-10,10))

    return model

#
# Define a function to execute on scenarios that solves the pyomo
# instance and returns the objective function value. Function
# invocations require that the function to be invoked always accepts
# the process-local scenario tree worker object as the first argument.
# InvocationType.PerScenario requires a second argument representing
# the scenario object to be processed. Refer to the help doc-string
# on InvocationType for more information.
#
def solve_model(worker, scenario):
    from pyomo.opt import SolverFactory

    with SolverFactory("glpk") as opt:
        opt.solve(scenario._instance)
        return value(scenario._instance.obj)

if __name__ == "__main__":

    # generate an absolute path to this file
    thisfile = os.path.abspath(__file__)

    # generate a list of options we can configure
    options = ScenarioTreeManagerClientPyro.register_options()

    # To see detailed information about options
    #for name in options.keys():
    #    print(options.about(name))

    # To see a more compact display of options
    #options.display()

    #
    # Set a few options
    #

    # the pysp_instance_creation_callback function
    # will be detected and used
    options.model_location = thisfile
    # setting this option to None implies there
    # is a pysp_scenario_tree_model_callback function
    # defined in the model file
    options.scenario_tree_location = None
    # use verbose output
    options.verbose = True
    #
    # Pyro-specific options
    #
    options.pyro_host = 'localhost'
    # we allow this option to be overridden from the
    # command line for Pyomo testing purposes
    options.pyro_port = \
        None if (len(sys.argv) == 1) else int(sys.argv[1])
    # set this option to the number of scenario tree
    # servers currently running
    # Note: it can be fewer than the number of scenarios
    options.pyro_required_scenariotreeservers = 3
    # Shutdown all pyro-related components when the scenario
    # tree manager closes. Note that with Pyro4, the nameserver
    # must be shutdown manually.
    options.pyro_shutdown = True

    # using the 'with' block will automatically call
    # manager.close() and gracefully shutdown the
    # scenario tree servers
    with ScenarioTreeManagerClientPyro(options) as manager:
        manager.initialize()

        results = manager.invoke_function(
            "solve_model",  # function to execute
            thisfile,       # file (or module) containing the function
            invocation_type=InvocationType.PerScenario)

        for scenario_name in sorted(results):
            print(scenario_name+": "+str(results[scenario_name]))
