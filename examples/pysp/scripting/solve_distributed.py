#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# This PySP example is setup to run as an indepedent python script
# that does the following:
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

# *** How to run this example ***:
#
# In a separate shell launch
# $ mpirun -np 1 pyomo_ns -n localhost : -np 1 dispatch_srvr localhost : -np 1 scenariotreeserver --pyro-hostname=localhost
#
# In this shell launch
# python solve_distributed.py

from pyomo.environ import *

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
    st_model.StageCostVariable[first_stage] = 'FirstStageCost'
    st_model.StageVariables[first_stage].add('x')

    # Second Stage
    st_model.StageCostVariable[second_stage] = 'SecondStageCost'
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
# Define a function to execute on scenarios that
# solves the pyomo instance and returns the objective
# function value. External function invocations require
# that the first two arguments of the function are
# always the worker object and the scenario tree.
# InvocationType.PerScenarioInvocation requires a third
# argument representing the scenario object to be processed
#
def solve_model(worker, scenario_tree, scenario):
    from pyomo.opt import SolverFactory

    with SolverFactory("glpk") as opt:
        opt.solve(scenario._instance)
        return value(scenario._instance.obj)

if __name__ == "__main__":
    import os
    from pyutilib.misc.config import ConfigBlock
    from pyomo.pysp.scenariotree.scenariotreemanager import \
        ScenarioTreeManagerSPPyro
    from pyomo.pysp.scenariotree.scenariotreeworkerbasic import \
        ScenarioTreeWorkerBasic
    from pyomo.pysp.scenariotree.scenariotreeserverutils import \
        InvocationType

    # generate an absolute path to this file
    thisfile = os.path.abspath(__file__)

    # generate a list of options we can configure
    options = ConfigBlock()
    ScenarioTreeManagerSPPyro.register_options(options)
    ScenarioTreeWorkerBasic.register_options(options)

    #
    # Set a few options
    #

    options.verbose = False
    options.pyro_hostname='localhost'
    # the pysp_instance_creation_callback function
    # will be detected and used
    options.model_location = thisfile
    # setting this option to None implies there
    # is a pysp_scenario_tree_model_callback function
    # defined in the model file
    options.scenario_tree_location = None
    # set this option to the number of scenario tree
    # servers currently running
    # Note: it can be fewer than the number of scenarios
    options.sppyro_required_servers = 1

    # using the 'with' block will automatically call
    # manager.close() and gracefully shutdown the
    # scenario tree servers
    with ScenarioTreeManagerSPPyro(options) as manager:
        action_handles = manager.initialize()
        # wait for actions to complete
        manager.complete_actions(action_handles)

        action_handles = \
            manager.transmit_external_function_invocation(
                thisfile,  # file (or module) containing the function
                "solve_model",  # function to execute
                invocation_type=InvocationType.PerScenarioInvocation,  # each scenario
                return_action_handles=True)  # collect results

        # wait for actions to complete
        results = manager.complete_actions(action_handles)
        for action_id in results:
            print(results[action_id])
