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
#  (1) Registers options for the ScenarioTreeManagerSolver and
#      BendersAlgorithm classes
#  (2) Creates a ScenarioTreeManagerSolver object (serial- or pyro-based)
#      for the farmer problem
#  (3) Creates a BendersAlgorithm object to solve the farmer problem
#  (4) Creates another BendersAlgorithm object to solve the
#      farmer problem, but uses some advanced features to interact
#      with the Benders solve process.

import os
import sys
import random

from pyomo.environ import *
from pyomo.pysp.scenariotree.manager_solver import \
    (ScenarioTreeManagerSolverClientSerial,
     ScenarioTreeManagerSolverClientPyro)
from pyomo.pysp.benders import BendersAlgorithm

# *** How to run this example using Pyro ***:
#
# To run this example using Pyro, launch the following
# command in another terminal:
#   $ mpirun -np 1 pyomo_ns -r -n localhost : \
#            -np 1 dispatch_srvr -n localhost : \
#            -np 3 scenariotreeserver --pyro-host=localhost --traceback
#
# In this shell launch:
#   $ python benders_scripting.py
# with using_pyro set to True

using_pyro = False
if using_pyro:
    manager_type = ScenarioTreeManagerSolverClientPyro
else:
    manager_type = ScenarioTreeManagerSolverClientSerial
options = manager_type.register_options()
BendersAlgorithm.register_options(options)

# To see detailed information about options
#for name in options.keys():
#    print(options.about(name))

# To see a more compact display of options
#options.display()

#
# General options for the scenario tree manager
#

# using absolute paths so we can automate testing
# of this example
examplesdir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
options.model_location = \
    os.path.join(examplesdir, "farmerWintegers", "models")
options.scenario_tree_location = \
    os.path.join(examplesdir, "farmerWintegers", "scenariodata")
options.solver = "cplex"
#options.verbose = True
if using_pyro:
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
    options.pyro_shutdown = False

# using the 'with' block will automatically call
# manager.close() and gracefully shutdown the
# scenario tree servers
with manager_type(options) as manager:
    manager.initialize()

    #
    # General options for the benders algorithm
    #

    # include one of the scenarios in the benders master problem
    options.master_include_scenarios = \
        [manager.scenario_tree.scenarios[0].name]
    # aggregate all scenarios into a single average cut
    options.multicut_level = 1

    # Using the 'with' block will automatically restore the
    # subproblems to their state before the benders initialization
    # (e.g., removing benders fixing constraints and reactivating
    # the first-stage cost in each scenario's objective). Note that
    # BendersAlgorithm sets up the scenarios for generating
    # cuts immediately upon initialization.
    with BendersAlgorithm(manager, options) as benders:
        # this must be called before solve()
        benders.build_master_problem()
        benders.solve()
        assert len(benders.cut_pool) > 0
        last_cut = benders.cut_pool[-1]

    print("\nRestarting benders algorithm")
    # Now setup and solve again, but use some more advanced
    # features
    with BendersAlgorithm(manager, options) as benders:

        # build the master problem, add the last cut
        # from the previous solve
        benders.build_master_problem()
        benders.add_cut(last_cut)

        # override the default percent_gap set
        # on the options object
        objective = benders.solve(percent_gap=100.0)
        assert objective == benders.incumbent_objective
        print("")
        print("Incumbent Objective: %s"
              % (benders.incumbent_objective))
        print("Optimality Gap:      %s %%"
              % (benders.optimality_gap*100))
        print("Iterations:          %s"
              % (benders.iterations))
        print("Resuming Benders...")
        print("")

        # override the default max_iterations set
        # on the options object
        objective = benders.solve(max_iterations=3)
        assert objective == benders.incumbent_objective
        print("")
        print("Incumbent Objective: %s"
              % (benders.incumbent_objective))
        print("Optimality Gap:      %s %%"
              % (benders.optimality_gap*100))
        print("Iterations:          %s"
              % (benders.iterations))
        print("Resuming Benders...")
        print("")

        # use the default percent_gap / max_iterations
        # set on the options object
        objective = benders.solve()
        assert objective == benders.incumbent_objective
        print("")
        print("Incumbent Objective: %s"
              % (benders.incumbent_objective))
        print("Optimality Gap:      %s %%"
              % (benders.optimality_gap*100))
        print("Iterations:          %s"
              % (benders.iterations))
        print("")

        print("Building a new master (without including any "
              "scenarios) and adding a random set of cuts "
              "before solving.")
        print("Current size of cut pool: %s"
              % (len(benders.cut_pool)))
        # save the cut pool, rebuild the master problem
        # and add a random sample of the cuts to initialize
        # the algorithm
        cut_pool = benders.cut_pool
        assert len(cut_pool) >= 5
        # override the default master_include_scenarios
        # set on the options object
        benders.build_master_problem(include_scenarios=None)
        assert len(benders.cut_pool) == 0

        random.shuffle(cut_pool)
        while len(benders.cut_pool) < 5:
            benders.add_cut(cut_pool.pop())
        benders.solve()
        print("")
        print("Incumbent Objective: %s"
              % (benders.incumbent_objective))
        print("Optimality Gap:      %s %%"
              % (benders.optimality_gap*100))
        print("Iterations:          %s"
              % (benders.iterations))
        print("Current size of cut pool: %s"
              % (len(benders.cut_pool)))
        print("")

