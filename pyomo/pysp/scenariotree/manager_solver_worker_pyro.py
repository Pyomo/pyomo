#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ("ScenarioTreeManagerSolverWorkerPyro",)

import time

from pyomo.opt import SolverFactory
from pyomo.pysp.util.configured_object import PySPConfiguredObject
from pyomo.pysp.util.config import (PySPConfigBlock,
                                    safe_declare_common_option)
from pyomo.pysp.scenariotree.manager_worker_pyro import \
    ScenarioTreeManagerWorkerPyro
from pyomo.pysp.scenariotree.manager_solver import \
    (_ScenarioTreeManagerSolverWorker,
     ScenarioTreeManagerSolver)
from six import iteritems

#
# A full implementation of the ScenarioTreeManagerSolver and
# ScenarioTreeManager interfaces designed to be used by Pyro-based
# client-side ScenarioTreeManagerSolver implementations.
#

class ScenarioTreeManagerSolverWorkerPyro(ScenarioTreeManagerWorkerPyro,
                                          _ScenarioTreeManagerSolverWorker,
                                          ScenarioTreeManagerSolver,
                                          PySPConfiguredObject):

    _declared_options = \
        PySPConfigBlock("Options declared for the "
                        "ScenarioTreeManagerSolverWorkerPyro class")

    def __init__(self, *args, **kwds):

        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            __init__(*args, **kwds)

    #
    # Abstract methods for ScenarioTreeManager:
    #

    # override what is implemented by ScenarioTreeSolverWorkerPyro
    def _init(self, *args, **kwds):
        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            _init(*args, **kwds)
        super(ScenarioTreeManagerSolverWorkerPyro, self).\
            _init_solver_worker()

    # TODO: functionality for returning suffixes
    def _solve_objects_for_client(self,
                                  object_type,
                                  objects,
                                  update_stages,
                                  ephemeral_solver_options,
                                  disable_warmstart):

        if self.get_option("verbose"):
            print("Received request to queue solves for %s" % (object_type))

        # suppress any verbose output within _solve_objects
        # as it will be repeated by the client
        self_verbose = self.get_option("verbose")
        self_output_times = self.get_option("output_times")
        setattr(self._options,
                self.get_full_option_name("verbose"),
                False)
        setattr(self._options,
                self.get_full_option_name("output_times"),
                False)
        manager_results = super(ScenarioTreeManagerSolverWorkerPyro, self).\
                          _solve_objects(object_type,
                                         objects,
                                         update_stages,
                                         ephemeral_solver_options,
                                         disable_warmstart,
                                         False, # check_status
                                         False) # async
        setattr(self._options,
                self.get_full_option_name("verbose"),
                self_verbose)
        setattr(self._options,
                self.get_full_option_name("output_times"),
                self_output_times)

        if object_type == 'bundles':
            if objects is None:
                objects = self._scenario_tree._scenario_bundle_map
        else:
            assert object_type == 'scenarios'
            if objects is None:
                objects = self._scenario_tree._scenario_map

        results = {}
        for object_name in objects:

            manager_object_results = \
                manager_results.results_for(object_name)
            # Convert enums to strings to avoid difficult
            # behavior related to certain Pyro serializer
            # settings
            manager_object_results['solver_status'] = \
                str(manager_object_results['solver_status'])
            manager_object_results['termination_condition'] = \
                str(manager_object_results['termination_condition'])
            manager_object_results['solution_status'] = \
                str(manager_object_results['solution_status'])

            if object_type == 'bundles':
                solution = {}
                for scenario_name in self._scenario_tree.\
                       get_bundle(object_name).scenario_names:
                    scenario = self._scenario_tree.get_scenario(
                        scenario_name)
                    solution[scenario_name] = \
                        scenario.copy_solution()
            else:
                scenario = self._scenario_tree.get_scenario(object_name)
                solution = scenario.copy_solution()


            results[object_name] = (manager_object_results, solution)

        return results

    def _update_fixed_variables_for_client(self, fixed_variables):

        print("Received request to update fixed statuses on "
              "scenario tree nodes")

        for node_name, node_fixed_vars in iteritems(fixed_variables):
            tree_node = self._scenario_tree.get_node(node_name)
            tree_node._fix_queue.update(node_fixed_vars)

        self.push_fix_queue_to_instances()
